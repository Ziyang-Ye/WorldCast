"""The state model (Sec. 3.4, App. "State model"): a client's position read off its own latents.

Trunk (per-frame conv encoder + 16-layer causal transformer, plus the control encoder), motion head
(per-latent displacement ``inc``, in units of 64 u) and place head (coarse cell, sub-cell and offset
-> ``A_world``). The closed loop runs it and fuses the two heads with Eq. (6)
(``worldcast.player_state.closed_loop``).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from worldcast.data.actions import STATE_MODEL_CONTROL_DIM, STATE_MODEL_SUBSTEPS

__all__ = [
    "WINDOW_LATENTS",
    "WINDOW_STRIDE",
    "POSITION_UNIT_U",
    "MAPS",
    "StateTables",
    "StateModel",
    "load_state_model",
    "map_id_of",
]

#: Latent frames per state-model window (10 s) and the stride between windows (the last latent frame
#: of window w is the first of window w + 1).
WINDOW_LATENTS = 41
WINDOW_STRIDE = 40
#: Unit of the motion head's displacement, world units.
POSITION_UNIT_U = 64.0
MAPS = ("de_ancient", "de_dust2", "de_mirage", "de_nuke")
#: Coarse cells of the place head (world units), split 4 x 4 x 1 into sub-cells; the offset is
#: bounded to half a sub-cell.
CELL_SIZE = (256.0, 256.0, 64.0)
FINE_DIV = (4, 4, 1)
FINE_SIZE = tuple(CELL_SIZE[i] / FINE_DIV[i] for i in range(3))
NFINE = FINE_DIV[0] * FINE_DIV[1] * FINE_DIV[2]
OFF_BOUND = tuple(f / 2.0 for f in FINE_SIZE)
MAP_DIM, CELL_DIM, FINE_DIM, BODY_DIM, E2_CHANNELS = 64, 64, 16, 512, 384
E2_GRID = (12, 21)
#: Coarse logits of other maps are masked to this value (finite: softmax stays exact).
MASK_NEG = -1e9
#: The learned gate's elapsed-time embedding (in the checkpoint; the closed loop uses Eq. (6)).
GATE_TIME_BUCKETS, GATE_TIME_DIM = 31, 16


def map_id_of(media_id: str) -> int:
    """Map index of a media id ``<match>-<map>-r<round>-p<slot>``."""
    parts = str(media_id).split("-")
    if len(parts) != 4:
        raise ValueError(f"unexpected media id {media_id!r}")
    return MAPS.index(parts[1])


def _pack_cell_key(map_id: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    a, m = np.asarray(ijk, np.int64), np.asarray(map_id, np.int64)
    return (
        (m << 40)
        | ((a[..., 0] + (1 << 11)) << 28)
        | ((a[..., 1] + (1 << 11)) << 16)
        | (a[..., 2] + (1 << 15))
    )


def _fine_centres() -> np.ndarray:
    t = np.zeros((NFINE, 3), np.float64)
    for fy in range(FINE_DIV[1]):
        for fx in range(FINE_DIV[0]):
            t[fy * FINE_DIV[0] + fx] = (
                (fx + 0.5) * FINE_SIZE[0],
                (fy + 0.5) * FINE_SIZE[1],
                0.5 * FINE_SIZE[2],
            )
    return t


@dataclass(frozen=True)
class StateTables:
    """The frozen tables the place head is built on (``configs/state_model/``).

    Attributes:
        cell_ijk (np.ndarray): ``[N, 3]`` int64 lattice index of every occupied cell.
        cell_map (np.ndarray): ``[N]`` int64 map index of every cell.
        map_mean (np.ndarray): ``[4, 3]`` float64 per-map position mean, world units.
        map_std (np.ndarray): ``[4, 3]`` float64 per-map position standard deviation, world units.
    """

    cell_ijk: np.ndarray
    cell_map: np.ndarray
    map_mean: np.ndarray
    map_std: np.ndarray

    @classmethod
    def load(cls, cells_json: str | Path, map_norm_json: str | Path) -> "StateTables":
        """Read the cell table and the map normalisation; raises if built on another lattice."""
        cells = json.loads(Path(cells_json).read_text())
        if tuple(cells["cell_size_u"]) != CELL_SIZE or tuple(cells["fine_div"]) != FINE_DIV:
            raise ValueError(f"{cells_json}: built on another cell lattice")
        if list(cells["map_order"]) != list(MAPS):
            raise ValueError(f"{cells_json}: map order {cells['map_order']}")
        ijk = [np.asarray(cells["maps"][m]["ijk"], np.int64).reshape(-1, 3) for m in MAPS]
        norm = json.loads(Path(map_norm_json).read_text())["maps"]
        return cls(
            cell_ijk=np.concatenate(ijk, 0),
            cell_map=np.concatenate([np.full(len(a), i, np.int64) for i, a in enumerate(ijk)]),
            map_mean=np.asarray([norm[m]["mean_xyz"] for m in MAPS], np.float64),
            map_std=np.asarray([norm[m]["std_xyz"] for m in MAPS], np.float64),
        )

    @property
    def n_cells(self) -> int:
        return int(self.cell_ijk.shape[0])


def _causal_mask(n: int) -> torch.Tensor:
    return torch.triu(torch.full((n, n), float("-inf")), 1)


def _conv_block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


class StateModel(nn.Module):
    """Trunk, control encoder, motion head and place head (350M parameters at the defaults).

    The learned gate (``mgat``, ``meta``, ``tbuck``) is part of the checkpoint and is loaded, but
    never run: the closed loop fuses with Eq. (6). Parameter names are the checkpoint's.

    Args:
        tables (StateTables): the place head's cell table and map normalisation.
        dim (int): trunk width.
        layers (int): trunk depth.
    """

    def __init__(self, tables: StateTables, *, dim: int = 1280, layers: int = 16) -> None:
        super().__init__()
        n_frames, n_ticks = WINDOW_LATENTS, WINDOW_LATENTS * STATE_MODEL_SUBSTEPS
        self.e1, self.e2 = _conv_block(48, 256, 2), _conv_block(256, 384, 1)
        self.e3, self.e4, self.e5 = (
            _conv_block(384, 512, 2),
            _conv_block(512, 512, 1),
            _conv_block(512, 512, 2),
        )
        self.proj = nn.Linear(512 * 3 * 6, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_frames, dim))
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                dim, 16, dim * 4, batch_first=True, norm_first=True, dropout=0.0
            ),
            layers,
        )
        self.head = nn.Linear(dim, 3)
        self.register_buffer("m", _causal_mask(n_frames))
        # control encoder
        self.tick_in = nn.Linear(STATE_MODEL_CONTROL_DIM, 256)
        self.tick_pos = nn.Parameter(torch.zeros(1, n_ticks, 256))
        self.ticktr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                256, 8, 1024, batch_first=True, norm_first=True, dropout=0.0
            ),
            4,
        )
        self.act_out = nn.Linear(512, dim)
        self.register_buffer("tm", _causal_mask(n_ticks))
        # place head
        n_cells = tables.n_cells
        key = _pack_cell_key(tables.cell_map, tables.cell_ijk)
        order = np.argsort(key)
        self.register_buffer("map_mu", torch.tensor(tables.map_mean, dtype=torch.float32))
        self.register_buffer("map_sd", torch.tensor(tables.map_std, dtype=torch.float32))
        self.register_buffer(
            "cell_min", torch.tensor(tables.cell_ijk * np.asarray(CELL_SIZE), dtype=torch.float32)
        )
        self.register_buffer("cell_map", torch.tensor(tables.cell_map, dtype=torch.long))
        mask = np.zeros((len(MAPS), n_cells), bool)
        mask[tables.cell_map, np.arange(n_cells)] = True
        self.register_buffer("cell_mask", torch.tensor(mask))
        self.register_buffer("cell_key_sorted", torch.tensor(key[order], dtype=torch.long))
        self.register_buffer("cell_key_order", torch.tensor(order, dtype=torch.long))
        self.register_buffer("fine_local", torch.tensor(_fine_centres(), dtype=torch.float32))
        self.register_buffer("cell_size", torch.tensor(CELL_SIZE, dtype=torch.float32))
        self.register_buffer("off_bound", torch.tensor(OFF_BOUND, dtype=torch.float32))
        self.memb = nn.Embedding(len(MAPS), MAP_DIM)
        self.mcnv = nn.Sequential(
            nn.Conv2d(E2_CHANNELS, 192, 3, 2, 1),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.cfeat = 96 * ((E2_GRID[0] - 1) // 2 + 1) * ((E2_GRID[1] - 1) // 2 + 1)
        self.mbody = nn.Sequential(
            nn.Linear(dim + E2_CHANNELS + self.cfeat + MAP_DIM, 1024),
            nn.SiLU(),
            nn.Linear(1024, BODY_DIM),
            nn.SiLU(),
        )
        self.mcls = nn.Linear(BODY_DIM, n_cells)
        self.cemb = nn.Embedding(n_cells, CELL_DIM)
        self.femb = nn.Embedding(NFINE, FINE_DIM)
        self.mfin = nn.Sequential(
            nn.Linear(BODY_DIM + CELL_DIM, 512), nn.SiLU(), nn.Linear(512, NFINE)
        )
        self.moff = nn.Sequential(
            nn.Linear(BODY_DIM + CELL_DIM + FINE_DIM, 512), nn.SiLU(), nn.Linear(512, 3)
        )
        # the learned gate (loaded, never run)
        self.mgat = nn.Sequential(
            nn.Linear(dim + 3 + MAP_DIM + 1 + 1 + GATE_TIME_DIM, 512), nn.SiLU(), nn.Linear(512, 3)
        )
        self.meta = nn.Parameter(torch.zeros(3))
        self.tbuck = nn.Embedding(GATE_TIME_BUCKETS, GATE_TIME_DIM)

    def control_encoder(self, controls: torch.Tensor) -> torch.Tensor:
        """Encode a window's controls.

        Args:
            controls (torch.Tensor): ``[B, 41, 16, 16]`` float32
                (:func:`worldcast.data.actions.state_model_controls`).

        Returns:
            torch.Tensor: ``[B, 41, dim]`` float32, added to the frame tokens.
        """
        b = controls.shape[0]
        n_ticks = WINDOW_LATENTS * STATE_MODEL_SUBSTEPS
        a = self.tick_in(controls.reshape(b, n_ticks, STATE_MODEL_CONTROL_DIM)) + self.tick_pos
        a = self.ticktr(a, mask=self.tm).view(b, WINDOW_LATENTS, STATE_MODEL_SUBSTEPS, 256)
        return self.act_out(torch.cat([a[:, :, -1], a.mean(2)], -1))

    def forward(
        self, latents: torch.Tensor, controls: torch.Tensor, map_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one window.

        Args:
            latents (torch.Tensor): ``[B, 41, 48, 24, 42]`` float32 latent frames.
            controls (torch.Tensor): ``[B, 41, 16, 16]`` float32 controls.
            map_id (torch.Tensor): ``[B]`` long map index (:func:`map_id_of`).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``inc`` ``[B, 41, 3]`` (displacement per latent
            frame, units of 64 u) and ``A_world`` ``[B, 41, 3]`` (place estimate, world units).
        """
        b, f = latents.shape[:2]
        h2 = self.e2(self.e1(latents.flatten(0, 1)))
        h = self.proj(self.e5(self.e4(self.e3(h2))).flatten(1)).view(b, f, -1) + self.pos
        z = self.tr(h + self.control_encoder(controls), mask=self.m)
        inc = self.head(z)
        mi = map_id.reshape(-1).long()
        pooled = h2.mean((-2, -1)).view(b, f, E2_CHANNELS)
        grid = self.mcnv(h2).flatten(1).view(b, f, self.cfeat)
        emb = self.memb(mi).view(b, 1, MAP_DIM).expand(b, f, MAP_DIM)
        body = self.mbody(torch.cat([z, pooled, grid, emb], -1))
        logits = self.mcls(body).masked_fill(~self.cell_mask[mi][:, None, :], MASK_NEG)
        # the arg max of the softmax, not of the logits, as trained (ties included)
        cell = torch.softmax(logits, -1).max(-1).indices
        fine = self.mfin(torch.cat([body, self.cemb(cell)], -1)).argmax(-1)
        offset = (
            torch.tanh(self.moff(torch.cat([body, self.cemb(cell), self.femb(fine)], -1)))
            * self.off_bound
        )
        return inc, self.cell_min[cell] + self.fine_local[fine] + offset


def load_state_model(checkpoint: str | Path, tables: StateTables, *, device) -> StateModel:
    """Build the model at the checkpoint's size and load it (strict).

    Args:
        checkpoint (str | Path): a plain state dict.
        tables (StateTables): the place head's tables.
        device (torch.device | str): where to put the model.

    Returns:
        StateModel: in eval mode.
    """
    state = torch.load(str(checkpoint), map_location="cpu")
    layers = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("tr.layers."))
    model = StateModel(tables, dim=int(state["proj.weight"].shape[0]), layers=layers)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
