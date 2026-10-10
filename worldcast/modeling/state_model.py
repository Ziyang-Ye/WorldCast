"""The state model (Sec. 3.4; App. "State model in detail"): a client's position read off its own
latents and controls.

A convolutional ``encoder`` maps each latent frame to a token, the ``action_encoder`` adds the
embedding of its controls, and the causal transformer ``trunk`` processes the sequence. The
``motion_head`` predicts the displacement ``Delta_f`` between consecutive latent frames (in units
of 64 u); the ``place_head`` classifies the frame into a map cell and a sub-cell and regresses the
offset within it (the :class:`Address`), which gives a position estimate. The closed loop combines
the two with the complementary filter of Eq. (4), whose ``A_f`` is that estimate relative to its
first value (:class:`worldcast.player_state.closed_loop.ComplementaryFilter`).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from worldcast.data.controls import STATE_MODEL_CONTROL_DIM, STATE_MODEL_SUBSTEPS
from worldcast.data.latents import LATENT_CHANNELS, LATENT_GRID
from worldcast.utils.weights import (
    indexed_count,
    leading_size,
    module_from_state,
    read_state_dict,
)

__all__ = [
    "CELL_SIZE",
    "DISPLACEMENT_UNIT_U",
    "MAPS",
    "OFFSET_BOUND",
    "STATE_MODEL_WINDOW_LATENT_FRAMES",
    "STATE_MODEL_WINDOW_STRIDE",
    "SUB_CELLS",
    "SUB_CELL_DIV",
    "SUB_CELL_SIZE",
    "ActionEncoder",
    "Address",
    "CellTable",
    "Encoder",
    "PlaceHead",
    "StateModel",
    "StateModelOutput",
    "Trunk",
    "load_cell_table",
    "load_state_model",
    "map_index",
]

#: Latent frames per state-model window (10 s) and the stride between windows (the last latent frame
#: of window w is the first of window w + 1).
STATE_MODEL_WINDOW_LATENT_FRAMES = 41
STATE_MODEL_WINDOW_STRIDE = 40
#: Engine ticks of a window: the action encoder's sequence, 16 per latent frame.
_WINDOW_TICKS = STATE_MODEL_WINDOW_LATENT_FRAMES * STATE_MODEL_SUBSTEPS
#: Unit of the motion head's displacement, u.
DISPLACEMENT_UNIT_U = 64.0
#: The maps of the place head, in the order of its map embedding and of the cell table.
MAPS = ("de_ancient", "de_dust2", "de_mirage", "de_nuke")
#: The address lattice (App. "Address"): cells of 256 x 256 x 64 u, each divided 4 x 4 x 1 into
#: sub-cells; the offset is bounded to half a sub-cell (+-32 u).
CELL_SIZE = (256.0, 256.0, 64.0)
SUB_CELL_DIV = (4, 4, 1)
SUB_CELL_SIZE = tuple(CELL_SIZE[i] / SUB_CELL_DIV[i] for i in range(3))
SUB_CELLS = SUB_CELL_DIV[0] * SUB_CELL_DIV[1] * SUB_CELL_DIV[2]
OFFSET_BOUND = tuple(f / 2.0 for f in SUB_CELL_SIZE)
#: Widths of the place head: map, cell and sub-cell embeddings, its body.
_MAP_DIM, _CELL_DIM, _SUB_CELL_DIM, _BODY_DIM = 64, 64, 16, 512
#: Channels of the early feature map (after the encoder's second block), which the place head
#: reads.
_EARLY_CHANNELS = 384


def _halved(grid: tuple[int, ...]) -> tuple[int, ...]:
    """The grid after a strided 3 x 3 convolution: half of it, rounded up."""
    return tuple((n + 1) // 2 for n in grid)


#: Grids of the feature maps: the encoder's early map ``(12, 21)``, the place head's own map of it
#: ``(6, 11)``, and the encoder's last map ``(3, 6)``, two strided blocks after the early one.
_EARLY_GRID = _halved(LATENT_GRID)
_PLACE_GRID = _halved(_EARLY_GRID)
_DEEP_GRID = _halved(_halved(_EARLY_GRID))
#: Width of the action encoder.
_ACTION_DIM = 256
#: Cell logits of other maps are masked to this value (finite: softmax stays exact).
_MASK_NEG = -1e9


class Address(NamedTuple):
    """A position on the place head's lattice: cell, sub-cell and offset (App. "Address").

    Attributes:
        cell (Tensor): ``[B, F]`` long, row of the cell table.
        sub_cell (Tensor): ``[B, F]`` long, in ``[0, 16)``.
        offset (Tensor): ``[B, F, 3]`` float32 offset from the sub-cell centre, u.
    """

    cell: torch.Tensor
    sub_cell: torch.Tensor
    offset: torch.Tensor


class StateModelOutput(NamedTuple):
    """What the two heads read off one window: the inputs of the complementary filter (Eq. (4)).

    Attributes:
        displacement (Tensor): ``[B, 41, 3]`` float32 ``Delta_f``, the motion head's displacement
            since the previous latent frame, in units of 64 u.
        place (Tensor): ``[B, 41, 3]`` float32, the place head's position estimate, u; Eq. (4)
            uses it relative to its first value, ``A_f = place_f - place_0 + p_0``.
    """

    displacement: torch.Tensor
    place: torch.Tensor


def map_index(map_name: str) -> int:
    """The index of a map in :data:`MAPS`: the ``map_id`` of :meth:`StateModel.forward`."""
    if map_name not in MAPS:
        raise ValueError(f"the state model knows the maps {MAPS}, not {map_name!r}")
    return MAPS.index(map_name)


def _sub_cell_centres() -> np.ndarray:
    t = np.zeros((SUB_CELLS, 3), np.float64)
    for fy in range(SUB_CELL_DIV[1]):
        for fx in range(SUB_CELL_DIV[0]):
            t[fy * SUB_CELL_DIV[0] + fx] = (
                (fx + 0.5) * SUB_CELL_SIZE[0],
                (fy + 0.5) * SUB_CELL_SIZE[1],
                0.5 * SUB_CELL_SIZE[2],
            )
    return t


@dataclass(frozen=True, eq=False)
class CellTable:
    """The place head's cell table (:func:`load_cell_table` of
    ``configs/state_model/cells.json``): the occupied cells of every map on the address lattice.

    Attributes:
        cell_ijk (np.ndarray): ``[N, 3]`` int64 lattice index of every occupied cell.
        cell_map (np.ndarray): ``[N]`` int64 map index of every cell.
    """

    cell_ijk: np.ndarray
    cell_map: np.ndarray

    @property
    def num_cells(self) -> int:
        return int(self.cell_ijk.shape[0])

    def cell_origin(self) -> torch.Tensor:
        """``[N, 3]`` float32: the corner of every cell, u."""
        return torch.tensor(self.cell_ijk * np.asarray(CELL_SIZE), dtype=torch.float32)

    def cell_mask(self) -> torch.Tensor:
        """``[maps, N]`` bool: the cells of each map."""
        mask = np.zeros((len(MAPS), self.num_cells), bool)
        mask[self.cell_map, np.arange(self.num_cells)] = True
        return torch.tensor(mask)


def load_cell_table(cells_json: str | Path) -> CellTable:
    """The cell table of a JSON file (``configs/state_model/cells.json``); raises if the file is
    not one, or was built on another lattice."""
    cells = json.loads(Path(cells_json).read_text())
    keys = ("cell_size_u", "sub_cells", "map_order", "maps")
    if not isinstance(cells, dict) or any(key not in cells for key in keys):
        raise ValueError(f"{cells_json}: not a cell table (a JSON object with {keys})")
    if tuple(cells["cell_size_u"]) != CELL_SIZE or tuple(cells["sub_cells"]) != SUB_CELL_DIV:
        raise ValueError(f"{cells_json}: built on another cell lattice")
    if list(cells["map_order"]) != list(MAPS):
        raise ValueError(f"{cells_json}: map order {cells['map_order']}, expected {MAPS}")
    ijk = [np.asarray(cells["maps"][m]["ijk"], np.int64).reshape(-1, 3) for m in MAPS]
    return CellTable(
        cell_ijk=np.concatenate(ijk, 0),
        cell_map=np.concatenate([np.full(len(a), i, np.int64) for i, a in enumerate(ijk)]),
    )


def _causal_mask(n: int) -> torch.Tensor:
    return torch.triu(torch.full((n, n), float("-inf")), 1)


def _conv_block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


def _causal_transformer(dim: int, heads: int, ffn: int, layers: int) -> nn.TransformerEncoder:
    """Pre-norm transformer layers without dropout; causal through the mask they are called with."""
    layer = nn.TransformerEncoderLayer(
        dim, heads, ffn, batch_first=True, norm_first=True, dropout=0.0
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Encoder(nn.Module):
    """Five convolutional blocks (three strided) from 48 to 512 channels, then a linear projection:
    one latent frame -> one token.

    Args:
        dim (int): token width.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _conv_block(LATENT_CHANNELS, 256, 2),
                _conv_block(256, _EARLY_CHANNELS, 1),
                _conv_block(_EARLY_CHANNELS, 512, 2),
                _conv_block(512, 512, 1),
                _conv_block(512, 512, 2),
            ]
        )
        self.proj = nn.Linear(512 * _DEEP_GRID[0] * _DEEP_GRID[1], dim)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode latent frames.

        Args:
            latents (Tensor): ``[N, 48, 24, 42]`` float32 latent frames.

        Returns:
            tuple[Tensor, Tensor]: tokens ``[N, dim]`` and the early feature map
            ``[N, 384, 12, 21]`` (after the second block), which the place head reads.
        """
        early = self.blocks[1](self.blocks[0](latents))
        deep = self.blocks[4](self.blocks[3](self.blocks[2](early)))
        return self.proj(deep.flatten(1)), early


class Trunk(nn.Module):
    """The causal transformer over a window's tokens (pre-norm, 16 heads, 4x feed-forward), with a
    learned position embedding.

    Args:
        dim (int): token width.
        layers (int): transformer layers.
    """

    def __init__(self, dim: int, layers: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, STATE_MODEL_WINDOW_LATENT_FRAMES, dim))
        self.transformer = _causal_transformer(dim, 16, dim * 4, layers)
        self.register_buffer("mask", _causal_mask(STATE_MODEL_WINDOW_LATENT_FRAMES))

    def forward(self, tokens: torch.Tensor, control_embedding: torch.Tensor) -> torch.Tensor:
        """``[B, 41, dim]`` tokens and control embeddings -> ``[B, 41, dim]`` trunk outputs."""
        return self.transformer(tokens + self.pos + control_embedding, mask=self.mask)


class ActionEncoder(nn.Module):
    """The sixteen control values of every engine tick, projected to 256 dimensions and encoded by a
    four-layer causal transformer; each latent frame's embedding is a projection of its last tick
    and of the mean over its 16 ticks.

    Args:
        dim (int): trunk width.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(STATE_MODEL_CONTROL_DIM, _ACTION_DIM)
        self.pos = nn.Parameter(torch.zeros(1, _WINDOW_TICKS, _ACTION_DIM))
        self.transformer = _causal_transformer(_ACTION_DIM, 8, 4 * _ACTION_DIM, 4)
        self.proj_out = nn.Linear(2 * _ACTION_DIM, dim)
        self.register_buffer("mask", _causal_mask(_WINDOW_TICKS))

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        """Encode a window's controls.

        Args:
            controls (Tensor): ``[B, 41, 16, 16]`` float32
                (:func:`worldcast.data.controls.state_model_controls`).

        Returns:
            Tensor: ``[B, 41, dim]`` float32, added to the frame tokens.
        """
        b = controls.shape[0]
        a = self.proj_in(controls.reshape(b, _WINDOW_TICKS, STATE_MODEL_CONTROL_DIM)) + self.pos
        a = self.transformer(a, mask=self.mask)
        a = a.view(b, STATE_MODEL_WINDOW_LATENT_FRAMES, STATE_MODEL_SUBSTEPS, _ACTION_DIM)
        return self.proj_out(torch.cat([a[:, :, -1], a.mean(2)], -1))


class PlaceHead(nn.Module):
    """Where on the map a latent frame was taken: the trunk tokens, the pooled and the unpooled
    early feature map and a map embedding give a cell, then a sub-cell conditioned on the cell's
    embedding, then the offset through a tanh (App. "Address").

    Args:
        table (CellTable): the cell table.
        dim (int): trunk width.
    """

    def __init__(self, table: CellTable, dim: int) -> None:
        super().__init__()
        num_cells = table.num_cells
        self.register_buffer("cell_origin", table.cell_origin())
        self.register_buffer("cell_mask", table.cell_mask())
        self.register_buffer(
            "sub_cell_centre", torch.tensor(_sub_cell_centres(), dtype=torch.float32)
        )
        self.register_buffer("offset_bound", torch.tensor(OFFSET_BOUND, dtype=torch.float32))
        self.map_embedding = nn.Embedding(len(MAPS), _MAP_DIM)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(_EARLY_CHANNELS, 192, 3, 2, 1),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.grid_features = 96 * _PLACE_GRID[0] * _PLACE_GRID[1]
        self.body = nn.Sequential(
            nn.Linear(dim + _EARLY_CHANNELS + self.grid_features + _MAP_DIM, 1024),
            nn.SiLU(),
            nn.Linear(1024, _BODY_DIM),
            nn.SiLU(),
        )
        self.cell_classifier = nn.Linear(_BODY_DIM, num_cells)
        self.cell_embedding = nn.Embedding(num_cells, _CELL_DIM)
        self.sub_cell_embedding = nn.Embedding(SUB_CELLS, _SUB_CELL_DIM)
        self.sub_cell_classifier = nn.Sequential(
            nn.Linear(_BODY_DIM + _CELL_DIM, 512), nn.SiLU(), nn.Linear(512, SUB_CELLS)
        )
        self.offset_regressor = nn.Sequential(
            nn.Linear(_BODY_DIM + _CELL_DIM + _SUB_CELL_DIM, 512), nn.SiLU(), nn.Linear(512, 3)
        )

    def address(
        self, trunk_tokens: torch.Tensor, early: torch.Tensor, map_id: torch.Tensor
    ) -> Address:
        """The address of every latent frame of a window.

        Args:
            trunk_tokens (Tensor): ``[B, 41, dim]`` trunk outputs.
            early (Tensor): ``[B 41, 384, 12, 21]`` the encoder's early feature maps.
            map_id (Tensor): ``[B]`` long map index (:func:`map_index`); cells of other maps are
                masked out.

        Returns:
            Address: per latent frame.
        """
        b, f = trunk_tokens.shape[:2]
        maps = map_id.reshape(-1).long()
        pooled = early.mean((-2, -1)).view(b, f, _EARLY_CHANNELS)
        grid = self.grid_conv(early).flatten(1).view(b, f, self.grid_features)
        embedding = self.map_embedding(maps).view(b, 1, _MAP_DIM).expand(b, f, _MAP_DIM)
        body = self.body(torch.cat([trunk_tokens, pooled, grid, embedding], -1))
        logits = self.cell_classifier(body).masked_fill(
            ~self.cell_mask[maps][:, None, :], _MASK_NEG
        )
        # the arg max of the softmax, not of the logits, as trained (ties included)
        cell = torch.softmax(logits, -1).max(-1).indices
        cell_embedding = self.cell_embedding(cell)
        sub_cell = self.sub_cell_classifier(torch.cat([body, cell_embedding], -1)).argmax(-1)
        offset = (
            torch.tanh(
                self.offset_regressor(
                    torch.cat([body, cell_embedding, self.sub_cell_embedding(sub_cell)], -1)
                )
            )
            * self.offset_bound
        )
        return Address(cell=cell, sub_cell=sub_cell, offset=offset)

    def forward(
        self, trunk_tokens: torch.Tensor, early: torch.Tensor, map_id: torch.Tensor
    ) -> torch.Tensor:
        """The position that :meth:`address` names: the cell's corner, plus the sub-cell's
        centre, plus the offset.

        Args:
            trunk_tokens (Tensor): ``[B, 41, dim]`` trunk outputs.
            early (Tensor): ``[B 41, 384, 12, 21]`` the encoder's early feature maps.
            map_id (Tensor): ``[B]`` long map index (:func:`map_index`).

        Returns:
            Tensor: ``[B, 41, 3]`` float32 positions, u.
        """
        address = self.address(trunk_tokens, early, map_id)
        return (
            self.cell_origin[address.cell] + self.sub_cell_centre[address.sub_cell] + address.offset
        )


class StateModel(nn.Module):
    """Encoder, trunk, action encoder, motion head and place head (App. "State model in detail").

    Args:
        table (CellTable): the place head's cell table.
        dim (int): token width, a multiple of the trunk's 16 heads.
        layers (int): trunk depth.
    """

    def __init__(self, table: CellTable, *, dim: int = 1280, layers: int = 16) -> None:
        super().__init__()
        # built in this order so that a fresh model draws its initial weights as trained
        self.encoder = Encoder(dim)
        self.trunk = Trunk(dim, layers)
        self.motion_head = nn.Linear(dim, 3)
        self.action_encoder = ActionEncoder(dim)
        self.place_head = PlaceHead(table, dim)

    def forward(
        self, latents: torch.Tensor, controls: torch.Tensor, map_id: torch.Tensor
    ) -> StateModelOutput:
        """Read one window.

        Args:
            latents (Tensor): ``[B, 41, 48, 24, 42]`` float32 latent frames.
            controls (Tensor): ``[B, 41, 16, 16]`` float32: per latent frame, the sixteen control
                values of each of its 16 engine ticks
                (:func:`worldcast.data.controls.state_model_controls`).
            map_id (Tensor): ``[B]`` long map index (:func:`map_index`).

        Returns:
            StateModelOutput: the motion head's displacement and the place head's estimate.
        """
        window = (STATE_MODEL_WINDOW_LATENT_FRAMES, LATENT_CHANNELS, *LATENT_GRID)
        if latents.ndim != 5 or tuple(latents.shape[1:]) != window:
            raise ValueError(
                f"the state model reads windows [B, {', '.join(map(str, window))}], got"
                f" {tuple(latents.shape)}"
            )
        ticks = (
            latents.shape[0],
            STATE_MODEL_WINDOW_LATENT_FRAMES,
            STATE_MODEL_SUBSTEPS,
            STATE_MODEL_CONTROL_DIM,
        )
        if tuple(controls.shape) != ticks:
            raise ValueError(f"controls must be {list(ticks)}, got {tuple(controls.shape)}")
        if not torch.is_tensor(map_id) or map_id.numel() != latents.shape[0]:
            raise ValueError("map_id must be a [B] tensor of map indices (map_index)")
        if map_id.is_floating_point() or not bool(((map_id >= 0) & (map_id < len(MAPS))).all()):
            raise ValueError(f"map_id must hold integer indices of {MAPS}, got {map_id.tolist()}")
        b, f = latents.shape[:2]
        tokens, early = self.encoder(latents.flatten(0, 1))
        trunk_tokens = self.trunk(tokens.view(b, f, -1), self.action_encoder(controls))
        return StateModelOutput(
            displacement=self.motion_head(trunk_tokens),
            place=self.place_head(trunk_tokens, early, map_id),
        )


def load_state_model(
    checkpoint: str | Path, table: CellTable, *, device: torch.device | str = "cpu"
) -> StateModel:
    """The state model of a weight file, at the file's size, in eval mode and without gradients.

    Args:
        checkpoint (str | Path): the released ``state_model.safetensors``, or a torch state dict.
        table (CellTable): the place head's cell table; a table other than the one the weights
            were trained on (their ``cell_origin`` and ``cell_mask``) is refused.
        device (torch.device | str): where to put the model.
    """
    state = read_state_dict(checkpoint)
    dim = leading_size(state, "encoder.proj.weight")
    layers = indexed_count(state, "trunk.transformer.layers.")
    for name, value in (("cell_origin", table.cell_origin()), ("cell_mask", table.cell_mask())):
        if not torch.equal(state[f"place_head.{name}"], value):
            raise ValueError(f"{checkpoint} was trained on another cell table ({name} differs)")
    model = module_from_state(lambda: StateModel(table, dim=dim, layers=layers), state)
    return model.to(device).eval().requires_grad_(False)
