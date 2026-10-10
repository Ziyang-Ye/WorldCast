"""A tiny training world for the training, data and evaluation tests: model dimensions, synthetic
loader items and window batches, datasets, prompt files and checkpoints."""

import copy
import math
from pathlib import Path
from typing import Any

import torch

from tests.modeling.support import TINY as TINY_MODEL
from tests.modeling.support import (
    TINY_CONTROLS,
    TINY_OBSERVER_SIGNALS_HIDDEN,
    random_c2w,
    randomize_,
)
from worldcast.config.training import TrainConfig
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import VIDEO_FRAMES_PER_LATENT
from worldcast.modeling.build import EMA_KEY

#: The tests' tiny generator with 2 layers and the paper's latent layout (48 channels, 24 x 42, 41
#: latents).
TINY = {**TINY_MODEL, "in_dim": 48, "out_dim": 48, "num_layers": 2}
TINY_HEAD = dict(dit_block=1, hidden=16)
TINY_DIMS = {
    **TINY,
    "controls.hidden": TINY_CONTROLS["hidden"],
    "controls.adaln_rank": TINY_CONTROLS["adaln_rank"],
    "observer_signals.hidden": TINY_OBSERVER_SIGNALS_HIDDEN,
    "visibility_probe.dit_block": TINY_HEAD["dit_block"],
    "visibility_probe.hidden": TINY_HEAD["hidden"],
}


P, F = 10, 41
T = 1 + 4 * (F - 1)
LATENT_H, LATENT_W = 24, 42


# ============================================================================================= data
class FixedBatches:
    """``next_batch()`` serving deep copies of a list of collated batches, in order (the loader
    stream's role)."""

    def __init__(self, batches: list[dict[str, Any]]):
        self.batches = batches
        self.index = 0
        self.epoch = 0
        self.sample_offset = 0

    def next_batch(self):
        batch = copy.deepcopy(self.batches[self.index % len(self.batches)])
        self.index += 1
        self.sample_offset += 1
        return batch

    def state_dict(self):
        return {"index": self.index}

    def load_state_dict(self, state):
        self.index = int(state["index"])


def make_item(seed: int, *, target_start: int | None = None) -> dict[str, Any]:
    """One loader item in the release format (``BucketWindows.__getitem__``; no batch axis),
    synthetic but shaped like the paper's: 41 latents of 48 x 24 x 42, 161 video frames, ten players
    (two teams of five) placed in and out of the client's view, one player dying mid-window,
    row-wise ``visible & valid`` labels, observer signals; with ``target_start`` also the memory
    frames' inputs (:func:`_memory_frame_inputs`)."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    def uni(*shape):
        return torch.rand(*shape, generator=g)

    client = int(torch.randint(0, P, (1,), generator=g))
    yaw0 = float(uni(1) * 360 - 180)
    states = torch.zeros(P, T, 6)
    client_xy = torch.cumsum(rnd(T, 2) * 3.0, dim=0) + rnd(1, 2) * 200.0
    for p in range(P):
        if p == client:
            states[p, :, 0:2] = client_xy
            states[p, :, 3] = yaw0
            states[p, :, 4] = float(uni(1) * 10 - 5)
        else:
            bearing = math.radians(yaw0 + float(rnd(1)) * (35.0 if p % 4 else 120.0))
            dist = 60.0 + 900.0 * float(uni(1)) + torch.cumsum(rnd(T) * 2.0, dim=0)
            states[p, :, 0] = client_xy[:, 0] + math.cos(bearing) * dist
            states[p, :, 1] = client_xy[:, 1] + math.sin(bearing) * dist
            states[p, :, 3] = uni(1) * 360 - 180 + torch.cumsum(rnd(T), dim=0)
            states[p, :, 4] = rnd(1) * 5
        states[p, :, 2] = rnd(1) * 10 + torch.cumsum(rnd(T) * 0.3, dim=0)
        states[p, :, 5] = 1.0
    dying = (client + 3) % P
    states[dying, T // 2 :, 5] = 0.0
    substeps = torch.zeros(P, T, 4, 13)
    substeps[..., :11] = (uni(P, T, 4, 11) < 0.25).float()
    substeps[..., 11] = rnd(P, T, 4) * 0.05
    substeps[..., 12] = rnd(P, T, 4) * 0.2
    valid = uni(P, T, 4) < 0.95
    vis_valid = uni(P, T) < 0.9
    visibility = ((uni(P, T) < 0.7) & vis_valid).float()
    item = {
        "latents": rnd(F, 48, LATENT_H, LATENT_W) * 0.5,
        "buttons": (uni(T, 11) < 0.25).float(),
        "view_deltas": rnd(T, 2) * 0.2,
        "weapon": torch.randint(0, 52, (T,), generator=g),
        "player_states": states,
    }
    rnd(P, T, 2)  # a draw the recorded references of the recipes' tests count on
    item |= {
        "player_weapon_ids": torch.randint(0, 52, (P, T), generator=g),
        "player_control_substeps": substeps,
        "player_control_substep_valid": valid,
        "player_team_ids": torch.tensor([2] * 5 + [3] * 5),
        "client_slot": torch.tensor(client),
        "client_visibility": visibility,
        "client_visibility_valid": vis_valid,
        **{key: torch.randint(0, 2, (F,), generator=g) for key in OBSERVER_SIGNAL_KEYS},
        "metadata": {"media_id": f"synthetic-{seed}", "start_frame": 0, "dataset_index": 0},
    }
    item["obs_scope_level"] = torch.randint(0, 3, (F,), generator=g)
    if target_start is not None:
        item.update(_memory_frame_inputs(g, seed, target_start, client_xy))
    return item


def _memory_frame_inputs(
    g: torch.Generator, seed: int, target_start: int, client_xy: torch.Tensor
) -> dict[str, Any]:
    """The memory frames' inputs of :func:`worldcast.data.memory_frames.memory_frame_item` for the
    target block at ``target_start``: a 4-latent teammate block near the client, its cameras and
    controls, the window's cameras and m_k (covered inside unseen)."""

    def rnd(*shape):
        return torch.randn(*shape, generator=g)

    def uni(*shape):
        return torch.rand(*shape, generator=g)

    unseen = uni(4, 12, 21) > 0.4
    memory_mask = unseen & (uni(4, 12, 21) > 0.5)
    inputs = {
        "window_target_start": torch.tensor(int(target_start)),
        "window_c2w": random_c2w(1, F, seed=seed + 7)[0],
        "window_tans": uni(F, 2) * 0.2 + torch.tensor([1.33, 0.75]),
        "memory_frames_latents": rnd(4, 48, LATENT_H, LATENT_W) * 0.5,
        **{
            "memory_frames_" + key: torch.randint(0, 2, (4,), generator=g)
            for key in OBSERVER_SIGNAL_KEYS
        },
        "memory_frames_buttons": (uni(16, 11) < 0.25).float(),
        "memory_frames_view_deltas": rnd(16, 2) * 0.2,
        "memory_frames_weapon_ids": torch.randint(0, 52, (16,), generator=g),
        "memory_frames_states": torch.cat(
            [
                client_xy[:16] + rnd(16, 2) * 50,
                rnd(16, 1) * 10,
                uni(16, 1) * 360 - 180,
                rnd(16, 1) * 5,
                torch.ones(16, 1),
            ],
            dim=-1,
        ),
    }
    rnd(16, 2)  # as in make_item: a draw the recorded references count on
    inputs |= {
        "memory_frames_control_substeps": torch.cat(
            [(uni(16, 4, 11) < 0.25).float(), rnd(16, 4, 2) * 0.1], dim=-1
        ),
        "memory_frames_control_substep_valid": uni(16, 4) < 0.95,
        "memory_frames_c2w": random_c2w(1, 4, seed=seed + 11)[0],
        "memory_frames_tans": uni(4, 2) * 0.2 + torch.tensor([1.33, 0.75]),
        "window_memory_mask": memory_mask,
    }
    return inputs


# ======================================================================================= runs
#: Tiny dimensions of a run built from its config (``build_trainer_parts``): the prompt embedding is
#: ``[1, 512, 24]``, so the text length is the paper's 512.
RUN_DIMS = {**TINY_DIMS, "text_len": 512}


class ItemDataset(torch.utils.data.Dataset):
    """Synthetic loader items (``make_item``) with sampling weights, for ``ResumableDataStream``."""

    def __init__(self, n: int, *, target_start: int | None = None, seed: int = 100):
        self.items = [make_item(seed + i, target_start=target_start) for i in range(n)]
        for i, item in enumerate(self.items):
            item["metadata"] = dict(item["metadata"], dataset_index=i)

    def __len__(self):
        return len(self.items)

    @property
    def sample_weights(self):
        return [1.0 + 0.1 * i for i in range(len(self.items))]

    def __getitem__(self, index):
        return copy.deepcopy(self.items[index])


def write_prompt_embedding(path: Path, text_dim: int = 24, seed: int = 9) -> Path:
    """A prompt-embedding file of the fixed prompt (``[1, 512, text_dim]``, 10 real tokens)."""
    from worldcast.modeling.wan22.text_encoder import (
        FIXED_PROMPT_TOKEN_IDS,
        save_prompt_embedding,
    )

    n = len(FIXED_PROMPT_TOKEN_IDS)
    embeds = torch.zeros(1, 512, text_dim)
    embeds[:, :n] = torch.randn(1, n, text_dim, generator=torch.Generator().manual_seed(seed))
    save_prompt_embedding(embeds, path)
    return path


def write_init_checkpoint(
    path: Path, cfg: TrainConfig, *, drop: tuple[str, ...] = (), key: str = EMA_KEY, seed: int = 4
) -> Path:
    """A release-layout checkpoint of a randomised tiny generator of ``cfg`` without the modules in
    ``drop`` (to stand for the previous stage's checkpoint)."""
    from worldcast.engine.checkpoint.training import release_state
    from worldcast.engine.stage import generator_config
    from worldcast.modeling.wan22.model import WorldCastGenerator

    torch.manual_seed(seed)
    model = randomize_(WorldCastGenerator(generator_config(cfg)), seed)
    state = {
        "generator." + k: v for k, v in model.state_dict().items() if not k.startswith(tuple(drop))
    }
    torch.save({key: release_state(state), "step": 0}, str(path))
    return path


# =========================================================================== window batches
TEAM_IDS = (2, 3)


def make_window_batch(
    seed: int,
    *,
    batch: int = 2,
    players: int = 7,
    latents: int = 9,
    client_slots=(1, 4),
    pitch_saturation: bool = True,
    continuous: bool = False,
) -> dict:
    """A seeded batch of one client's window at video rate (``T = 1 + 4 (F - 1)``) for the
    foreground weight.

    The other players stand at 20-2500 u from the client, mostly in front (some behind, some outside
    the frustum), so the projected radius covers both clamps (0.5 and 12 rows on the 24 x 42 grid)
    and beta covers 1..3. Visibility labels are row-wise ``visible & valid`` with unknown rows. With
    ``pitch_saturation`` the client's pitch deltas drive the running sum past +89 degrees and back,
    where the clamp of the sum and the row-by-row clamp differ. With ``continuous`` the batch also
    carries gathered continuous camera columns ``[B, F, P, 7]`` (a gathered window).
    """
    g = torch.Generator().manual_seed(seed)
    T = 1 + VIDEO_FRAMES_PER_LATENT * (latents - 1)
    team_ids = torch.tensor([[TEAM_IDS[p % 2] for p in range(players)]] * batch, dtype=torch.long)
    client_slot = torch.tensor(list(client_slots)[:batch], dtype=torch.long)
    states = torch.zeros(batch, players, T, 6)
    substeps = torch.zeros(batch, players, T, SUBSTEPS, BUTTONS + 2)
    valid = torch.zeros(batch, players, T, SUBSTEPS, dtype=torch.bool)
    for b in range(batch):
        client = int(client_slot[b])
        _place_players(g, states[b], substeps[b], valid[b], client)
        if pitch_saturation:
            # late in the window: ~+20 deg per row for 6 rows (the sum passes +89), then back down
            start = T - 13
            substeps[b, client, start : start + 6, :, BUTTONS] += 1.0
            substeps[b, client, start + 6 : start + 12, :, BUTTONS] -= 1.0
            valid[b, client] = True
    vis_valid = torch.rand(batch, players, T, generator=g) < 0.85
    visibility = (torch.rand(batch, players, T, generator=g) < 0.6) & vis_valid
    out = {
        "player_states": states,
        "player_control_substeps": substeps,
        "player_control_substep_valid": valid,
        "player_team_ids": team_ids,
        "player_weapon_ids": (torch.rand(batch, players, T, generator=g) * 52).long().clamp(max=51),
        "client_slot": client_slot,
        "client_visibility": visibility,
        "client_visibility_valid": vis_valid,
    }
    if continuous:
        cont = torch.zeros(batch, latents, players, 7)
        cont[..., 0] = torch.rand(batch, latents, players, generator=g) * 360.0 - 180.0
        cont[..., 1] = torch.rand(batch, latents, players, generator=g) * 60.0 - 30.0
        cont[..., 4:7] = torch.randn(batch, latents, players, 3, generator=g) * 100.0
        out["player_continuous_columns"] = cont
    return out


#: Substeps per video frame and buttons of a player's packed controls.
SUBSTEPS, BUTTONS = 4, 11
#: Distances, u, of the players of :func:`make_window_batch` from the client: four in front, one
#: (the fifth) behind, one (the sixth) outside the frustum.
DISTANCES = (20.0, 60.0, 250.0, 2500.0, 900.0, 400.0)


def _place_players(
    g: torch.Generator,
    states: torch.Tensor,
    substeps: torch.Tensor,
    valid: torch.Tensor,
    client: int,
) -> None:
    """Fill one sample's ``states [P, T, 6]``, ``substeps [P, T, 4, 13]`` and ``valid [P, T, 4]``:
    the client walking, the other players around it at :data:`DISTANCES`."""
    players, T = int(states.shape[0]), int(states.shape[1])

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    client_xy = torch.cumsum(randn(T, 2) * 4.0, dim=0) + randn(1, 2) * 100.0
    client_yaw0 = float(rand(1) * 360.0 - 180.0)
    client_pitch0 = float(rand(1) * 20.0 - 10.0)
    for p in range(players):
        if p == client:
            xy, z = client_xy, torch.zeros(T)
            yaw, pitch = client_yaw0 + torch.zeros(T), client_pitch0 + torch.zeros(T)
        else:
            k = p if p < client else p - 1
            bearing = (
                client_yaw0
                + (float(randn(1)) * 20.0 if k != 4 else 170.0)
                + (70.0 if k == 5 else 0.0)
            )
            scale = 1.0 + 0.1 * float(randn(1))
            dist = DISTANCES[k % len(DISTANCES)] * scale + torch.cumsum(randn(T), dim=0)
            rad = math.radians(bearing)
            xy = client_xy + torch.tensor([math.cos(rad), math.sin(rad)])[None] * dist[:, None]
            z = randn(1) * 20.0 + torch.cumsum(randn(T) * 0.5, dim=0)
            yaw = rand(1) * 360.0 - 180.0 + torch.cumsum(randn(T) * 3.0, dim=0)
            pitch = randn(1) * 5.0 + torch.zeros(T)
        states[p, :, 0:2] = xy
        states[p, :, 2] = z
        states[p, :, 3] = yaw
        states[p, :, 4] = pitch
        states[p, :, 5] = 1.0
        substeps[p, :, :, :BUTTONS] = (rand(T, SUBSTEPS, BUTTONS) < 0.3).float()
        substeps[p, :, :, BUTTONS] = randn(T, SUBSTEPS) * 0.1  # pitch delta / 5 deg
        substeps[p, :, :, BUTTONS + 1] = randn(T, SUBSTEPS) * 0.4  # yaw delta / 5 deg
        valid[p] = rand(T, SUBSTEPS) < 0.9
