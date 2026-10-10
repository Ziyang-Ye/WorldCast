"""The window of a block (Sec. 3.3, "Conditioning the generator"): first frame | memory frames (4) |
recent context | target frames (4).

With the paper's 12 latent frames of recent context, window positions 0, 1-4, 5-16 and 17-20 hold
the client's latent frame 0 (the round's recorded first frame), the memory frames of the retrieved
memory entry, the recent context s-12 .. s-1 and the target frames s .. s+3: 21 latent frames, or
17 (first frame | recent context | target frames) when nothing is retrieved. RoPE and the control
history read these positions. :func:`gather_window` gathers every per-frame entry of a round batch
with the latents, through one closed table (:data:`WINDOW_ENTRIES`).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.latents import (
    BLOCK,
    RECENT,
    VIDEO_FRAMES_PER_BLOCK,
    video_frame_count,
    video_frames_of,
)
from worldcast.data.memory_frames import MEMORY_FRAMES_PREFIX
from worldcast.data.recordings import ALIVE_INDEX
from worldcast.modeling.controls import CONTROL_KEYS
from worldcast.modeling.ray_embedding import RayConditions
from worldcast.modeling.wan22.attention import causal_blocks

__all__ = [
    "CONTINUOUS_COLUMNS_KEY",
    "FIRST_FRAME_LATENT",
    "KV_CACHE_LATENTS",
    "MEMORY_CONTINUOUS_COLUMNS_KEY",
    "ROUND_CONTINUOUS_COLUMNS_KEY",
    "WINDOW_ENTRIES",
    "WindowEntry",
    "WindowLayout",
    "gather_window",
]

#: The first frame of every window is the client's latent frame 0, the recorded first frame.
FIRST_FRAME_LATENT = 0
#: KV cache capacity of a client, in latent frames: the 41 latent frames of the generator's
#: training window, as the reference runs allocated it; a rollout writes at most 25 of them.
KV_CACHE_LATENTS = 41

#: Batch key of the players' continuous state columns ``[B, F, P, 7]`` of the whole round
#: (``worldcast.player_state.continuous_row_state``), before the window is gathered ...
ROUND_CONTINUOUS_COLUMNS_KEY = "round_continuous_columns"
#: ... of those columns on the memory frames ``[B, 4, P, 7]``
#: (``worldcast.player_state.memory_continuous_columns``), the round entry's memory input ...
MEMORY_CONTINUOUS_COLUMNS_KEY = MEMORY_FRAMES_PREFIX + "continuous_columns"
#: ... and of the same columns of a gathered window ``[B, F', P, 7]``.
CONTINUOUS_COLUMNS_KEY = "player_continuous_columns"


#: The frame axis of a per-frame entry, by what it counts: latent frames ``[B, F, ...]``, the
#: client's video frames ``[B, T, ...]`` (``T = 1 + 4 (F - 1)``) or every player's video frames
#: ``[B, P, T, ...]``.
_FRAME_AXIS = {"latent": 1, "video": 1, "player_video": 2}


@dataclass(frozen=True)
class WindowEntry:
    """How :func:`gather_window` gathers one per-frame entry of a round batch.

    Attributes:
        frames (str): what the entry's frame axis counts: ``"latent"`` (``[B, F, ...]``),
            ``"video"`` (the client's ``[B, T, ...]``, ``T = 1 + 4 (F - 1)``) or
            ``"player_video"`` (every player's ``[B, P, T, ...]``).
        memory_input (str | None): the batch key of the entry on the memory frames: the memory
            entry's four latent frames ``[B, 4, ...]``, or the 16 video frames ``[B, 16, ...]`` of
            its source player, which fill the client's slot of a per-player entry. ``None``, for a
            per-player entry only: the entry is zero on the memory frames.
        other_players (str | None): for a per-player entry with a memory input, what every other
            player holds on the memory frames: ``"first_frame"`` (its first-frame value, repeated)
            or ``"zero"``. ``None`` for every other entry.
        optional (bool): a batch may lack the entry.
    """

    frames: str
    memory_input: str | None = None
    other_players: str | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if self.frames not in _FRAME_AXIS:
            raise ValueError(f"frames must be one of {tuple(_FRAME_AXIS)}, got {self.frames!r}")
        per_player = self.frames == "player_video"
        if self.memory_input is None and not per_player:
            raise ValueError("only a per-player entry can be zero on the memory frames")
        filled = per_player and self.memory_input is not None
        allowed = ("first_frame", "zero") if filled else (None,)
        if self.other_players not in allowed:
            raise ValueError(f"other_players must be one of {allowed}, got {self.other_players!r}")

    @property
    def axis(self) -> int:
        """The entry's frame axis."""
        return _FRAME_AXIS[self.frames]


def _memory(name: str) -> str:
    return MEMORY_FRAMES_PREFIX + name


#: The closed table of the per-frame entries of a round batch. By latent frame: the latents, the
#: observer signals and the continuous columns; by video frame: the client's controls, then every
#: player's entries. The visibility labels have no memory input: nobody is labelled visible on
#: the memory frames.
WINDOW_ENTRIES: dict[str, WindowEntry] = {
    "latents": WindowEntry("latent", _memory("latents")),
    **{key: WindowEntry("latent", _memory(key), optional=True) for key in OBSERVER_SIGNAL_KEYS},
    ROUND_CONTINUOUS_COLUMNS_KEY: WindowEntry(
        "latent", MEMORY_CONTINUOUS_COLUMNS_KEY, optional=True
    ),
    **{
        key: WindowEntry("video", _memory(name))
        for key, name in zip(CONTROL_KEYS, ("buttons", "view_deltas", "weapon_ids"))
    },
    "player_states": WindowEntry("player_video", _memory("states"), "first_frame"),
    "player_weapon_ids": WindowEntry("player_video", _memory("weapon_ids"), "first_frame"),
    "player_control_substeps": WindowEntry("player_video", _memory("control_substeps"), "zero"),
    "player_control_substep_valid": WindowEntry(
        "player_video", _memory("control_substep_valid"), "zero"
    ),
    "client_visibility": WindowEntry("player_video"),
    "client_visibility_valid": WindowEntry("player_video"),
}
#: Tensors of a round batch that carry no frame axis: passed on as they are.
_PER_ROUND = ("client_slot", "player_team_ids")
#: The window's own inputs (``window_*``) and the memory frames' inputs: never passed on.
_WINDOW_PREFIXES = ("window_", MEMORY_FRAMES_PREFIX)


@dataclass(frozen=True)
class WindowLayout:
    """Where the parts of a block's window sit: first frame | memory frames | recent context |
    target frames.

    Attributes:
        recent (int): latent frames of recent context (the paper's 12; 32 in the windows of stage
            2s).
        with_memory (bool): the window holds the four memory frames of a retrieved memory entry.
    """

    recent: int = RECENT
    with_memory: bool = True

    def __post_init__(self) -> None:
        if self.recent <= 0 or self.recent % BLOCK:
            raise ValueError(f"recent must be a positive multiple of {BLOCK}, got {self.recent}")

    @property
    def memory_positions(self) -> list[int]:
        """Window positions of the memory frames: ``[1, 2, 3, 4]``, or ``[]`` without them."""
        return list(range(1, 1 + BLOCK)) if self.with_memory else []

    @property
    def num_frames(self) -> int:
        """Latent frames of the window, ``1 + 4 + recent + 4`` with memory frames (21 at the
        paper's 12 recent frames) and ``1 + recent + 4`` without (17)."""
        return 1 + len(self.memory_positions) + self.recent + BLOCK

    @property
    def target_positions(self) -> list[int]:
        """Window positions of the target frames, the last block (17-20 of 21, 13-16 of 17)."""
        return list(range(self.num_frames - BLOCK, self.num_frames))

    @property
    def context_ranges(self) -> list[tuple[int, int]]:
        """The context as it is written before the target frames, ``[(start, end), ...]`` in
        window positions: the first frame alone, then blocks of four (``[(0, 1), (1, 5), (5, 9),
        (9, 13), (13, 17)]`` with memory frames). Written in this order, each range attends to the
        cache so far and to itself (there is no other mask)."""
        return causal_blocks(self.num_frames - BLOCK)

    def own_latents(self, target_start: int) -> list[int]:
        """The client's latent frames after the first one, for the target block ``s =
        target_start``: the recent context ``s - recent .. s - 1``, then the target frames ``s ..
        s + 3``."""
        first = int(target_start) - self.recent
        if first < 1:
            raise ValueError(
                f"target block {int(target_start)} leaves no room for {self.recent} recent latent"
                " frames after the first frame"
            )
        return list(range(first, int(target_start) + BLOCK))


# ======================================================================================= the gather
def _require(batch: Mapping[str, Any], key: str, leading: tuple[int, ...] = ()) -> torch.Tensor:
    """The tensor ``batch[key]``, whose shape starts with ``leading``."""
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"the window needs a tensor batch[{key!r}]")
    if tuple(value.shape[: len(leading)]) != leading:
        raise ValueError(f"{key} must start with {list(leading)}, got {tuple(value.shape)}")
    return value


def _memory_frames(batch: Mapping[str, Any], key: str, b: int, client: int) -> torch.Tensor:
    """Sample ``b``'s entry ``key`` on the memory frames: four latent frames, or their 16 video
    frames."""
    entry = WINDOW_ENTRIES[key]
    if entry.frames == "latent":
        return batch[entry.memory_input][b]
    reference = batch[key][b]
    if entry.frames == "video":
        return batch[entry.memory_input][b].to(reference.dtype)
    shape = (reference.shape[0], VIDEO_FRAMES_PER_BLOCK, *reference.shape[2:])
    if entry.other_players == "first_frame":  # alive stays monotone
        frames = reference[:, 0:1].expand(shape).clone()
    else:
        frames = torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    if entry.memory_input is not None:
        frames[client] = batch[entry.memory_input][b].to(reference.dtype)
    return frames


def _passed_through(batch: Mapping[str, Any]) -> dict[str, Any]:
    """The entries the gather leaves alone; a tensor outside the closed table is an error."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key in WINDOW_ENTRIES or key.startswith(_WINDOW_PREFIXES):
            continue
        if isinstance(value, torch.Tensor) and key not in _PER_ROUND:
            raise KeyError(
                f"the window does not know batch key {key!r} (shape {tuple(value.shape)}); a"
                " per-frame condition left ungathered would ride next to gathered latents"
            )
        out[key] = value
    return out


def _gather_sample(
    batch: Mapping[str, Any], b: int, layout: WindowLayout, own_latents: list[int], client: int
) -> dict[str, torch.Tensor]:
    """Sample ``b`` of the window: each entry's first frame | memory frames | recent context and
    target frames, along its frame axis."""
    latent_frames = ([FIRST_FRAME_LATENT], own_latents)
    video_frames = tuple(
        [t for f in frames for t in video_frames_of(f)] for frames in latent_frames
    )

    def own(value: torch.Tensor, axis: int, index: list[int]) -> torch.Tensor:
        index = torch.as_tensor(index, dtype=torch.long, device=value.device)
        return value[b].index_select(axis - 1, index)

    sample: dict[str, torch.Tensor] = {}
    for key, entry in WINDOW_ENTRIES.items():
        if key not in batch:
            continue
        first, rest = latent_frames if entry.frames == "latent" else video_frames
        parts = [own(batch[key], entry.axis, first)]
        if layout.with_memory:
            parts.append(_memory_frames(batch, key, b, client))
        sample[key] = torch.cat(parts + [own(batch[key], entry.axis, rest)], dim=entry.axis - 1)
    return sample


def _window_rays(
    batch: Mapping[str, Any],
    layout: WindowLayout,
    own_latents: list[list[int]],
    target_start: torch.Tensor,
) -> RayConditions:
    """The cameras of the gathered window, float32: the client's own, with the memory frames' (the
    source player's poses) at their positions; the anchor is the camera of the first target
    frame."""
    own_c2w = batch["window_c2w"].float()
    batch_size, frames = own_c2w.shape[:2]
    own_tans = _require(batch, "window_tans", (batch_size, frames, 2)).float()
    if not bool(torch.isfinite(own_tans).all()) or bool((own_tans <= 0).any()):
        raise ValueError("window_tans must be finite and positive")
    memory_c2w = memory_tans = None
    if layout.with_memory:
        memory_c2w = _require(batch, _memory("c2w"), (batch_size, BLOCK, 4, 4)).float()
        memory_tans = _require(batch, _memory("tans"), (batch_size, BLOCK, 2)).float()

    def in_window(own: torch.Tensor, memory: torch.Tensor | None, b: int) -> torch.Tensor:
        parts = [own[b, [FIRST_FRAME_LATENT]]]
        parts += [memory[b]] if layout.with_memory else []
        return torch.cat(parts + [own[b, own_latents[b]]], dim=0)

    samples = range(batch_size)
    return RayConditions(
        frame_c2w=torch.stack([in_window(own_c2w, memory_c2w, b) for b in samples]),
        frame_tans=torch.stack([in_window(own_tans, memory_tans, b) for b in samples]),
        anchor_c2w=torch.stack([own_c2w[b, int(target_start[b])] for b in samples]),
    )


def _check_round_batch(batch: Mapping[str, Any], layout: WindowLayout) -> tuple[int, int, int]:
    """``(B, F, P)`` of a round batch whose entries, and their memory inputs if the layout has
    memory frames, have the frames of its latents and the players of its states."""
    if CONTINUOUS_COLUMNS_KEY in batch:
        raise ValueError(f"the batch already carries {CONTINUOUS_COLUMNS_KEY}: it is a window")
    latents, states = _require(batch, "latents"), _require(batch, "player_states")
    if latents.ndim != 5 or states.ndim != 4:
        raise ValueError("latents must be [B, F, C, H, W] and player_states [B, P, T, 6]")
    batch_size, frames, players = int(latents.shape[0]), int(latents.shape[1]), int(states.shape[1])
    for key, entry in WINDOW_ENTRIES.items():
        if key not in batch and entry.optional:
            continue
        count, in_memory = (
            (frames, BLOCK)
            if entry.frames == "latent"
            else (video_frame_count(frames), VIDEO_FRAMES_PER_BLOCK)
        )
        leading = (batch_size, players, count) if entry.axis == 2 else (batch_size, count)
        value = _require(batch, key, leading)
        if layout.with_memory and entry.memory_input is not None:
            shape = (batch_size, in_memory, *value.shape[entry.axis + 1 :])
            if tuple(_require(batch, entry.memory_input).shape) != shape:
                raise ValueError(f"{entry.memory_input} must be {list(shape)}, as {key} has it")
    continuous = batch.get(ROUND_CONTINUOUS_COLUMNS_KEY)
    if continuous is not None and tuple(continuous.shape[:3]) != (batch_size, frames, players):
        raise ValueError(
            f"the continuous columns are {tuple(continuous.shape)}, expected [{batch_size},"
            f" {frames}, {players}, 7]"
        )
    _require(batch, "window_c2w", (batch_size, frames, 4, 4))
    return batch_size, frames, players


def gather_window(
    batch: Mapping[str, Any], layout: WindowLayout
) -> tuple[dict[str, Any], RayConditions]:
    """Gather one window out of the client's round batch.

    On the memory frames' 16 video frames the client's slot carries the source player's states
    and controls, every other player repeats its first-frame state with zero controls, and the
    visibility labels are zero, so the label-gated player state field draws no other player there.

    Args:
        batch (Mapping[str, Any]): the round batch: ``B`` rounds, the client's ``F`` latent frames,
            ``T = 1 + 4 (F - 1)`` video frames, ``P`` players; every tensor on one device.

            - The entries of :data:`WINDOW_ENTRIES`: ``latents`` ``[B, F, C, H, W]`` (the client's
              store of clean latents); the observer signals ``[B, F]`` and
              :data:`ROUND_CONTINUOUS_COLUMNS_KEY` ``[B, F, P, 7]`` (both optional); ``buttons``
              ``[B, T, 11]``, ``view_deltas`` ``[B, T, 2]``, ``weapon`` ``[B, T]``;
              ``player_states`` ``[B, P, T, 6]``, ``player_weapon_ids`` ``[B, P, T]``,
              ``player_control_substeps`` ``[B, P, T, 4, 13]``, ``player_control_substep_valid``
              ``[B, P, T, 4]``, ``client_visibility`` and ``client_visibility_valid``
              ``[B, P, T]``.
            - ``client_slot`` ``[B]`` long and ``window_target_start`` ``[B]`` long (``s``), the
              client's cameras ``window_c2w`` ``[B, F, 4, 4]`` and ``window_tans`` ``[B, F, 2]``.
            - With memory frames, each entry's :attr:`WindowEntry.memory_input`, shaped as the
              entry with the 4 latent frames, or the 16 video frames of one player, in place of
              its frames, and the memory entry's cameras ``memory_frames_c2w`` ``[B, 4, 4, 4]``
              and ``memory_frames_tans`` ``[B, 4, 2]``; without them these inputs are ignored.
            - ``client_slot``, ``player_team_ids`` and whatever is not a tensor are passed on; any
              other tensor is an error.
        layout (WindowLayout): the window's layout.

    Returns:
        tuple[dict[str, Any], RayConditions]: the gathered batch (``layout.num_frames`` latent
        frames, 21 or 17 at the paper's layout, and their 81 or 65 video frames; the ``window_*``
        and ``memory_frames_*`` inputs dropped; the continuous columns under
        :data:`CONTINUOUS_COLUMNS_KEY` ``[B, F', P, 7]`` float32) and the cameras of its frames,
        which the ray embedding reads.
    """
    batch_size, frames, players = _check_round_batch(batch, layout)
    client_slot = _require(batch, "client_slot").to(torch.long).reshape(batch_size)
    if int(client_slot.min()) < 0 or int(client_slot.max()) >= players:
        raise ValueError(f"client_slot {client_slot.tolist()} is outside the {players} players")
    target_start = _require(batch, "window_target_start").to(torch.long).reshape(batch_size)
    if int(target_start.max()) + BLOCK > frames:
        raise ValueError(
            f"target block {int(target_start.max())} ends past the round's {frames} latent frames"
        )

    out = _passed_through(batch)
    own_latents = [layout.own_latents(int(target_start[b])) for b in range(batch_size)]
    samples = [
        _gather_sample(batch, b, layout, own_latents[b], int(client_slot[b]))
        for b in range(batch_size)
    ]
    for key in samples[0]:
        out[key] = torch.stack([sample[key] for sample in samples], dim=0)
    if ROUND_CONTINUOUS_COLUMNS_KEY in out:
        out[CONTINUOUS_COLUMNS_KEY] = out.pop(ROUND_CONTINUOUS_COLUMNS_KEY).float()
    # one life per round: the gathered alive column must never rise along the frame axis
    alive = out["player_states"][..., ALIVE_INDEX]
    if bool((alive[:, :, 1:] > alive[:, :, :-1]).any()):
        raise RuntimeError(
            "the gathered window holds a resurrection (alive rises along the frames)"
        )
    return out, _window_rays(batch, layout, own_latents, target_start)
