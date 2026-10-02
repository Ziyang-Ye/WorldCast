"""The window of a reconstituted block: sink | memory slot (4) | recent (12) | target (4).

Window positions 0, 1-4, 5-16 and 17-20 hold the client's latent 0, the retrieved memory entry, its
latents s-12 .. s-1 and the target s .. s+3 (17 latent frames, sink | recent | target, when nothing
is retrieved); RoPE and the control history read these positions. Every per-frame condition is
gathered with the latents through one closed key table.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from worldcast.data.obs_signals import OBS_SIGNAL_KEYS

__all__ = [
    "BLOCK",
    "CONTINUOUS_COLUMNS_KEY",
    "CONTINUOUS_ROWS_KEY",
    "CONTINUOUS_WIDTH",
    "ROWS_PER_LATENT",
    "SINK_LATENT",
    "WP_CONDITION_KEYS",
    "VideoConditioning",
    "WindowGeometry",
    "WindowLayout",
    "compact_window",
    "compaction_layout",
    "context_block_ranges",
    "ordinary_window_conditions",
    "pixel_rows",
    "prepare_window",
    "video_conditioning",
]

#: Latent frames per block (the target, the memory slot, each recent block).
BLOCK = 4
#: Pixel rows per latent frame: row 0 for latent 0, rows 4f-3 .. 4f for latent f.
ROWS_PER_LATENT = 4
#: The sink is always the client's latent 0, the recorded first latent of the round.
SINK_LATENT = 0

#: The round's continuous peer-state columns ``[B, F, P, 7]`` (yaw, pitch, corpse, tsd, frozen
#: xyz), integrated once by the caller and gathered into ``CONTINUOUS_COLUMNS_KEY``.
CONTINUOUS_ROWS_KEY = "peer_continuous_rows"
CONTINUOUS_COLUMNS_KEY = "peer_continuous_columns"
CONTINUOUS_WIDTH = 7

#: The closed key table: latent-rate and pixel-row keys and their frame axis (in gather order).
_LATENT_AXIS: dict[str, int] = {"latents": 1, **{key: 1 for key in OBS_SIGNAL_KEYS}}
_PIXEL_AXIS: dict[str, int] = {
    "button_condition": 1,
    "camera_condition": 1,
    "weapon_condition": 1,
    "player_states": 2,
    "player_buttons": 2,
    "player_camera": 2,
    "player_weapon_ids": 2,
    "player_action_substeps": 2,
    "player_action_substep_valid": 2,
    "observer_visibility": 2,
    "observer_visibility_valid": 2,
}
_UNTOUCHED = ("prompts", "metadata", "observer_slot", "player_team_ids", "group_weight")
#: Window material and per-block keys: never gathered, never passed on.
_WP_PREFIX = "wp_"
#: The retrieved memory entry's material.
_SLOT_PREFIX = "wp_slot_"
#: Batch key -> suffix of its slot material ``wp_slot_<suffix>`` (16 pixel rows of the source).
_SLOT_PIXEL_SOURCE = {
    "button_condition": "buttons",
    "camera_condition": "camera_quantized",
    "weapon_condition": "weapon_ids",
    "player_states": "states",
    "player_buttons": "buttons",
    "player_camera": "camera_abs",
    "player_weapon_ids": "weapon_ids",
    "player_action_substeps": "action_substeps",
    "player_action_substep_valid": "action_substep_valid",
}
#: Per-player keys whose slot rows hold the source player in the observer row only...
_OBSERVER_ONLY = (
    "player_states",
    "player_buttons",
    "player_camera",
    "player_weapon_ids",
    "player_action_substeps",
    "player_action_substep_valid",
)
#: ... of which the other players repeat their sink-row state (the rest are zero actions).
_PEER_STATE_FROM_SINK = ("player_states", "player_camera", "player_weapon_ids")
_VISIBILITY_KEYS = ("observer_visibility", "observer_visibility_valid")
#: The geometry conditions the ray module reads.
WP_CONDITION_KEYS = (
    "state_wp_memory_c2w",
    "state_wp_anchor_c2w",
    "state_wp_memory_frames",
    "state_wp_frame_c2w",
    "state_wp_frame_tans",
)


def pixel_rows(latent_index: int) -> list[int]:
    """Pixel rows of latent ``f`` in the ``1 + 4 (F-1)`` layout: ``[0]``, else ``4f-3 .. 4f``."""
    f = int(latent_index)
    if f < 0:
        raise ValueError("latent index must be non-negative")
    return [0] if f == 0 else [ROWS_PER_LATENT * f - 3 + k for k in range(ROWS_PER_LATENT)]


@dataclass(frozen=True)
class WindowLayout:
    """The client's latent indices in one window and the window positions of its parts.

    Attributes:
        sink (list[int]): ``[0]``.
        recent (list[int]): the client's latents ``s - recent .. s - 1``.
        target (list[int]): the client's latents ``s .. s + 3``.
        slot_positions (list[int]): window positions of the memory slot (``[1, 2, 3, 4]``, or
            ``[]`` when nothing is retrieved).
        target_positions (list[int]): window positions of the target (17-20, or 13-16).
        num_frames (int): 21 with a memory slot, 17 without.
    """

    sink: list[int]
    recent: list[int]
    target: list[int]
    slot_positions: list[int]
    target_positions: list[int]
    num_frames: int


def compaction_layout(target_start: int, *, recent: int, with_slot: bool) -> WindowLayout:
    """The window of target block ``s = target_start``; needs ``s - recent >= 1``."""
    s = int(target_start)
    recent = int(recent)
    if recent <= 0 or recent % BLOCK:
        raise ValueError(f"recent must be a positive multiple of {BLOCK}, got {recent}")
    if s - recent < 1:
        raise ValueError(
            f"target block {s} leaves no room for {recent} recent latents after the sink"
        )
    n = 1 + (BLOCK if with_slot else 0) + recent + BLOCK
    return WindowLayout(
        sink=[SINK_LATENT],
        recent=list(range(s - recent, s)),
        target=list(range(s, s + BLOCK)),
        slot_positions=list(range(1, 1 + BLOCK)) if with_slot else [],
        target_positions=list(range(n - BLOCK, n)),
        num_frames=n,
    )


def context_block_ranges(num_frames: int) -> list[tuple[int, int]]:
    """``[(start, n), ...]`` of the context written before the target, in window positions.

    The sink alone, then blocks: ``[(0, 1), (1, 4), (5, 4), (9, 4), (13, 4)]`` for 21 latent frames.
    Written in this order, each range attends to the cache so far and to itself (there is no
    other mask).
    """
    return [(0, 1)] + [(start, BLOCK) for start in range(1, num_frames - BLOCK, BLOCK)]


@dataclass
class WindowGeometry:
    """The cameras of one window, what the ray code reads (``K`` is 4 or 0, ``F'`` 21 or 17).

    Attributes:
        num_frames (int): ``F'``.
        memory_frames (Tensor): ``[B, K]`` long window positions of the memory slot (1-4).
        memory_c2w (Tensor): ``[B, K, 4, 4]`` float32 cameras of the memory slot (the source
            player's recorded poses).
        anchor_c2w (Tensor): ``[B, 4, 4]`` float32 camera of the target's first latent frame; the
            rays are relative to it.
        frame_c2w (Tensor): ``[B, F', 4, 4]`` float32 camera of every latent frame in window order
            (sink, memory slot, recent, target); its slot rows are ``memory_c2w``.
        frame_tans (Tensor): ``[B, F', 2]`` float32 ``tan(hfov/2), tan(vfov/2)`` of each frame.
        target_start (Tensor): ``[B]`` long, the client's latent index ``s``.
    """

    num_frames: int
    memory_frames: torch.Tensor
    memory_c2w: torch.Tensor
    anchor_c2w: torch.Tensor
    frame_c2w: torch.Tensor
    frame_tans: torch.Tensor
    target_start: torch.Tensor

    def conditions(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """The five ``state_wp_*`` generator inputs on ``device``."""
        return {
            "state_wp_memory_c2w": self.memory_c2w.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "state_wp_anchor_c2w": self.anchor_c2w.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "state_wp_memory_frames": self.memory_frames.to(device=device, non_blocking=True),
            "state_wp_frame_c2w": self.frame_c2w.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            "state_wp_frame_tans": self.frame_tans.to(device=device, dtype=torch.float32),
        }


def _require(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"window compaction needs batch[{key!r}]")
    return value


def _slot_tensor(batch: Mapping[str, Any], name: str, prefix_shape: tuple) -> torch.Tensor:
    value = _require(batch, _SLOT_PREFIX + name)
    if tuple(value.shape[: len(prefix_shape)]) != tuple(prefix_shape):
        raise ValueError(
            f"{_SLOT_PREFIX + name} must start with {prefix_shape}, got {tuple(value.shape)}"
        )
    return value


def _slot_pieces(
    batch: Mapping[str, Any], b: int, *, batch_size: int, players: int, observer: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Row ``b``'s memory entry on the slot positions: latent-rate pieces, pixel-row pieces and
    the cameras ``[4, 4, 4]``."""
    rows = ROWS_PER_LATENT * BLOCK
    lat = {"latents": _slot_tensor(batch, "latents", (batch_size, BLOCK))[b]}
    for key in OBS_SIGNAL_KEYS:
        if key in batch:
            lat[key] = _slot_tensor(batch, "obs_" + key, (batch_size, BLOCK))[b]
    pix: dict[str, torch.Tensor] = {}
    for key, src in _SLOT_PIXEL_SOURCE.items():
        piece = _slot_tensor(batch, src, (batch_size, rows))[b]
        ref = batch[key][b]
        if key not in _OBSERVER_ONLY:
            pix[key] = piece.to(ref.dtype)
            continue
        if key in _PEER_STATE_FROM_SINK:
            # every other player: its sink-row state repeated (alive stays monotone)
            full = ref[:, 0:1].expand(players, rows, *ref.shape[2:]).clone()
        else:
            # every other player: no action on the slot rows
            full = torch.zeros((players, rows) + tuple(ref.shape[2:]), dtype=ref.dtype)
        full[observer] = piece.to(ref.dtype)
        pix[key] = full
    for key in _VISIBILITY_KEYS:
        ref = batch[key][b]
        pix[key] = torch.zeros((players, rows) + tuple(ref.shape[2:]), dtype=ref.dtype)
    c2w = _slot_tensor(batch, "c2w", (batch_size, BLOCK, 4, 4))[b].to(torch.float32)
    return lat, pix, c2w


def _slot_continuous(
    sink_cols: torch.Tensor, slot_states: torch.Tensor, observer: int
) -> torch.Tensor:
    """Continuous columns ``[4, P, 7]`` of the slot latents from the sink row ``[1, P, 7]``.

    Every other player keeps its sink-row columns. The observer row takes the source player's
    engine angles at each slot latent's last pixel row (``slot_states`` ``[16, 6]``), alive, no
    corpse, and its own position as the frozen xyz.
    """
    cols = sink_cols.expand(BLOCK, *sink_cols.shape[1:]).clone()
    lat_rows = slot_states.to(torch.float32)[ROWS_PER_LATENT - 1 :: ROWS_PER_LATENT]  # [4, 6]
    cols[:, observer, 0] = lat_rows[:, 3]
    cols[:, observer, 1] = lat_rows[:, 4]
    cols[:, observer, 2] = 0.0
    cols[:, observer, 3] = 0.0
    cols[:, observer, 4:7] = lat_rows[:, :3]
    return cols


def compact_window(
    batch: Mapping[str, Any], *, recent: int, with_slot: bool
) -> tuple[dict[str, Any], WindowGeometry]:
    """Gather one window out of the client's round batch.

    On the memory slot's 16 pixel rows the observer row carries the source player's recorded rows,
    every other player repeats its sink-row state with zero actions, and the visibility labels are
    zero, so the label-gated player state field draws no peer there.

    Args:
        batch (Mapping[str, Any]): the round batch (``B`` rounds, the client's ``F`` latent frames,
            ``T = 1 + 4 (F-1)`` pixel rows, ``P`` players): ``latents`` ``[B, F, C, H, W]`` (the
            client's store: float32 clean latents), the ``obs_*`` keys ``[B, F]``, the pixel-row
            keys of ``_PIXEL_AXIS`` (``button_condition`` ``[B, T, 11]``, ``player_states``
            ``[B, P, T, 6]``, ...), ``observer_slot`` ``[B]`` long, ``wp_target_start`` ``[B]`` long
            (``s``), ``wp_own_c2w`` ``[B, F, 4, 4]``, ``wp_own_tans`` ``[B, F, 2]``, optionally
            ``peer_continuous_rows`` ``[B, F, P, 7]``, and with ``with_slot`` the memory entry's
            ``wp_slot_*`` material: ``latents`` ``[B, 4, C, H, W]``, ``obs_<key>`` ``[B, 4]``, the
            16-row source keys of ``_SLOT_PIXEL_SOURCE``, ``c2w`` ``[B, 4, 4, 4]``, ``tans``
            ``[B, 4, 2]``.
        recent (int): recent latent frames (12).
        with_slot (bool): a memory entry was retrieved.

    Returns:
        tuple[dict[str, Any], WindowGeometry]: the compacted batch (latent axis 21 or 17, pixel
        axis 81 or 65, the other ``wp_*`` keys dropped, the continuous columns under
        ``peer_continuous_columns`` ``[B, F', P, 7]``) and its cameras.
    """
    if CONTINUOUS_COLUMNS_KEY in batch:
        raise ValueError(
            f"the batch already carries {CONTINUOUS_COLUMNS_KEY}: it is already compacted"
        )
    latents = _require(batch, "latents")
    if latents.ndim != 5:
        raise ValueError("latents must be [B, F, C, H, W]")
    B, F = int(latents.shape[0]), int(latents.shape[1])
    T = 1 + ROWS_PER_LATENT * (F - 1)
    states = _require(batch, "player_states")
    if states.ndim != 4 or int(states.shape[2]) != T:
        raise ValueError(f"player_states must be [B, P, {T}, 6], got {tuple(states.shape)}")
    P = int(states.shape[1])
    observer_slot = _require(batch, "observer_slot").to(torch.long).reshape(B)
    target_start = _require(batch, "wp_target_start").to(torch.long).reshape(B)
    own_c2w = _require(batch, "wp_own_c2w")
    if tuple(own_c2w.shape) != (B, F, 4, 4):
        raise ValueError(f"wp_own_c2w must be [B, {F}, 4, 4], got {tuple(own_c2w.shape)}")

    # the closed table: every tensor key is gathered, passed through, or window material
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key in _UNTOUCHED or key.startswith(_WP_PREFIX):
            continue
        if key in _LATENT_AXIS or key in _PIXEL_AXIS or key == CONTINUOUS_ROWS_KEY:
            continue
        if isinstance(value, torch.Tensor):
            raise KeyError(
                f"window compaction does not know batch key {key!r} (shape {tuple(value.shape)}); "
                "a per-frame condition left uncompacted would ride next to compacted latents"
            )
        out[key] = value
    for key in _UNTOUCHED:
        if key in batch:
            out[key] = batch[key]
    for key in list(_LATENT_AXIS) + list(_PIXEL_AXIS):
        if key not in batch and key not in OBS_SIGNAL_KEYS:
            raise KeyError(f"window compaction: batch has no {key!r}")

    layouts = [
        compaction_layout(int(target_start[b]), recent=recent, with_slot=with_slot)
        for b in range(B)
    ]
    n_frames = layouts[0].num_frames
    K = BLOCK if with_slot else 0
    latent_pieces: dict[str, list[torch.Tensor]] = {k: [] for k in _LATENT_AXIS if k in batch}
    pixel_pieces: dict[str, list[torch.Tensor]] = {key: [] for key in _PIXEL_AXIS}
    memory_c2w = torch.zeros(B, K, 4, 4, dtype=torch.float32)
    continuous = batch.get(CONTINUOUS_ROWS_KEY)
    if continuous is not None:
        continuous = continuous.float()
        if tuple(continuous.shape) != (B, F, P, CONTINUOUS_WIDTH):
            raise RuntimeError(
                f"the continuous columns are {tuple(continuous.shape)}, expected "
                f"[{B}, {F}, {P}, {CONTINUOUS_WIDTH}]"
            )
    continuous_pieces: list[torch.Tensor] = []

    for b in range(B):
        lay = layouts[b]
        own_rows_ctx = pixel_rows(SINK_LATENT)
        own_rows_rest = [r for f in lay.recent + lay.target for r in pixel_rows(f)]
        obs = int(observer_slot[b])

        def own_slice(value: torch.Tensor, axis: int, index: list[int]) -> torch.Tensor:
            return value[b].index_select(axis - 1, torch.as_tensor(index, dtype=torch.long))

        slot_lat: dict[str, torch.Tensor] = {}
        slot_pix: dict[str, torch.Tensor] = {}
        if with_slot:
            slot_lat, slot_pix, slot_c2w = _slot_pieces(
                batch, b, batch_size=B, players=P, observer=obs
            )
            memory_c2w[b] = slot_c2w

        for key, axis in _LATENT_AXIS.items():
            if key not in batch:
                continue
            head = own_slice(batch[key], axis, lay.sink)
            rest = own_slice(batch[key], axis, lay.recent + lay.target)
            parts = [head] + ([slot_lat[key]] if with_slot else []) + [rest]
            latent_pieces[key].append(torch.cat(parts, dim=axis - 1))
        for key, axis in _PIXEL_AXIS.items():
            head = own_slice(batch[key], axis, own_rows_ctx)
            rest = own_slice(batch[key], axis, own_rows_rest)
            parts = [head] + ([slot_pix[key]] if with_slot else []) + [rest]
            pixel_pieces[key].append(torch.cat(parts, dim=axis - 1))
        if continuous is not None:
            # the same latent indices as the rows: sink | slot | recent + target
            own_idx = torch.as_tensor(lay.sink + lay.recent + lay.target, dtype=torch.long)
            own_cols = continuous[b].index_select(0, own_idx)  # [1 + recent + 4, P, 7]
            parts = [own_cols[:1]]
            if with_slot:
                parts.append(
                    _slot_continuous(continuous[b, 0:1], slot_pix["player_states"][obs], obs)
                )
            parts.append(own_cols[1:])
            continuous_pieces.append(torch.cat(parts, dim=0))

    for key, pieces in latent_pieces.items():
        out[key] = torch.stack(pieces, dim=0)
    for key, pieces in pixel_pieces.items():
        out[key] = torch.stack(pieces, dim=0)
    if continuous is not None:
        out[CONTINUOUS_COLUMNS_KEY] = torch.stack(continuous_pieces, dim=0)  # [B, F', P, 7]
    # one life per round: the compacted alive column must never rise along the frame axis
    alive = out["player_states"][..., 5]
    if bool((alive[:, :, 1:] > alive[:, :, :-1]).any()):
        raise RuntimeError(
            "window compaction produced a resurrection (alive rises along the frame axis)"
        )

    memory_frames = (
        torch.tensor(layouts[0].slot_positions, dtype=torch.long).view(1, K).expand(B, K).clone()
    )
    anchor = torch.stack([own_c2w[b, int(target_start[b])] for b in range(B)]).to(torch.float32)
    frame_c2w = torch.stack(
        [
            torch.cat(
                [own_c2w[b, layouts[b].sink].to(torch.float32)]
                + ([memory_c2w[b]] if with_slot else [])
                + [own_c2w[b, layouts[b].recent + layouts[b].target].to(torch.float32)],
                dim=0,
            )
            for b in range(B)
        ]
    )  # [B, F', 4, 4]
    own_tans = _require(batch, "wp_own_tans").float()
    if (
        tuple(own_tans.shape) != (B, F, 2)
        or not bool(torch.isfinite(own_tans).all())
        or bool((own_tans <= 0).any())
    ):
        raise ValueError("wp_own_tans must be finite positive [B, F, 2]")
    slot_tans = _slot_tensor(batch, "tans", (B, BLOCK, 2)).float() if with_slot else None
    frame_tans = torch.stack(
        [
            torch.cat(
                [own_tans[b, layouts[b].sink]]
                + ([slot_tans[b]] if with_slot else [])
                + [own_tans[b, layouts[b].recent + layouts[b].target]]
            )
            for b in range(B)
        ]
    )
    geometry = WindowGeometry(
        num_frames=n_frames,
        memory_frames=memory_frames,
        memory_c2w=memory_c2w,
        anchor_c2w=anchor,
        frame_c2w=frame_c2w,
        frame_tans=frame_tans,
        target_start=target_start.clone(),
    )
    return out, geometry


def ordinary_window_conditions(
    batch: Mapping[str, Any], device: torch.device | str
) -> dict[str, torch.Tensor]:
    """The camera conditions of a contiguous window (the plain prefix): no memory slot, the
    anchor at the client's latent 0.

    Args:
        batch (Mapping[str, Any]): ``wp_own_c2w`` ``[B, L, 4, 4]`` and ``wp_own_tans``
            ``[B, L, 2]`` over the round.
        device (torch.device | str): where the conditions go.

    Returns:
        dict[str, Tensor]: the five ``state_wp_*`` conditions with ``K = 0``.
    """
    own = _require(batch, "wp_own_c2w")
    if own.ndim != 4 or tuple(own.shape[-2:]) != (4, 4):
        raise ValueError(f"wp_own_c2w must be [B, L, 4, 4], got {tuple(own.shape)}")
    b = int(own.shape[0])
    tans = batch.get("wp_own_tans")
    if tans is None or tuple(tans.shape) != (*own.shape[:2], 2):
        raise ValueError("ordinary geometry requires wp_own_tans[B, L, 2]")
    return {
        "state_wp_memory_c2w": torch.zeros(b, 0, 4, 4, dtype=torch.float32, device=device),
        "state_wp_anchor_c2w": own[:, 0].to(device=device, dtype=torch.float32, non_blocking=True),
        "state_wp_memory_frames": torch.zeros(b, 0, dtype=torch.long, device=device),
        "state_wp_frame_c2w": own.to(device=device, dtype=torch.float32, non_blocking=True),
        "state_wp_frame_tans": tans.to(device=device, dtype=torch.float32),
    }


def prepare_window(
    batch: Mapping[str, Any], *, recent: int, with_slot: bool, device: torch.device | str
) -> tuple[dict[str, Any], WindowGeometry, dict[str, torch.Tensor]]:
    """:func:`compact_window` and the window's camera conditions on ``device``.

    Returns:
        tuple: the compacted batch, its :class:`WindowGeometry` and its ``state_wp_*`` conditions.
    """
    compacted, geometry = compact_window(batch, recent=recent, with_slot=with_slot)
    return compacted, geometry, geometry.conditions(device)


@dataclass
class VideoConditioning:
    """The latents, prompts and controls of one batch, on the device.

    Attributes:
        clean_latent (Tensor): ``[B, F, C, H, W]`` in the model dtype.
        prompts (list[str]): ``B`` prompts.
        actions (dict[str, Tensor]): ``button_condition`` ``[B, T, 11]`` and ``camera_condition``
            ``[B, T, 2]`` in the model dtype, ``weapon_condition`` ``[B, T]`` long, over every
            pixel row of the window (the control history is never sliced).
    """

    clean_latent: torch.Tensor
    prompts: list[str]
    actions: dict[str, torch.Tensor]


def video_conditioning(
    batch: Mapping[str, Any], *, device: torch.device | str, dtype: torch.dtype
) -> VideoConditioning:
    """The batch's latents and full control history on ``device``."""
    latents = batch["latents"]
    if latents.ndim != 5:
        raise ValueError("latents must be single-POV [B, F, C, H, W]")
    batch_size, pixel_frames = latents.shape[0], batch["player_states"].shape[2]
    prompts = batch["prompts"]
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, Sequence) or len(prompts) != batch_size:
        raise ValueError(f"prompts must contain {batch_size} strings")

    def action(key: str, trailing: int | None) -> torch.Tensor:
        value = batch[key]
        expected_ndim = 2 if trailing is None else 3
        if value.ndim != expected_ndim or value.shape[:2] != (batch_size, pixel_frames):
            raise ValueError(
                f"{key} must keep the full [B, {pixel_frames}, ...] action window, "
                f"got {tuple(value.shape)}"
            )
        if trailing is not None and value.shape[-1] != trailing:
            raise ValueError(f"{key} must end in {trailing} channels")
        return value

    actions = {
        "button_condition": action("button_condition", 11).to(
            device=device, dtype=dtype, non_blocking=True
        ),
        "camera_condition": action("camera_condition", 2).to(
            device=device, dtype=dtype, non_blocking=True
        ),
        "weapon_condition": action("weapon_condition", None)
        .long()
        .to(device=device, non_blocking=True),
    }
    return VideoConditioning(
        clean_latent=latents.to(device=device, dtype=dtype, non_blocking=True).contiguous(),
        prompts=[str(p) for p in prompts],
        actions=actions,
    )
