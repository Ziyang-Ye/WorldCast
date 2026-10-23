"""The visibility labels of one recording: every other player against the player's view, per frame.

One recording's labels are ``[10, T]`` arrays over its ``T`` video frames (32 fps), one row per
player slot. The reader (:func:`worldcast.data.labels.visibility_rows`) takes ``_binary_visible``,
``_binary_eval_valid``, ``_in_frustum`` and ``_binary_offscreen``; the other arrays say how the
labels were reached.
"""

import json
from dataclasses import dataclass

import numpy as np

from worldcast.data.game import TICK_RATE
from worldcast.data.recordings import NUM_PLAYERS

from ._visibility_geometry import (
    BODY_PROBES,
    Occluders,
    apply_player_capsules,
    line_of_sight,
    player_capsule_blocks,
)

__all__ = [
    "MAX_TICK_LAG",
    "SCHEMA",
    "FrameState",
    "PlayerTicks",
    "decode_meta",
    "encode_meta",
    "frame_state",
    "frame_ticks",
    "hidden_by_players",
    "observer_labels",
]

#: The name of the label format (the ``_meta`` of a file carries it).
SCHEMA = "opencs2-geometry-visibility-binary-v1"
#: A video frame takes the last tick at or before its own; a table may miss this many ticks.
MAX_TICK_LAG = 1
#: ``_slot_team_num`` of a slot without a recording.
TEAM_UNKNOWN = -1
_PROBE_MASK = np.uint16((1 << len(BODY_PROBES)) - 1)
_HEAD_MASK = np.uint16(0b111)


@dataclass(frozen=True)
class PlayerTicks:
    """One player's tick table as the labels read it.

    Attributes:
        tick (np.ndarray): ``[n]`` int64, strictly increasing engine ticks.
        states (np.ndarray): ``[n, 5]`` float64 ``x, y, z, yaw, pitch`` (feet; u, degrees).
        alive (np.ndarray): ``[n]`` bool, ``is_alive``.
        crouched (np.ndarray): ``[n]`` bool, ``duck`` is held.
        team (int): the team of the table's first row (2 T, 3 CT).
        death_tick (int | None): the tick the player died at; None if it lived to the round's end.
    """

    tick: np.ndarray
    states: np.ndarray
    alive: np.ndarray
    crouched: np.ndarray
    team: int
    death_tick: int | None


@dataclass(frozen=True)
class FrameState:
    """A player's state at each queried tick.

    Attributes:
        valid (np.ndarray): ``[T]`` bool, the table covers the tick and the state is finite.
        alive (np.ndarray): ``[T]`` bool, valid, alive in its row and before its death tick.
        known_dead (np.ndarray): ``[T]`` bool, dead (by its death tick or its row): known, not
            merely missing.
        crouched (np.ndarray): ``[T]`` bool.
        states (np.ndarray): ``[T, 5]`` float64.
    """

    valid: np.ndarray
    alive: np.ndarray
    known_dead: np.ndarray
    crouched: np.ndarray
    states: np.ndarray


def frame_ticks(first_tick: int, fps: float, frames: int) -> np.ndarray:
    """``[frames]`` int64 engine tick of every video frame: the capture's first tick, then every
    ``TICK_RATE / fps`` ticks."""
    per_frame = TICK_RATE / float(fps)
    if not np.isclose(per_frame, round(per_frame), atol=1e-9):
        raise ValueError(f"the tick rate {TICK_RATE} is not a multiple of the frame rate {fps}")
    return int(first_tick) + int(round(per_frame)) * np.arange(int(frames), dtype=np.int64)


def frame_state(player: PlayerTicks, queries: np.ndarray) -> FrameState:
    """The player's state at the ticks ``queries``: the last row at or before each, within
    :data:`MAX_TICK_LAG` ticks (no interpolation)."""
    rows = np.searchsorted(player.tick, queries, side="right") - 1
    covered = rows >= 0
    rows = np.clip(rows, 0, len(player.tick) - 1)
    lag = queries - player.tick[rows]
    covered &= (lag >= 0) & (lag <= MAX_TICK_LAG)
    valid = covered & np.isfinite(player.states[rows]).all(axis=1)
    explicit_dead = (
        np.zeros(len(queries), dtype=bool)
        if player.death_tick is None
        else queries >= player.death_tick
    )
    row_dead = valid & ~player.alive[rows]
    return FrameState(
        valid=valid,
        alive=valid & player.alive[rows] & ~explicit_dead,
        known_dead=explicit_dead | row_dead,
        crouched=player.crouched[rows] & valid,
        states=player.states[rows],
    )


def _popcount(masks: np.ndarray) -> np.ndarray:
    return sum(((masks >> shift) & 1).astype(np.uint8) for shift in range(len(BODY_PROBES)))


def observer_labels(
    observer_slot: int,
    roster: dict[int, PlayerTicks],
    *,
    fps: float,
    frames: int,
    round_end_tick: int,
    scoped: np.ndarray,
    occluders: Occluders,
    capsule_unknown: bool,
) -> tuple[dict[str, np.ndarray], int]:
    """The labels of one recording: its player against every other player of the round.

    A frame is evaluated when the player is alive, the round is on and the view is unscoped, and
    the target is alive; the target is tested with the map and with the other players' capsules.
    Known and evaluated labels are separate: false outside ``_binary_eval_valid`` is unknown, not
    occluded.

    Args:
        observer_slot (int): the recording's player slot.
        roster (dict[int, PlayerTicks]): every recorded player of the round by slot.
        fps (float): the video's frame rate.
        frames (int): the video's frame count ``T``.
        round_end_tick (int): the last tick of the round; later frames are not evaluated.
        scoped (np.ndarray): ``[T]`` bool, frames seen through a scope; not evaluated.
        occluders (Occluders): the map.
        capsule_unknown (bool): a target that only another player's capsule hides, while the map
            leaves a probe clear, is unknown instead of occluded.

    Returns:
        tuple[dict[str, np.ndarray], int]: the arrays of the label file (without ``_meta``) and
        the number of cells ``capsule_unknown`` moved from occluded to unknown.
    """
    me = roster[observer_slot]
    queries = frame_ticks(me.tick[0], fps, frames)
    round_active = queries <= round_end_tick
    observer = frame_state(me, queries)
    view_valid = ~scoped
    resolved = {slot: frame_state(player, queries) for slot, player in roster.items()}
    states = np.zeros((frames, NUM_PLAYERS, 5))
    alive = np.zeros((frames, NUM_PLAYERS), dtype=bool)
    crouched = np.zeros((frames, NUM_PLAYERS), dtype=bool)
    known = np.zeros((frames, NUM_PLAYERS), dtype=bool)
    for slot, state in resolved.items():
        states[:, slot], alive[:, slot], crouched[:, slot] = (
            state.states,
            state.alive,
            state.crouched,
        )
        known[:, slot] = state.valid | state.known_dead

    shape = (NUM_PLAYERS, frames)
    arrays = {
        "_frame_tick": queries.astype(np.int32),
        "_round_active": round_active,
        "_observer_state_valid": observer.valid,
        "_observer_alive": observer.alive,
        "_observer_duck": observer.crouched,
        "_scoped": scoped.copy(),
        "_view_geometry_valid": view_valid,
        "_view_texture_usable": np.ones(frames, dtype=bool),
        "_frame_tick_calibration_valid": np.ones(frames, dtype=bool),
        "_slot_expected": np.ones(NUM_PLAYERS, dtype=bool),
        "_slot_present": np.zeros(NUM_PLAYERS, dtype=bool),
        "_slot_team_num": np.full(NUM_PLAYERS, TEAM_UNKNOWN, dtype=np.int8),
        "_same_team": np.zeros(NUM_PLAYERS, dtype=bool),
    }
    for name in (
        "_target_state_valid _target_alive _target_known_dead _target_duck _in_frustum"
        " _static_los_lower _static_los_upper _map_label_valid _los_lower _los_upper"
        " _geometry_label_valid _label_valid _blocker_state_complete _binary_eval_valid"
        " _binary_visible _binary_invisible _binary_train_valid _binary_offscreen"
    ).split():
        arrays[name] = np.zeros(shape, dtype=bool)
    for name in "_tested_count _clear_count_lower _clear_count_upper".split():
        arrays[name] = np.zeros(shape, dtype=np.uint8)
    for name in (
        "_projected_mask _clear_mask_lower _clear_mask_upper _alpha_ambiguous_mask"
        " _player_proxy_ambiguous_mask _unknown_blocker_ambiguous_mask"
        " _player_possible_ambiguous_mask _possible_ambiguous_mask _player_proxy_blocker_slots"
        " _unknown_blocker_slots _binary_player_blocked_mask _binary_clear_mask"
    ).split():
        arrays[name] = np.zeros(shape, dtype=np.uint16)

    for slot, player in roster.items():
        arrays["_slot_present"][slot] = True
        arrays["_slot_team_num"][slot] = player.team
        arrays["_same_team"][slot] = player.team == me.team
        if slot == observer_slot:
            continue
        target = resolved[slot]
        pair_valid = (
            observer.valid
            & observer.alive
            & round_active
            & view_valid
            & target.valid
            & target.alive
        )
        map_test = line_of_sight(
            observer.states,
            target.states,
            observer_crouched=observer.crouched,
            target_crouched=target.crouched,
            valid=pair_valid,
            occluders=occluders,
        )
        others = np.asarray([s for s in range(NUM_PLAYERS) if s not in (observer_slot, slot)])
        capsules = player_capsule_blocks(
            observer.states,
            observer.crouched,
            map_test.probes,
            map_test.projected,
            states[:, others],
            alive[:, others],
            crouched[:, others],
            known[:, others],
            others,
        )
        test = apply_player_capsules(map_test, capsules)
        in_frustum = map_test.in_frustum
        label_valid = (
            (observer.valid & view_valid)
            & observer.alive
            & round_active
            & (target.known_dead | (target.valid & target.alive & test.label_valid))
        )
        player_blocked = test.proxy_ambiguous & map_test.clear_upper
        clear = map_test.clear_upper & np.bitwise_and(~player_blocked, _PROBE_MASK)
        count = _popcount(clear)
        raw_visible = ((clear & _HEAD_MASK) != 0) | (count >= 2)
        decisive = (count == 0) | raw_visible
        eval_valid = pair_valid & in_frustum & capsules.complete & decisive
        visible = eval_valid & raw_visible
        row = {
            "_target_state_valid": target.valid,
            "_target_alive": target.alive,
            "_target_known_dead": target.known_dead,
            "_target_duck": target.crouched,
            "_in_frustum": in_frustum,
            "_static_los_lower": map_test.los_lower,
            "_static_los_upper": map_test.los_upper,
            "_map_label_valid": map_test.label_valid,
            "_los_lower": test.los_lower,
            "_los_upper": map_test.los_upper,
            "_geometry_label_valid": test.label_valid,
            "_label_valid": label_valid,
            "_tested_count": map_test.tested_count,
            "_clear_count_lower": test.clear_count_lower,
            "_clear_count_upper": map_test.clear_count_upper,
            "_projected_mask": map_test.projected,
            "_clear_mask_lower": test.clear_lower,
            "_clear_mask_upper": map_test.clear_upper,
            "_alpha_ambiguous_mask": map_test.ambiguous,
            "_player_proxy_ambiguous_mask": test.proxy_ambiguous,
            "_unknown_blocker_ambiguous_mask": test.unknown_ambiguous,
            "_player_possible_ambiguous_mask": test.possible_ambiguous,
            "_possible_ambiguous_mask": map_test.ambiguous | test.possible_ambiguous,
            "_player_proxy_blocker_slots": np.bitwise_or.reduce(
                capsules.blocker_slots, axis=1, initial=np.uint16(0)
            ),
            "_unknown_blocker_slots": capsules.unknown_slots,
            "_blocker_state_complete": capsules.complete,
            "_binary_player_blocked_mask": player_blocked,
            "_binary_clear_mask": clear,
            "_binary_eval_valid": eval_valid,
            "_binary_visible": visible,
            "_binary_invisible": eval_valid & ~raw_visible,
            "_binary_train_valid": eval_valid,
            "_binary_offscreen": pair_valid & ~in_frustum,
        }
        for name, values in row.items():
            arrays[name][slot] = values
        arrays[f"slot_{slot}"] = visible

    moved = 0
    if capsule_unknown:
        only_players = hidden_by_players(arrays)
        for name in ("_binary_eval_valid", "_binary_train_valid", "_binary_invisible"):
            arrays[name] = arrays[name] & ~only_players
        moved = int(only_players.sum())
    return arrays, moved


def hidden_by_players(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """``[10, T]`` bool: the cells labelled occluded that only another player's capsule closes,
    while the map leaves a probe clear (the cells ``capsule_unknown`` makes unknown).

    Args:
        arrays (dict[str, np.ndarray]): a label file's arrays.
    """
    occluded = arrays["_binary_eval_valid"] & ~arrays["_binary_visible"] & arrays["_in_frustum"]
    return (
        occluded
        & arrays["_target_alive"]
        & (arrays["_binary_player_blocked_mask"] != 0)
        & (arrays["_static_los_upper"] | arrays["_static_los_lower"])
    )


def encode_meta(meta: dict) -> np.ndarray:
    """``_meta``: the canonical JSON of ``meta`` as a ``uint8`` array."""
    text = json.dumps(
        meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8)


def decode_meta(meta: np.ndarray) -> dict:
    """The dict of a ``_meta`` array."""
    return json.loads(np.asarray(meta, dtype=np.uint8).tobytes().decode("ascii"))
