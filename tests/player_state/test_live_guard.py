"""``live.wall_keys: guard``: the key release that cannot leave a player stuck. It clears a block's
movement keys where a wall the depth head sees stops the camera (the plane kept from one included)
and nowhere else: never for the edge of the walkable extent (the position estimate is wrong by
construction there), for at most ``CLEAR_BLOCKS`` blocks in a row, the next block passing its keys
as a probe; not once the player's keys change or are all released, or the turn takes the way off
the stop; and not after the motion head read the picture moving while the keys were cleared.
``held``, ``edge`` and ``walls`` are as they were (``test_live.py``).
"""

import numpy as np
import pytest
import torch

from tests.player_state.test_live import (
    FORWARD,
    LATENTS,
    LEFT,
    STRIP,
    CameraReader,
    LostReader,
    Room,
    block_motion,
    keys,
    live_loop,
    play,
    play_released,
    prepare,
    release,
    static_wall_loop,
    stop_row,
)
from worldcast.config.inference import WALL_KEY_MODES, LiveConfig
from worldcast.player_state.live import CLEAR_BLOCKS


def guarded(reader=None):
    """A client that pushes into the wall 300 u ahead (the depth head sees it) under ``guard``."""
    return static_wall_loop(reader or LostReader(), wall_keys="guard")


def rows_of(loop, blocks, pressed=None):
    """The row each block of ``blocks`` clears the forward key from (16: none), the blocks played as
    the rollout's cut plays them."""
    return [
        stop_row(play_released(loop, s, keys(FORWARD) if pressed is None else pressed))
        for s in blocks
    ]


def test_guard_is_a_setting_that_is_checked_and_the_other_values_are_what_they_were():
    assert (
        WALL_KEY_MODES == ("held", "edge", "walls", "guard")
        and LiveConfig(wall_keys="guard").wall_keys == "guard"
    )
    with pytest.raises(ValueError, match="live.wall_keys"):
        LiveConfig(wall_keys="stuck")
    assert LiveConfig().wall_keys == "held"


def test_the_edge_of_the_extent_never_clears_the_keys_in_guard_and_does_in_walls():
    """On a strip of the map that ends at x = 768 u the blocks run on with their keys to the edge;
    there ``walls`` clears them from the row the edge stops the camera at, and every block after it,
    where ``guard`` keeps them: the camera stops at the edge (the position is not moved past it) and
    the player's keys stay in the controls."""
    rows = {}
    for mode in ("walls", "guard"):
        loop = live_loop(Room(), 0, reader=CameraReader(), walkable=STRIP, wall_keys=mode)
        rows[mode] = [stop_row(play_released(loop, s, keys(FORWARD))) for s in range(1, LATENTS, 4)]
        x = loop.batch["player_states"][0, 0, :, 0].numpy()
        assert 750.0 < x.max() <= 768.0  # the camera stops at the edge in both
    assert min(rows["walls"]) == 0 and rows["walls"].count(0) >= 2
    assert rows["guard"] == [16] * len(rows["guard"])


def test_a_stuck_stretch_is_bounded_clear_for_clear_blocks_then_one_block_of_probe():
    """The wall is found at block 5; its block and the blocks after it are cleared, ``CLEAR_BLOCKS``
    in a row, and the block after them passes the keys (the camera is still stopped there: the plane
    stands), then the clearing starts again: the player never holds the key more than
    ``CLEAR_BLOCKS`` blocks with nothing happening."""
    loop = guarded()
    assert rows_of(loop, [1]) == [16]
    rows = rows_of(loop, range(5, LATENTS, 4))
    assert len(rows) >= 7 and 0 <= rows[0] < 16
    cleared = [row < 16 for row in rows]
    longest = run = 0
    for c in cleared:
        run = run + 1 if c else 0
        longest = max(longest, run)
    assert longest == CLEAR_BLOCKS
    assert cleared[:CLEAR_BLOCKS] == [True] * CLEAR_BLOCKS and cleared[CLEAR_BLOCKS] is False
    assert cleared[CLEAR_BLOCKS + 1 : 2 * CLEAR_BLOCKS + 1] == [True] * CLEAR_BLOCKS
    assert [r for r, c in zip(rows, cleared) if not c] == [16] * cleared.count(False)
    assert cleared.count(False) >= 1
    # the probe block's camera is still stopped by the plane: the keys pass, the camera stands
    probe = 5 + 4 * CLEAR_BLOCKS
    camera, _ = block_motion(loop, probe)
    assert np.abs(camera).max() < 1e-3


def test_a_stop_with_no_wall_in_the_picture_ends_in_one_probe_when_the_view_moves():
    """The depth head stopped a block where the picture has no wall: the cleared keys keep the
    camera and the picture standing for ``CLEAR_BLOCKS`` blocks; the probe block passes the keys,
    the generator walks, the motion head reads the picture moving past a camera that was held back
    and concedes (``moved_past_wall``), so that the next block is not stopped."""

    class Walks(LostReader):
        """The motion head reads the view moving 55 u a latent frame whenever the player's keys were
        not cleared."""

        def read(self, latents, latent_frames):
            place, steps = super().read(latents, latent_frames)
            if self.probing:
                steps[:] = np.array([55.0, 0.0, 0.0], np.float32) / 64.0
            return place, steps

    reader = Walks()
    reader.probing = False
    loop = guarded(reader)
    rows_of(loop, [1])
    cleared = 0
    for s in range(5, 5 + 4 * 12, 4):
        taken = play_released(loop, s, keys(FORWARD))
        if stop_row(taken) < 16:
            cleared += 1
        else:
            reader.probing = (
                True  # the block after the probe is read by the motion head as the view that walked
            )
        if reader.probing and loop.position.moved_past_wall:
            break
    assert cleared == CLEAR_BLOCKS
    assert loop._moved_on is True or loop.position.moved_past_wall


def test_changed_or_released_keys_end_the_stretch_at_once_and_a_new_press_has_the_whole_budget():
    loop = guarded()
    rows_of(loop, [1, 5])
    assert stop_row(play_released(loop, 9, keys(FORWARD))) == 0 and loop._clear_run == 2
    # the keys are released: the block is the player's, the count starts again
    prepare(loop, 13)
    idle = keys()
    assert release(loop, 13, idle) is idle and loop._clear_run == 0
    loop.take_controls(
        13,
        idle,
        np.zeros((16, 2), np.float32),
        np.full(16, 1, np.int64) * loop.batch["player_weapon_ids"][0, 0, 0].item(),
        np.arange(16) / 16.0 + 13.0,
    )
    # pressed again at the same wall: cleared at once, with the budget of a new press
    assert stop_row(play_released(loop, 17, keys(FORWARD), clock=2.0)) == 0 and loop._clear_run == 1
    # another key pressed with it is another movement: the stretch ends, the count is 0 or 1 of a
    # new one, never the old run's
    prepare(loop, 21)
    both = keys(FORWARD, LEFT)
    out = release(loop, 21, both)
    assert (
        loop._clear_keys == (True, False, True, False)
        and loop._clear_run <= 1
        and (out is both or loop._clear_run == 1)
    )


def test_a_turn_off_the_stop_frees_the_block_at_once():
    loop = guarded()
    rows_of(loop, [1, 5, 9])
    pressed = keys(FORWARD)
    prepare(loop, 13)
    assert (
        release(loop, 13, pressed, turn=(0.0, 4.0)) is pressed
    )  # 64 degrees over the rows: the way is off the stop
    assert loop._clear_run == 0 and loop._clear_off is False
    # a turn of a few degrees keeps facing the wall: still held
    loop = guarded()
    rows_of(loop, [1, 5, 9])
    prepare(loop, 13)
    assert stop_row(release(loop, 13, keys(FORWARD), turn=(0.0, 0.3))) == 0


def test_the_picture_moving_while_the_keys_are_cleared_ends_the_clearing_until_the_keys_change():
    class Moving(LostReader):
        """The motion head reads the view walking at 55 u a latent frame whatever the camera is."""

        def read(self, latents, latent_frames):
            place, steps = super().read(latents, latent_frames)
            steps[:] = np.array([55.0, 0.0, 0.0], np.float32) / 64.0
            return place, steps

    loop = guarded(Moving())
    rows_of(loop, [1])
    taken = play_released(loop, 5, keys(FORWARD))
    assert 0 <= stop_row(taken) <= 8 and loop._cleared_from is not None
    prepare(
        loop, 9
    )  # the motion head read block 5's frames: the picture walked while the keys were cleared
    assert loop._clear_off is True
    pressed = keys(FORWARD)
    assert release(loop, 9, pressed) is pressed  # not cleared: the clearing is no use here
    # the player lets go: the next press is a new try
    idle = keys()
    assert release(loop, 9, idle) is idle and loop._clear_off is False


def test_a_picture_that_stands_does_not_end_the_clearing():
    loop = guarded(CameraReader())  # its motion head reads the camera, held at the wall
    rows_of(loop, [1])
    assert 0 <= stop_row(play_released(loop, 5, keys(FORWARD))) < 16
    prepare(loop, 9)
    assert loop._clear_off is False and stop_row(release(loop, 9, keys(FORWARD))) == 0


def test_a_block_that_leaves_the_ground_keeps_its_keys_and_does_not_count():
    loop = guarded()
    rows_of(loop, [1, 5])
    run = loop._clear_run
    jumping = keys(FORWARD)
    jumping[:2, 4] = 1.0  # jump
    prepare(loop, 9)
    assert release(loop, 9, jumping) is jumping and loop._clear_run == 0 and run >= 1


def test_guard_is_the_released_game_where_nothing_stops_the_player():
    """No wall, no edge: every block's rows are those of ``held``, the keys the player's."""
    guard, held = live_loop(Room(), 0, wall_keys="guard"), live_loop(Room(), 0)
    for s, pressed in zip(range(1, LATENTS, 4), [keys(FORWARD), keys(FORWARD, LEFT)] * 4):
        prepare(guard, s)
        assert release(guard, s, pressed) is pressed
        guard.take_controls(
            s,
            pressed,
            np.zeros((16, 2), np.float32),
            np.full(16, guard.batch["player_weapon_ids"][0, 0, 0].item(), np.int64),
            np.arange(16) / 16.0 + s,
        )
        play(held, s, pressed)
    for name in ("displacement", "prior_displacement", "yaw", "pitch"):
        assert np.array_equal(getattr(guard.motion, name), getattr(held.motion, name)), name
    assert torch.equal(guard.batch["player_states"], held.batch["player_states"])
