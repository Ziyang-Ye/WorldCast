"""The controls: ticks aligned to video frames, to ordered substeps and to the state model's
latent frames; the camera encoding, the buttons as bits and the weapon vocabulary."""

import numpy as np
import pytest

from worldcast.data.controls import (
    CONTROL_BUTTONS,
    OPENCS2_BUTTONS,
    OPENCS2_WEAPONS,
    AlignmentError,
    align_ticks_to_ordered_substeps,
    align_ticks_to_video_frames,
    check_camera_encoding,
    check_weapon_ids,
    count_until,
    encode_turn,
    encode_weapon,
    frame_tick_ranges,
    normalize_weapon_name,
    quantize_camera_delta,
    state_model_controls,
)

#: 33 ticks at 64 Hz: ``forward`` held throughout, ``attack`` at tick 6 only (with a keyboard turn,
#: which is not a button), a yaw turn of 0.5 degrees per tick, a pitch turn of -0.25 degrees at tick
#: 3, and the weapon changing from tick 7 on.
N = 33
TIMES = np.arange(N) / 64.0
HELD = [["forward", "attack", "turn_left"] if tick == 6 else ["forward"] for tick in range(N)]
PITCH = np.where(np.arange(N) == 3, -0.25, 0.0).astype(np.float32)
YAW = np.full(N, 0.5, np.float32)
WEAPONS = ["weapon_ak47"] * 7 + ["weapon_awp"] * (N - 7)
TICKS = dict(timestamps=TIMES, held_buttons=HELD, delta_pitch=PITCH, delta_yaw=YAW)
GAP = 2.5 / 64.0


def test_a_frame_owns_the_ticks_up_to_its_own_time():
    assert count_until(TIMES[:3], 1 / 64) == 2
    assert count_until(TIMES[:3], 1 / 64 - 1e-9) == 2  # within the slack of the time base
    assert count_until(TIMES[:3], 1 / 64 - 1e-6) == 1
    ranges = dict(timestamps=TIMES, source_fps=32.0, max_tick_gap_seconds=GAP)
    # a video frame is two source frames, four ticks: frame f owns the ticks 4f - 3 .. 4f
    begin, end = frame_tick_ranges(**ranges, start_frame=0, num_frames=4, stride=2)
    assert (begin.tolist(), end.tolist()) == ([0, 1, 5, 9], [1, 5, 9, 13])
    begin, end = frame_tick_ranges(**ranges, start_frame=4, num_frames=2, stride=2)
    assert (begin.tolist(), end.tolist()) == ([5, 9], [9, 13])
    # a latent frame is eight source frames, sixteen ticks
    begin, end = frame_tick_ranges(**ranges, start_frame=0, num_frames=3, stride=8)
    assert (begin.tolist(), end.tolist()) == ([0, 1, 17], [1, 17, 33])


def test_ticks_that_do_not_cover_a_frame_are_refused():
    window = dict(start_frame=0, num_frames=3, source_fps=32.0, stride=2)
    with pytest.raises(AlignmentError, match="no tick rows align to frame 0"):
        frame_tick_ranges(timestamps=TIMES[8:], max_tick_gap_seconds=GAP, **window)
    # without the ticks 3 and 4, the last tick of frame 1 is 2 / 64 s before the frame
    holed = np.delete(TIMES, [3, 4])
    with pytest.raises(AlignmentError, match="leaves the end of frame 1 uncovered by 0.031250s"):
        frame_tick_ranges(timestamps=holed, max_tick_gap_seconds=1.5 / 64.0, **window)
    # without the ticks 2 and 3, frame 1 ends on a tick but its ticks are 3 / 64 s apart
    holed = np.delete(TIMES, [2, 3])
    with pytest.raises(AlignmentError, match="a tick gap of 0.046875s exceeds 0.039062s"):
        frame_tick_ranges(timestamps=holed, max_tick_gap_seconds=GAP, **window)
    with pytest.raises(AlignmentError, match="strictly increasing"):
        state_model_controls(
            **{**TICKS, "timestamps": TIMES[::-1]}, start_frame=0, source_fps=32.0, latent_frames=2
        )


def test_the_clients_controls_per_video_frame():
    controls = align_ticks_to_video_frames(
        **TICKS,
        input_weapons=WEAPONS,
        start_frame=0,
        num_frames=5,
        source_fps=32.0,
        max_tick_gap_seconds=GAP,
        camera_encoding="noclip",
    )
    assert controls.buttons.shape == (5, 11) and controls.buttons.dtype == np.float32
    assert controls.buttons[:, CONTROL_BUTTONS.index("forward")].tolist() == [1, 1, 1, 1, 1]
    assert controls.buttons[:, CONTROL_BUTTONS.index("attack")].tolist() == [0, 0, 1, 0, 0]
    assert controls.buttons.sum() == 6  # nothing else is held
    # the turn of a frame's ticks, mu-law encoded: frame 0 has one tick, the others four
    assert controls.view_deltas[:, 1].tolist() == pytest.approx([0.05, 0.175, 0.175, 0.175, 0.175])
    assert controls.view_deltas[:, 0].tolist() == pytest.approx([0.0, -0.025, 0.0, 0.0, 0.0])
    # the weapon held at the frame's last tick: frame 2 owns the ticks 5 .. 8
    assert [OPENCS2_WEAPONS[i] for i in controls.weapon] == ["ak47", "ak47", "awp", "awp", "awp"]


def test_the_players_controls_in_four_ordered_substeps_per_video_frame():
    substeps = align_ticks_to_ordered_substeps(
        **TICKS, start_frame=0, num_frames=4, source_fps=32.0, max_tick_gap_seconds=GAP
    )
    assert substeps.values.shape == (4, 4, 13) and substeps.values.dtype == np.float32
    # frame 0 holds tick 0 alone, in its last substep; a later frame one tick per substep
    assert substeps.valid.tolist() == [[False, False, False, True]] + [[True] * 4] * 3
    assert substeps.values[0, :, CONTROL_BUTTONS.index("forward")].tolist() == [0, 0, 0, 1]
    # frame 1 = ticks 1 .. 4 in order: the pitch turn of tick 3 is in substep 2; turns are / 5
    assert substeps.values[1, :, -2].tolist() == pytest.approx([0.0, 0.0, -0.05, 0.0])
    assert substeps.values[1, :, -1].tolist() == pytest.approx([0.1, 0.1, 0.1, 0.1])
    # frame 2 = ticks 5 .. 8: the attack of tick 6 is in substep 1
    assert substeps.values[2, :, CONTROL_BUTTONS.index("attack")].tolist() == [0, 1, 0, 0]


def test_two_ticks_of_one_substep_add_their_turns_in_float32():
    """As trained. On the recordings' 64 Hz ticks a substep holds one tick; a tick 2 ms late shares
    the substep of the next one, and their turns of 0.1 and 0.5 degrees add up in float32: to the
    float32 above 0.12, not to the one nearest it."""
    times = np.array([0.0, 1 / 64, 2 / 64 + 0.002, 3 / 64, 4 / 64])
    substeps = align_ticks_to_ordered_substeps(
        timestamps=times,
        held_buttons=[["forward"]] * 5,
        delta_pitch=np.zeros(5, np.float32),
        delta_yaw=np.array([0.0, 0.0, 0.1, 0.5, 0.0], np.float32),
        start_frame=0,
        num_frames=2,
        source_fps=32.0,
        max_tick_gap_seconds=GAP,
    )
    assert substeps.valid[1].tolist() == [True, False, True, True]
    assert float(substeps.values[1, 2, -1]) == 0.12000000476837158
    assert float(np.float32(0.12)) == 0.11999999731779099


def test_the_state_models_controls_in_sixteen_substeps_per_latent_frame():
    controls = state_model_controls(**TICKS, start_frame=0, source_fps=32.0, latent_frames=3)
    assert controls.shape == (3, 16, 16) and controls.dtype == np.float32
    forward, attack = OPENCS2_BUTTONS.index("forward"), OPENCS2_BUTTONS.index("attack")
    pitch, yaw, valid = 13, 14, 15
    # latent frame 0 holds tick 0 alone; latent frame 1 the ticks 1 .. 16, one per substep
    assert controls[0, :, valid].tolist() == [0.0] * 15 + [1.0]
    assert controls[1, :, valid].tolist() == [1.0] * 16
    assert controls[1, :, forward].tolist() == [1.0] * 16
    assert controls[1, :, attack].tolist() == [0.0] * 5 + [1.0] + [0.0] * 10  # tick 6
    assert controls[1, :4, pitch].tolist() == pytest.approx([0.0, 0.0, -0.05, 0.0])  # tick 3
    assert controls[1, :, yaw].tolist() == pytest.approx([0.1] * 16)
    assert not controls[1, :, pitch][3:].any() and not controls[2, :, pitch].any()
    # the state model reads a latent frame only with a tick within 1.5 ticks of its end
    with pytest.raises(AlignmentError, match="uncovered"):
        state_model_controls(
            **{key: value[:31] for key, value in TICKS.items()},
            start_frame=0,
            source_fps=32.0,
            latent_frames=3,
        )


def test_an_unknown_button_is_refused():
    held = [["forward"]] * 4 + [["fly"]] + [["forward"]] * (N - 5)
    with pytest.raises(AlignmentError, match=r"unknown OpenCS2 buttons: \['fly'\]"):
        align_ticks_to_ordered_substeps(
            **{**TICKS, "held_buttons": held},
            start_frame=0,
            num_frames=4,
            source_fps=32.0,
            max_tick_gap_seconds=GAP,
        )


def test_the_camera_turn_is_mu_law_encoded_in_half_degree_buckets():
    encoded = quantize_camera_delta([0.0, 20.0], clip=True)
    assert encoded.dtype == np.float32 and encoded.tolist() == [0.0, 1.0]
    # 20 sign(x / 20) log1p(2.7 |x / 20|) / log1p(2.7), rounded to 0.5 degrees, over 20
    assert quantize_camera_delta([-1.0, 1.0], clip=True).tolist() == pytest.approx([-0.1, 0.1])
    assert quantize_camera_delta([-0.25, 2.0], clip=True).tolist() == pytest.approx([-0.025, 0.175])
    turns = np.linspace(-20, 20, 81)
    encoded = np.stack([quantize_camera_delta([t, t], clip=True) for t in turns])
    assert (np.diff(encoded[:, 0]) >= 0).all()


def test_noclip_keeps_a_turn_of_more_than_20_degrees():
    assert quantize_camera_delta([30.0, -45.0], clip=True).tolist() == [1.0, -1.0]
    assert quantize_camera_delta([30.0, -45.0], clip=False).tolist() == [1.25, -1.5]
    below = [5.0, -19.5]
    assert np.array_equal(
        quantize_camera_delta(below, clip=True), quantize_camera_delta(below, clip=False)
    )
    with pytest.raises(ValueError):
        quantize_camera_delta([np.nan, 0.0], clip=False)
    with pytest.raises(ValueError):
        check_camera_encoding("carry")


def test_a_turn_is_encoded_under_the_name_of_an_encoding():
    turn = [30.0, -45.0]
    assert encode_turn(turn).tolist() == quantize_camera_delta(turn, clip=False).tolist()
    assert encode_turn(turn, "noclip").tolist() == [1.25, -1.5]
    assert encode_turn(turn, "clip").tolist() == [1.0, -1.0]
    with pytest.raises(ValueError, match="camera_encoding must be one of"):
        encode_turn(turn, "carry")


def test_weapon_names_are_normalised_to_the_vocabulary():
    assert normalize_weapon_name("weapon_AK47 ") == "ak47"
    assert normalize_weapon_name(None) == "<none>" and normalize_weapon_name("laser") == "<unk>"
    assert OPENCS2_WEAPONS[encode_weapon("weapon_awp")] == "awp"
    assert len(OPENCS2_WEAPONS) == 52
    check_weapon_ids(np.array([0, 51], np.uint8))
    check_weapon_ids(np.zeros(0, np.int64))
    for weapon in (-1, 52):
        with pytest.raises(ValueError, match=r"weapon ids must lie in \[0, 52\)"):
            check_weapon_ids(np.array([2, weapon]))
