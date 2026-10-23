"""The records a worker and its client exchange: the wire form of a player state, of the field's
markers, of how much of the player's turn a frame holds and of the controls of its row, a block's
live controls, the outbox."""

import numpy as np
import pytest

from worldcast.data.controls import CONTROL_BUTTONS
from worldcast.demo.clients import (
    ClientMessage,
    LiveControls,
    LiveFrame,
    LivePlayerState,
    Outbox,
    button_bits,
    make_client,
)
from worldcast.demo.config import load_config
from worldcast.demo.library import synthetic_library
from worldcast.demo.mock import MockClient
from worldcast.engine.inference.controls import LiveBlockControls
from worldcast.engine.inference.field_rows import FieldRows


def test_a_player_state_round_trips_on_the_wire():
    state = LivePlayerState(
        3, 12.34567, 100.126, -7.0, 64.0, 179.999, -12.5, alive=False, vx=1.5, src=1, drawn=False
    )
    wire = state.wire()
    assert wire == [3, 12.3457, 100.13, -7.0, 64.0, 180.0, -12.5, 0, 1.5, 0.0, 1, 0]
    assert LivePlayerState.from_wire(wire) == LivePlayerState(
        3, 12.3457, 100.13, -7.0, 64.0, 180.0, -12.5, alive=False, vx=1.5, src=1, drawn=False
    )
    # a player's own state: no label of its own, null on the wire
    own = LivePlayerState(0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert own.wire()[-2:] == [0, None] and LivePlayerState.from_wire(own.wire()) == own


def test_the_wire_codes_of_a_players_source_are_the_field_rows_codes():
    assert (FieldRows.OWN, FieldRows.LIVE, FieldRows.RECORDED) == (0, 1, 2)


#: A frame's pixels and state, for its wire forms.
PIXELS, STATE = np.zeros((1, 1, 3), np.uint8), LivePlayerState(0, 0.0, 0, 0, 0, 0, 0)


def test_the_markers_go_on_the_wire_rounded():
    frame = LiveFrame(
        0, PIXELS, STATE, (0.0, 0.0), 0.0, markers=[(5, 0.359375, 0.52666667, 1600.04)]
    )
    assert frame.marker_wire() == [[5, 0.3594, 0.5267, 1600.0]]


def test_the_turn_a_frame_holds_goes_on_the_wire_as_the_page_sends_it():
    """The turn with the four decimals of the page's input samples (``static/input.js``), the
    camera's pitch as the other angles. A client says both of every frame: neither has a
    default."""
    frame = LiveFrame(0, PIXELS, STATE, (-3.000049, 721.12346), 12.3456)
    assert frame.look_wire() == {"tn": [-3.0, 721.1235], "cp": 12.35}
    with pytest.raises(TypeError, match="'turned' and 'camera_pitch'"):
        LiveFrame(0, PIXELS, STATE)


def test_the_controls_of_a_frames_row_go_on_the_wire_as_the_page_sends_them():
    """A frame's header says the buttons held in its row of the controls (a button held for any
    share of the frame is held) and the weapon, as the page's input samples do (``b``, ``w``):
    the instant turn keeps the frames of a moving viewmodel out of its panorama."""
    row = np.zeros(len(CONTROL_BUTTONS), np.float32)
    row[CONTROL_BUTTONS.index("forward")] = 1.0
    row[CONTROL_BUTTONS.index("reload")] = 0.25
    bits = button_bits(row)
    assert bits == (1 << CONTROL_BUTTONS.index("forward")) | (1 << CONTROL_BUTTONS.index("reload"))
    frame = LiveFrame(0, PIXELS, STATE, (0.0, 0.0), 0.0, buttons=bits, weapon=7)
    assert frame.controls_wire() == {"b": bits, "w": 7}
    assert LiveFrame(0, PIXELS, STATE, (0.0, 0.0), 0.0).controls_wire() == {"b": 0, "w": 0}


def test_live_controls_are_a_blocks_live_controls():
    rows = 4
    controls = LiveControls(
        buttons=np.zeros((rows, 11)),
        view_deltas=np.zeros((rows, 2)),
        weapon=np.zeros(rows),
        turn=np.zeros((rows, 2)),
        times=np.arange(rows) / 16.0,
        input_seq=np.full(rows, -1),
        turned=np.zeros((rows, 2)),
    )
    assert isinstance(controls, LiveBlockControls) and controls.num_frames == rows
    assert controls.turn.dtype == np.float32 and controls.times.dtype == np.float64
    assert controls.turned.dtype == np.float64
    with pytest.raises(ValueError, match=r"have turn \[4, 2\] and times \[4\]"):
        LiveControls(
            buttons=np.zeros((rows, 11)),
            view_deltas=np.zeros((rows, 2)),
            weapon=np.zeros(rows),
            turn=np.zeros((rows, 2)),
            times=np.zeros(rows + 1),
            input_seq=np.full(rows, -1),
            turned=np.zeros((rows, 2)),
        )
    with pytest.raises(ValueError, match=r"have turned \[4, 2\]; got \(4,\)"):
        LiveControls(
            buttons=np.zeros((rows, 11)),
            view_deltas=np.zeros((rows, 2)),
            weapon=np.zeros(rows),
            turn=np.zeros((rows, 2)),
            times=np.zeros(rows),
            input_seq=np.full(rows, -1),
            turned=np.zeros(rows),
        )


def test_the_outbox_hands_each_message_out_once():
    outbox = Outbox()
    first, second = (ClientMessage(0, kind, {}, b"") for kind in ("live", "scene"))
    outbox.put(first)
    outbox.put(second)
    assert outbox.take_messages() == [first, second] and outbox.take_messages() == []


def test_the_outbox_notifies_once_a_message_is_queued():
    """The worker sends each message as soon as it is queued (``test_coordinator.py``)."""
    outbox, queued = Outbox(), []
    outbox.notify = lambda: queued.append(outbox.take_messages())
    message = ClientMessage(0, "live", {}, b"")
    outbox.put(message)
    assert queued == [[message]]


def test_the_default_client_is_the_mock():
    client = make_client(load_config(), synthetic_library())
    assert isinstance(client, MockClient) and client.frames_per_block == 4
    assert client.cfg.player_state_source == "live"
