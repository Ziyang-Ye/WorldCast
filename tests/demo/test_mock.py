"""The mock client: a block's frames, the player it moves, the messages it sends and reads, the
other players as the WorldCast client shows them (only those a human plays), the drift it shows
standing in for ``gt``, the clips of a library."""

import asyncio
import math
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from tests.demo.support import FORWARD, create_room, demo, live_controls, take_seat, until
from worldcast.data.latents import FRAME_SIZE
from worldcast.demo import mock
from worldcast.demo.config import MockConfig
from worldcast.demo.library import Library, RoundStart, Seat, synthetic_library
from worldcast.engine.inference.field_rows import FieldRows

INSTANT = MockConfig(frames_per_block=16, denoise_ms=0.0, decode_ms=0.0, jitter_ms=0.0)
LIVE, RECORDED = FieldRows.LIVE, FieldRows.RECORDED
#: Where the synthetic arena's players 1, 2 and 3 stand in the view of player 0, who looks along
#: 30 degrees from (-600, -300): ``(slot, u, v, depth)`` of their feet. Player 1 is 300 u ahead
#: and 30 degrees to the side, outside the view's width. The feet are 64 u below the level eye:
#: ``v = 0.5 + 64 / depth / (2 tan(vfov / 2))``, ``tan(vfov / 2) = tan(hfov / 2) * 9 / 16``.
SEEN_FROM_0 = {
    1: (1, -0.149521, 0.642223, 300.0),
    2: (2, 0.716507, 0.541056, 1039.230485),
    3: (3, 0.522509, 0.531859, 1339.230485),
}


def _client(slot: int, **settings) -> mock.MockClient:
    """A mock client of the synthetic arena's seat ``slot``."""
    library = synthetic_library()
    arena = library.get("arena")
    client = mock.MockClient(replace(INSTANT, **settings), library)
    client.start(arena, arena.seat(slot))
    return client


def test_a_block_publishes_its_rows_at_the_cut_and_a_memory_entry():
    client = _client(slot=0)
    x0 = client.body.state.x
    block = client.step(lambda: live_controls(forward=True))
    first = next(block)
    (live,) = client.take_messages()  # the block's rows leave before its first frame
    frames = [first, *block]
    assert [f.index for f in frames] == list(range(16))
    assert frames[0].rgb.shape == (*FRAME_SIZE, 3)
    assert frames[0].rgb.dtype == np.uint8 and frames[0].input_seq == -1
    assert frames[-1].state.x > x0 + 100.0 and frames[-1].state.t == 1.0  # a second of running
    assert live.kind == "live" and live.meta["rows"] == [f.state.wire() for f in frames]
    (entry,) = client.take_messages()
    assert entry.kind == "scene" and entry.meta == {"frames": [0, 16]}
    assert len(entry.payload) == 4 * 4 * 48 * 24 * 42  # a block's latents, float32
    assert {f.status for f in frames} == {"live"}


def test_a_frame_says_how_far_its_view_has_turned_and_its_cameras_pitch():
    """How much of the player's turn a frame holds: the turn the controls folded
    in through the frame's row, by which the view has turned since the session's start, about the
    frame's own camera (the mock's state is each frame's view). The pitch stops at the engine's
    89 degrees, the turn the frame says goes on."""
    client = _client(slot=0)
    frames = [
        frame
        for k in range(2)
        for frame in client.step(lambda k=k: live_controls(first=16 * k, pitch=4.0, yaw=1.5))
    ]
    assert [frame.turned for frame in frames] == [(4.0 * (j + 1), 1.5 * (j + 1)) for j in range(32)]
    turned = [frame.state.yaw - 30.0 for frame in frames]  # from the spawn's yaw
    assert turned == pytest.approx([1.5 * (j + 1) for j in range(32)], abs=1e-9)
    pitch = [min(4.0 * (j + 1), 89.0) for j in range(32)]  # from the spawn's level view
    assert [frame.camera_pitch for frame in frames] == [frame.state.pitch for frame in frames]
    assert [frame.camera_pitch for frame in frames] == pytest.approx(pitch, abs=1e-9)


def test_only_the_players_a_human_plays_are_shown_as_the_worldcast_client_shows_them():
    client, other = _client(slot=0), _client(slot=2)
    first = next(client.step(lambda: live_controls()))
    # the seats nobody plays are nowhere, not at their spawns either
    assert first.others == [] and first.markers == []
    walked = [
        list(other.step(lambda k=k: live_controls(first=16 * k, forward=True))) for k in (0, 1)
    ]
    for message in other.take_messages():
        client.receive_message(message)
    frames = list(client.step(lambda: live_controls(first=16)))
    # player 2 at its own rows (as the wire carries them), frame by frame
    seen = [next(p for p in frame.others if p.slot == 2) for frame in frames]
    assert {p.src for p in seen} == {LIVE}
    assert [p.wire()[1:5] for p in seen] == [f.state.wire()[1:5] for f in walked[1]]
    # player 2 leaves: hidden, and nobody else is in the world
    client.remove_player(2)
    last = list(client.step(lambda: live_controls(first=32)))[-1]
    assert last.others == [] and last.markers == []


def test_standing_in_for_gt_the_players_stay_on_their_spots_while_the_view_moves():
    client = _client(slot=0, player_state_source="gt")
    frames = list(client.step(lambda: live_controls(forward=True)))
    assert client.body.state.x > -600.0 + 100.0  # the keys moved the view
    assert not np.array_equal(frames[1].rgb, frames[-1].rgb)
    # but the players are drawn from the recording's camera, the spawn
    assert {(f.state.x, f.state.y, f.state.yaw) for f in frames} == {(-600.0, -300.0, 30.0)}
    for frame in frames:
        assert frame.markers == [pytest.approx(SEEN_FROM_0[slot], abs=1e-5) for slot in (1, 2, 3)]
    assert [m.kind for m in client.take_messages()] == ["scene"]  # nothing positional
    assert {f.status for f in frames} == {"gt"}


def test_a_library_gives_the_clip(tmp_path):
    """A seat with a clip of three frames: the mock plays them looped, and the keys still move the
    player."""
    (tmp_path / "clip").mkdir()
    for i in range(3):
        Image.new("RGB", (64, 36), (80 * i, 0, 0)).save(tmp_path / "clip" / f"{i:06d}.jpg")
    seat = Seat(slot=4, team="T", spawn=(150.0, 0.0, 0.0, 0.0, 0.0), clip="clip")
    round_start = RoundStart("r", "de_nuke", "Round", 0, (seat,))
    client = mock.MockClient(INSTANT, Library({"r": round_start}, tmp_path))
    client.start(round_start, seat)
    frames = list(client.step(lambda: live_controls(forward=True)))
    assert frames[0].rgb.shape == (36, 64, 3)  # the clip's frames, looped
    assert [round(int(f.rgb[0, 0, 0]) / 80) for f in frames[:4]] == [0, 1, 2, 0]
    # a second of running along +x from rest: 64 substeps, the velocity easing to 240 u/s at 10 / s
    assert frames[-1].state.x == pytest.approx(150.0 + 217.83, abs=0.01)


# ------------------------------------------------------------------------------------- the drift
async def towards_an_idle_player(source: str) -> list[dict]:
    """Player 0 walks towards player 3, who plays and stands still; the frame headers of player 0
    with the field's markers in them."""
    settings = {"client.mock.player_state_source": source}
    async with demo(2, **settings) as (local, http):
        room = await create_room(local, http)
        idle = await take_seat(local, http, room, 3)
        walker = await take_seat(local, http, room, 0, hold=FORWARD, turn=0.0)
        await walker.show_markers()

        def seen() -> list[dict]:
            return [h for h in walker.frames if "m" in h and any(p[0] == 3 for p in h["p"])]

        await until(lambda: len(seen()) >= 40, "forty frames that show player 3")
        await walker.close()
        await idle.close()
        return seen()


@pytest.mark.parametrize("source", ["live", "gt"])
def test_an_idle_player_stays_put_while_another_walks(source):
    """Under ``live``, player 3 stays where it stands and comes nearer in player 0's view as
    player 0 walks; standing in for ``gt``, it stays on one spot of the view and at one depth,
    however far player 0 walks: the drift of the WorldCast client's ``gt`` under live keys."""
    headers = asyncio.run(towards_an_idle_player(source))
    rows = [next(p for p in h["p"] if p[0] == 3) for h in headers]
    assert {(row[2], row[3]) for row in rows} == {(600.0, 300.0)}  # in the world: where it stands
    own = [h["s"] for h in headers]
    walked = math.hypot(own[-1][2] - own[0][2], own[-1][3] - own[0][3])
    marker = [next(m for m in h["m"] if m[0] == 3) for h in headers]
    if source == "live":
        # its seat was taken before player 0 started: shown from its first live state on
        assert {row[10] for row in rows} == {LIVE}
        assert walked > 300.0 and marker[0][3] - marker[-1][3] > 300.0
        assert all(b[3] <= a[3] for a, b in zip(marker, marker[1:]))  # nearer, frame by frame
    else:
        assert {row[10] for row in rows} == {RECORDED}
        assert walked == 0.0 and {tuple(m) for m in marker} == {(3, 0.5225, 0.5319, 1339.2)}
        assert {h["st"] for h in headers} == {"gt"}
