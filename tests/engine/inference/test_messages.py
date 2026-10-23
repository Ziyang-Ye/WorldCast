"""The shared world state over messages: the parts of a client's message and their bytes."""

import dataclasses
import hashlib
import json
import pickle
import struct

import numpy as np
import pytest

from tests.engine.inference.support import block_key, block_latents
from worldcast.data.controls import CONTROL_BUTTONS, buttons_from_bits
from worldcast.data.latents import block_span
from worldcast.engine.inference.messages import MessagePart, MessageWorldState
from worldcast.engine.inference.world_state import (
    DONE_OK,
    PublishedBlock,
    StepRecord,
    WorldStateError,
)
from worldcast.utils.fingerprints import sha256_float32

STEP = StepRecord("A", t_target=40, withdrawn=(8,), resident=(16, 24))
PUBLISHED = PublishedBlock("A", 0, 1, 8, 32, sha256_float32(block_latents(0)))


def live(media_id: str = "A", f0: int = 25, t_last: float = 10.0, x: float = 100.0) -> MessagePart:
    """A ``live`` part of block ``f0`` whose last frame is at session time ``t_last``: walking
    along +x from ``x`` at 250 u/s, facing +x, ``forward`` held, the AK-47 (id 2)."""
    k = np.arange(16)
    rows = np.zeros((16, 5), np.float32)
    rows[:, 0] = x + 250.0 * k / 16
    times = t_last - (15 - k) / 16
    buttons = np.full(16, 1, np.uint16)
    turn = np.zeros((16, 2), np.float32)
    weapon = np.full(16, 2, np.uint8)
    return MessagePart(
        media_id, "live", f0=f0, times=times, rows=rows, buttons=buttons, turn=turn, weapon=weapon
    )


def live_fields(**changed) -> dict:
    """The kind and the fields of :func:`live`'s part, with ``changed``."""
    part = live()
    names = ("f0", "times", "rows", "buttons", "turn", "weapon")
    return {"kind": "live", **{name: getattr(part, name) for name in names}, **changed}


def scene(f0: int, seed: int = 0) -> MessagePart:
    """The ``scene`` part of A's memory entry of block ``f0``, latents :func:`block_latents`
    ``(seed)``, of a rollout from source frame 0."""
    latents = block_latents(seed)
    block = PublishedBlock("A", 0, f0, *block_span(0, f0), sha256_float32(latents))
    return MessagePart("A", "scene", block=block, latents=latents)


#: The header of :func:`live`'s part.
LIVE_HEADER = {
    "media_id": "A",
    "kind": "live",
    "f0": 25,
    "times": ["<f8", [16]],
    "rows": ["<f4", [16, 5]],
    "buttons": ["<u2", [16]],
    "turn": ["<f4", [16, 2]],
    "weapon": ["|u1", [16]],
}
#: One part of every kind.
PARTS = [
    MessagePart("A", "scene", block=PUBLISHED, latents=block_latents(0)),
    MessagePart("A", "step", step=STEP),
    MessagePart("A", "position", t_target=40, latent_frames=(1, 2), xyz=np.ones((2, 3))),
    MessagePart("A", "done", status=DONE_OK),
    live(),
]


def test_a_client_sends_its_message_in_parts():
    sent: list[MessagePart] = []
    owner = MessageWorldState(client="A", poll_s=0.01, outbox=sent.append)
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))
    owner.publish_step(t_target=40, withdrawn=[], resident=[8])
    owner.publish_position(40, [1, 2, 3, 4], np.arange(12.0).reshape(4, 3))
    owner.mark_done(status=DONE_OK)
    assert [part.kind for part in sent] == ["scene", "step", "position", "done"]
    assert {part.media_id for part in sent} == {"A"}
    assert sent[0].block == PUBLISHED and np.array_equal(sent[0].latents, block_latents(0))


def test_the_parts_carry_the_world_state_over_the_wire():
    reader = MessageWorldState(client="B", poll_s=0.01)

    def deliver(part: MessagePart) -> None:
        reader.receive(MessagePart.decode(part.encode()))

    owner = MessageWorldState(client="A", poll_s=0.01, outbox=deliver)
    owner.publish_block(window_start=0, f0=1, latents=block_latents(0))
    owner.publish_step(t_target=40, withdrawn=[], resident=[8])
    owner.publish_position(40, [1, 2, 3, 4], np.arange(12.0).reshape(4, 3))
    owner.mark_done(status=DONE_OK)
    assert reader.published_blocks("A") == []  # applied at the next refresh
    reader.refresh()
    assert reader.published_blocks("A") == [PUBLISHED]
    assert np.array_equal(reader.block_latents(block_key("A", 0, 1), 33), block_latents(0))
    assert reader.has_step("A", 40) and reader.done_status("A") == DONE_OK
    latent_frames, xyz = reader.read_position("A", 40)
    assert latent_frames == [1, 2, 3, 4] and np.array_equal(xyz, np.arange(12.0).reshape(4, 3))


def test_latents_that_are_not_the_stamped_ones_are_refused_at_the_read():
    reader = MessageWorldState(client="B", poll_s=0.01)
    reader.receive(MessagePart("A", "scene", block=PUBLISHED, latents=block_latents(1)))
    reader.refresh()
    with pytest.raises(WorldStateError, match="sha256 of the latents read"):
        reader.block_latents(block_key("A", 0, 1), 33)


def test_a_client_does_not_receive_its_own_parts():
    world_state = MessageWorldState(client="A", poll_s=0.01)
    world_state.receive(MessagePart("A", "done", status=DONE_OK))
    world_state.refresh()
    assert world_state.done_status("A") is None


def test_a_live_part_is_the_rows_the_camera_uses_with_the_blocks_controls():
    sent: list[MessagePart] = []
    owner = MessageWorldState(client="A", poll_s=0.01, outbox=sent.append)
    buttons = np.zeros((16, 11))
    buttons[:, CONTROL_BUTTONS.index("forward")] = 1.0
    buttons[3, CONTROL_BUTTONS.index("attack")] = 1.0
    rows = np.arange(80.0).reshape(16, 5)
    times = 10.0 + np.arange(16) / 16
    owner.publish_live_state(
        25, times=times, rows=rows, buttons=buttons, turn=np.full((16, 2), 0.5), weapon=[2] * 16
    )
    (part,) = sent
    assert part.media_id == "A" and part.kind == "live" and part.f0 == 25
    assert part.buttons.dtype == np.uint16
    assert part.buttons.tolist() == [1, 1, 1, 129] + [1] * 12  # bit 7: attack
    assert np.array_equal(buttons_from_bits(part.buttons), buttons)
    assert part.rows.dtype == np.float32 and np.array_equal(part.rows, rows)
    assert part.times.dtype == np.float64 and np.array_equal(part.times, times)
    assert part.turn.dtype == np.float32 and part.turn.tolist() == [[0.5, 0.5]] * 16
    assert part.weapon.dtype == np.uint8 and part.weapon.tolist() == [2] * 16
    assert owner.live_part("A", 25) is part and owner.recent_live("A")[0] is part
    rows[:] = 0.0
    assert part.rows[1, 0] == 5.0  # the part holds a copy


def test_a_live_part_takes_about_a_kilobyte_per_block():
    data = live().encode()
    (length,) = struct.unpack_from(">I", data)
    assert json.loads(data[4 : 4 + length]) == LIVE_HEADER
    times = np.arange(16.0).tobytes()  # 128 of the 624 bytes of arrays
    assert MessagePart.decode(wire(LIVE_HEADER, times, bytes(624 - 128))).f0 == 25
    assert np.array_equal(MessagePart.decode(live_wire()).rows, live().rows)
    assert len(live("2393400-de_ancient-r18-p00").encode()) == 801  # 6.4 kbit/s at 1 block/s


def test_live_parts_are_ordered_by_session_time_not_by_arrival():
    reader = MessageWorldState(client="B", poll_s=0.01)
    t_last = {25: 13.0, 29: 10.0, 33: 12.0, 37: 11.0}
    parts = {f0: live(f0=f0, t_last=t) for f0, t in t_last.items()}
    for f0 in (33, 25, 37, 29):
        reader.receive(parts[f0])
    assert reader.recent_live("A") == []  # applied at the next refresh
    reader.refresh()
    assert [part.f0 for part in reader.recent_live("A")] == [33, 25]  # by block: [33, 37]
    assert [part.f0 for part in reader.recent_live("A", n=1)] == [25]
    assert [part.f0 for part in reader.recent_live("A", n=10)] == [29, 37, 33, 25]
    assert reader.recent_live("C") == [] and reader.live_part("C", 25) is None
    assert all(reader.live_part("A", f0) is part for f0, part in parts.items())


def test_a_memory_entry_keeps_the_live_part_of_its_session():
    reader = MessageWorldState(client="B", poll_s=0.01)
    first, retaken = live(f0=25, t_last=10.0), live(f0=25, t_last=40.0, x=2000.0)
    for part in (first, scene(25), live(f0=29, t_last=11.0), retaken, scene(25, seed=1)):
        reader.receive(part)
    reader.refresh()
    assert reader.live_part("A", 25) is first  # the memory entry at that key is the first's
    (block,) = reader.published_blocks("A")
    assert block.f0 == 25 and block.latents_sha256 == sha256_float32(block_latents(0))
    assert reader.recent_live("A", n=1)[0] is retaken
    assert [part.times[-1] for part in reader.recent_live("A", n=3)] == [10.0, 11.0, 40.0]


def test_a_seat_taken_again_before_its_first_entries_keys_them_at_its_own_rows():
    """The first six memory entries leave together at block 21's commit: a session that leaves
    before it leaves ``live`` parts without entries, and the next session's entries are keyed at
    the next session's rows."""
    reader = MessageWorldState(client="B", poll_s=0.01)
    for f0, t_last in ((1, 1.0), (5, 2.0), (9, 3.0)):
        reader.receive(live(f0=f0, t_last=t_last, x=100.0))
    reader.receive(MessagePart("A", "done", status=DONE_OK))
    for k, f0 in enumerate(range(1, 25, 4)):
        reader.receive(live(f0=f0, t_last=30.0 + k, x=2000.0))
    for f0 in range(1, 25, 4):
        reader.receive(scene(f0, seed=f0))
    reader.refresh()
    assert reader.done_status("A") is None
    published = reader.published_blocks("A")
    assert [block.f0 for block in published] == [1, 5, 9, 13, 17, 21]
    assert [reader.live_part("A", block.f0).rows[0, 0] for block in published] == [2000.0] * 6


def test_a_seat_taken_again_reopens_its_done_and_a_leavers_late_part_does_not():
    reader = MessageWorldState(client="B", poll_s=0.01)
    for part in (live(f0=25, t_last=10.0), MessagePart("A", "done", status=DONE_OK)):
        reader.receive(part)
    reader.refresh()
    assert reader.done_status("A") == DONE_OK
    reader.receive(live(f0=25, t_last=10.0))  # the same part again
    reader.receive(live(f0=21, t_last=9.0))  # an earlier part, late
    reader.receive(live(f0=29, t_last=11.0))  # the leaver's last part, after its done
    reader.refresh()
    assert reader.done_status("A") == DONE_OK and reader.is_done("A")
    assert [part.f0 for part in reader.recent_live("A", n=10)] == [21, 25, 29]
    reader.receive(live(f0=1, t_last=30.0))  # the seat taken again: its rollout starts over
    reader.refresh()
    assert reader.done_status("A") is None and not reader.is_done("A")
    assert [part.f0 for part in reader.recent_live("A", n=10)] == [21, 25, 29, 1]
    reader.receive(live(f0=5, t_last=31.0))
    reader.receive(MessagePart("A", "done", status=DONE_OK))
    reader.receive(live(f0=9, t_last=32.0))  # this session's last part, after its done
    reader.refresh()
    assert reader.done_status("A") == DONE_OK


#: One part of each kind the ``live`` kind came after, and its bytes from before: their length,
#: the JSON header and the sha256 of all of them.
PINNED_PARTS = {
    "scene": (
        MessagePart(
            "A",
            "scene",
            block=PublishedBlock("A", 0, 1, 8, 32, "0" * 64),
            latents=np.arange(8, dtype=np.float32).reshape(4, 1, 1, 2),
        ),
        252,
        '{"media_id":"A","kind":"scene","block":{"media_id":"A","window_start":0,"f0":1,'
        '"t_first":8,"t_last":32,"latents_sha256":"' + "0" * 64 + '"},'
        '"latents":["<f4",[4,1,1,2]]}',
        "18a0468539e8dcd1560c9f504f9b6b377b2d930a6d072063c814e6b795622dd4",
    ),
    "step": (
        MessagePart("A", "step", step=STEP),
        107,
        '{"media_id":"A","kind":"step","step":{"media_id":"A","t_target":40,"withdrawn":[8],'
        '"resident":[16,24]}}',
        "077d96f1857f2b42a2c42e0c45c13215baf5351d43e67c99ef611221865fa397",
    ),
    "position": (
        MessagePart(
            "A", "position", t_target=40, latent_frames=(1, 2), xyz=np.arange(6.0).reshape(2, 3)
        ),
        142,
        '{"media_id":"A","kind":"position","t_target":40,"latent_frames":[1,2],'
        '"xyz":["<f8",[2,3]]}',
        "4a17a063da09e06d330432407bdf01ec3f8043ecf67aec0c494e671732f87dcc",
    ),
    "done": (
        MessagePart("A", "done", status=DONE_OK),
        48,
        '{"media_id":"A","kind":"done","status":"ok"}',
        "1f637cf4d4afb5a9152dde6057b1ab9fe54b35792fed44317a5d3f157a99078e",
    ),
}


@pytest.mark.parametrize("kind", PINNED_PARTS)
def test_the_other_kinds_keep_their_bytes(kind):
    part, length, header, sha256 = PINNED_PARTS[kind]
    data = part.encode()
    assert data[:4] == struct.pack(">I", len(header))
    assert data[4 : 4 + len(header)].decode() == header
    assert len(data) == length and hashlib.sha256(data).hexdigest() == sha256


@pytest.mark.parametrize("sent", PARTS, ids=[part.kind for part in PARTS])
def test_a_part_survives_the_wire(sent):
    read = MessagePart.decode(sent.encode())
    for field in dataclasses.fields(MessagePart):
        a, b = getattr(sent, field.name), getattr(read, field.name)
        if isinstance(a, np.ndarray):
            assert b.dtype == a.dtype and np.array_equal(a, b) and b.flags.writeable
        else:
            assert a == b and type(a) is type(b)


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        (dict(kind="note"), "a kind of"),
        (dict(kind="done"), r"a 'done' part carries \('status',\), got \[\]"),
        (
            dict(kind="position", t_target=40),
            r"a 'position' part carries \('t_target', 'latent_frames', 'xyz'\), got \['t_target'\]",
        ),
        (dict(kind="done", status="ok", f0=1), "a 'done' part carries"),
        (live_fields(f0="1"), r"\['f0'\] of another type"),
        (dict(kind="step", step=StepRecord("B", 40, (), ())), "of 'A' with a record of 'B'"),
        (
            dict(kind="live", f0=1, rows=np.zeros((16, 5), np.float32)),
            r"a 'live' part carries \('f0', 'times', 'rows', 'buttons', 'turn', 'weapon'\),"
            r" got \['f0', 'rows'\]",
        ),
        (live_fields(xyz=np.zeros((4, 3))), "a 'live' part carries"),
        (live_fields(weapon=[2] * 16), r"\['weapon'\] of another type"),
        (
            live_fields(buttons=np.ones((16, 11), np.float32)),
            r"a 'live' part carries times float64 \(16,\), rows float32 \(16, 5\), buttons uint16"
            r" \(16,\), turn float32 \(16, 2\), weapon uint8 \(16,\); got"
            r" \['buttons float32 \(16, 11\)'\]",
        ),
        (live_fields(times=np.arange(16.0, dtype=np.float32)), r"got \['times float32 \(16,\)'\]"),
        (live_fields(rows=np.zeros((4, 5), np.float32)), r"got \['rows float32 \(4, 5\)'\]"),
        (live_fields(weapon=np.full(16, 2)), r"got \['weapon int64 \(16,\)'\]"),
        (live_fields(weapon=np.full(16, 52, np.uint8)), r"weapon ids must lie in \[0, 52\)"),
        (live_fields(times=np.r_[np.arange(15.0), np.nan]), r"non-finite \['times'\]"),
        (live_fields(times=np.r_[np.arange(15.0), 14.0]), "the times of a 'live' part increase"),
        (live_fields(times=np.full(16, 3.0)), "the times of a 'live' part increase"),
        (
            live_fields(
                rows=np.full((16, 5), np.inf, np.float32), turn=np.full((16, 2), np.nan, np.float32)
            ),
            r"non-finite \['rows', 'turn'\]",
        ),
    ],
)
def test_a_part_carries_the_fields_of_its_kind(fields, error):
    with pytest.raises(ValueError, match=error):
        MessagePart("A", **fields)


@pytest.mark.parametrize(
    ("changed", "error"),
    [
        (dict(weapon=[300] * 16), r"weapon ids must lie in \[0, 52\)"),  # not 44, wrapped
        (dict(weapon=[-1] * 16), r"weapon ids must lie in \[0, 52\)"),  # not 255
        (dict(turn=np.full((16, 2), np.nan)), r"non-finite \['turn'\]"),
        (dict(buttons=np.zeros((16, 13))), r"buttons are \[n, 11\], got \(16, 13\)"),
    ],
)
def test_a_client_publishes_no_live_part_its_readers_cannot_read(changed, error):
    sent: list[MessagePart] = []
    owner = MessageWorldState(client="A", poll_s=0.01, outbox=sent.append)
    controls = dict(buttons=np.zeros((16, 11)), turn=np.zeros((16, 2)), weapon=[2] * 16)
    with pytest.raises(ValueError, match=error):
        owner.publish_live_state(
            25, times=np.arange(16.0), rows=np.zeros((16, 5)), **{**controls, **changed}
        )
    assert sent == [] and owner.live_part("A", 25) is None


def wire(header, *arrays: bytes) -> bytes:
    """A part's bytes with ``header`` (JSON, or the bytes given) and the arrays' bytes."""
    head = header if isinstance(header, bytes) else json.dumps(header).encode()
    return struct.pack(">I", len(head)) + head + b"".join(arrays)


def live_wire(**changed: np.ndarray) -> bytes:
    """The bytes of :func:`live`'s part with the arrays ``changed``, written without the part's
    checks."""
    arrays = live_fields(**changed)
    return wire(
        LIVE_HEADER, *(arrays[n].tobytes() for n in ("times", "rows", "buttons", "turn", "weapon"))
    )


POSITION = {
    "media_id": "A",
    "kind": "position",
    "t_target": 40,
    "latent_frames": [1],
    "xyz": ["<f4", [1, 2]],
}
STEP_HEADER = {"media_id": "A", "kind": "step", "step": dataclasses.asdict(STEP)}
#: Bytes that are not a part, by what is wrong with them.
MALFORMED = {
    "a pickled numpy function": pickle.dumps(np.testing.assert_equal),
    "empty": b"",
    "a header that is not JSON": wire(b"{nope"),
    "a header that is a list": wire([1, 2]),
    "a header that is a string": wire("position"),
    "deeply nested JSON": wire(b"[" * 100_000 + b"]" * 100_000),
    "an unknown kind": wire({"media_id": "A", "kind": "note"}),
    "a kind that is a list": wire({"media_id": "A", "kind": ["done"]}),
    "a missing field": wire({"media_id": "A", "kind": "position", "t_target": 40}),
    "a field of another kind": wire({**POSITION, "status": "ok"}, bytes(8)),
    "trailing bytes": wire(POSITION, bytes(9)),
    "a short array": wire(POSITION, bytes(7)),
    "a negative dimension": wire({**POSITION, "xyz": ["<f4", [-1]]}, bytes(8)),
    "an overflowing shape": wire({**POSITION, "xyz": ["<f4", [2**40, 2**40]]}, bytes(8)),
    "an object array": wire({**POSITION, "xyz": ["|O", [1]]}, bytes(8)),
    "a structured array": wire({**POSITION, "xyz": ["f4,f4", [1]]}, bytes(8)),
    "an array that is a mapping": wire({**POSITION, "xyz": {"dtype": "<f4"}}, bytes(8)),
    "an array that is a number": wire({**POSITION, "xyz": 3}, bytes(8)),
    "a shape of floats": wire({**POSITION, "xyz": ["<f4", [2.0]]}, bytes(8)),
    "a client that is a list": wire({**POSITION, "media_id": ["A"]}, bytes(8)),
    "a frame that is a string": wire({**POSITION, "t_target": "40"}, bytes(8)),
    "a frame that is a boolean": wire({**POSITION, "t_target": True}, bytes(8)),
    "latent frames that are strings": wire({**POSITION, "latent_frames": ["1"]}, bytes(8)),
    "a status that is a mapping": wire({"media_id": "A", "kind": "done", "status": {}}),
    "a record that is a list": wire({**STEP_HEADER, "step": [40]}),
    "a record with a string for a frame": wire(
        {**STEP_HEADER, "step": {**STEP_HEADER["step"], "t_target": "40"}}
    ),
    "a record with another field": wire({**STEP_HEADER, "step": {**STEP_HEADER["step"], "x": 1}}),
    "a record of another client": wire(
        {**STEP_HEADER, "step": {**STEP_HEADER["step"], "media_id": "B"}}
    ),
    "a live part with a weapon of int64": wire(
        {**LIVE_HEADER, "weapon": ["<i8", [16]]}, bytes(128 + 320 + 32 + 128 + 128)
    ),
    "a live part of four rows": wire(
        {
            **LIVE_HEADER,
            "times": ["<f8", [4]],
            "rows": ["<f4", [4, 5]],
            "buttons": ["<u2", [4]],
            "turn": ["<f4", [4, 2]],
            "weapon": ["|u1", [4]],
        },
        bytes(32 + 80 + 8 + 32 + 4),
    ),
    "a live part with weapon 52": live_wire(weapon=np.full(16, 52, np.uint8)),
    "a live part with a NaN time": live_wire(times=np.r_[np.arange(15.0), np.nan]),
    "a live part whose times step back": live_wire(times=np.r_[np.arange(15.0), 14.0]),
}


@pytest.mark.parametrize("data", MALFORMED.values(), ids=MALFORMED)
def test_decoding_raises_value_error_for_everything_that_is_not_a_part(data):
    with pytest.raises(ValueError):
        MessagePart.decode(data)


def test_decoding_runs_nothing_from_the_bytes(tmp_path):
    """A pickle that would write a file when loaded is refused before anything of it runs."""
    target = tmp_path / "written.npy"

    class Writes:
        def __reduce__(self):
            return np.save, (str(target), np.zeros(1))

    payload = pickle.dumps(Writes())
    with pytest.raises(ValueError):
        MessagePart.decode(payload)
    assert not target.exists()
    pickle.loads(payload)
    assert target.exists()  # the payload does what it says when it is unpickled


# ----------------------------------------------------------------------------------- what it keeps
def blocks_of(media_id: str, k: int) -> list[MessagePart]:
    """The parts of ``media_id``'s ``k``-th block, as they arrive: the ``live`` part, the ``scene``
    part (a block of 4 latent frames from source frame 0), the ``step`` and the ``position`` parts
    of the block's time."""
    f0 = 1 + 4 * k
    latents = block_latents(k)
    block = PublishedBlock(media_id, 0, f0, *block_span(0, f0), sha256_float32(latents))
    return [
        live(media_id, f0=f0, t_last=1.0 + k),
        MessagePart(media_id, "scene", block=block, latents=latents),
        MessagePart(
            media_id, "step", step=StepRecord(media_id, block.t_first, (), (block.t_first,))
        ),
        MessagePart(
            media_id, "position", t_target=block.t_first, latent_frames=(f0,), xyz=np.ones((1, 3))
        ),
    ]


def held(world_state: MessageWorldState, media_id: str, blocks: int) -> dict[str, list[int]]:
    """The f0 of the blocks of ``media_id`` that each index of the world state still holds, of
    its first ``blocks`` blocks."""
    f0s = [1 + 4 * k for k in range(blocks)]
    t_firsts = {f0: block_span(0, f0)[0] for f0 in f0s}
    return {
        "scene": [b.f0 for b in world_state.published_blocks(media_id)],
        "step": [f for f in f0s if world_state.has_step(media_id, t_firsts[f])],
        "position": [f for f in f0s if world_state.read_position(media_id, t_firsts[f])],
        "live": [f for f in f0s if world_state.live_part(media_id, f) is not None],
        "recent": [p.f0 for p in world_state.recent_live(media_id, blocks)],
    }


def test_a_world_state_that_retains_keeps_the_newest_blocks_of_each_client_in_every_index():
    keep = 4
    kept = MessageWorldState(client="B", poll_s=0.01, retain=keep)
    everything = MessageWorldState(client="B", poll_s=0.01)
    for world_state in (kept, everything):
        for k in range(3 * keep):
            for media_id in ("A", "C"):
                for part in blocks_of(media_id, k):
                    world_state.receive(part)
            world_state.refresh()
    newest = [1 + 4 * k for k in range(2 * keep, 3 * keep)]
    everyone = [1 + 4 * k for k in range(3 * keep)]
    for media_id in ("A", "C"):
        assert held(kept, media_id, 3 * keep) == {
            index: newest for index in ("scene", "step", "position", "live", "recent")
        }
        assert held(everything, media_id, 3 * keep) == {
            index: everyone for index in ("scene", "step", "position", "live", "recent")
        }
    # the own parts are applied as they are sent, and retained alike
    for k in range(3 * keep):
        kept.publish_live_state(
            1 + 4 * k,
            times=live(f0=1 + 4 * k, t_last=1.0 + k).times,
            rows=np.zeros((16, 5)),
            buttons=np.zeros((16, len(CONTROL_BUTTONS))),
            turn=np.zeros((16, 2)),
            weapon=np.full(16, 2),
        )
    assert [p.f0 for p in kept.recent_live("B", 3 * keep)] == newest


def test_a_block_keeps_the_live_part_of_its_entry_until_its_scene_part_leaves():
    """The ``live`` part of a block that keyed its memory entry goes with the entry's ``scene``
    part, not before: a reader that admits the block one block after its parts arrived must find
    the part it was keyed with, however many newer ``live`` parts arrived."""
    keep = 4
    world_state = MessageWorldState(client="B", poll_s=0.01, retain=keep)
    parts = [blocks_of("A", k) for k in range(2 * keep)]
    for live_part, scene_part, *_ in parts[:keep]:  # each block's live part, then its scene part
        world_state.receive(live_part)
        world_state.receive(scene_part)
    world_state.refresh()
    pinned = {f0: world_state.live_part("A", f0) for f0 in (1, 5, 9, 13)}
    assert all(part is not None for part in pinned.values())
    for live_part, *_ in parts[keep:]:  # newer live parts push the older ones out of the index ...
        world_state.receive(live_part)
    world_state.refresh()
    assert [p.f0 for p in world_state.recent_live("A", 2 * keep)] == [17, 21, 25, 29]
    # ... but not the parts the entries were keyed with
    assert all(world_state.live_part("A", f0) is part for f0, part in pinned.items())
    for _, scene_part, *_ in parts[keep:]:  # the scene parts of the newer blocks push theirs out
        world_state.receive(scene_part)
    world_state.refresh()
    assert [world_state.live_part("A", f0) for f0 in pinned] == [None] * keep
    assert world_state.live_part("A", 29) is not None
