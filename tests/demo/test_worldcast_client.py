"""The demo's WorldCast client on the CPU with a tiny random model, under live controls
(``player_state.source: live``): its warm-up (a replay session of the library's first seat), alone
(its round-index row, the controls taken right before each block's denoising, its frames and
messages, no end to its session, the ``done`` its stop leaves queued), as two players cross-wired
(each frame shows the rows the player state field read, and only players a human plays; a standing
player stays put while the other walks; a leaver is hidden from the next block; a late joiner sees
the earlier player at once), and as two players of a room, the second joining after the first has
published memory entries and receiving the first's ``done`` when it leaves. The demo refuses
the other sources."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from aiohttp import ClientSession

import tests.engine.inference.support as sw
from tests.demo.support import create_room, live_controls, take_seat, until, write_labels
from tests.tools.support import synthetic_round
from worldcast.config.inference import InferenceConfig
from worldcast.config.loader import config_to_dict
from worldcast.data import MediaIndex
from worldcast.demo import worldcast_client
from worldcast.demo.clients import LIVE_STATE, MEMORY_ENTRY, LiveFrame
from worldcast.demo.config import LiveSceneStateConfig, WorldCastClientConfig, load_config
from worldcast.demo.library import Library, RoundStart, Seat
from worldcast.demo.local import LocalDemo
from worldcast.demo.worldcast_client import (
    REFRESH_WARM_UP_ROWS,
    REFRESH_WARM_UP_WINDOWS,
    WARM_UP_WINDOWS,
    WINDOW_SHAPES,
    WorldCastClient,
    field_states,
    index_row,
)
from worldcast.engine.inference.field_rows import FieldRows
from worldcast.engine.inference.messages import MessagePart
from worldcast.engine.inference.taehv import TinyDecoder

OWN, LIVE = FieldRows.OWN, FieldRows.LIVE
#: The standing round's player who stands 1600 u ahead of player 0 and 600 u to its left
#: (``tests/engine/inference/support.py``, ``STANDING``).
STANDING = 5


def round_start(*slots: int) -> RoundStart:
    seats = tuple(Seat(slot=s, team="T", spawn=(1.0, 2.0, 3.0, 0.0, 0.0)) for s in slots)
    return RoundStart("r", sw.MAP, "synthetic", 0, seats, match_id=sw.MATCH, round=sw.ROUND)


def demo_client(tmp_path: Path, cfg: InferenceConfig, overrides: dict) -> WorldCastClientConfig:
    """The demo's client on the inference config ``cfg`` (a YAML file of it), with the tiny decoder
    and its ``overrides``."""
    (tmp_path / "inference.yaml").write_text(yaml.safe_dump(config_to_dict(cfg)))
    torch.manual_seed(0)
    torch.save(TinyDecoder().state_dict(), tmp_path / "taew2_2.pth")
    return WorldCastClientConfig(
        configs=[str(tmp_path / "inference.yaml")],
        overrides=overrides,
        taehv_path=str(tmp_path / "taew2_2.pth"),
        serving={"compile": False},
    )


def live(tmp_path: Path) -> dict:
    """The overrides of live play: the source and a tiny state model."""
    return {"player_state.source": "live", **sw.write_state_model(tmp_path / "sm.pt")}


def client_config(
    tmp_path, monkeypatch, *, max_blocks: int, seconds: float = 8.5
) -> tuple[dict, WorldCastClientConfig]:
    """The synthetic round (``seconds`` long) and the demo's client on it under live controls."""
    world, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=max_blocks, seconds=seconds)
    sw.empty_first_frame(world)  # the seats a library offers show nobody in their first frame
    return world, demo_client(tmp_path, cfg, live(tmp_path))


def warming_client(tmp_path, monkeypatch, refresh=False) -> tuple[WorldCastClient, list]:
    """A client on a library whose first seat is player 0 (``refresh``: with ``play.refresh``),
    and the sessions its rollout starts: (media id, clients, the player state's source)."""
    _, config = client_config(tmp_path, monkeypatch, max_blocks=100, seconds=20.0)
    start = round_start(0, 3)
    library = Library({start.id: start})
    client = WorldCastClient(config, LiveSceneStateConfig(), library, refresh=refresh)
    started = []
    start_rollout = client.client.start

    def record_start(row, **kwargs):
        started.append((row.media_id, row.clients, client.client.cfg.player_state.source))
        start_rollout(row, **kwargs)

    monkeypatch.setattr(client.client, "start", record_start)
    return client, started


def window_shapes(client: WorldCastClient) -> list[int | None]:
    return [record.read and record.read.window_frames for record in client.client.records]


def test_a_decoder_on_its_own_device_has_the_worker_start_the_next_block_meanwhile(
    tmp_path, monkeypatch
):
    """The client says how many blocks the worker may have in flight, and requests a block when
    it is stepped, not when its frames are first taken: the worker steps the next block while
    another thread takes the frames of this one."""
    _, config = client_config(tmp_path, monkeypatch, max_blocks=2)
    served, requested = [], []

    class Requests:
        def __init__(self, cfg, serving, on_message):
            served.append(serving)

        def step(self, controls):
            requested.append(controls)
            return iter(())

    monkeypatch.setattr(worldcast_client, "Client", Requests)
    library = Library({"r": round_start(0)})
    alone = WorldCastClient(config, LiveSceneStateConfig(), library)
    second_card = {"decoder": "wan", "decoder_device": "cuda:1"}
    own = replace(config, serving={**config.serving, **second_card})
    apart = WorldCastClient(own, LiveSceneStateConfig(), library)
    assert (alone.blocks_in_flight, apart.blocks_in_flight) == (1, 2)
    assert (served[0].decoder, served[0].decoder_device) == ("taehv", None)
    assert (served[1].decoder, served[1].decoder_device) == ("wan", "cuda:1")
    frames = apart.step(lambda: None)
    assert len(requested) == 1  # requested by the call
    assert list(frames) == []


def test_the_warm_up_replays_the_first_seat_until_each_window_shape_ran(
    tmp_path, monkeypatch, caplog
):
    client, started = warming_client(tmp_path, monkeypatch)
    play = client.client.cfg
    with caplog.at_level(logging.WARNING, logger=worldcast_client.__name__):
        client.warm_up()
    assert WINDOW_SHAPES == (17, 21) and WARM_UP_WINDOWS == 16
    # player 0 alone, on its recorded player states and controls, though the play is live
    assert started == [(sw.media_id(0), (), "gt")]
    assert client.client.cfg is play and play.player_state.source == "live"
    # the six blocks of latents 1-24, then window blocks until each shape ran: 2 of the round's 16
    assert window_shapes(client) == [None] * 6 + [21, 17]
    assert client.client.finished and client.take_messages() == []  # stopped; nothing is sent
    assert caplog.records == []


def test_the_warm_up_of_the_generator_without_scene_state_runs_each_shape_of_its_window(
    tmp_path, monkeypatch, caplog
):
    """With ``model.scene_state: false`` the client plays a round of 45 latent frames, and its
    warm-up runs the ordinary window as it grows, 29, 33, 37 and 41 latent frames, then stops."""
    world, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=100, seconds=20.0)
    sw.empty_first_frame(world)
    weights = sw.make_weights(tmp_path / "noscene", scene_state=False)
    cfg = cfg.with_overrides({"paths.checkpoint": weights["checkpoint"]})
    config = demo_client(tmp_path, cfg, {**live(tmp_path), "model.scene_state": False})
    start = round_start(0, 3)
    client = WorldCastClient(config, LiveSceneStateConfig(), Library({start.id: start}))
    assert (client.cfg.model.scene_state, client.cfg.run.latent_frames) == (False, 45)
    with caplog.at_level(logging.WARNING, logger=worldcast_client.__name__):
        client.warm_up()
    assert worldcast_client.window_shapes(False) == (29, 33, 37, 41)
    assert window_shapes(client) == [None] * 6 + [29, 33, 37, 41]
    assert caplog.records == []


def test_the_warm_up_stops_after_its_window_blocks_and_names_a_shape_that_did_not_run(
    tmp_path, monkeypatch, caplog
):
    client, _ = warming_client(tmp_path, monkeypatch)
    monkeypatch.setattr(worldcast_client, "WARM_UP_WINDOWS", 1)
    with caplog.at_level(logging.WARNING, logger=worldcast_client.__name__):
        client.warm_up()
    assert window_shapes(client) == [None] * 6 + [21]
    assert [record.getMessage() for record in caplog.records] == [
        "warm-up on r: no window of 17 latent frames ran; its generator calls are warm, the rest"
        " of a session's first one is not"
    ]


def test_with_refresh_the_warm_up_plays_the_seat_live_and_refreshes_its_window_blocks(
    tmp_path, monkeypatch
):
    """After the replay, the seat is played live through the first window blocks, each refreshed
    from a frame of its second latent frame and whole (the decoder resumes from its snapshots);
    nothing is sent."""
    client, started = warming_client(tmp_path, monkeypatch, refresh=True)
    monkeypatch.setattr(worldcast_client, "WARM_UP_WINDOWS", 1)
    asked = []
    refresh = client.client.refresh

    def spy(controls, switch):
        asked.append(switch(0.0))
        return refresh(controls, switch)

    monkeypatch.setattr(client.client, "refresh", spy)
    client.warm_up()
    assert started == [(sw.media_id(0), (), "gt"), (sw.media_id(0), (), "live")]
    windows = REFRESH_WARM_UP_WINDOWS
    firsts = [1 + 16 * (6 + k) for k in range(windows)]
    assert asked == [first + row for first in firsts for row in REFRESH_WARM_UP_ROWS]
    records = [r for r in client.client.records if r.refreshes]
    assert [len(r.refreshes) for r in records] == [2] * windows
    assert [r.refreshes[0].first_frame for r in records] == [f + 6 for f in firsts]
    assert client.take_messages() == []


def test_with_a_replica_the_warm_up_checks_it_on_every_window_block(tmp_path, monkeypatch):
    """With ``refresh_devices`` (two cards of the CPU) the replay denoises each window block again
    on the replica, and the live part refreshes each window block on the cards in turn, the
    block's own controls on one of them giving its first frames again: every check equal."""
    _, config = client_config(tmp_path, monkeypatch, max_blocks=100, seconds=20.0)
    config = replace(config, serving={**config.serving, "refresh_devices": ["cpu", "cpu"]})
    start = round_start(0, 3)
    client = WorldCastClient(
        config, LiveSceneStateConfig(), Library({start.id: start}), refresh=True
    )
    monkeypatch.setattr(worldcast_client, "WARM_UP_WINDOWS", 1)
    assert client.refresh_cards == 2
    client.warm_up()
    records = [r for r in client.client.records if r.refreshes]
    assert [[refresh.card for refresh in r.refreshes] for r in records] == [[0, 1], [1, 0]]
    replayed = len(client.client.replicas_equal) - REFRESH_WARM_UP_WINDOWS
    assert replayed >= 1 and client.client.replicas_equal == [True] * (replayed + 2)
    assert client.take_messages() == []


@pytest.mark.parametrize("fault", ["other", "fails"])
def test_a_replica_the_warm_up_finds_otherwise_is_let_go(tmp_path, monkeypatch, caplog, fault):
    """A replica whose denoising gives other bits than the client's card (``other``: other
    kernels), or fails (``fails``: a CUDA graph that cannot be captured), is let go after the
    replay: the client warms up its refreshes and plays on its own card alone, and says so."""
    _, config = client_config(tmp_path, monkeypatch, max_blocks=100, seconds=20.0)
    config = replace(config, serving={**config.serving, "refresh_devices": ["cpu", "cpu"]})
    start = round_start(0, 3)
    client = WorldCastClient(
        config, LiveSceneStateConfig(), Library({start.id: start}), refresh=True
    )
    monkeypatch.setattr(worldcast_client, "WARM_UP_WINDOWS", 1)
    (replica,) = client.client.replicas
    denoise = replica.sampler.denoise

    def faulty(*args, **kwargs):
        if fault == "fails":
            raise RuntimeError("operation failed due to a previous error during capture")
        return denoise(*args, **kwargs) + 1e-3

    monkeypatch.setattr(replica.sampler, "denoise", faulty)
    with caplog.at_level(logging.WARNING):
        client.warm_up()
    assert False in client.client.replicas_equal and client.refresh_cards == 1
    assert client.client.replicas == () and "refreshes run on cpu alone" in caplog.text
    records = [r for r in client.client.records if r.refreshes]
    assert [[refresh.card for refresh in r.refreshes] for r in records] == [[0, 0], [0, 0]]


def test_a_client_plays_its_round_block_by_block(tmp_path, monkeypatch):
    world, config = client_config(tmp_path, monkeypatch, max_blocks=1)
    start = round_start(0)
    client = WorldCastClient(config, LiveSceneStateConfig(), Library({start.id: start}))
    assert client.cfg.scene_state.bound == 32  # the live session's bound, not the paper's 64
    client.start(start, start.seat(0))
    blocks, taken = [], []
    for _ in range(8):

        def controls(first=sum(len(b) for b in blocks)):
            taken.append(first)
            return live_controls(first=first, yaw=0.25)

        blocks.append(list(client.step(controls)))
    # the first frame, then sixteen frames a block: the six blocks of latents 1-24, the one more
    # that fills the round and the one after it, which the round slid for: a session has no end
    assert [len(b) for b in blocks] == [17] + [16] * 7
    assert taken == [0, 17, 33, 49, 65, 81, 97, 113]
    frames = [frame for block in blocks for frame in block]
    assert [f.index for f in frames] == list(range(129)) and frames[0].rgb.shape == (384, 672, 3)
    # the first frame and frame 1 share the block's first row of controls; frame 16 has its last
    assert [frames[i].state.t for i in (0, 1, 16, 17)] == [1 / 16, 1 / 16, 1.0, 18 / 16]
    # and how far the player had turned through it: a quarter degree a frame, 4 degrees a second;
    # the first frame is the recording's, before any turn
    turned = [frames[i].turned for i in (0, 1, 16, 17)]
    assert turned == [(0.0, 0.0), (0.0, 0.25), (0.0, 4.0), (0.0, 4.5)]
    assert {f.turned[1] - 4.0 * f.state.t for f in frames[1:]} == {0.0}
    # the first frame shows the recording's first row: the round start
    recorded = world["tables"][sw.media_id(0)]
    assert (frames[0].state.x, frames[0].state.y) == pytest.approx(
        (recorded.x[0], recorded.y[0]), abs=1e-3
    )
    assert {f.state.src for f in frames} == {OWN}
    # the own position's estimator: a random state model's places are rejected until it says so
    assert [frames[0].status, frames[-1].status] == ["guarded", "degraded"]
    kinds = [m.kind for m in client.take_messages()]
    assert {kind: kinds.count(kind) for kind in kinds} == {"live": 8, "scene": 8, "step": 3}
    client.stop()  # the client's done stays queued, for the worker to send after the others
    (done,) = client.take_messages()
    assert (done.slot, done.kind, MessagePart.decode(done.payload).status) == (0, "done", "ok")


@pytest.mark.parametrize(
    ("shown", "refused"),
    [
        (dict(visible_at={0: (3, 7)}), "players 3, 7"),
        (dict(visible_at={}, unknown_at={0: (4,)}), "players 4"),
        (dict(visible_at={4: (3,)}, behind_at={0: (2, 6)}), None),
    ],
    ids=["visible", "in-view-unknown", "other-frames-and-behind"],
)
def test_a_seat_whose_first_frame_may_show_another_player_is_not_played(
    tmp_path, monkeypatch, shown, refused
):
    """The library offers only seats that show nobody in their first frame; the client reads the
    labels again at the start, so a library built another way cannot put the people of a recorded
    round into the world's first frame."""
    world, config = client_config(tmp_path, monkeypatch, max_blocks=1)
    start = round_start(0)
    client = WorldCastClient(config, LiveSceneStateConfig(), Library({start.id: start}))
    write_labels(Path(world["visibility_label_root"]), 0, **shown)
    if refused is None:  # only players around the first frame, or behind the camera at it
        client.start(start, start.seat(0))
        client.stop()
        return
    with pytest.raises(ValueError, match=f"seat 0 of r: {refused} may be seen in its first frame"):
        client.start(start, start.seat(0))
    assert client.client._rollout is None  # nothing started


@pytest.mark.parametrize(
    "weapons, named",
    [(("AWP",), "awp"), (("ak47", "weapon_aug", "knife", "AUG"), "aug, aug")],
)
def test_a_seat_whose_loadout_holds_a_weapon_that_zooms_is_not_played(
    tmp_path, monkeypatch, weapons, named
):
    """Live play does not track the scope, and the generator draws a black scope view from the
    first frame while the scope is unknown for the weapon held. The library leaves such seats out;
    a library built before that rule, or by hand, still offers them, and the client refuses them
    at the start, naming the weapons by the model's names however the loadout spells them."""
    _, config = client_config(tmp_path, monkeypatch, max_blocks=1)
    start = round_start(0)
    seat = Seat(slot=0, team="T", spawn=(1.0, 2.0, 3.0, 0.0, 0.0), loadout=weapons)
    client = WorldCastClient(config, LiveSceneStateConfig(), Library({start.id: start}))
    with pytest.raises(ValueError, match=f"seat 0 of r: its loadout holds {named}, which zooms"):
        client.start(start, seat)
    assert client.client._rollout is None  # nothing started


@pytest.mark.parametrize("source", ["gt", "predicted"])
def test_the_demo_refuses_the_paths_that_do_not_play_live(tmp_path, monkeypatch, source):
    """Under live keys ``gt`` is the drift, and ``predicted`` reads the recorded controls."""
    _, config = client_config(tmp_path, monkeypatch, max_blocks=1)
    config.overrides["player_state.source"] = source
    with pytest.raises(ValueError, match=f"plays player_state.source live, not {source} "):
        WorldCastClient(config, LiveSceneStateConfig(), Library())


def test_index_row(tmp_path):
    world = sw.make_world(tmp_path / "world", clients=(0,))
    media_index = MediaIndex.load(world["media_index"])
    row = index_row(round_start(0, 3), 3, media_index)
    assert row.media_id == sw.media_id(3) and row.clients == ()
    assert (row.start_frame, row.player_slot) == (0, 3)
    unrecorded = RoundStart("arena", "arena", "", 0, ())
    with pytest.raises(ValueError, match="no recorded round"):
        index_row(unrecorded, 0, media_index)
    with pytest.raises(KeyError, match="player slot 11 has no recording in round"):
        index_row(round_start(11), 11, media_index)


# ----------------------------------------------------------------------- two players, cross-wired
@pytest.fixture
def standing(tmp_path, monkeypatch) -> tuple[WorldCastClientConfig, Library]:
    """The round where everyone stands still (``tests/engine/inference/support.py``,
    ``standing_world``), 33 latent frames, and a library of its seats 0, 3 and 5."""
    torch.set_num_threads(1)
    world = sw.standing_world(tmp_path / "world")
    sw.empty_first_frame(world)
    sw.patch_ticks(monkeypatch, world["tables"])
    sw.patch_extent(monkeypatch)
    weights = sw.make_weights(tmp_path / "weights")
    cfg = sw.config(world, weights, tmp_path / "run", max_blocks=sw.MAX_BLOCKS)
    start = round_start(0, 3, STANDING)
    return demo_client(tmp_path, cfg, live(tmp_path)), Library({start.id: start})


def player(standing, slot: int) -> WorldCastClient:
    """A client of the standing round's seat ``slot``, whose state model reads its own position
    where its keys put it."""
    config, library = standing
    start = library.get("r")
    client = WorldCastClient(config, LiveSceneStateConfig(), library)
    client.start(start, start.seat(slot))
    sw.own_position_oracle(client.client._rollout)
    return client


def block(client: WorldCastClient, k: int, *, forward: bool = False) -> list[LiveFrame]:
    """The frames of the client's ``k``-th block, its frames at the room times of video frames
    ``16 k + 1 .. 16 k + 16``."""
    return list(client.step(lambda: live_controls(first=16 * k, forward=forward)))


def deliver(sender: WorldCastClient, receiver: WorldCastClient, kinds=None) -> None:
    """Hand the messages ``sender`` queued (those of ``kinds``) to ``receiver``."""
    for message in sender.take_messages():
        if kinds is None or message.kind in kinds:
            receiver.receive_message(message)


def latent_of(frame: LiveFrame) -> int:
    return -(-frame.index // 4)


def test_a_frame_says_the_cameras_pitch_at_itself(standing):
    """A frame's header says the camera's pitch at the frame itself; the
    player state field's row of a frame is its latent frame's, read at the latent frame's last
    video frame. A player who looks down half a degree a frame from a level view."""
    frames = list(player(standing, 0).step(lambda: live_controls(pitch=0.5)))
    assert [f.index for f in frames] == list(range(17))
    assert [f.camera_pitch for f in frames] == pytest.approx([0.5 * k for k in range(17)], abs=1e-4)
    assert [f.state.pitch for f in frames] == pytest.approx(
        [0.0] + [2.0 * latent for latent in range(1, 5) for _ in range(4)], abs=1e-4
    )
    assert [frames[k].turned for k in (0, 1, 3, 16)] == [
        (0.0, 0.0),
        (0.5, 0.0),
        (1.5, 0.0),
        (8.0, 0.0),
    ]


def test_a_standing_player_stays_put_while_the_other_walks(standing):
    """Player 0 walks towards player 5, who plays and stands still; each steps a block, then the
    other: player 5 is nowhere until its first part reaches player 0, then drawn where it stands,
    nearer frame by frame, and hidden again once it has left before player 0's last block. Nobody
    else is in the world: the round's other players, seat 3 included, are recorded and nowhere.
    Every frame shows the rows the player state field read."""
    walker, idle = player(standing, 0), player(standing, STANDING)
    frames = {walker: [], idle: []}

    def play(client, k, **kwargs):
        shown = block(client, k, **kwargs)
        for frame in shown:  # every frame shows the rows the player state field read
            rows = client.client.field_rows(latent_of(frame))
            assert (frame.state, frame.others) == field_states(rows, frame.state.t)
        frames[client] += shown

    for k in range(6 + sw.MAX_BLOCKS):
        if k == 6 + sw.MAX_BLOCKS - 1:
            walker.remove_player(STANDING)  # its done may come late: hidden all the same
        play(walker, k, forward=True)
        deliver(walker, idle)
        play(idle, k)
        deliver(idle, walker)
    seen = {latent_of(frame): frame for frame in frames[walker]}

    def sources(slot: int) -> list[int | None]:
        return [next((p.src for p in seen[f].others if p.slot == slot), None) for f in range(33)]

    assert sources(STANDING) == [None] * 5 + [LIVE] * 24 + [None] * 4
    assert {p.slot for frame in frames[walker] for p in frame.others} == {STANDING}
    # its visibility label at each block's first denoising step
    drawn = [next(p.drawn for p in seen[f].others if p.slot == STANDING) for f in range(5, 29)]
    assert {type(label) for label in drawn} == {bool}
    rows = [p for frame in frames[walker] for p in frame.others if p.slot == STANDING]
    assert {(p.x, p.y, p.z) for p in rows} == {(1600.0, 600.0, 0.0)}
    walked = [frame.state for frame in frames[walker]]
    assert walked[-1].x - walked[0].x > 1000.0 and {s.src for s in walked} == {OWN}
    # the field's marker of player 5: first 1357 u ahead (player 0 has walked 243 u by player 5's
    # first block), 23.8 degrees to the left, then nearer, frame by frame, until player 0 has
    # walked past it
    markers = [m for frame in frames[walker] for m in frame.markers if m[0] == STANDING]
    assert markers[0] == pytest.approx((STANDING, 0.334224, 0.531436, 1357.2568), abs=1e-4)
    depth = [m[3] for m in markers]
    assert all(b <= a for a, b in zip(depth, depth[1:])) and depth[-1] < 600.0
    # the idle player draws the walker at the walker's own rows, from its first block on
    for frame in frames[idle][1:]:
        (drawn,) = [p for p in frame.others if p.slot == 0]
        own = next(f.state for f in frames[walker] if f.index == frame.index)
        assert drawn.src == LIVE and (drawn.x, drawn.y) == pytest.approx((own.x, own.y), abs=1e-3)
    for client in frames:
        client.stop()


def test_a_late_joiner_sees_the_earlier_player_at_once(standing):
    """Player 0 plays three blocks alone; player 5 joins with what the coordinator kept of it (its
    live parts and memory entries) and draws it from its first block on."""
    early = player(standing, 0)
    for k in range(3):
        block(early, k, forward=True)
    late = player(standing, STANDING)
    deliver(early, late, kinds=(LIVE_STATE, MEMORY_ENTRY))
    frames = block(late, 3)
    sources = [[p.src for p in frame.others if p.slot == 0] for frame in frames]
    assert sources == [[]] + [[LIVE]] * 16  # latent frame 0, the first frame, holds nobody
    for client in (early, late):
        client.stop()


# ------------------------------------------------------------------------------------ in a room
class ReceivingClient(WorldCastClient):
    """A WorldCast client that records the kinds of the messages it took in."""

    def start(self, round_start, seat):
        self.received = []
        super().start(round_start, seat)

    def receive_message(self, message):
        super().receive_message(message)
        self.received.append(message.kind)


async def two_players(client: WorldCastClientConfig):
    """Player 0 plays; once its first memory entry is kept, player 3 joins the room and plays."""
    start = round_start(0, 3)
    library = Library({start.id: start})
    settings = {"coordinator.host": "127.0.0.1", "coordinator.port": 0, "worker.port": 0}
    settings.update({"client.kind": "worldcast", "client.worldcast.taehv_path": client.taehv_path})
    clients = [ReceivingClient(client, LiveSceneStateConfig(), library) for _ in range(2)]
    async with LocalDemo(load_config((), settings), library, clients) as local:
        async with ClientSession() as http:
            coordinator = local.coordinator
            await until(lambda: len(coordinator.workers) == 2, "the workers to connect")
            room = await create_room(local, http, "r")
            first = await take_seat(local, http, room, 0)
            await until(lambda: coordinator.rooms[room].entries.get(0), "a kept memory entry")
            late = await take_seat(local, http, room, 3)
            late_client = next(w.client for w in local.workers if w.session.seat.slot == 3)

            def shown(browser, slot: int) -> list[int]:
                return [state.src for state in browser.states(slot)]

            await until(
                lambda: LIVE in shown(late, 0), "the late player's frames to show the first"
            )
            await until(lambda: "scene" in late_client.received, "the kept entries to arrive")
            await until(
                lambda: LIVE in shown(first, 3), "the first player's frames to show the late"
            )
            await first.close()  # the first player leaves: its client's done reaches the late one
            await until(lambda: "done" in late_client.received, "the first player's done")
            await late.close()
            return first, late, len(coordinator.workers)


def test_a_late_player_joins_a_room_that_holds_memory_entries(tmp_path, monkeypatch):
    _, client = client_config(tmp_path, monkeypatch, max_blocks=4)
    first, late, workers = asyncio.run(two_players(client))
    assert workers == 2
    assert first.welcome["frames_per_block"] == 16 and first.welcome["fps"] == 16.0
    # the late player's seat was nowhere in the first player's frames until it played, and the
    # late player saw the first one from its first block on
    assert {state.src for state in first.states(3)} == {LIVE}
    assert {state.src for state in late.states(0)} == {LIVE}
