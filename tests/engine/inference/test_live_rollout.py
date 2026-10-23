"""A client under live controls (``player_state.source: live``), on the CPU with the tiny models.

* The drift: while the client walks, and turns, a player standing still stays where it stands;
  the player state field draws it through the client's live camera, and under ``gt`` the same keys
  leave it pinned to one token (the bug live play fixes). Turning turns every camera of the
  client.
* What a client publishes is what it used: its ``live`` part, the rows its target frames were
  drawn with, the cameras of its rays, its memory entries and the rows of the page; a window
  prepared with the last controls held takes the rows, the weapon and the labels of its cut; a
  turn past the pitch range is cut in the view controls too.
* Two clients cross-wired through the messages; a late joiner; a leaver; a seat whose recorded
  player dies, taken by a human; the labels of latents 1-24 from the rows of the cut.
* An idle client keeps the clip of its stretch in its context (``live.idle_context``),
  alone in its world and beside other players, and what it publishes is what it keeps.
* Refusals, and the paper's paths keep their rollout.
"""

import json
import types
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

import tests.engine.inference.support as sw
import worldcast.engine.inference.live_rollout as live_rollout_module
import worldcast.engine.inference.rollout as rollout_module
import worldcast.player_state.live as live_module
import worldcast.player_state.predicted_visibility as predicted_visibility_module
import worldcast.sampling.window as W
from tests.player_state.support import flat_depth, round_batch, same_columns
from worldcast.data.camera import c2w_from_state_rows
from worldcast.data.controls import (
    CONTROL_BUTTONS,
    button_bits,
    buttons_from_bits,
    encode_weapon,
    quantize_camera_delta,
)
from worldcast.data.latents import (
    BLOCK,
    FIRST_TARGET,
    last_video_frames,
    video_frames_of,
)
from worldcast.data.memory_frames import MemoryFrames
from worldcast.data.recordings import ALIVE_INDEX, load_round_index_row
from worldcast.engine.inference.client import Client, client_world_state, run_client
from worldcast.engine.inference.controls import BlockControls, LiveBlockControls
from worldcast.engine.inference.field_rows import FieldRows
from worldcast.engine.inference.live_rollout import LiveRollout
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.messages import (
    LIVE_STATES_PER_ENTRY,
    MessagePart,
    MessageWorldState,
)
from worldcast.engine.inference.rollout import Rollout
from worldcast.engine.inference.serving import ServingOptions
from worldcast.player_state.extrapolation import (
    PhysicsPrior,
    PositionTrack,
    PredictedStateTable,
    physics_prior_track,
)
from worldcast.player_state.live import LiveLoop, LiveMotion, LiveVisibility
from worldcast.player_state.live_peers import block_substeps
from worldcast.player_state.predicted_visibility import PredictedVisibility
from worldcast.player_state.projection import project_players, project_view, table_angles
from worldcast.player_state.states import continuous_row_state, pack_substeps
from worldcast.player_state.visibility import fold_video_frames
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.window import WINDOW_ENTRIES
from worldcast.scene_state import SceneState

OWN, OTHER = 0, 5
LATENTS = 1 + BLOCK * (6 + sw.MAX_BLOCKS)
BLOCKS = list(range(1, LATENTS, BLOCK))
STANDING_AT = torch.tensor([1600.0, 600.0, 0.0])
PRIOR = PhysicsPrior.load(sw.STATE_TABLES / "physics_prior.json")


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    """The standing round, the tiny weights and a tiny state model. The round index holds the
    clients of slots 0 and 5. Slot 5's recording dies at 3 s in ``dies``."""
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("live")
    weights = sw.make_weights(tmp / "weights")
    live = {"player_state.source": "live", **sw.write_state_model(tmp / "state_model.pt")}
    worlds = {
        "standing": sw.standing_world(tmp / "standing", clients=(OWN, OTHER)),
        "dies": sw.standing_world(tmp / "dies", clients=(OWN, OTHER), dies={OTHER: 3.0}),
        "empty": sw.standing_world(tmp / "empty", clients=(OWN, OTHER)),
    }
    sw.empty_first_frame(worlds["empty"])  # a seat the demo's library offers
    return dict(tmp=tmp, weights=weights, live=live, worlds=worlds, models={})


def config(room, *, world="standing", source="live", settings=None):
    overrides = dict(room["live"]) if source != "gt" else {}
    overrides["player_state.source"] = source
    overrides.update(settings or {})
    cfg = sw.config(room["worlds"][world], room["weights"], room["tmp"] / "out", max_blocks=3)
    return cfg.with_overrides(overrides)


def models(room, cfg):
    key = cfg.player_state.source
    if key not in room["models"]:
        room["models"][key] = ClientModels.load(cfg, ServingOptions(decoder="none"))
    return room["models"][key]


def rollout(
    room,
    monkeypatch,
    relay=None,
    *,
    row=0,
    world="standing",
    source="live",
    extent=False,
    settings=None,
):
    """A rollout of the client of ``row``; under live controls its state model reads its own
    position where its keys put it, on open ground: the synthetic worlds are no map, unless
    ``extent`` keeps the walkable extent of the round's map; ``settings``: dotted settings of
    the config."""
    cfg = config(room, world=world, source=source, settings=settings)
    if not extent:
        sw.patch_extent(monkeypatch)
    sw.patch_ticks(monkeypatch, room["worlds"][world]["tables"])
    media = load_round_index_row(cfg.paths.round_index, row).media_id
    world_state = relay.join(media) if relay else MessageWorldState(client=media, poll_s=0.01)
    made = sw.make_rollout(cfg, world_state, models=models(room, cfg), row=row)
    if source == "live":
        sw.own_position_oracle(made)
    return made


def spy(rollout) -> list[dict]:
    """The conditions of every denoising call of ``rollout``, in order (one per block)."""
    given = []
    denoise = rollout.sampler.denoise

    def call(noisy, conditions, **kwargs):
        given.append(conditions)
        return denoise(noisy, conditions, **kwargs)

    rollout.sampler.denoise = call
    return given


def spy_steps(rollout) -> list[tuple[dict, dict]]:
    """The conditions of every block of ``rollout``, in order: its first denoising step's, and its
    later steps' (labelled again on the first step's x0)."""
    given = []
    denoise = rollout.sampler.denoise

    def call(noisy, conditions, *, retest, **kwargs):
        def retested(x0):
            given[-1] = conditions, retest(x0)
            return given[-1][1]

        given.append((conditions, None))
        return denoise(noisy, conditions, retest=retested, **kwargs)

    rollout.sampler.denoise = call
    return given


def target(conditions: dict, f0: int) -> slice:
    """The positions of block ``f0``'s latent frames in its conditions."""
    if f0 < FIRST_TARGET:
        return slice(f0, f0 + BLOCK)
    frames = int(conditions["player_state_table"].shape[1])
    return slice(frames - BLOCK, frames)


def keys_track(taken: list) -> tuple[np.ndarray, np.ndarray]:
    """Where a player's keys put it at each latent frame, from the origin looking along +x: the
    physics prior over the whole session, the yaw the turns summed; xyz ``[F, 3]``, yaw ``[F]``."""
    values = np.concatenate([block_substeps(c.buttons, c.turn)[0] for c in taken])
    values = np.concatenate([np.zeros_like(values[:1]), values])
    packed = pack_substeps(torch.from_numpy(values), torch.ones(len(values), 4, dtype=torch.bool))
    yaw = np.cumsum(np.concatenate([[0.0], *(c.turn[:, 1] for c in taken)]))
    xyz = physics_prior_track(
        packed[None], torch.from_numpy(yaw).float()[None], torch.zeros(1, 1, 3), prior=PRIOR
    )
    rows = last_video_frames(0, LATENTS)
    return xyz[0].double().numpy()[rows], yaw[rows]


def drawn_at(conditions: list[dict], slot: int = OTHER) -> tuple[np.ndarray, np.ndarray]:
    """Where the player state field puts the player of ``slot`` at every latent frame the blocks
    were denoised at: its feet ``[F, 2]`` (tokens) and its depth ``[F]`` (u)."""
    uv, depth = np.zeros((LATENTS, 2)), np.zeros(LATENTS)
    for f0, given in zip(BLOCKS, conditions):
        view = project_view(
            given["player_state_table"],
            given["player_controls"],
            given["client_slot"],
            grid_h=12,
            grid_w=21,
        )
        uv[f0 : f0 + BLOCK] = view.uv[0, target(given, f0), slot].numpy()
        depth[f0 : f0 + BLOCK] = view.depth[0, target(given, f0), slot].numpy()
    return uv, depth


def beside_idle_player(room, monkeypatch) -> tuple[sw.Relay, LiveRollout, LiveRollout]:
    """The client of slot OWN and, in its world, the client of slot OTHER, who stands still where
    its recording stands: the standing player nobody else's world holds anything beside. They are
    cross-wired through the messages."""
    relay = sw.Relay()
    return (
        relay,
        rollout(room, monkeypatch, relay, row=0),
        rollout(room, monkeypatch, relay, row=1),
    )


def step_pair(relay, live: LiveRollout, idle: LiveRollout, f0: int, k: int, controls) -> None:
    """Block ``f0`` of the idle client, then of ``live`` on ``controls``, each passed on."""
    sw.step(idle, f0, lambda: sw.live_controls(k))
    relay.deliver(idle.media_id)
    sw.step(live, f0, lambda: controls)
    relay.deliver(live.media_id)


# ------------------------------------------------------------------------------------- the drift
@pytest.mark.parametrize(
    ("held", "yaw", "first", "leaves"),
    [
        # it slides to the left and leaves the view by the left edge at latent frame 20
        (("forward",), 0.0, [7.515, 6.323], 20),
        # turning left 25 degrees per block, it slides to the right and leaves by the right edge
        (("forward", "move_left"), 25.0, [8.532, 6.312], 10),
    ],
    ids=["walk", "walk-and-turn"],
)
def test_a_standing_player_stays_put_while_the_client_walks(
    room, monkeypatch, held, yaw, first, leaves
):
    """The standing player, whom a human plays, never moves, and the field draws it through the
    camera the client's keys put it at, latent frame by latent frame. It is in the client's world
    from the first block on, nobody is in the first frame, and nobody else is in the world."""
    relay, live, idle = beside_idle_player(room, monkeypatch)
    given = spy(live)
    taken = [sw.live_controls(k, *held, yaw=yaw) for k in range(len(BLOCKS))]
    for k, (f0, controls) in enumerate(zip(BLOCKS, taken)):
        step_pair(relay, live, idle, f0, k, controls)
    for conditions in given:
        assert (conditions["player_state_table"][0, :, OTHER, :3] == STANDING_AT).all()
    in_world = live.closed.visibility.live
    assert in_world[1:, OTHER].all() and not in_world[0].any()
    assert not in_world[:, [slot for slot in range(10) if slot != OTHER]].any()
    assert live.field_rows(LATENTS - 1).source.tolist() == (
        [FieldRows.OWN] + [FieldRows.HIDDEN] * 4 + [FieldRows.LIVE] + [FieldRows.HIDDEN] * 4
    )
    xyz, heading = keys_track(taken)
    uv, depth = drawn_at(given)
    expected, _, _, ahead = project_players(
        STANDING_AT.expand(1, LATENTS, 1, 3),
        torch.from_numpy(xyz).float()[None],
        torch.from_numpy(heading).float()[None],
        torch.zeros(1, LATENTS),
        grid_h=12,
        grid_w=21,
    )
    front = depth[1:] > 1.0
    np.testing.assert_allclose(
        uv[1:][front], expected[0, 1:, 0][front].numpy(), atol=1e-3, rtol=1e-4
    )
    np.testing.assert_allclose(depth[1:], ahead[0, 1:, 0].numpy(), atol=1e-2)
    assert uv[1].tolist() == pytest.approx(first, abs=1e-3)
    in_view = (uv[1:, 0] >= 0.0) & (uv[1:, 0] <= 21.0) & front
    assert in_view.tolist() == [True] * (leaves - 1) + [False] * (LATENTS - leaves)


def test_a_client_stops_at_the_edge_of_its_maps_extent(room, monkeypatch):
    """The walkable extent comes from the state model's place head and the round's map: a client
    that keeps running along +x from the round start leaves the track its keys put it on where
    the map ends, and stands there."""
    live = rollout(room, monkeypatch, extent=True)
    taken = [sw.live_controls(k, "forward") for k in range(len(BLOCKS))]
    for f0, controls in zip(BLOCKS, taken):
        sw.step(live, f0, lambda controls=controls: controls)
    track = live.closed.cameras.track.xyz[: BLOCKS[-1]]  # up to the last block's first latent frame
    xyz, _ = keys_track(taken)
    assert all(live.closed.walkable.contains(p[:2]) for p in track)
    assert (track[-2:] == track[-1]).all() and xyz[-1, 0] - track[-1, 0] > 300.0
    assert track[-1, 0] == pytest.approx(1529.0, abs=0.5)


def at_the_edge(room, monkeypatch, wall_keys: str):
    """A client that runs along +x from the round start for good on its map's extent, with the
    controls each block was cut with (:meth:`LiveRollout._take_controls`), the ``live`` parts it
    sent, the conditions of its denoising calls and whether each window block was idle."""
    live = rollout(
        room,
        monkeypatch,
        extent=True,
        settings={"live.wall_keys": wall_keys, "live.idle_context": "freeze"},
    )
    cut, sent, idle = [], [], []
    take = live._take_controls

    def taking(f0, controls):
        cut.append(controls.buttons.copy())
        return take(f0, controls)

    live._take_controls, live.world_state.outbox = taking, sent.append
    conditions = spy(live)
    for f0, k in zip(BLOCKS, range(len(BLOCKS))):
        controls = sw.live_controls(k, "forward")
        sw.step(live, f0, lambda controls=controls: controls)
        idle.append(live._idle)
    return live, cut, [p for p in sent if p.kind == "live"], conditions, idle


def test_a_client_at_the_edge_of_its_extent_runs_on_without_its_keys_when_wall_keys_says_so(
    room, monkeypatch
):
    """The edge of the map stops the client's camera in block 25, which keeps its forward key for
    eight rows; block 29 is pinned at the edge from its first row. With ``edge`` the keys are
    cleared from there in every place the controls go: the generator's conditions (the first six
    blocks' round and a window's target frames), the ``live`` part the other clients extend the
    player by, and the idle stretch's test (the pinned block is idle, the one before it is not);
    ``held`` leaves them as they are."""
    forward = CONTROL_BUTTONS.index("forward")
    held, cut, _, _, idle = at_the_edge(room, monkeypatch, "held")
    assert all((buttons[:, forward] == 1).all() for buttons in cut) and not any(idle)
    live, cut, parts, conditions, idle = at_the_edge(room, monkeypatch, "edge")
    kept = [int((buttons[:, forward] > 0.5).sum()) for buttons in cut]
    assert kept[:6] == [16] * 6 and 0 < kept[6] < 16 and kept[7] == 0
    assert idle == [False] * 7 + [True]
    for buttons, part, given, f0 in zip(cut, parts, conditions, BLOCKS):
        assert np.array_equal(buttons_from_bits(part.buttons), buttons)
        rows = (
            given["buttons"][0, -16:]
            if f0 >= FIRST_TARGET
            else given["buttons"][0, video_frames_of(f0, BLOCK)]
        )
        assert np.array_equal(rows.numpy(), buttons)
        assert np.array_equal(buttons[:, forward] > 0.5, np.arange(16) < kept[BLOCKS.index(f0)])


def test_under_the_gt_states_the_keys_leave_the_standing_player_on_one_token(room, monkeypatch):
    """The drift live play removes: under ``gt`` the client's camera follows its recording, which
    stands still, so walking leaves the standing player on one token at one depth."""
    gt = rollout(room, monkeypatch, source="gt")
    pinned = spy(gt)
    for k, f0 in enumerate(BLOCKS):
        sw.step(gt, f0, lambda k=k: sw.live_controls(k, "forward"))
    uv, depth = drawn_at(pinned)
    assert np.allclose(uv[1:], [7.547, 6.320], atol=1e-3) and np.allclose(depth[1:], 1600.0)


def test_a_seat_that_shows_nobody_in_its_first_frame_has_nobody_there_in_every_block(
    room, monkeypatch
):
    """Every window attends the round's first frame (position 0) and the field says nobody is in
    it: a window block labels it against a world without the recorded players, the first six
    blocks read the dataset's labels of latent frame 0, which show nobody in a seat the library
    offers. On a seat whose first frame shows people the two disagree: only the first six blocks'
    conditions carry them, which is why the library does not offer it and the demo's client
    refuses it."""
    seen = {}
    for world in ("empty", "standing"):
        live = rollout(room, monkeypatch, world=world)
        given = spy(live)
        for k, f0 in enumerate(BLOCKS):
            sw.step(live, f0, lambda k=k: sw.live_controls(k, "forward"))
        assert len(given) == len(BLOCKS)
        seen[world] = [
            conditions["player_visible"][0, 0].nonzero().flatten().tolist() for conditions in given
        ]
    first_six = [f0 < FIRST_TARGET for f0 in BLOCKS]
    assert seen["empty"] == [[]] * len(BLOCKS)
    assert [bool(people) for people in seen["standing"]] == first_six


def test_turning_turns_every_camera_of_the_client(room, monkeypatch):
    """90 degrees to the left per block: the field's camera, the rays, retrieval and the cameras
    of the visibility labels."""
    retrieved = []
    retrieve = SceneState.retrieve

    def spy_retrieve(self, **kwargs):
        retrieved.append(torch.as_tensor(kwargs["next_c2w"]).clone())
        return retrieve(self, **kwargs)

    monkeypatch.setattr(SceneState, "retrieve", spy_retrieve)
    live = rollout(room, monkeypatch)
    given = spy(live)
    for k, f0 in enumerate(BLOCKS):
        sw.step(live, f0, lambda k=k: sw.live_controls(k, yaw=90.0))
    expected = (90.0 * np.arange(LATENTS) / BLOCK + 180.0) % 360.0 - 180.0

    def heading(forward: torch.Tensor) -> np.ndarray:
        forward = forward.double()
        return np.degrees(np.arctan2(forward[..., 1].numpy(), forward[..., 0].numpy()))

    def same(a, b) -> bool:
        return bool(np.all(np.abs((a - b + 180.0) % 360.0 - 180.0) < 1e-3))

    field, rays = np.zeros(LATENTS), np.zeros(LATENTS)
    for f0, conditions in zip(BLOCKS, given):
        yaw, _ = table_angles(conditions["player_state_table"], conditions["player_controls"])
        field[f0 : f0 + BLOCK] = yaw[0, target(conditions, f0), OWN].numpy()
        rays[f0 : f0 + BLOCK] = heading(
            conditions["ray_frame_c2w"][0, target(conditions, f0), :3, 2]
        )
    assert same(field[1:], expected[1:]) and same(rays[1:], expected[1:])
    cameras = live.closed.visibility.cameras[last_video_frames(1, LATENTS - 1), :3, 2]
    assert same(heading(cameras), expected[1:])
    # retrieval looks where a block is prepared to look: with the last turn held; the block after
    # the last is prepared too
    assert len(retrieved) == 3
    for f0, next_c2w in zip((25, 29), retrieved):
        assert same(heading(next_c2w[:, :3, 2]), expected[f0 : f0 + BLOCK])


def test_what_a_client_publishes_is_what_it_used(room, monkeypatch):
    sent = []
    live = rollout(room, monkeypatch)
    live.world_state.outbox = sent.append
    given = spy(live)
    rows = {}
    for k, f0 in enumerate(BLOCKS):
        sw.step(live, f0, lambda k=k: sw.live_controls(k, "forward", "move_left", yaw=25.0))
        rows.update((f, live.field_rows(f)) for f in range(f0, f0 + BLOCK))
    parts = [part for part in sent if part.kind == "live"]
    assert [part.f0 for part in parts] == BLOCKS
    own = {block.f0: block.c2w for block in live.known.blocks if block.media_id == live.media_id}
    assert sorted(own) == BLOCKS
    for part, conditions in zip(parts, given):
        f0, used = part.f0, part.rows[3::4]
        cameras = c2w_from_state_rows(np.column_stack([used, np.ones(4, np.float32)]))
        at = target(conditions, f0)
        # the rows the field drew the block with; their angles are the continuous columns'
        table = conditions["player_state_table"][0, at, OWN]
        assert np.array_equal(table[:, :3].numpy(), used[:, :3])
        if f0 >= FIRST_TARGET:
            np.testing.assert_allclose(table[:, 6:8].numpy(), used[:, 3:5], atol=1e-3)
        # the cameras of its rays and of its memory entry
        assert torch.equal(conditions["ray_frame_c2w"][0, at], cameras)
        assert torch.equal(own[f0], cameras)
        # the page's rows
        for i, f in enumerate(range(f0, f0 + BLOCK)):
            assert np.array_equal(rows[f].states[OWN, :3], used[i, :3].astype(np.float64))
            assert rows[f].states[OWN, 3:5] == pytest.approx(used[i, 3:5], abs=1e-3)
    last = rows[LATENTS - 1]
    assert last.source.tolist() == [FieldRows.OWN] + [FieldRows.HIDDEN] * 9
    assert last.status == "guarded" and last.drawn.dtype == bool and last.drawn.shape == (10,)


def test_a_window_takes_the_controls_of_its_cut(room, monkeypatch):
    """Block 25 is prepared standing still with the AK-47, as the client stood; at its cut the
    client strafes right turning 90 degrees right with the AWP, and at block 29's cut it walks
    forward with the AK-47 again. Each window's target frames are drawn, labelled, keyed and
    published with the rows and controls of its cut; the anchor of its rays stays the camera its
    context was written with."""
    sent = []
    relay, live, idle = beside_idle_player(room, monkeypatch)
    live.closed.visibility.depth_fn = flat_depth(5000.0)
    live.world_state.outbox = sent.append
    steps = spy_steps(live)
    prepared = {}
    prepare = live._prepare

    def keep(f0):
        prepared[f0] = prepare(f0)
        return prepared[f0]

    live._prepare = keep
    cuts = {25: (("move_right",), -90.0, "awp"), 29: (("forward",), 0.0, "ak47")}
    for k, f0 in enumerate(BLOCKS):
        held, yaw, weapon = cuts.get(f0, ((), 0.0, "ak47"))
        controls = sw.live_controls(k, *held, yaw=yaw, weapon=weapon)
        sw.step(idle, f0, lambda k=k: sw.live_controls(k))
        relay.deliver(idle.media_id)
        sw.step(live, f0, lambda controls=controls: controls)
    parts = {part.f0: part for part in sent if part.kind == "live"}
    own = {block.f0: block.c2w for block in live.known.blocks if block.media_id == live.media_id}
    for f0, block, scope in ((25, steps[6], [0] * 4), (29, steps[7], [1] * 4)):
        before, at, used = prepared[f0], target(block[0], f0), parts[f0].rows
        cameras = c2w_from_state_rows(np.column_stack([used[3::4], np.ones(4, np.float32)]))
        # the rows the window was prepared with are not the cut's
        provisional = before.batch["player_states"][0, OWN, -16:, :3].numpy()
        assert np.abs(provisional - used[:, :3]).max() > 100.0
        assert torch.equal(own[f0], cameras)  # the keys of its memory entry
        assert before.conditions["obs_scope_valid"][0, at].tolist() == [1 - v for v in scope]
        for conditions in block:
            # the rows of the cut: drawn, the cameras of the rays, published
            table = conditions["player_state_table"][0, at, OWN, :3].numpy()
            assert np.array_equal(table, used[3::4, :3])
            assert torch.equal(conditions["ray_frame_c2w"][0, at], cameras)
            assert torch.equal(conditions["ray_anchor_c2w"], before.conditions["ray_anchor_c2w"])
            assert not torch.equal(conditions["ray_anchor_c2w"][0], cameras[0])
            # the weapon of the cut: the AWP's scope is unknown, the AK-47's off
            assert conditions["obs_scope_valid"][0, at].tolist() == scope
            # the window's first frame says nobody is in the world, the idle player included
            assert not conditions["player_visible"][0, 0].any()
    # the standing player 20 degrees to the left, labelled visible when block 25 was prepared,
    # leaves the view in the block's middle as the client turns at the cut
    at = target(steps[6][0], 25)
    assert prepared[25].conditions["player_visible"][0, at, OTHER].all()
    assert steps[6][0]["player_visible"][0, at, OTHER].tolist() == [1, 1, 0, 0]
    drawn = [bool(live.closed.visibility.drawn(f)[OTHER]) for f in range(25, 29)]
    assert drawn == [True, True, False, False]


def test_a_turn_past_the_pitch_range_is_cut_in_the_view_controls_too(room, monkeypatch):
    """160 degrees down over block 1: the pitch stops at 89 degrees in every camera, and the view
    controls the generator reads are those of the turn that was made."""
    live = rollout(room, monkeypatch)
    given = spy(live)
    sw.step(live, 1, lambda: sw.live_controls(0, pitch=160.0))
    rows = video_frames_of(1, BLOCK)
    pitch = live.closed.motion.pitch[rows]
    assert pitch.tolist() == pytest.approx([10.0 * k for k in range(1, 9)] + [89.0] * 8)
    assert live.continuous_rows[0, 1:5, OWN, 1].tolist() == pytest.approx([40.0, 80.0, 89.0, 89.0])
    turned = np.diff(np.concatenate([[0.0], pitch]))
    codes = np.stack([quantize_camera_delta([p, 0.0], clip=False) for p in turned])
    assert codes[:, 0].tolist() == pytest.approx([0.65] * 8 + [0.6] + [0.0] * 7)
    assert np.array_equal(live.round_batch["view_deltas"][0, rows].numpy(), codes)
    assert np.array_equal(given[0]["view_deltas"][0, rows].numpy(), codes)


@pytest.mark.parametrize(("gain", "yaw_code"), [(1.0, 0.175), (1.1, 0.2)])
def test_the_pitch_cut_keeps_the_yaw_the_input_path_fed_and_the_camera_the_turn_made(
    room, monkeypatch, gain, yaw_code
):
    """The input path may feed the generator a gain on the yaw (``play.input_mapping: hybrid``,
    ``tests/demo/test_inputs.py``): the rows the pitch cut encodes again keep their yaw code, with
    no gain as much as with one, and the camera integrates the turn the player made, not the
    controls."""
    live = rollout(room, monkeypatch)
    given = spy(live)
    made = sw.live_controls(0, pitch=160.0, yaw=32.0)  # 2 degrees of yaw a frame
    fed = np.stack([quantize_camera_delta(row * (1.0, gain), clip=False) for row in made.turn])
    assert fed[:, 1].tolist() == pytest.approx([yaw_code] * 16)
    sw.step(live, 1, lambda: replace(made, view_deltas=fed))
    rows = video_frames_of(1, BLOCK)
    codes = given[0]["view_deltas"][0, rows].numpy()
    assert np.array_equal(codes[:, 1], fed[:, 1])  # the yaw as fed, in every row
    assert codes[:, 0].tolist() == pytest.approx([0.65] * 8 + [0.6] + [0.0] * 7)
    assert live.closed.motion.yaw[rows].tolist() == pytest.approx([2.0 * k for k in range(1, 17)])


# --------------------------------------------------------------------------------- live players
def test_two_clients_play_together(room, monkeypatch):
    """Cross-wired through the messages in random step orders, the walker's parts up to a block
    late and one of its blocks lost (the part after it on time: late as well, the walker's newest
    rows would be ``PEER_HIDE_S`` old and it hidden): the idle client stays put on the walker's
    screen, the walker is drawn within the extrapolation of its newest rows (the idle client's
    clock is 0.1 s ahead, 1.6 video frames: 24.5 u at the walker's speed), and the block whose part
    was lost is never admitted."""
    relay = sw.Relay()
    walker = rollout(room, monkeypatch, relay, row=0)
    idle = rollout(room, monkeypatch, relay, row=1)
    rng = np.random.default_rng(0)
    for k, f0 in enumerate(BLOCKS):
        for client in rng.permutation([0, 1]):
            if client == 0:
                sw.step(walker, f0, lambda k=k: sw.live_controls(k, "forward"))
                late = int(rng.integers(0, 2)) if f0 < FIRST_TARGET and f0 != 13 + BLOCK else 0
                waiting = len(relay.queues[walker.media_id])
                relay.deliver(walker.media_id, waiting - late, lost=[("live", 13)])
            else:
                sw.step(idle, f0, lambda k=k: sw.live_controls(k, clock=0.1))
                relay.deliver(idle.media_id)
    # each is in the other's world from the other's first part on, and nobody before it
    in_world = walker.closed.in_world(LATENTS - 1)
    assert in_world[OTHER] and walker.closed.visibility.live[17:, OTHER].all()
    assert not walker.closed.visibility.live[0].any() and not idle.closed.visibility.live[0].any()
    assert (walker.round_batch["player_states"][0, OTHER, 17:, :3] == STANDING_AT).all()
    drawn = idle.round_batch["player_states"][0, OWN, :, :3].numpy()
    walked = walker.round_batch["player_states"][0, OWN, :, :3].numpy()
    error = np.abs(drawn - walked).max(1)
    shown = idle.closed.visibility.live[:, OWN].numpy()
    assert shown[17:].all() and error[17:].max() < 2 * 245.0  # at most two blocks behind
    # every part on time from latent frame 25 on: the walker is drawn at its rows, or a block
    # behind, where the idle client steps first (keys held 8 video frames, then coasting)
    assert error[97:].max() < 128.0 and (np.abs(error[97:] - 24.5) < 0.5).any()
    assert (walker.media_id, 8 * 13) in idle.known.skipped
    assert (idle.round_batch["player_states"][0, :, :, ALIVE_INDEX] == 1.0).all()


def test_a_late_joiner_sees_the_earlier_player_at_once(room, monkeypatch):
    relay = sw.Relay()
    early = rollout(room, monkeypatch, relay, row=0)
    for k, f0 in enumerate(BLOCKS[:3]):
        sw.step(early, f0, lambda k=k: sw.live_controls(k, "forward"))
        relay.deliver(early.media_id)
    clock = 3 * 16 * sw.FRAME_S  # the joiner's first block at the time of the early client's 4th
    late = rollout(room, monkeypatch, relay, row=1)
    for k, f0 in enumerate(BLOCKS[3:]):
        sw.step(late, BLOCKS[k], lambda k=k: sw.live_controls(k, clock=clock))
        relay.deliver(late.media_id)
        sw.step(early, f0, lambda k=k: sw.live_controls(3 + k, "forward"))
        relay.deliver(early.media_id)
    # the joiner draws the early client from its own first block on
    assert late.closed.visibility.live[1:81, OWN].all()
    # the early client had nobody in its world until the joiner's first part, then draws the
    # joiner from its first block on
    shown = early.closed.visibility.live[:, OTHER]
    assert not shown[: video_frames_of(13, 1)[0]].any()
    assert shown[video_frames_of(13, LATENTS - 13)].all()


def test_a_leaver_is_hidden_from_the_next_block(room, monkeypatch):
    relay = sw.Relay()
    a, b = rollout(room, monkeypatch, relay, row=0), rollout(room, monkeypatch, relay, row=1)
    for k, f0 in enumerate(BLOCKS[:4]):
        for client in (a, b):
            sw.step(client, f0, lambda k=k: sw.live_controls(k))
            relay.deliver(client.media_id)
    a.world_state.receive(MessagePart(b.media_id, "done", status="ok"))
    sw.step(a, 17, lambda: sw.live_controls(4))
    shown = a.closed.visibility.live[:, OTHER]
    assert shown[video_frames_of(13, BLOCK)].all() and not shown[video_frames_of(17, BLOCK)].any()
    assert (a.round_batch["player_states"][0, OTHER, :, ALIVE_INDEX] == 1.0).all()
    assert a.field_rows(16).source[OTHER] == FieldRows.LIVE
    assert a.field_rows(17).source[OTHER] == FieldRows.HIDDEN


def test_a_seat_whose_recording_dies_is_nowhere_and_can_be_taken(room, monkeypatch):
    """Slot 5's recording dies at 3 s: nobody is in its place before or after, never a corpse; a
    human who takes the seat later is drawn without a resurrection."""
    relay = sw.Relay()
    a = rollout(room, monkeypatch, relay, row=0, world="dies")
    for k, f0 in enumerate(BLOCKS[:4]):
        sw.step(a, f0, lambda k=k: sw.live_controls(k))
    assert (a.round_batch["player_states"][0, OTHER, :, ALIVE_INDEX] == 1.0).all()
    shown = a.closed.visibility.live[:, OTHER]
    assert not shown.any()
    assert {a.field_rows(f).source[OTHER] for f in (11, 12)} == {FieldRows.HIDDEN}
    b = rollout(room, monkeypatch, relay, row=1, world="dies")
    for k, f0 in enumerate(BLOCKS[4:]):
        sw.step(b, BLOCKS[k], lambda k=k: sw.live_controls(k, clock=64 * sw.FRAME_S))
        relay.deliver(b.media_id)
        sw.step(a, f0, lambda k=k: sw.live_controls(4 + k))
    assert not shown[:65].any() and shown[65:].all()
    assert a.field_rows(LATENTS - 1).source[OTHER] == FieldRows.LIVE


def test_the_cut_writes_only_the_blocks_frames_as_the_whole_round_gives_them(room, monkeypatch):
    """Latents 1-24: the round's conditions written again at the block's latent frames, and left
    before them, are a rebuild of the whole round's there; the labels the block is denoised with,
    tested against the surface of its context read before the cut, are those of a test from
    scratch after it. The continuous columns of the block, written on their own, are the whole
    round's (the yaw to float32 rounding, every other column bit for bit). A window: the cameras of
    the cut are those the block's start gives from the rows of the cut."""
    live = rollout(room, monkeypatch)
    rebuilt, columns, cameras, labels = [], [], [], {}
    take, visibility = live._take_controls, live.closed.visibility
    visibility.depth_fn = flat_depth(5000.0)

    def spy_take(f0, controls):
        take(f0, controls)
        # copies: on the CPU the controls' conditions are the history later blocks write into
        snapshot = (live.round_conditions, live._round_conditions())
        rebuilt.append((f0, *({k: v.clone() for k, v in c.items()} for c in snapshot)))
        columns.append((f0, live.continuous_rows.clone(), continuous_row_state(live.round_batch)))
        if f0 < FIRST_TARGET:
            context = PredictedVisibility._target_context(visibility, f0, live.output[0])
            visible, _ = PredictedVisibility.target_labels(visibility, *context)
            labels[f0] = fold_video_frames(visible, f0, BLOCK)

    live._take_controls = spy_take
    target = live._live_target

    def spy_target(f0, prepared):
        out = target(f0, prepared)
        cameras.append((out.cameras, live.closed.cameras.for_block(f0)))
        return out

    live._live_target = spy_target
    for k, f0 in enumerate(BLOCKS):
        sw.step(live, f0, lambda k=k: sw.live_controls(k, "forward", "move_left", yaw=37.3))
    assert [f0 for f0, _, _ in rebuilt] == BLOCKS and len(cameras) == 2
    for f0, written, whole in rebuilt[:6]:
        upto = slice(None, f0 + BLOCK)
        for key, value in whole.items():
            if key == "player_visible":  # written with the block's denoising
                continue
            if value.ndim >= 2 and value.shape[1] == live.latent_frames:
                assert torch.equal(written[key][:, upto], value[:, upto]), (f0, key)
            else:
                assert torch.equal(written[key], value), (f0, key)
        drawn = np.stack([visibility.drawn(f) for f in range(f0, f0 + BLOCK)])
        assert np.array_equal(drawn, labels[f0].numpy()), f0
    for f0, kept, whole in columns:  # the yaw to rounding: a turn of 2.33 degrees a row
        same_columns(kept[:, : f0 + BLOCK], whole[:, : f0 + BLOCK])
    for at_cut, whole in cameras:
        assert torch.equal(at_cut, whole)


class Dispatched(TorchDispatchMode):
    """Counts the ATen operations dispatched while it is entered, and the elements they write: of
    the outputs of every operation but a view, outside the reprojection of the context's surface
    into the target frames' cameras (``_zbuffer``), which runs on the client's device and writes
    the same at every block of a window."""

    def __init__(self) -> None:
        super().__init__()
        self.count, self.elements, self.on_device = 0, 0, False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.count += 1
        if not (self.on_device or func.is_view):
            self.elements += sum(t.numel() for t in tree_leaves(out) if isinstance(t, torch.Tensor))
        return out


def dispatched(work) -> Dispatched:
    with Dispatched() as counted:
        work()
    return counted


#: What the cut never computes again, nor any of it: the whole round's conditions or continuous
#: columns, a window's inputs with its latents, the depth of the context (read when the block is
#: prepared).
REBUILDS = (
    (Rollout, "_round_conditions"),
    (Rollout, "_inputs"),
    (rollout_module, "continuous_row_state"),
    (live_rollout_module, "continuous_row_state"),
    (PredictedVisibility, "_depth"),
)


def reference_workload() -> Dispatched:
    """The counts of the cut path's reference workload: the integral of the camera angles of a
    round of the demo's 441 latent frames (``continuous_row_state``), one of the things the
    release's cut redid at every block."""
    batch = round_batch(0, batch=1, latents=441)
    return dispatched(lambda: continuous_row_state(batch))


def count_cuts(
    monkeypatch, clients
) -> tuple[dict[int, Dispatched], list[str], list[int], Callable[[int], None]]:
    """Count the cut path, from a block's controls taken to its first denoising step: per block,
    what :class:`Dispatched` counts from a call of the returned ``count(f0)`` (in the controls
    callback) on; the names of :data:`REBUILDS` called meanwhile; and the rows of the own motion
    of ``clients`` each prior call integrates."""
    cuts, counting, rebuilt, own_rows = {}, [], [], []
    denoise = Sampler.denoise

    def first_step(self, *args, **kwargs):
        if counting:
            counting.pop().__exit__(None, None, None)
        return denoise(self, *args, **kwargs)

    monkeypatch.setattr(Sampler, "denoise", first_step)
    for owner, name in REBUILDS:

        def watched(*args, original=getattr(owner, name), name=name, **kwargs):
            if counting:
                rebuilt.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, watched)
    zbuffer = predicted_visibility_module._zbuffer

    def on_device(*args, **kwargs):
        for cut in counting:
            cut.on_device = True
        out = zbuffer(*args, **kwargs)
        for cut in counting:
            cut.on_device = False
        return out

    monkeypatch.setattr(predicted_visibility_module, "_zbuffer", on_device)
    for client in clients:
        motion = client.closed.motion
        prior_input = motion.prior_input

        def spy_input(first_row, substeps, prior_input=prior_input):
            own_rows.append(int(substeps.shape[0]))
            return prior_input(first_row, substeps)

        motion.prior_input = spy_input

    def count(f0: int) -> None:
        cuts[f0] = Dispatched().__enter__()
        counting.append(cuts[f0])

    return cuts, rebuilt, own_rows, count


def long_rollout(room, monkeypatch, world_state, *, row: int = 0) -> LiveRollout:
    """A live rollout of the standing round's client of ``row``, ten blocks long."""
    cfg = sw.config(room["worlds"]["standing"], room["weights"], room["tmp"] / "long", max_blocks=4)
    cfg = cfg.with_overrides(room["live"])
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    made = sw.make_rollout(cfg, world_state, models=models(room, cfg), row=row)
    sw.own_position_oracle(made)
    return made


def test_the_cut_costs_few_operations_whatever_the_session(room, monkeypatch):
    """The cut path, from a block's controls taken to its first denoising step, counted in ATen
    operations and the elements they write (counts, not wall times) against a reference workload
    counted alike (:func:`reference_workload`). A block's cut costs under a third of it in both,
    the same at every block; one where a part of the other player, 0.3 s behind, arrived after the
    block was prepared integrates the part's extension too, under a half. It calls nothing of
    :data:`REBUILDS`; the own motion integrates only the block's 16 rows."""
    relay = sw.Relay()
    walker, other = (
        long_rollout(room, monkeypatch, relay.join(sw.media_id(slot)), row=row)
        for row, slot in ((0, OWN), (1, OTHER))
    )
    cuts, rebuilt, own_rows, count = count_cuts(monkeypatch, (walker, other))
    blocks = list(range(1, 41, BLOCK))
    for k, f0 in enumerate(blocks):
        # every second block the other client steps first: its parts of this block and the one
        # before reach the walker after the walker prepared its block
        for client in (other, walker) if k % 2 == 0 else (walker, other):
            controls = sw.live_controls(
                k, "forward", yaw=12.0 * (k % 3 - 1), clock=-0.3 * (client is other)
            )

            def taken(controls=controls, counted=client is walker, f0=f0):
                if counted:
                    count(f0)
                return controls

            sw.step(client, f0, taken)
            relay.deliver(client.media_id)
    reference = reference_workload()
    newer = [cuts[f0] for k, f0 in enumerate(blocks) if f0 >= FIRST_TARGET and k % 2 == 0]
    own = [cuts[f0] for k, f0 in enumerate(blocks) if f0 >= FIRST_TARGET and k % 2 == 1]
    first_six = [cuts[f0] for f0 in blocks if f0 < FIRST_TARGET]
    assert len(newer) == len(own) == 2 and len(first_six) == 6 and rebuilt == []
    for measure in ("count", "elements"):
        counted = {
            name: [getattr(cut, measure) for cut in cuts]
            for name, cuts in (("newer", newer), ("own", own), ("first_six", first_six))
        }
        bound = getattr(reference, measure)
        assert max(counted["own"] + counted["first_six"]) < bound / 3, (measure, counted, bound)
        assert max(counted["newer"]) < bound / 2, (measure, counted, bound)
    assert max(cut.count for cut in own) - min(cut.count for cut in own) <= 50
    # each client's own rows of every block, prepared, then cut, and of the block after the last,
    # prepared on a round that slid
    assert set(own_rows) == {16} and len(own_rows) == 2 * (2 * len(blocks) + 1)


def test_the_cut_with_three_newer_peers_costs_under_half_the_reference(room, monkeypatch):
    """Three other live players, a quarter, a half and 0.9 s behind the walker, whose parts of
    every block arrive between its prepare and its cut: each extension is integrated only up to
    the row of the block's last time (4, 8 and 15 rows past the newest, after the 32 warm rows), in
    the one prior call of the cut, and the cut costs under half the reference workload."""
    world_state = MessageWorldState(client=sw.media_id(OWN), poll_s=0.01)
    walker = long_rollout(room, monkeypatch, world_state)
    cuts, rebuilt, own_rows, count = count_cuts(monkeypatch, (walker,))
    calls, integrate = [], live_module.prior_displacements

    def spy_integrate(inputs, *, prior):
        calls.append(sorted(int(rows.shape[0]) for rows, _, _ in inputs))
        return integrate(inputs, prior=prior)

    monkeypatch.setattr(live_module, "prior_displacements", spy_integrate)
    behind = {1: 0.25, 2: 0.5, OTHER: 0.9}
    forward = np.zeros((16, len(CONTROL_BUTTONS)), np.float32)
    forward[:, CONTROL_BUTTONS.index("forward")] = 1.0
    blocks = list(range(1, 41, BLOCK))
    at_cut = {}
    for k, f0 in enumerate(blocks):
        controls = sw.live_controls(k, "forward", yaw=12.0 * (k % 3 - 1))

        def taken(controls=controls, f0=f0):
            for slot, late in behind.items():
                times = controls.times - late
                rows = np.zeros((16, 5), np.float32)
                rows[:, 0], rows[:, 1] = 200.0 * times, 300.0 * slot
                world_state.receive(
                    MessagePart(
                        sw.media_id(slot),
                        "live",
                        f0=f0,
                        times=times,
                        rows=rows,
                        buttons=button_bits(forward),
                        turn=np.zeros((16, 2), np.float32),
                        weapon=np.full(16, encode_weapon("ak47"), np.uint8),
                    )
                )
            count(f0)
            at_cut[f0] = len(calls)
            return controls

        sw.step(walker, f0, taken)
    reference = reference_workload()
    window = [cuts[f0] for f0 in blocks if f0 >= FIRST_TARGET]
    assert rebuilt == [] and set(own_rows) == {16}
    # from the second block on the cut integrates the own 16 rows and the three extensions
    assert [calls[at_cut[f0]] for f0 in blocks[1:]] == [[16, 36, 40, 47]] * (len(blocks) - 1)
    assert max(cut.count for cut in window) < reference.count / 2
    assert max(cut.elements for cut in window) < reference.elements / 2


def test_the_labels_of_latents_1_24_come_from_the_rows_of_the_cut(room, monkeypatch):
    """The first block is prepared with no turn held and turns 90 degrees to the right at its
    cut: the standing player 20 degrees to the left leaves the view during the block, and the
    block's labels say so."""
    relay, live, idle = beside_idle_player(room, monkeypatch)
    live.closed.visibility.depth_fn = flat_depth(5000.0)
    given = spy(live)
    step_pair(relay, live, idle, 1, 0, sw.live_controls(0, yaw=-90.0))
    labels = given[0]["player_visible"][0, 1:5, OTHER].tolist()
    assert labels[0] == 1.0 and labels[-1] == 0.0
    assert live.closed.visibility.drawn(1).tolist() == [
        bool(v) for v in given[0]["player_visible"][0, 1].tolist()
    ]


# ----------------------------------------------------------------------------- the paper's paths
@pytest.mark.parametrize("source", ["gt", "predicted", "live"])
def test_the_client_runs_the_rollout_of_its_source(room, monkeypatch, source):
    cfg = config(room, source=source)
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    client = Client(cfg, ServingOptions(decoder="none", lockstep=False), models=models(room, cfg))
    client.start(load_round_index_row(cfg.paths.round_index, 0))
    assert type(client._running.rollout) is (LiveRollout if source == "live" else Rollout)
    if source == "live":
        # the tiny state model reads the own position off the generated latents
        assert client.models.state_model is not None
        frames = [f for k in range(2) for f in client.step(sw.live_controls(k, "forward"))]
        assert [frame.index for frame in frames] == list(range(9))  # decoder 'none': latents
        rows = client.field_rows(8)
        assert rows.status == "guarded"
        # a client alone: the nine recorded players of the round are nowhere
        assert rows.source.tolist() == [FieldRows.OWN] + [FieldRows.HIDDEN] * 9
        with pytest.raises(ValueError, match="from its player's live controls"):
            list(client.step(None))
    client.stop()


def test_live_controls_are_refused_where_they_cannot_run(room, monkeypatch, tmp_path):
    cfg = config(room).with_overrides({"paths.world_state_dir": str(tmp_path / "world_state")})
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    with pytest.raises(ValueError, match="the offline client runs the recorded ones: gt or"):
        run_client(cfg)
    directory = client_world_state(cfg, sw.media_id(0))
    with pytest.raises(ValueError, match=r"messages \(a MessageWorldState\), not on a Directory"):
        sw.make_rollout(cfg, directory, models=models(room, cfg))
    messages = MessageWorldState(client=sw.media_id(0), poll_s=0.01)
    live = sw.make_rollout(cfg, messages, models=models(room, cfg))
    recorded = BlockControls(np.zeros((16, 11)), np.zeros((16, 2)), np.zeros(16))
    with pytest.raises(ValueError, match=r"live controls \(LiveBlockControls\); got BlockControls"):
        live.denoise(1, lambda: recorded)
    # a map the state model does not know
    for index in ("media_index", "round_index"):
        elsewhere = tmp_path / f"{index}.jsonl"
        text = Path(getattr(cfg.paths, index)).read_text()
        elsewhere.write_text(text.replace('"map_name": "de_nuke"', '"map_name": "de_cache"'))
        cfg = cfg.with_overrides({f"paths.{index}": str(elsewhere)})
    with pytest.raises(
        ValueError, match="knows the maps de_ancient, de_dust2, de_mirage, de_nuke;"
    ):
        sw.make_rollout(cfg, messages, models=models(room, cfg))


# ================================================================ a session that has no end
#: The players dead from the first row of the long tests' rounds, and the one slot with no
#: recording at all.
DEAD, ABSENT = (3, 4), 9
#: Where the second player's seat is in the standing round: ahead of the client, facing it.
PEER_AT = (320.0, 0.0, 180.0)
#: Where it is in the world the client cannot see it in: behind the client.
PEER_BEHIND = (-320.0, 0.0, 0.0)
#: The blocks a long test plays: the round of 29 latent frames slides from the eighth on, the round
#: of 61 from the sixteenth.
LONG_BLOCKS = 20


def script(k: int, *, clock: float = 0.0) -> LiveBlockControls:
    """The controls of session block ``k``: walk out for six blocks and back for six (the way back
    sees the first views again, so that old entries are retrieved), turn a little, strafe now and
    then, change weapon, jump, turn about twice."""
    held = ["forward"] if (k // 6) % 2 == 0 else ["back"]
    if (k // 5) % 3 == 1:
        held.append("move_left")
    if (k // 7) % 3 == 2:
        held.append("jump")
    weapon = ("ak47", "knife", "usp_silencer", "awp")[(k // 4) % 4]
    yaw, pitch = 12.0 * np.sin(k / 3.0), 3.0 * np.sin(k / 7.0)
    if k in (9, 16):  # a view nothing in the bank has seen
        yaw += 150.0
    controls = sw.live_controls(k, *held, yaw=float(yaw), pitch=float(pitch), weapon=weapon)
    return replace(controls, times=controls.times + clock)


def stay_script(k: int, *, client: bool) -> LiveBlockControls:
    """The controls of block ``k`` of two players that stay within a few hundred u of each other:
    the client mostly stands and now and then steps, jumps, turns and changes weapon, the other
    strafes, so that each stays in the other's view."""
    if client:
        held = ["forward"] if k % 9 == 0 else ["back"] if k % 9 == 4 else []
        held += ["jump"] if k % 11 == 5 else []
        yaw, pitch = 5.0 * np.sin(k / 3.0), 2.0 * np.sin(k / 5.0)
        weapon = ("ak47", "knife", "awp")[(k // 5) % 3]
        return sw.live_controls(k, *held, yaw=float(yaw), pitch=float(pitch), weapon=weapon)
    held = ["move_left"] if (k // 3) % 2 == 0 else ["move_right"]
    return sw.live_controls(k, *held, weapon="usp_silencer" if k % 7 < 3 else "ak47")


def peer_script(k: int, how: str) -> LiveBlockControls:
    """The controls of block ``k`` of the second player: ``strafing`` (:func:`stay_script`),
    ``standing`` (nothing held) or ``leaving`` (it stands, and strafes to its left for ever from
    its 15th block on)."""
    if how == "standing":
        return sw.live_controls(k)
    if how == "leaving":
        return sw.live_controls(k, *(["move_left"] if k >= 14 else []))
    return stay_script(k, client=False)


def wavy_depth(latents) -> np.ndarray:
    """A depth head that sees a surface 3,000 to 7,000 u in front of each latent frame, how far
    depending smoothly on the frame's latents: nothing hides a player in view, and the surface
    differs from frame to frame, so that retrieval has entries to prefer."""
    sums = torch.as_tensor(latents).float().flatten(1).sum(1).numpy().astype(np.float64)
    log_depth = np.log(5000.0 * (1.0 + 0.4 * np.sin(0.37 * sums)))
    return np.broadcast_to(log_depth[:, None, None, None], (len(sums), 4, 24, 42)).astype(
        np.float32
    )


@dataclass
class Session:
    """A live client's blocks: the rollout after them, what its sampler was given and returned at
    each block, what each block read, and the shift of the round after each."""

    rollout: LiveRollout
    calls: list[dict]
    reads: list
    shifts: list[int]


class Endless:
    """The sessions of the long tests, each run once when a test asks for it. The rounds are 31 s
    long (a round of 121 latent frames): ``moving``, the recorded players walk and turn, two seats
    are dead from the first row and one has no recording; ``standing``, nine recorded players stand
    still and the one at slot 5 is the second client, ahead of the first."""

    def __init__(self, room: dict) -> None:
        self.room, self.sessions, self.parts = room, {}, {}
        tmp = room["tmp"] / "endless"
        self.worlds = {
            "moving": sw.make_world(tmp / "moving", clients=(OWN, OTHER), seconds=31.0),
        }
        moving = self.worlds["moving"]
        for slot in DEAD:
            table = moving["tables"][sw.media_id(slot)]
            moving["tables"][sw.media_id(slot)] = replace(
                table, is_alive=np.zeros(len(table), bool)
            )
        index = Path(moving["media_index"])
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        index.write_text("".join(json.dumps(r) + "\n" for r in rows if r["player_slot"] != ABSENT))
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sw.STANDING, OTHER, PEER_AT)
            self.worlds["standing"] = sw.standing_world(
                tmp / "standing", clients=(OWN, OTHER), dies={s: 0.0 for s in DEAD}, seconds=31.0
            )
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sw.STANDING, OTHER, PEER_BEHIND)
            self.worlds["behind"] = sw.standing_world(
                tmp / "behind", clients=(OWN, OTHER), dies={s: 0.0 for s in DEAD}, seconds=31.0
            )
        for world in self.worlds.values():
            sw.empty_first_frame(world)

    def start(
        self,
        patch,
        world: str,
        latent_frames: int,
        *,
        row: int = 0,
        bound: int = 4,
        wavy: bool = False,
        relay: sw.Relay | None = None,
        overrides: dict | None = None,
    ) -> LiveRollout:
        """A rollout of the client of ``row`` of ``world`` on a round of ``latent_frames``, with
        the world state a client of the demo keeps (``relay``: joined to a room); ``wavy``: the
        depth head is :func:`wavy_depth`; ``overrides``: dotted settings of the config."""
        cfg = sw.config(
            self.worlds[world], self.room["weights"], self.room["tmp"] / "long", max_blocks=0
        )
        cfg = cfg.with_overrides(
            {
                **self.room["live"],
                "run.latent_frames": latent_frames,
                "scene_state.bound": bound,
                **(overrides or {}),
            }
        )
        sw.patch_extent(patch)
        sw.patch_ticks(patch, self.worlds[world]["tables"])
        loaded = models(self.room, cfg)
        if wavy:
            patch.setattr(loaded.depth, "log_depth", wavy_depth)
        media = load_round_index_row(cfg.paths.round_index, row).media_id
        world_state = relay.join(media) if relay else MessageWorldState(client=media, poll_s=0.01)
        world_state.retain = LIVE_STATES_PER_ENTRY * bound
        made = sw.make_rollout(cfg, world_state, models=loaded, row=row)
        sw.own_position_oracle(made)
        return made

    def peer(self, world: str, blocks: int, how: str = "strafing") -> list[list[bytes]]:
        """What the client of slot 5 sends, per block of a session of its own (a round of 61); its
        controls are :func:`peer_script` ``how``."""
        if (world, blocks, how) not in self.parts:
            sent: list[bytes] = []
            with pytest.MonkeyPatch.context() as patch:
                peer = self.start(patch, world, 61, row=1, wavy=True)
                peer.world_state.outbox = lambda part: sent.append(part.encode())
                per_block = []
                for k in range(blocks):
                    controls = peer_script(k, how)
                    sw.step(peer, 1 + BLOCK * k, lambda controls=controls: controls)
                    per_block.append(sent[:])
                    sent.clear()
            self.parts[(world, blocks, how)] = per_block
        return self.parts[(world, blocks, how)]

    def session(
        self, world: str, latent_frames: int, *, peer: bool = False, whole_round: bool = False
    ) -> Session:
        """``LONG_BLOCKS`` blocks of the client of slot 0 on a round of ``latent_frames`` latent
        frames; ``peer``: the second client's parts of its own session reach it from the eleventh
        block on; ``whole_round``: the continuous columns of the whole round are recomputed at every
        block, as the paper's rollout does."""
        key = (world, latent_frames, peer, whole_round)
        if key in self.sessions:
            return self.sessions[key]
        parts = self.peer(world, LONG_BLOCKS) if peer else None
        with pytest.MonkeyPatch.context() as patch:
            rollout = self.start(patch, world, latent_frames, wavy=peer)
            if whole_round:
                rollout._refresh_continuous_rows = types.MethodType(
                    Rollout._refresh_continuous_rows, rollout
                )
            calls, reads, shifts = spy_calls(rollout), [], []
            for k in range(LONG_BLOCKS):
                if parts and k >= 10:
                    for data in parts[k]:
                        rollout.world_state.receive(MessagePart.decode(data))
                controls = stay_script(k, client=True) if peer else script(k)
                f0 = 1 + BLOCK * k
                reads.append(rollout.commit(f0, rollout.denoise(f0, lambda: controls)))
                rollout.prepare_next(f0)
                shifts.append(rollout.shift)
        self.sessions[key] = Session(rollout, calls, reads, shifts)
        return self.sessions[key]


@pytest.fixture(scope="module")
def endless(room):
    return Endless(room)


def spy_calls(rollout) -> list[dict]:
    """What the sampler of ``rollout`` is given and returns at every block: its noisy input, its x0
    and the conditions of its first denoising step."""
    calls = []
    denoise = rollout.sampler.denoise

    def call(noisy, conditions, **kwargs):
        x0 = denoise(noisy, conditions, **kwargs)
        kept = {k: v.clone() if torch.is_tensor(v) else v for k, v in conditions.items()}
        calls.append(dict(noisy=noisy.clone(), x0=x0.clone(), conditions=kept))
        return x0

    rollout.sampler.denoise = call
    return calls


def differences(a, b, *, tolerance: float = 0.0, path: str = "") -> tuple[list[str], float]:
    """Where two nested dicts of tensors differ: floats by more than ``tolerance``, anything else at
    all; and the largest difference of two floats."""
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: the keys {sorted(set(a) ^ set(b))}"], 0.0
        found, worst = [], 0.0
        for key in sorted(a):
            more, most = differences(a[key], b[key], tolerance=tolerance, path=f"{path}{key}.")
            found, worst = found + more, max(worst, most)
        return found, worst
    if not torch.is_tensor(a):
        return ([] if a == b else [f"{path[:-1]}: {a} != {b}"]), 0.0
    if a.shape != b.shape:
        return [f"{path[:-1]}: {tuple(a.shape)} against {tuple(b.shape)}"], 0.0
    if not a.is_floating_point():
        return ([] if torch.equal(a, b) else [path[:-1]]), 0.0
    gap = float((a - b).abs().max()) if a.numel() else 0.0
    return ([] if gap <= tolerance else [f"{path[:-1]}: {gap:.3g}"]), gap


def compare(short: Session, long: Session, *, tolerance: float = 0.0) -> tuple[list[str], float]:
    """Every block of two sessions of the same client against each other: the first six blocks'
    noisy input and x0 (their conditions span rounds of other lengths), and every window block's
    noisy input, x0 and conditions."""
    found, worst = [], 0.0
    for k, (a, b) in enumerate(zip(short.calls, long.calls)):
        keys = ("noisy", "x0") if k < 6 else ("noisy", "x0", "conditions")
        more, most = differences(
            {v: a[v] for v in keys}, {v: b[v] for v in keys}, tolerance=tolerance
        )
        found += [f"block {k + 1}: {m}" for m in more]
        worst = max(worst, most)
    return found, worst


def entry_ages(session: Session) -> list[int]:
    """How many latent frames older than its round's first block the entry each window block
    retrieved is: positive for an entry that left the round."""
    return [
        4 * max(k - 6, 0) - read.entry[1] // 8
        for k, read in enumerate(session.reads)
        if read is not None and read.entry is not None
    ]


def test_a_round_that_slides_generates_what_a_longer_round_generates(endless):
    """The same client on a round of 29 latent frames, which slides from its eighth block, and on a
    round of 61, which slides from its sixteenth: the same walls, jumps, weapons and entries
    withdrawn (the bound is 4), the same seats dead from the first row and absent, a recorded
    world that moves; every noisy input, x0 and condition tensor of every block is the same bits,
    and entries older than the round are retrieved."""
    short, long = endless.session("moving", 29), endless.session("moving", 61)
    assert short.shifts[5:8] == [0, 4, 8] and short.shifts[-1] == 56 and long.shifts[-1] == 24
    assert compare(short, long)[0] == []
    assert [r.window_frames for r in short.reads[6:]] == [r.window_frames for r in long.reads[6:]]
    assert {r.window_frames for r in short.reads[6:]} == {17, 21}
    assert [r.entry for r in short.reads[6:]] == [r.entry for r in long.reads[6:]]
    assert max(entry_ages(short)) > 0


def test_a_round_that_slides_beside_another_live_player_generates_what_a_longer_round_does(endless):
    """The same with the second client in view of the first (a depth head that lets it be seen),
    whose parts, from its own session on a round of 61, reach it from its eleventh block on: the
    rows the peer writes, the visibility geometry and the entries of the peer's blocks, retrieved
    from their payload, are the same bits."""
    short = endless.session("standing", 29, peer=True)
    long = endless.session("standing", 61, peer=True)
    assert compare(short, long)[0] == []
    assert short.rollout.closed.visibility.live[:, OTHER].any()
    seen = [int(c["conditions"]["player_visible"][..., 1:, OTHER].sum()) for c in short.calls[10:]]
    assert max(seen) > 0
    assert {media for media, _ in short.rollout.known.latents} == {
        sw.media_id(m) for m in (OWN, OTHER)
    }


@pytest.mark.parametrize(
    ("world", "peer"), [("moving", False), ("standing", True)], ids=["alone", "beside"]
)
def test_a_slid_round_is_the_whole_round_of_the_paper_to_the_rounding_of_the_yaw(
    endless, world, peer
):
    """The same client on a round of 121 latent frames, which does not slide in 20 blocks and
    recomputes the continuous columns of the whole round at every block as the paper's rollout does,
    and on a round of 29, which slides 13 times: every noisy input, x0 and condition tensor within
    3e-3, and every integer, flag and label the same; the largest float difference is the yaw
    column's (measured 1.5e-5 to 6e-5)."""
    release = endless.session(world, 121, peer=peer, whole_round=True)
    slid = endless.session(world, 29, peer=peer)
    assert release.shifts[-1] == 0 and slid.shifts[-1] == 56
    found, gap = compare(slid, release, tolerance=3e-3)
    assert found == [] and gap < 1e-4


def test_every_other_slot_of_a_session_stands_at_its_first_row(endless, room, monkeypatch):
    """On a world whose recorded players move, after the rollout opens and after 20 blocks on a
    round that slid 13 times: no slot but the client's has a row, a control or a weapon past its
    first video frame that differs from its first (no other player is in the world, and the rows
    the slide vacates repeat the last)."""
    slid = endless.session("moving", 29).rollout
    opened = endless.start(monkeypatch, "moving", 29)
    assert slid.shift > 0
    recorded = opened.recorded_states[0]
    assert (recorded[:, 1:, :5] != recorded[:, :1, :5]).any()  # the recording itself moves
    for rollout in (opened, slid):
        batch = rollout.round_batch
        for slot in [s for s in range(10) if s != OWN]:
            states, weapon = batch["player_states"][0, slot], batch["player_weapon_ids"][0, slot]
            assert (states[1:, :5] == states[:1, :5]).all(), slot
            assert (batch["player_control_substeps"][0, slot, 1:] == 0.0).all(), slot
            assert not batch["player_control_substep_valid"][0, slot, 1:].any(), slot
            assert (weapon[1:] == weapon[0]).all(), slot
            for facing in rollout.closed._batch_facing:
                assert (facing[slot] == facing[slot, :1]).all(), slot


# --------------------------------------------------------------------- what a session keeps
def short_world(room: dict) -> dict:
    """The round of the bounded session's test: 8.5 s of recording, which covers the 113 video
    frames (7 s) of a round of 29 latent frames from the start and not a frame more."""
    world = sw.make_world(room["tmp"] / "short", clients=(OWN, OTHER), seconds=8.5)
    sw.empty_first_frame(world)
    return world


def sizes(rollout: LiveRollout) -> dict[str, int]:
    """How much a live rollout holds of everything that could grow with the blocks played."""
    media, closed = rollout.media_id, rollout.closed
    return dict(
        known=sum(block is not None for block in rollout.known.blocks),
        latents=len(rollout.known.latents),
        parts=len(rollout.known.parts),
        published=len(rollout.scene._published),
        bank=len(rollout.scene.bank.resident()),
        seen=len(rollout.known._seen),
        pending=len(rollout.known._pending),
        scenes=len(rollout.world_state.published_blocks(media)),
        lives=len(rollout.world_state.recent_live(media, 10_000)),
        steps=len(rollout.world_state.read_steps(media, upto=10**9)),
        positions=len(rollout.world_state._positions),
        ticks=len(closed.reader._live_active),
        windows=len(closed.reader._windows),
        frames=len(closed.reader._frames),
        track=len(closed.cameras.track),
        drawn=len(closed.visibility._drawn),
        rows=len(rollout._rows),
        pitches=len(rollout._pitch),
        uid_at=len(rollout.scene._uid_at),
        empty=len(rollout.scene._empty),
        followed=sum(len(steps) for steps in rollout.scene._followed.values()),
    )


def test_a_session_keeps_a_round_that_slides_reads_no_recording_and_starts_as_before(
    room, monkeypatch
):
    """40 blocks (160 latent frames, 4 state-model windows) on a round of 29 latent frames with
    the bound at 3, from a recording that ends right after the round's first 7 s and a frame reader
    that raises: nothing a block reads comes from the recording after the first frame, and after
    every block the round's arrays are the shapes they were and everything else that could grow
    with the blocks played holds a number of things that does not depend on how many have been (the
    list of known blocks, a pointer a block, is the one residual). The first six blocks run on the
    round before the slide: the round's conditions have its 29 latent frames, the six entries are
    published together with the sixth block, the cache holds 25 latent frames after it; every later
    block is denoised at the round's latent frame 25, in a window of 21 latent frames or 17."""
    world = short_world(room)
    cfg = sw.config(world, room["weights"], room["tmp"] / "short-out", max_blocks=0)
    cfg = cfg.with_overrides({**room["live"], "run.latent_frames": 29, "scene_state.bound": 3})
    sw.patch_extent(monkeypatch)
    sw.patch_ticks(monkeypatch, world["tables"])

    def refuse(*args, **kwargs):
        raise AssertionError("a live session read the recording")

    monkeypatch.setattr(MemoryFrames, "__call__", refuse)
    media = load_round_index_row(cfg.paths.round_index, 0).media_id
    world_state = MessageWorldState(client=media, poll_s=0.01, retain=LIVE_STATES_PER_ENTRY * 3)
    sent = []
    world_state.outbox = sent.append
    rollout = sw.make_rollout(cfg, world_state, models=models(room, cfg))
    calls, shapes, held = spy_calls(rollout), None, []
    for k in range(40):
        f0 = 1 + BLOCK * k
        rollout.commit(f0, rollout.denoise(f0, lambda k=k: script(k)))
        if k == 5:  # the sixth block is committed
            assert len([p for p in sent if p.kind == "scene"]) == 6
            assert rollout.sampler.cache.end == 25 * 252  # 25 latent frames of 12 x 21 tokens
        rollout.prepare_next(f0)
        now = {
            key: tuple(value.shape)
            for key, value in rollout.round_batch.items()
            if torch.is_tensor(value)
        }
        now["store"] = tuple(rollout.store.shape)
        shapes = shapes or now
        assert now == shapes, k
        held.append(sizes(rollout))
        if k < 6:  # the first six blocks, on the round before it slides
            assert rollout.shift == 0
            conditions = rollout.round_conditions["player_state_table"]
            assert conditions.shape[1] == 29
        if k == 4:
            assert not [p for p in sent if p.kind == "scene"]
    assert rollout.shift == 4 * 34
    ceilings = {key: max(h[key] for h in held[20:]) for key in held[0]}
    assert ceilings == dict(
        known=4,  # the bound's 3 entries and the newest block, which is not published yet
        latents=4,
        parts=4,
        published=0,
        bank=3,
        seen=0,
        pending=0,
        scenes=6,  # twice the bound
        lives=6,
        steps=6,
        positions=0,
        ticks=658,  # a state-model window of ticks: 10.25 s at 64 a second, and two
        windows=1,
        frames=41,  # the window's latent frames
        track=25,
        drawn=24,
        rows=32,  # the page's 8 blocks
        pitches=128,
        uid_at=0,  # nobody else is in the world: an own entry leaves through the bound
        empty=0,
        followed=0,
    )
    window = {call["conditions"]["player_state_table"].shape[1] for call in calls[6:]}
    assert window == {17, 21}
    live = [p for p in sent if p.kind == "live"]
    for call, part in zip(calls[6:], live[6:]):
        table = call["conditions"]["player_state_table"][0, -4:, OWN, :3]
        assert torch.equal(table, torch.from_numpy(part.rows[3::4, :3]))  # the block's own rows


def play_together(clients: list, first: int, blocks: int, relay: sw.Relay, reads: dict) -> None:
    """Room blocks ``first .. first + blocks - 1``: each client in turn plays its own block (the
    room's block less the one it joined at, ``start``, in the room's time), and what it sent is
    passed on to the others; ``reads`` collects what each block read."""
    for k in range(first, first + blocks):
        for client, start in clients:
            j, f0 = k - start, 1 + BLOCK * (k - start)
            controls = script(j + 3 * (start == 0 and client is not clients[0][0]), clock=start)
            read = client.commit(f0, client.denoise(f0, lambda controls=controls: controls))
            client.prepare_next(f0)
            reads[client.media_id].append(read)
            relay.deliver(client.media_id)


def test_two_clients_without_an_end_retrieve_each_others_entries_and_keep_two_bounds_of_parts(
    endless, monkeypatch
):
    """Two clients start together on the standing round (the second ahead of the first) and play 20
    blocks on rounds that slide, cross-wired through a coordinator that keeps for a joiner the
    newest ``B`` entries and ``2 B`` live parts of each: each draws the other from its first part
    on and retrieves entries of the other's blocks, keeping their latents and ``live`` parts as
    the entries' payload; what each world state keeps of each client stays at ``2 B`` blocks."""
    relay = sw.Relay(bound=4)
    first = endless.start(monkeypatch, "standing", 29, relay=relay)
    second = endless.start(monkeypatch, "standing", 29, row=1, relay=relay)
    reads = {first.media_id: [], second.media_id: []}
    play_together([(first, 0), (second, 0)], 0, LONG_BLOCKS, relay, reads)
    for client, other, slot in ((first, second, OTHER), (second, first, OWN)):
        owners = {r.entry[0] for r in reads[client.media_id][6:] if r.entry is not None}
        assert owners == {first.media_id, second.media_id}, client.media_id
        assert slot in client.closed.peers and client.closed.visibility.live[5:97, slot].all()
        assert other.media_id in {media for media, _ in client.known.latents}
        assert set(client.known.parts) == set(client.known.latents)
        for media in (first.media_id, second.media_id):
            assert len(client.world_state.published_blocks(media)) <= 2 * 4
            assert len(client.world_state.recent_live(media, 10_000)) <= 2 * 4
        assert len(client.known._seen) <= 2 * 4 and len(client.known._pending) <= 2 * 4
        assert client.known.vanished == 0 and client.world_state._positions == {}
        scene = client.scene
        assert (
            len(scene._uid_at) + len(scene._empty) <= 4 + 1
        )  # the other's entries held, and the one in flight
        assert (
            len(scene._followed[other.media_id]) <= 2 * 4
        )  # the step records its world state keeps


def together(endless, monkeypatch, latent_frames: int) -> dict[str, Session]:
    """Two clients start together on the standing round and play ``LONG_BLOCKS`` blocks on rounds
    of ``latent_frames`` latent frames, cross-wired as :func:`play_together` has them."""
    relay = sw.Relay(bound=4)
    pair = [
        endless.start(monkeypatch, "standing", latent_frames, row=row, relay=relay, wavy=True)
        for row in (0, 1)
    ]
    calls = {client.media_id: spy_calls(client) for client in pair}
    reads = {client.media_id: [] for client in pair}
    play_together([(client, 0) for client in pair], 0, LONG_BLOCKS, relay, reads)
    return {
        media: Session(client, calls[media], reads[media], [])
        for media, client in ((c.media_id, c) for c in pair)
    }


def test_two_clients_on_rounds_that_slide_generate_what_two_on_longer_rounds_do(
    endless, monkeypatch
):
    """The retrieval of another client's entry after the round slid: two clients on rounds of 29
    latent frames and two on rounds of 61, cross-wired, each retrieving entries of the other's
    blocks (their latents and ``live`` parts kept as the entries' payload, which a round of 61 holds
    in its store until it slides too): every noisy input, x0 and condition tensor of both clients is
    the same bits."""
    short, long = together(endless, monkeypatch, 29), together(endless, monkeypatch, 61)
    for media in short:
        assert compare(short[media], long[media])[0] == [], media
        other = (set(short) - {media}).pop()
        slid = [k for k, r in enumerate(short[media].reads) if r and r.entry and k >= 8]
        assert short[media].rollout.shift > long[media].rollout.shift > 0
        assert sum(short[media].reads[k].entry[0] == other for k in slid) >= 3, media


def test_a_late_joiner_admits_no_block_of_a_client_that_started_before_it(endless, monkeypatch):
    """The causal cut stays in latent time: a client that joins when the other is at its 17th block
    starts at its own first block, its latent clock its own play time, so the other's blocks the
    coordinator keeps for it (the newest four) are in its future, and by the time its clock passes
    them the world state has retained them out (it keeps the newest 2 B blocks of each client: a
    lead of 2 B - 1 blocks or more is the cliff), so it counts them as expired and uses none; the
    other admits the joiner's blocks as they come, and each draws the other from its first part."""
    relay = sw.Relay(bound=4)
    first = endless.start(monkeypatch, "standing", 29, relay=relay)
    reads = {first.media_id: []}
    play_together([(first, 0)], 0, 16, relay, reads)
    second = endless.start(monkeypatch, "standing", 29, row=1, relay=relay)
    reads[second.media_id] = []
    play_together([(first, 0), (second, 16)], 16, 10, relay, reads)
    assert {b.media_id for b in second.known.blocks if b is not None} == {second.media_id}
    assert second.known.expired > 0
    assert {b.media_id for b in first.known.blocks if b is not None} == {
        first.media_id,
        second.media_id,
    }
    assert first.closed.visibility.live[video_frames_of(13, 1)[0] :, OTHER].any()
    assert second.closed.visibility.live[1:, OWN].all()
    entries = [r.entry for r in reads[second.media_id] if r is not None and r.entry is not None]
    assert all(owner == second.media_id for owner, _ in entries)
    assert all(
        len(second.world_state.published_blocks(media)) <= 2 * 4
        for media in (first.media_id, second.media_id)
    )


def test_a_late_joiner_a_few_blocks_behind_admits_the_entries_of_the_client_before_it(
    endless, monkeypatch
):
    """The other side of the cliff: a client that joins when the other is at its 4th block (the
    bound is 4: a lead under 2 B - 1 blocks) has the other's blocks in its past within a few
    blocks of its own, the world state still holds them, so it admits them and retrieves entries of
    them, as two clients that started together do."""
    relay = sw.Relay(bound=4)
    first = endless.start(monkeypatch, "standing", 29, relay=relay)
    reads = {first.media_id: []}
    play_together([(first, 0)], 0, 3, relay, reads)
    second = endless.start(monkeypatch, "standing", 29, row=1, relay=relay)
    reads[second.media_id] = []
    play_together([(first, 0), (second, 3)], 3, 16, relay, reads)
    assert {b.media_id for b in second.known.blocks if b is not None} == {
        first.media_id,
        second.media_id,
    }
    assert second.known.expired == 0
    entries = [r.entry for r in reads[second.media_id] if r is not None and r.entry is not None]
    assert first.media_id in {owner for owner, _ in entries}


# ------------------------------------------------------------------- every array that slides
#: What a live rollout's slide moves, by the owner that holds it (a dotted path from the rollout):
#: the arrays with a frame axis, named as :func:`frame_arrays` names them, or their prefix.
SLID = {
    "": ("store", "continuous_rows")
    + tuple(f"round_batch.{key}" for key in (*WINDOW_ENTRIES, "window_c2w", "window_tans")),
    "closed": LiveLoop.SLID,
    "closed.motion": LiveMotion.SLID,
    "closed.cameras.track": PositionTrack.SLID,
    "closed.table": PredictedStateTable.SLID,
    "closed.visibility": LiveVisibility.SLID,
}
#: What it leaves, and why.
NOT_SLID = {
    "": {
        "output": "nothing reads it after the first six blocks: the frames are decoded from x0",
        "client_c2w": "a view of round_batch.window_c2w, which slides",
        "recorded_states": "the recording, which no block of a live session reads",
        "round_conditions": "read by the first six blocks only, before the round slides",
    },
    "closed": {"batch": "the rollout's round batch, which slides with it"},
    "closed.cameras": {"_cache": "rebuilt when the motion's version changes"},
    "closed.table": {"states": "the round batch's table"},
    "closed.visibility": {"states": "the round batch's table"},
}


def frame_arrays(owner, path: str) -> dict[str, torch.Tensor | np.ndarray]:
    """The arrays an object holds, directly, in a tuple or list (``name[i]``) or as the values of a
    dict (``name.key``), by name."""
    found = {}

    def walk(value, name: str) -> None:
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            found[name] = value
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                walk(item, f"{name}[{i}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{name}.{key}")

    for attribute, value in vars(owner).items():
        walk(value, attribute)
    return found


def named_in(name: str, names) -> bool:
    return any(name == n or name.startswith((f"{n}[", f"{n}.")) for n in names)


def test_every_array_of_a_live_rollout_with_a_frame_axis_slides_or_says_why_not(room, monkeypatch):
    """A rollout with a round of 37 latent frames (145 video frames) and 17 positions published:
    every array it, its live loop, motion, cameras, position track, table, visibility geometry and
    state-model reader hold that has an axis of one of these lengths is named by its owner as slid
    (the owner's ``SLID``, or this test's table of the rollout) or as not, with the reason; so an
    array that a later change adds cannot be left out of the slide unseen. One slide then moves
    each array that is named slid: the frames after the first two (video frames after the first
    five) up by a block, the vacated tail repeating the last, the position track dropping the
    block (the entry noise is a stream that holds no frame); and the labels drawn follow their
    latent frames."""
    live = rollout(room, monkeypatch)
    for k, f0 in enumerate(BLOCKS[:5]):
        sw.step(live, f0, lambda k=k: sw.live_controls(k, "forward", yaw=7.0))
    closed = live.closed
    owners = {
        "": live,
        "closed": closed,
        "closed.motion": closed.motion,
        "closed.cameras": closed.cameras,
        "closed.cameras.track": closed.cameras.track,
        "closed.table": closed.table,
        "closed.visibility": closed.visibility,
        "closed.reader": closed.reader,
    }
    frames, video = live.latent_frames, 4 * (live.latent_frames - 1) + 1
    lengths = {frames, frames - 1, video, len(closed.cameras.track)}
    assert lengths == {37, 36, 145, 17}
    slid = {}
    for path, owner in owners.items():
        for name, array in frame_arrays(owner, path).items():
            if not lengths & set(array.shape):
                continue
            where = f"{path or 'rollout'}.{name}"
            if named_in(name, SLID.get(path, ())):
                slid[where] = (array, [i for i, n in enumerate(array.shape) if n in lengths][0])
            else:
                assert named_in(name, NOT_SLID.get(path, {})), f"{where} has a frame axis"
    assert len(slid) == 31  # 2 of the rollout, 16 of its round batch, 13 of the loop's
    # a marker in every frame of every array that slides, a unit apart (a flag: one in three)
    before = {}
    for where, (array, axis) in slid.items():
        index = torch.arange(array.shape[axis])
        mark = index.view([-1 if i == axis else 1 for i in range(array.ndim)])
        mark = mark.numpy() if isinstance(array, np.ndarray) else mark
        flag = array.dtype == (torch.bool if torch.is_tensor(array) else np.bool_)
        array[...] = (mark % 3 == 0) if flag else mark
        before[where] = array.clone() if torch.is_tensor(array) else array.copy()
    closed.visibility._drawn = {f: np.full(10, f) for f in (0, 1, 5, 6, 10)}
    live._slide()
    for where, (array, axis) in slid.items():
        if where.endswith(".xyz"):
            continue
        latent = array.shape[axis] == frames
        rows, keep = (BLOCK, 2) if latent else (4 * BLOCK, 5)
        expected = before[where]
        rolled = expected.clone() if torch.is_tensor(expected) else expected.copy()
        W.roll(rolled, axis, rows, keep)
        assert not torch.equal(torch.as_tensor(expected), torch.as_tensor(rolled)), where
        assert torch.equal(torch.as_tensor(array), torch.as_tensor(rolled)), where
    track, was = closed.cameras.track.xyz, before["closed.cameras.track.xyz"]
    assert len(track) == 13 and np.array_equal(track[:2], was[:2])
    assert np.array_equal(track[2:], was[6:])
    assert live.noise is None and live.shift == 4
    assert sorted(closed.visibility._drawn) == [0, 1, 2, 6]
    assert closed.visibility._drawn[6][0] == 10 and closed.reader.offset == closed.offset == 4


# ------------------------------------------------------------------- the client beyond its round
def test_a_live_client_goes_on_past_its_round_and_the_page_reads_what_each_block_committed(
    room, monkeypatch
):
    """A client on a round of 29 latent frames plays 10 blocks (latent frame 40): it is never
    finished, its frames are the blocks' x0 (decoded by value: the output holds the first six
    blocks and the newest), and the rows and pitch the page reads at any frame of the newest
    blocks, beyond the round included, are what the block's commit took, which no later block
    moves."""
    cfg = config(room).with_overrides({"run.latent_frames": 29, "run.max_blocks": 0})
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    sw.patch_extent(monkeypatch)
    client = Client(cfg, ServingOptions(decoder="none", lockstep=False), models=models(room, cfg))
    client.start(load_round_index_row(cfg.paths.round_index, 0))
    calls = spy_calls(client._running.rollout)
    frames = []
    for k in range(10):
        frames += list(client.step(sw.live_controls(k, "forward", yaw=5.0)))
        assert not client.finished
    assert [record.f0 for record in client.records] == list(
        range(9, 38, BLOCK)
    )  # the page's 8 newest blocks
    assert [frame.index for frame in frames] == list(range(41))
    assert [frame.f0 for frame in frames[-4:]] == [37] * 4
    for frame in frames[1:]:
        block, i = calls[(frame.f0 - 1) // BLOCK]["x0"], (frame.index - 1) % BLOCK
        assert np.array_equal(frame.image, block[0, i].numpy()), frame.index
    rows = client.field_rows(40)
    assert rows.status in ("guarded", "reacquiring", "degraded")
    assert rows.source[OWN] == FieldRows.OWN
    assert client.field_rows(33).states[OWN, 0] < rows.states[OWN, 0]  # it walks on
    assert 0.0 <= abs(client.camera_pitch(160)) < 89.0
    with pytest.raises(ValueError, match="has not been generated"):
        client.field_rows(41)
    # the page's 8 newest blocks are kept: latent frames 9-40, video frames 33-160
    assert client.field_rows(9).source[OWN] == FieldRows.OWN
    assert abs(client.camera_pitch(33)) < 89.0
    with pytest.raises(ValueError, match=r"latent frame 8 is not kept.*latent frames 9-40"):
        client.field_rows(8)
    with pytest.raises(ValueError, match=r"video frame 32 is not kept.*video frames 33-160"):
        client.camera_pitch(32)
    client.stop()


def test_the_world_state_of_a_client_with_a_tiny_bound_keeps_the_first_six_blocks_parts(
    room, monkeypatch
):
    """The first six entries are published together after the sixth block, each with the ``live``
    part of its block, whatever the bound: a world state that keeps 2 B blocks (2 for B = 1) would
    have dropped the parts of the older ones and an entry that stayed in the bank would lack its
    controls."""
    overrides = {"run.latent_frames": 29, "run.max_blocks": 0, "scene_state.bound": 1}
    cfg = config(room).with_overrides(overrides)
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    sw.patch_extent(monkeypatch)
    client = Client(cfg, ServingOptions(decoder="none", lockstep=False), models=models(room, cfg))
    client.start(load_round_index_row(cfg.paths.round_index, 0))
    for k in range(9):
        list(client.step(sw.live_controls(k, "forward", yaw=5.0)))
    rollout = client._running.rollout
    assert client._running.messages.retain == 6
    assert rollout.known.parts and all(part is not None for part in rollout.known.parts.values())
    client.stop()


@pytest.mark.parametrize("source", ["gt", "predicted"])
def test_the_paper_clients_finish_with_their_round_and_decode_the_output(room, monkeypatch, source):
    """A client of the paper's sources is finished when its round is: the blocks it was asked for
    are its round's, and each frame is its output's."""
    cfg = config(room, source=source)
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    client = Client(cfg, ServingOptions(decoder="none", lockstep=False), models=models(room, cfg))
    client.start(load_round_index_row(cfg.paths.round_index, 0))
    assert type(client._running.rollout) is Rollout and not client._running.rollout.open_ended
    frames = []
    while not client.finished:
        frames += list(client.step())
    assert len(frames) == client.latents.shape[1] == 37 and len(client.records) == 9
    for frame in frames:
        assert np.array_equal(frame.image, client.latents[0, frame.index].numpy())
    assert list(client.step()) == []
    client.stop()


# ================================================================== an idle client
@dataclass
class Stood:
    """``blocks`` blocks of a client on the round of 29 latent frames: the rollout after them, each
    block's x0 and the store after its commit (before the round slides), what each read, and what
    the sampler was given at each."""

    rollout: LiveRollout
    x0s: list[torch.Tensor]
    stores: list[torch.Tensor]
    reads: list
    calls: list[dict]

    def seen(self) -> list[bool]:
        """Per window block: whether the second player is labelled visible on a video frame of its
        recent context or target frames (the conditions of its first denoising step)."""
        return [
            bool(call["conditions"]["player_visible"][0, -4 * BLOCK :, OTHER].any())
            for call in self.calls[FIRST_WINDOW:]
        ]


def play_idle(
    endless: Endless,
    overrides: dict,
    *,
    blocks: int = 14,
    controls: Callable[[int], LiveBlockControls] = lambda k: sw.live_controls(k),
    patch: Callable[[LiveRollout], None] = lambda rollout: None,
    world: str = "standing",
    peer: Callable[[int], list[bytes]] | None = None,
    latent_frames: int = 29,
) -> Stood:
    """The client of ``world`` on a round of ``latent_frames`` under ``controls`` (default: nothing
    held, no turn), ``overrides`` the dotted settings of the run; ``patch`` is given the rollout
    before it starts. ``peer`` gives the parts of the second client that reach it before its block
    ``k`` (its depth head then lets a player in view be seen)."""
    x0s, stores, reads = [], [], []
    with pytest.MonkeyPatch.context() as monkeypatch:
        rollout = endless.start(
            monkeypatch, world, latent_frames, overrides=overrides, wavy=peer is not None
        )
        patch(rollout)
        calls = spy_calls(rollout)
        for k in range(blocks):
            for data in peer(k) if peer else ():
                rollout.world_state.receive(MessagePart.decode(data))
            f0, taken = 1 + BLOCK * k, controls(k)
            x0 = rollout.denoise(f0, lambda: taken)
            reads.append(rollout.commit(f0, x0))
            stores.append(rollout.store.clone())
            x0s.append(x0[0].clone())
            rollout.prepare_next(f0)
    return Stood(rollout, x0s, stores, reads, calls)


#: The first window block of a client (its seventh), which an idle stretch starts at, and the block
#: the store first keeps the clip at: the clip is three blocks.
FIRST_WINDOW, FIRST_KEPT = 6, 9
NEWEST = slice(FIRST_TARGET, FIRST_TARGET + BLOCK)


def test_the_release_keeps_the_blocks_of_an_idle_client_in_its_context(endless):
    stood = play_idle(endless, {})
    assert stood.rollout.idle_context is None
    for k in range(FIRST_WINDOW, 14):
        assert torch.equal(stood.stores[k][NEWEST], stood.x0s[k])


def test_freeze_cuts_the_loop_of_an_idle_client_through_its_own_output(endless):
    release = play_idle(endless, {})
    frozen = play_idle(endless, {"live.idle_context": "freeze"})
    context = frozen.rollout.idle_context
    assert (context.stretch, context.kept) == (8, 5)
    # a block is generated from the context before it: the blocks up to the first the store keeps
    # as the clip are the release's bit for bit, the blocks after it are not
    for k in range(FIRST_KEPT + 1):
        assert torch.equal(frozen.x0s[k], release.x0s[k])
    assert all(not torch.equal(frozen.x0s[k], release.x0s[k]) for k in range(FIRST_KEPT + 1, 14))
    # what the store keeps of a block is the clip, in turn; what is shown of it is what it generated
    clip = frozen.x0s[FIRST_WINDOW:FIRST_KEPT]
    for k in range(FIRST_WINDOW, FIRST_KEPT):
        assert torch.equal(frozen.stores[k][NEWEST], frozen.x0s[k])
    rollout = frozen.rollout
    for k in range(FIRST_KEPT, 14):
        kept = clip[(k - FIRST_WINDOW) % 3]
        assert torch.equal(frozen.stores[k][NEWEST], kept) and not torch.equal(kept, frozen.x0s[k])
    # so is what the memory bank holds of the blocks after the clip it still has
    held = {
        k: v
        for (_, f0), v in rollout.known.latents.items()
        if (k := (f0 - 1) // BLOCK) >= FIRST_KEPT
    }
    assert held and all(torch.equal(v, clip[(k - FIRST_WINDOW) % 3]) for k, v in held.items())
    # the recent context of every block after the clip is the clip, the blocks of it in some order
    last = frozen.stores[13]
    assert sorted(float(last[f0].sum()) for f0 in (17, 21, 25)) == sorted(
        float(c[0].sum()) for c in clip
    )


def test_the_clip_does_not_depend_on_the_length_of_the_round_the_client_keeps(endless):
    """After 16 blocks a round of 45 latent frames has slid by 24 latent frames, one of 29 by 40:
    the same blocks, with the clip kept in the store wherever the newest block is."""
    overrides = {"live.idle_context": "freeze"}
    short = play_idle(endless, overrides, blocks=16)
    long = play_idle(endless, overrides, blocks=16, latent_frames=45)
    assert short.rollout.shift == 40 and long.rollout.shift == 24
    assert short.rollout.idle_context.kept == long.rollout.idle_context.kept > 0
    assert all(torch.equal(a, b) for a, b in zip(short.x0s, long.x0s))


def test_a_stretch_ends_where_the_client_turns_and_a_new_clip_is_made_after_it(endless):
    def turns_at_ten(k: int) -> LiveBlockControls:
        return sw.live_controls(k, yaw=20.0 if k == 10 else 0.0)

    stood = play_idle(endless, {"live.idle_context": "freeze"}, blocks=15, controls=turns_at_ten)
    assert torch.equal(stood.stores[10][NEWEST], stood.x0s[10])  # kept as generated
    assert stood.rollout.idle_context.stretch == 4  # the blocks 11-14: the new clip and one more
    assert torch.equal(stood.stores[14][NEWEST], stood.x0s[11])  # the new clip's first block
    assert stood.rollout.idle_context.kept == 2  # block 9 and block 14


def test_a_client_that_stops_walking_makes_its_clip_where_its_camera_has_stopped(endless):
    def walks_at_ten(k: int) -> LiveBlockControls:
        return sw.live_controls(k, *(["forward"] if k == 10 else []))

    stood = play_idle(endless, {"live.idle_context": "freeze"}, blocks=18, controls=walks_at_ten)
    # the camera is still coasting in block 11, after the key is released: the stretch is the
    # blocks of the stopped camera (12 on) and its clip is blocks 12-14 (the store has kept block 9
    # of the stretch before the walk, and blocks 15-17 of this one)
    assert (stood.rollout.idle_context.stretch, stood.rollout.idle_context.kept) == (6, 4)
    for k in range(10, 15):
        assert torch.equal(stood.stores[k][NEWEST], stood.x0s[k])
    clip = stood.x0s[12:15]
    for k in range(15, 18):
        assert torch.equal(stood.stores[k][NEWEST], clip[(k - 12) % 3])


def present_until(block: int) -> Callable[[LiveRollout], None]:
    """The second player is in the world up to the session's block ``block`` and gone after it."""

    def patch(rollout: LiveRollout) -> None:
        present, absent = np.zeros(10, bool), np.zeros(10, bool)
        present[OTHER] = True
        rollout.closed.in_world = lambda latent: (
            present if (latent + rollout.shift - 1) // BLOCK <= block else absent
        )

    return patch


def test_a_client_is_alone_in_its_world_unless_idle_with_peers_says_otherwise(endless):
    stood = play_idle(endless, {"live.idle_context": "freeze"}, patch=present_until(99))
    assert stood.rollout.idle_context.stretch == stood.rollout.idle_context.kept == 0
    for k in range(FIRST_WINDOW, 14):
        assert torch.equal(stood.stores[k][NEWEST], stood.x0s[k])


def test_a_client_alone_in_its_world_is_idle_once_a_player_who_left_is_out_of_its_window(endless):
    """The second player is in the world up to block 8: its figure is in the recent context of the
    blocks after it, so the first idle block is block 12, whose window (blocks 9-12) has nobody in
    it."""
    stood = play_idle(endless, {"live.idle_context": "freeze"}, blocks=18, patch=present_until(8))
    assert (stood.rollout.idle_context.stretch, stood.rollout.idle_context.kept) == (6, 3)
    for k in range(FIRST_WINDOW, 15):
        assert torch.equal(stood.stores[k][NEWEST], stood.x0s[k])
    clip = stood.x0s[12:15]
    for k in range(15, 18):
        assert torch.equal(stood.stores[k][NEWEST], clip[(k - 12) % 3])


#: The settings of a client that keeps the clip while other players are in the world.
FREEZE_WITH_PEERS = {"live.idle_context": "freeze", "live.idle_with_peers": True}
#: The block of the first client at which the second client's parts reach it.
JOINS = 10


def joins(parts: list[list[bytes]]) -> Callable[[int], list[bytes]]:
    """The second client's parts that reach the first before its block ``k``."""
    return lambda k: parts[k] if k >= JOINS else []


def shown(flags: list[bool], mark: str) -> str:
    """One character per block: ``mark`` where the flag is set."""
    return "".join(mark if flag else "." for flag in flags)


def replaced(stood: Stood) -> list[bool]:
    """Per window block: whether the store kept something else than what the block generated."""
    return [
        not torch.equal(stood.stores[k][NEWEST], stood.x0s[k])
        for k in range(FIRST_WINDOW, len(stood.x0s))
    ]


def test_a_player_nobody_sees_ends_a_stretch_only_when_idle_with_peers_is_off(endless):
    """The second client stands behind the first and joins at its eleventh block: in the first
    client's world, in no frame of its windows."""
    parts = endless.peer("behind", 20, "standing")
    alone = play_idle(
        endless, {"live.idle_context": "freeze"}, blocks=20, world="behind", peer=joins(parts)
    )
    peers = play_idle(endless, FREEZE_WITH_PEERS, blocks=20, world="behind", peer=joins(parts))
    for stood in (alone, peers):
        assert shown(stood.seen(), "X") == "." * 14
        assert stood.rollout.closed.in_world(FIRST_TARGET - 1)[OTHER]
    # the stretch of a client alone in its world ends with the first part of another player
    assert shown(replaced(alone), "F") == "...F" + "." * 10
    assert (alone.rollout.idle_context.stretch, alone.rollout.idle_context.kept) == (0, 1)
    # with idle_with_peers it goes on over the 14 blocks of standing, 11 of them kept as the clip's
    assert shown(replaced(peers), "F") == "...F" + "F" * 10
    assert (peers.rollout.idle_context.stretch, peers.rollout.idle_context.kept) == (14, 11)
    clip = peers.x0s[FIRST_WINDOW:FIRST_KEPT]
    for k in range(FIRST_KEPT, 20):
        assert torch.equal(peers.stores[k][NEWEST], clip[(k - FIRST_WINDOW) % 3])


def test_a_stretch_ends_where_a_player_is_drawn_and_a_new_one_begins_once_its_context_is_clear(
    endless,
):
    """The second client stands ahead of the first from the eleventh block on, in its view, and
    strafes out of it from its own fifteenth: the first client keeps what it generates of the blocks
    that draw it, and makes a new clip once nobody is drawn in the window of a block."""
    parts = endless.peer("standing", 30, "leaving")
    stood = play_idle(endless, FREEZE_WITH_PEERS, blocks=30, peer=joins(parts))
    assert shown(stood.seen(), "X") == "....XXXXXXXXXX.........."
    # the stretch of blocks 7-10 ends where the player is drawn (block 11); the first block whose
    # window shows nobody (block 21) is the first of the next, whose clip is blocks 21-23
    assert shown(replaced(stood), "F") == "...F.............FFFFFFF"
    assert (stood.rollout.idle_context.stretch, stood.rollout.idle_context.kept) == (10, 8)
    clip = stood.x0s[20:23]
    for k in range(23, 30):
        assert torch.equal(stood.stores[k][NEWEST], clip[(k - 20) % 3])


def test_a_player_in_view_keeps_the_blocks_of_an_idle_client_as_generated(endless):
    parts = endless.peer("standing", 20, "standing")
    stood = play_idle(endless, FREEZE_WITH_PEERS, blocks=20, peer=joins(parts))
    assert shown(stood.seen(), "X") == "....XXXXXXXXXX"
    assert shown(replaced(stood), "F") == "...F" + "." * 10
    assert stood.rollout.idle_context.stretch == 0


def test_an_idle_client_publishes_every_block_and_the_latents_it_keeps(endless):
    sent: list[MessagePart] = []

    def listen(rollout: LiveRollout) -> None:
        rollout.world_state.outbox = sent.append

    stood = play_idle(endless, {"live.idle_context": "freeze"}, patch=listen)
    assert [part.f0 for part in sent if part.kind == "live"] == [1 + BLOCK * k for k in range(14)]
    entries = {part.block.f0: part.latents for part in sent if part.kind == "scene"}
    assert sorted(entries) == [1 + BLOCK * k for k in range(14)]
    for k in range(FIRST_WINDOW, 14):
        published = entries[1 + BLOCK * k]
        assert np.array_equal(published, stood.stores[k][NEWEST].numpy())
        assert np.array_equal(published, stood.x0s[k].numpy()) == (k < FIRST_KEPT)


def test_a_late_joiner_is_sent_the_entries_an_idle_client_keeps(endless, monkeypatch):
    """What the coordinator keeps for a client that joins later (the newest ``B`` entries and
    ``2 B`` ``live`` parts of each client) are the clip's blocks of an idle client,
    each with the ``live`` part that keys it."""
    relay = sw.Relay(bound=4)
    standing = endless.start(
        monkeypatch, "standing", 29, relay=relay, overrides={"live.idle_context": "freeze"}
    )
    x0s = []
    for k in range(14):
        f0, controls = 1 + BLOCK * k, sw.live_controls(k)
        x0s.append(standing.denoise(f0, lambda controls=controls: controls)[0].clone())
        standing.commit(f0, x0s[-1][None])
        standing.prepare_next(f0)
        relay.deliver(standing.media_id)
    kept = [MessagePart.decode(data) for data in relay.kept]
    entries = {part.block.f0: part.latents for part in kept if part.kind == "scene"}
    assert sorted(entries) == [1 + BLOCK * k for k in range(10, 14)]
    assert {part.f0 for part in kept if part.kind == "live"} >= set(entries)
    clip = x0s[FIRST_WINDOW:FIRST_KEPT]
    for k in range(10, 14):
        assert np.array_equal(entries[1 + BLOCK * k], clip[(k - FIRST_WINDOW) % 3].numpy())


def test_a_client_reads_the_entries_an_idle_client_keeps(endless, monkeypatch):
    """The first client stands, the second strafes behind it: what the second admits of the first's
    blocks is what the first published, which is what the first keeps."""
    relay = sw.Relay(bound=4)
    standing = endless.start(
        monkeypatch, "behind", 29, relay=relay, wavy=True, overrides=FREEZE_WITH_PEERS
    )
    walking = endless.start(monkeypatch, "behind", 29, row=1, relay=relay, wavy=True)
    sent: list[MessagePart] = []
    forward = standing.world_state.outbox

    def publish(part: MessagePart) -> None:
        sent.append(part)
        forward(part)

    standing.world_state.outbox = publish
    for k in range(LONG_BLOCKS):
        for client, controls in (
            (standing, sw.live_controls(k)),
            (walking, stay_script(k, client=False)),
        ):
            f0 = 1 + BLOCK * k
            client.commit(f0, client.denoise(f0, lambda controls=controls: controls))
            client.prepare_next(f0)
            relay.deliver(client.media_id)
    assert (standing.idle_context.stretch, standing.idle_context.kept) == (14, 11)
    published = {part.block.f0: part.latents for part in sent if part.kind == "scene"}
    read = {f0: v for (media, f0), v in walking.known.latents.items() if media == standing.media_id}
    assert len(read) >= 2 and set(read) <= set(published)
    for f0, latents in read.items():
        assert np.array_equal(latents.numpy(), published[f0])
    kept = [f0 for f0 in read if (standing.media_id, f0) in standing.known.latents]
    assert kept and all(
        torch.equal(read[f0], standing.known.latents[(standing.media_id, f0)]) for f0 in kept
    )


def own_entry_retrieved(rollout: LiveRollout) -> None:
    """The scene state of the tiny round finds nothing for an idle client; here it
    retrieves its own newest entry, whose cameras are the block's own where the client stands."""
    original = rollout.scene.retrieve

    def retrieve(**kwargs):
        result = original(**kwargs)
        resident = rollout.scene.bank.resident()
        if result.entry is not None or not resident:
            return result
        return replace(result, entry=resident[-1], coverage=5, n_missing=9, n_candidates=1)

    rollout.scene.retrieve = retrieve


def test_a_long_stretch_keeps_its_context_from_the_seventh_block_to_the_hundredth(endless):
    """A client stands from its seventh block on (the stretch's first), its retrieval finding its
    own newest entry, and walks at the 106th idle block's place: the first frame, the memory frames
    and the recent context its cache is written from, and the store the next window is gathered
    from, are those of the seventh block when the hundredth is prepared (the clip is three blocks,
    so the context comes back every three), and the block that walks is generated from the same
    context and kept as generated, so that the blocks after it continue from the clip and it."""
    seventh, hundredth, walks = FIRST_WINDOW + 6, FIRST_WINDOW + 99, FIRST_WINDOW + 105
    clip_blocks, tail_blocks = range(FIRST_WINDOW, FIRST_KEPT), (seventh - 1, hundredth - 1, walks)
    contexts: dict[int, torch.Tensor] = {}  # what each of the blocks watched is prefilled from
    tails: dict[int, torch.Tensor] = {}  # the store's last three blocks after each of them
    x0s: dict[int, torch.Tensor] = {}
    windows: dict[int, int] = {}
    prefills = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        rollout = endless.start(
            monkeypatch, "standing", 29, overrides={"live.idle_context": "freeze"}
        )
        own_entry_retrieved(rollout)
        prefill = rollout.sampler.prefill_context

        def watch(latents, *args, **kwargs):
            block = FIRST_WINDOW + len(prefills)  # the n-th prefill is for the n-th window block
            prefills.append(block)
            if block in (seventh, seventh + 1, hundredth, walks, walks + 1):
                contexts[block] = latents.clone()
            return prefill(latents, *args, **kwargs)

        rollout.sampler.prefill_context = watch
        for k in range(walks + 2):
            f0, taken = 1 + BLOCK * k, sw.live_controls(k, *(["forward"] if k >= walks else []))
            x0 = rollout.denoise(f0, lambda: taken)
            read = rollout.commit(f0, x0)
            if k in (*clip_blocks, *tail_blocks):
                tails[k], x0s[k] = rollout.store[17:29].clone(), x0[0].clone()
            windows[k] = None if read is None else read.window_frames
            rollout.prepare_next(f0)
    clip = [x0s[k] for k in clip_blocks]
    assert windows[seventh] == windows[hundredth] == windows[walks] == 21
    # the store: the clip, in order, whenever the block before is the clip's last
    assert torch.equal(tails[hundredth - 1], tails[seventh - 1])
    assert torch.equal(tails[seventh - 1], torch.cat(clip))
    # the cache's context: the first frame, the memory frames (the block before, which is the
    # clip's last) and the recent context, at the seventh block and the hundredth
    assert torch.equal(contexts[hundredth], contexts[seventh])
    assert torch.equal(contexts[hundredth][0, 1:5], clip[2])
    assert torch.equal(contexts[hundredth][0, 5:17], torch.cat(clip))
    assert not torch.equal(contexts[seventh + 1], contexts[seventh])  # the clip in another order
    # the block that walks is generated from the same context, and the store keeps it as generated
    # after the clip's last two blocks
    assert torch.equal(contexts[walks], contexts[seventh])
    assert rollout.idle_context.stretch == 0
    assert torch.equal(tails[walks][0:4], clip[1]) and torch.equal(tails[walks][4:8], clip[2])
    assert torch.equal(tails[walks][8:12], x0s[walks])
    # and the next block's context is those three
    assert torch.equal(contexts[walks + 1][0, 5:17], tails[walks])
