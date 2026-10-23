"""A block refreshed under live controls (``live.refresh``), on the CPU with the tiny models.

* A refresh on the controls of the block's first cut gives its first x0 bit for bit, on other
  controls another x0, and back on the first controls the first again; the RNG is where the first
  denoising left it.
* A client whose window blocks are cut on other controls, refreshed on others and refreshed back on
  its own plays the rollout of a client that never refreshes, bit for bit: every block's latents,
  every part it publishes (its ``live`` part once a block, with the committed version's rows), the
  page's rows, and the latents of the client beside it, which reads its parts.
* The version committed is the newest: a client refreshed onto other controls plays on as a client
  cut on those controls from the start.
* Only the newest block of a window is refreshed; the first six blocks are not.
* A client that runs into a wall under the demo's wall settings (``live.wall_keys: guard``,
  ``live.dead_zone_u``, ``live.hop: spread``) is refreshed from the guard's state before the
  block's first cut: refreshes that end on the controls cut leave its rollout bit for bit, the
  guard clearing the same keys, and whatever the controls refreshed onto, no row it publishes
  passes the wall.
* The client decodes a refresh from the latent frame of the switch on the decoder's state before
  it in the version on screen (the tiny decoder and the Wan2.2 VAE): a refresh on the same controls
  gives the first version's frames from the switch on, byte for byte, wherever it switches, and the
  next block's frames are those of a client that never refreshed; a refresh whose frames were all
  sent yields none.
* A refresh delivered while the block's first version is decoded takes the decode over at the next
  latent frame (the step's iterator yields its frames), byte for byte the same way.
"""

import threading

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from tests.engine.inference.test_live_rollout import BLOCKS, OTHER, OWN, config, rollout
from worldcast.data.controls import CONTROL_BUTTONS, buttons_from_bits
from worldcast.data.game import BODY_HEIGHT
from worldcast.data.latents import FIRST_TARGET
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import Client
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.serving import ServingOptions
from worldcast.engine.inference.taehv import TinyDecoder
from worldcast.player_state.live_position import BODY_RADIUS_U

REFRESH = {"live.refresh": True}
#: The demo's settings for a wall in the way (``configs/demo/worldcast.yaml``).
WALL = {"live.wall_keys": "guard", "live.dead_zone_u": 4.0, "live.hop": "spread"}
#: A wall across +x, 150 u ahead of where the client that runs along +x from the origin starts its
#: first window block (latent frame 25, 1406 u on): that block stops at it, the next pushes on.
WALL_X = 1556.0
FORWARD = CONTROL_BUTTONS.index("forward")


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    """The standing round with the clients of slots 0 and 5, the tiny weights and a tiny state
    model, as ``test_live_rollout.py`` has them."""
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("refresh")
    weights = sw.make_weights(tmp / "weights")
    live = {"player_state.source": "live", **sw.write_state_model(tmp / "state_model.pt")}
    world = sw.standing_world(tmp / "standing", clients=(OWN, OTHER))
    return dict(tmp=tmp, weights=weights, live=live, worlds={"standing": world}, models={})


def controls(k: int, version: int = 0):
    """Block ``k``'s controls: the final ones walk forward and turn left; the others strafe right
    turning right (1), or jump with the AWP turning hard left (2)."""
    if version == 1:
        return sw.live_controls(k, "move_right", yaw=-30.0, pitch=4.0)
    if version == 2:
        return sw.live_controls(k, "jump", "forward", yaw=60.0, weapon="awp")
    return sw.live_controls(k, "forward", yaw=12.0)


def test_a_refresh_on_the_same_controls_gives_the_first_version_bit_for_bit(room, monkeypatch):
    live = rollout(room, monkeypatch, settings=REFRESH)
    for k, f0 in enumerate(BLOCKS[:6]):
        sw.step(live, f0, lambda k=k: controls(k))
        assert not live.can_refresh(f0)  # the first six blocks are committed at once
    f0 = BLOCKS[6]
    first = live.denoise(f0, lambda: controls(6)).clone()
    after = torch.get_rng_state()
    assert live.can_refresh(f0)
    same = live.refresh(f0, lambda: controls(6))
    assert torch.equal(same, first) and torch.equal(torch.get_rng_state(), after)
    other = live.refresh(f0, lambda: controls(6, 1))
    assert not torch.equal(other, first) and torch.equal(torch.get_rng_state(), after)
    assert torch.equal(live.refresh(f0, lambda: controls(6)), first)
    assert torch.equal(live.output[0, f0 - live.shift : f0 - live.shift + 4], first[0])
    with pytest.raises(RuntimeError, match="not open"):
        live.refresh(f0 + 4, lambda: controls(7))
    live.commit(f0, first)
    assert not live.can_refresh(f0)


def play(room, monkeypatch, *, versions: tuple[int, ...]) -> dict:
    """The client of slot 0 beside the idle client of slot 5, cross-wired. Each window block of
    slot 0 is cut on ``controls(k, versions[0])`` and refreshed on each later version in turn; the
    first six blocks are cut on the last version. Returns every block's x0 of both clients, slot
    0's parts and the page's rows of its blocks."""
    relay = sw.Relay()
    refreshing = len(versions) > 1
    live = rollout(room, monkeypatch, relay, row=0, settings=REFRESH if refreshing else None)
    idle = rollout(room, monkeypatch, relay, row=1)
    sent, x0s, rows = [], {}, {}
    outbox = live.world_state.outbox
    live.world_state.outbox = lambda part: (sent.append(part.encode()), outbox(part))
    for k, f0 in enumerate(BLOCKS):
        x0s["idle", f0] = idle.denoise(f0, lambda k=k: sw.live_controls(k)).clone()
        idle.commit(f0, x0s["idle", f0])
        idle.prepare_next(f0)
        relay.deliver(idle.media_id)
        first = versions[0] if f0 >= FIRST_TARGET else versions[-1]
        x0 = live.denoise(f0, lambda k=k: controls(k, first))
        assert live.can_refresh(f0) == (refreshing and f0 >= FIRST_TARGET)
        if live.can_refresh(f0):
            for version in versions[1:]:
                x0 = live.refresh(f0, lambda k=k, version=version: controls(k, version))
        x0s["live", f0] = x0.clone()
        live.commit(f0, x0)
        live.prepare_next(f0)
        rows.update((f, live.field_rows(f)) for f in range(f0, f0 + 4))
        relay.deliver(live.media_id)
    return dict(x0=x0s, sent=sent, rows=rows)


@pytest.mark.parametrize("versions", [(1, 2, 0), (0, 0)], ids=["refreshed-back", "same"])
def test_refreshes_that_end_on_the_controls_cut_leave_the_rollout_bit_for_bit(
    room, monkeypatch, versions
):
    plain = play(room, monkeypatch, versions=(0,))
    refreshed = play(room, monkeypatch, versions=versions)
    for key, x0 in plain["x0"].items():
        assert torch.equal(refreshed["x0"][key], x0), key
    assert refreshed["sent"] == plain["sent"]  # one live part a block, the same bytes
    for latent, rows in plain["rows"].items():
        again = refreshed["rows"][latent]
        assert np.array_equal(again.states, rows.states) and again.status == rows.status
        assert np.array_equal(again.source, rows.source)
        assert (again.drawn is None) == (rows.drawn is None)
        assert rows.drawn is None or np.array_equal(again.drawn, rows.drawn)


def test_the_version_committed_is_the_newest(room, monkeypatch):
    """A client cut on controls 0 and refreshed onto controls 1 plays as a client cut on controls 1
    from the start, and so does the client beside it, which reads its parts."""
    plain = play(room, monkeypatch, versions=(1,))
    refreshed = play(room, monkeypatch, versions=(0, 1))
    for key, x0 in plain["x0"].items():
        assert torch.equal(refreshed["x0"][key], x0), key
    assert refreshed["sent"] == plain["sent"]


def pushing(k: int, version: int = 0):
    """Block ``k``'s controls into the wall: forward (0), or forward turning right and looking down
    (1), which reaches the wall at a slant."""
    if version == 1:
        return sw.live_controls(k, "forward", yaw=-6.0, pitch=3.0)
    return sw.live_controls(k, "forward")


def into_the_wall(room, monkeypatch, *, versions: tuple[int, ...]) -> dict:
    """The client of slot 0 under :data:`WALL` running at a wall across +x at :data:`WALL_X`, which
    its depth head sees from every latent frame (a plane of points 1 u apart across, 4 u apart up to
    the body's height). Each window block is cut on ``pushing(k, versions[0])`` and refreshed on
    each later version in turn; the first six blocks are cut on the last version. Returns every
    block's x0, the pairs of a block's x0 on its first cut and again whenever a refresh came back to
    its controls, and the ``live`` parts the client published."""
    refreshing = len(versions) > 1
    live = rollout(room, monkeypatch, settings={**WALL, **(REFRESH if refreshing else {})})
    across, up = np.meshgrid(np.arange(-1000.0, 1001.0), np.arange(0.0, BODY_HEIGHT + 1.0, 4.0))
    wall = np.stack([np.full(across.size, WALL_X), across.ravel(), up.ravel()], axis=1)
    monkeypatch.setattr(live.closed.visibility, "surface", lambda latent, latents: wall.copy())
    parts, x0s, again = [], {}, []
    live.world_state.outbox = parts.append
    for k, f0 in enumerate(BLOCKS):
        first = versions[0] if f0 >= FIRST_TARGET else versions[-1]
        x0 = cut = live.denoise(f0, lambda k=k: pushing(k, first)).clone()
        if live.can_refresh(f0):
            for version in versions[1:]:
                x0 = live.refresh(f0, lambda k=k, version=version: pushing(k, version)).clone()
                if version == first:
                    again.append((cut, x0))
        x0s[f0] = x0
        live.commit(f0, x0)
        live.prepare_next(f0)
    return dict(x0=x0s, again=again, parts=[part for part in parts if part.kind == "live"])


def test_a_client_refreshed_while_it_runs_into_a_wall_stops_there_as_its_first_cut_does(
    room, monkeypatch
):
    """Block 25 stops at the wall, its hull's radius before it: the guard clears its forward key
    from the row it gets there, and block 29's, which pushes on, from its first row (the block
    glides on to where the state model reads the player, who slowed down past the stop without
    the key). Refreshed onto other controls and back, every block is denoised as on its first cut,
    and the client publishes the same ``live`` parts, bit for bit, as one that never refreshes.
    Refreshed onto controls that reach the wall at a slant for good, it publishes other rows, none
    of them past the wall either: the forward key is cleared there too."""
    plain = into_the_wall(room, monkeypatch, versions=(0,))
    window = [part for part in plain["parts"] if part.f0 >= FIRST_TARGET]
    cleared = [int(np.argmin(buttons_from_bits(p.buttons)[:, FORWARD])) for p in window]
    assert len(cleared) == 2 and 0 < cleared[0] < 16 and cleared[1] == 0
    stop = WALL_X - BODY_RADIUS_U
    assert stop - 1.0 < window[0].rows[:, 0].max() <= stop + 1e-3
    assert np.ptp(window[1].rows[:, 0]) < BODY_RADIUS_U / 2
    for versions in ((1, 0), (0, 1, 0)):
        refreshed = into_the_wall(room, monkeypatch, versions=versions)
        assert len(refreshed["again"]) == (2 if versions[0] == versions[-1] else 0)
        assert all(torch.equal(first, back) for first, back in refreshed["again"])
        for f0, x0 in plain["x0"].items():
            assert torch.equal(refreshed["x0"][f0], x0), (versions, f0)
        assert [p.encode() for p in refreshed["parts"]] == [p.encode() for p in plain["parts"]]
    slant = into_the_wall(room, monkeypatch, versions=(0, 1))
    assert [p.encode() for p in slant["parts"]] != [p.encode() for p in plain["parts"]]
    assert (buttons_from_bits(slant["parts"][-1].buttons)[:, FORWARD] == 0).any()
    for run in (plain, slant):
        x = np.concatenate([part.rows[:, 0] for part in run["parts"]])
        assert WALL_X - 2 * BODY_RADIUS_U < x.max() < WALL_X  # at a slant it stops farther off


def client(room, monkeypatch, decoder: str, tiny_vae, *, refresh: bool) -> Client:
    """The engine's client of slot 0 alone, its state model reading where its keys put it, with
    the tiny decoder (random weights) or the tiny Wan2.2 VAE."""
    cfg = config(room, settings={"live.refresh": True} if refresh else None)
    sw.patch_extent(monkeypatch)
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    taehv = room["tmp"] / "taew2_2.pth"
    if not taehv.exists():
        torch.manual_seed(0)
        torch.save(TinyDecoder().state_dict(), taehv)
    serving = ServingOptions(
        decoder=decoder,
        taehv_path=str(taehv) if decoder == "taehv" else None,
        cuda_graphs=False,
        lockstep=False,
    )
    key = ("client", decoder)
    if key not in room["models"]:
        room["models"][key] = ClientModels.load(cfg, ServingOptions(decoder="none"))
        if decoder == "wan":
            room["models"][key].wan_vae = tiny_vae
    made = Client(cfg, serving, models=room["models"][key])
    made.start(load_round_index_row(cfg.paths.round_index, 0))
    sw.own_position_oracle(made._rollout)
    return made


@pytest.mark.parametrize("decoder", ["taehv", "wan"])
def test_a_refresh_is_decoded_from_its_switch_on_the_state_of_the_version_on_screen(
    room, monkeypatch, tiny_vae, decoder
):
    after = {}
    for refresh in (False, True):
        made = client(room, monkeypatch, decoder, tiny_vae, refresh=refresh)
        for k in range(6):
            list(made.step(lambda k=k: controls(k)))
            assert not made.can_refresh
        first = list(made.step(lambda: controls(6)))
        assert made.can_refresh == refresh
        if refresh:
            start = first[0].index
            assert [f.index for f in first] == list(range(start, start + 16))
            assert {f.version for f in first} == {0} and first[0].rows is not None
            # the same controls, switched at row 6 (latent 1) and row 3 (latent 0), around a
            # refresh on other controls switched at row 9 (latent 2)
            for row, version, same in ((6, 0, True), (9, 1, False), (3, 0, True), (16, 1, None)):
                again = list(
                    made.refresh(lambda v=version: controls(6, v), lambda s, r=row: start + r)
                )
                assert [f.index for f in again] == list(range(start + row, start + 16))
                for new, old in zip(again, first[row:]):
                    assert np.array_equal(new.image, old.image) == same
                    assert new.version == len(made.records[-1].refreshes)
            records = made.records[-1].refreshes
            assert [r.first_frame for r in records] == [start + 6, start + 9, start + 3, -1]
            assert all(r.t_start <= r.t_controls <= r.t_cut <= r.t_latents for r in records)
            list(made.refresh(lambda: controls(6), lambda s: start + 16))  # committed: the first
        after[refresh] = [f.image for f in made.step(lambda: controls(7))]
        made.stop()
    assert all(np.array_equal(a, b) for a, b in zip(after[True], after[False]))


def newest(frames) -> dict:
    """The last frame taken of each index (a newer version comes after an older one)."""
    return {f.index: f for f in frames}


def test_a_refresh_asked_for_before_any_frame_of_its_block_is_taken_swaps_the_whole_block(
    room, monkeypatch, tiny_vae
):
    """The worker asks for a refresh as soon as the block's step is asked for: the refresh is cut
    right after the block's first denoising and gives the whole block (nothing was sent), the frames
    of a client cut on its controls from the start, byte for byte; so is the next block. Its frames
    come from the step's iterator if it takes the decode over, from its own if not."""
    frames = {}
    for refresh in (False, True):
        made = client(room, monkeypatch, "taehv", tiny_vae, refresh=refresh)
        for k in range(6):
            list(made.step(lambda k=k: controls(k)))
        if refresh:
            asked = made.step(lambda: controls(6))  # nothing taken yet
            start = 1 + 16 * 6
            again = made.refresh(lambda: controls(6, 1), lambda seconds: start)
            taken = list(asked) + list(again)
            puts = list(dict.fromkeys(f.index for f in taken))
            assert puts == list(range(start, start + 16))
            frames[refresh] = list(newest(taken).values())
            assert {f.version for f in frames[True]} == {1}
            assert made.records[-1].refreshes[0].first_frame == start
        else:
            frames[refresh] = list(made.step(lambda: controls(6, 1)))
        frames[refresh] += list(made.step(lambda: controls(7)))
        made.stop()
    assert all(np.array_equal(a.image, b.image) for a, b in zip(frames[True], frames[False]))
    assert len(frames[True]) == len(frames[False]) == 32


@pytest.mark.parametrize("decoder", ["taehv", "wan"])
@pytest.mark.parametrize(
    "taken, row, replaced", [(4, 2, 2), (4, 9, 0), (8, 5, 3), (12, 1, 11), (12, 16, 0)]
)
def test_a_refresh_generated_while_the_block_is_decoded_takes_the_decode_over(
    room, monkeypatch, tiny_vae, decoder, taken, row, replaced
):
    """A refresh delivered while the block's first version is decoded takes over at the next
    latent frame: the step's iterator yields its frames from its switch on, never past the first
    frame not yet decoded (``taken`` frames were, a latent frame's at least: with the decoder on
    the generator's device the refresh waits for the first), so the frames before come again
    (``replaced``), and its own iterator yields none. On the same controls they are the first
    version's, byte for byte, wherever it takes over; so are the next block's."""
    reference = {}
    for refresh in (False, True):
        made = client(room, monkeypatch, decoder, tiny_vae, refresh=refresh)
        for k in range(6):
            list(made.step(lambda k=k: controls(k)))
        if not refresh:
            reference = [f.image for f in made.step(lambda: controls(6))]
            reference += [f.image for f in made.step(lambda: controls(7))]
            made.stop()
            continue
        start = 1 + 16 * 6
        asked = made.step(lambda: controls(6))
        first = [next(asked) for _ in range(taken)]
        ready = threading.Event()
        switched = []

        def switch(seconds):
            switched.append(seconds)
            return start + row

        again = made.refresh(lambda: controls(6), switch, ready.set)
        assert ready.wait(60.0)
        rest = list(asked)
        assert list(again) == [] and len(switched) == 1
        assert [f.index for f in first] == list(range(start, start + taken))
        assert {f.version for f in rest} == {1}
        expected = list(range(start + min(row, taken), start + 16))
        assert [f.index for f in rest] == expected
        assert len([f for f in rest if f.index < start + taken]) == replaced
        assert made.records[-1].refreshes[0].first_frame == expected[0]
        frames = newest(first + rest)
        assert sorted(frames) == list(range(start, start + 16))
        images = [frames[k].image for k in sorted(frames)]
        images += [f.image for f in made.step(lambda: controls(7))]
        made.stop()
    assert len(images) == len(reference) == 32
    assert all(np.array_equal(a, b) for a, b in zip(images, reference))


@pytest.mark.parametrize("decoder", ["taehv", "wan"])
@pytest.mark.parametrize("row", [2, 9, 16])
def test_a_refresh_decode_stops_at_a_newer_refresh(room, monkeypatch, tiny_vae, decoder, row):
    """A refresh's decode stops at the next latent frame once a newer refresh of the block is
    delivered, which decodes from its own switch on the decoder's state of the version on screen;
    one that decodes nothing leaves the decoder where the version on screen ended. A refresh on
    other controls stopped after a latent frame, then one back on the block's first controls:
    from its switch on the first version's frames, byte for byte, and the next block's too."""
    reference = {}
    for refresh in (False, True):
        made = client(room, monkeypatch, decoder, tiny_vae, refresh=refresh)
        for k in range(6):
            list(made.step(lambda k=k: controls(k)))
        if not refresh:
            reference = [f.image for f in made.step(lambda: controls(6))]
            reference += [f.image for f in made.step(lambda: controls(7))]
            made.stop()
            continue
        start = 1 + 16 * 6
        first = list(made.step(lambda: controls(6)))
        ready = [threading.Event(), threading.Event()]
        other = made.refresh(lambda: controls(6, 1), lambda s: start, ready[0].set)
        assert ready[0].wait(60.0)
        stopped = [next(other) for _ in range(4)]
        back = made.refresh(lambda: controls(6), lambda s: start + row, ready[1].set)
        assert ready[1].wait(60.0)
        stopped += list(other)
        again = list(back)
        assert [f.index for f in stopped] == list(range(start, start + 4))
        assert [f.index for f in again] == list(range(start + row, start + 16))
        assert {f.version for f in stopped} == {1} and {f.version for f in again} <= {2}
        for new in again:
            assert np.array_equal(new.image, first[new.index - start].image)
        after = [f.image for f in made.step(lambda: controls(7))]
        made.stop()
    assert len(after) == 16
    assert all(np.array_equal(a, b) for a, b in zip(after, reference[16:]))
