"""A live client's generator on two cards (``ServingOptions.refresh_devices``), on the CPU with the
tiny models: two cards on the CPU, the client's and a replica's.

* Every block's context is prefilled on the replica as on the client's card: the same keys and
  values. A cut denoised on either card gives the x0 a refresh of the same controls gives a client
  without a replica, bit for bit, also when later cuts were made before it was denoised, and the
  newest cut committed plays on as that client does.
* A client with a replica plays as one without: refreshes on either card, the frames and the next
  block's, byte for byte.
* A refresh asked for while the block's first version is denoised is cut right after its cut and
  denoised on the replica beside it; the newest version shows, the whole block.
* A version done after a newer one shows nothing, and the newest cut is committed.
* A refresh's preview comes first, its frames marked, from the x0 of its preview step; the
  version's own frames come after it, those of a refresh without a preview, on either card.
* The warm-up's replay denoises each window block again on the replica: equal.
* Refusals: one card named, a card that is no device, cards without live refresh, a first card
  that is not the client's.
"""

import threading
import types
from concurrent.futures import Future

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from tests.engine.inference.test_live_rollout import BLOCKS, OTHER, OWN, config, rollout
from tests.engine.inference.test_refresh import REFRESH, controls, newest
from worldcast.data.latents import FIRST_TARGET
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import Client, RefreshRecord, _RunningRollout
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.replicas import GeneratorReplica
from worldcast.engine.inference.rollout import context_noise
from worldcast.engine.inference.serving import ServingOptions
from worldcast.engine.inference.taehv import TinyDecoder
from worldcast.modeling.wan22.attention import sdpa_attention

#: The first frame of the first window block, block 6 (latent frames 25-28).
START = 1 + 16 * 6


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    """The standing round with the client of slot 0, the tiny weights and a tiny state model, as
    ``test_refresh.py`` has them."""
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("replicas")
    weights = sw.make_weights(tmp / "weights")
    live = {"player_state.source": "live", **sw.write_state_model(tmp / "state_model.pt")}
    world = sw.standing_world(tmp / "standing", clients=(OWN, OTHER))
    return dict(tmp=tmp, weights=weights, live=live, worlds={"standing": world}, models={})


def replica_of(room) -> GeneratorReplica:
    """A replica of the room's client on the CPU, as a client's own (the warm-up's float32)."""
    cfg = config(room, settings=REFRESH)
    serving = ServingOptions(decoder="none", cuda_graphs=False)
    return GeneratorReplica(
        cfg,
        serving,
        "cpu",
        attention=sdpa_attention,
        dtype=torch.float32,
        context_noise=context_noise(cfg),
    )


def same_cache(live, replica) -> bool:
    mine, its = live.sampler.cache, replica.sampler.cache
    end = mine.end
    return end == its.end and all(
        torch.equal(a[:, :end], b[:, :end])
        for a, b in zip([*mine.keys, *mine.values], [*its.keys, *its.values])
    )


def test_a_cut_denoised_on_either_card_gives_the_x0_of_a_refresh(room, monkeypatch):
    """Three cuts of block 25 made before any is denoised (controls 0, 1 and 2), denoised in
    another order on the two cards: each the x0 of a client without a replica refreshed onto the
    same controls; the RNG where a denoising leaves it. The newest cut committed, the next block
    is that client's next block, and the replica's cache holds the client's after each prefill."""
    replica = replica_of(room)  # before the rollouts: its weights' init draws from the RNG
    plain = rollout(room, monkeypatch, settings=REFRESH)
    for k, f0 in enumerate(BLOCKS[:6]):
        sw.step(plain, f0, lambda k=k: controls(k))
    f0, after = BLOCKS[6], BLOCKS[7]
    want = {0: plain.denoise(f0, lambda: controls(6)).clone()}
    for version in (1, 2):
        want[version] = plain.refresh(f0, lambda v=version: controls(6, v)).clone()
    rng = torch.get_rng_state()
    plain.commit(f0, want[2])
    plain.prepare_next(f0)
    want["next"] = plain.denoise(after, lambda: controls(7)).clone()

    live = rollout(room, monkeypatch, settings=REFRESH)
    live.replicas = (replica,)
    for k, block in enumerate(BLOCKS[:6]):
        sw.step(live, block, lambda k=k: controls(k))
    replica.prefilled()
    assert same_cache(live, replica)
    cuts = [live.cut(f0, lambda v=version: controls(6, v)) for version in (0, 1, 2)]
    assert torch.equal(live.denoise_cut(cuts[1], replica), want[1])
    assert torch.equal(live.denoise_cut(cuts[2]), want[2])
    assert torch.equal(live.denoise_cut(cuts[0]), want[0])
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(live.denoise_cut(cuts[2], replica), want[2])
    live.commit(f0, want[2])
    live.prepare_next(f0)
    replica.prefilled()
    assert same_cache(live, replica)
    cut = live.cut(after, lambda: controls(7))
    assert torch.equal(live.denoise_cut(cut, replica), want["next"])


def client(room, monkeypatch, *, cards: int) -> Client:
    """The engine's client of slot 0 with refresh on and the tiny decoder (random weights), its
    state model reading where its keys put it, on ``cards`` cards of the CPU."""
    cfg = config(room, settings=REFRESH)
    sw.patch_extent(monkeypatch)
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    taehv = room["tmp"] / "taew2_2.pth"
    if not taehv.exists():
        torch.manual_seed(0)
        torch.save(TinyDecoder().state_dict(), taehv)
    serving = ServingOptions(
        decoder="taehv",
        taehv_path=str(taehv),
        cuda_graphs=False,
        lockstep=False,
        refresh_devices=("cpu",) * cards if cards > 1 else (),
    )
    key = ("client", "taehv")
    if key not in room["models"]:
        room["models"][key] = ClientModels.load(cfg, ServingOptions(decoder="none"))
    made = Client(cfg, serving, models=room["models"][key])
    made.start(load_round_index_row(cfg.paths.round_index, 0))
    sw.own_position_oracle(made._rollout)
    return made


def first_six(made: Client) -> None:
    for k in range(6):
        list(made.step(lambda k=k: controls(k)))


def test_a_client_with_a_replica_plays_as_one_without(room, monkeypatch):
    """Refreshes of block 25 on the replica, the client's card and the replica again, each on
    other controls and switched elsewhere, then the next block: the frames of a client without a
    replica, byte for byte, and each refresh's record names its card."""
    frames = {}
    for cards in (1, 2):
        made = client(room, monkeypatch, cards=cards)
        first_six(made)
        seen = [list(made.step(lambda: controls(6)))]
        for card, row, version in ((1, 6, 1), (0, 9, 2), (1, 3, 0)):
            on = {"card": card} if cards > 1 else {}
            again = made.refresh(lambda v=version: controls(6, v), lambda s, r=row: START + r, **on)
            seen.append(list(again))
        seen.append(list(made.step(lambda: controls(7))))
        frames[cards] = seen
        if cards > 1:
            assert made.refresh_cards == 2
            assert [r.card for r in made.records[-2].refreshes] == [1, 0, 1]
        made.stop()
    for one, two in zip(frames[1], frames[2]):
        assert [f.index for f in one] == [f.index for f in two]
        assert [f.version for f in one] == [f.version for f in two]
        assert all(np.array_equal(a.image, b.image) for a, b in zip(one, two))


def reference(room, monkeypatch, version: int) -> list[np.ndarray]:
    """The images of block 25 cut on ``controls(6, version)`` from the start and of the next
    block, on a client without a replica."""
    made = client(room, monkeypatch, cards=1)
    first_six(made)
    images = [f.image for f in made.step(lambda: controls(6, version))]
    images += [f.image for f in made.step(lambda: controls(7))]
    made.stop()
    return images


def test_a_refresh_asked_for_during_the_first_denoising_runs_on_the_replica_beside_it(
    room, monkeypatch
):
    """A refresh asked for with the block's step goes to the free replica, is cut right after the
    block's first cut, while the first version is still denoised (held until the refresh's cut is
    done), and takes the whole block over: the frames of a client cut on its controls from the
    start, byte for byte, and the next block's."""
    want = reference(room, monkeypatch, 1)
    made = client(room, monkeypatch, cards=2)
    first_six(made)
    live, cut_again, cuts = made._rollout, threading.Event(), []
    cut, denoise_cut = live.cut, live.denoise_cut

    def cutting(f0, controls):
        done = cut(f0, controls)
        cuts.append(f0)
        if len(cuts) == 2:
            cut_again.set()
        return done

    def denoising(block_cut, replica=None, on_step=None):
        if replica is None:  # the first version waits until the refresh is cut beside it
            assert cut_again.wait(60.0)
        return denoise_cut(block_cut, replica, on_step)

    monkeypatch.setattr(live, "cut", cutting)
    monkeypatch.setattr(live, "denoise_cut", denoising)
    asked = made.step(lambda: controls(6))
    again = made.refresh(lambda: controls(6, 1), lambda seconds: START)
    shown = newest(list(asked) + list(again))
    assert sorted(shown) == list(range(START, START + 16))
    assert {f.version for f in shown.values()} == {1}
    record = made.records[-1]
    (refresh,) = record.refreshes
    assert refresh.card == 1 and refresh.t_cut < record.t_latents
    images = [shown[k].image for k in sorted(shown)]
    images += [f.image for f in made.step(lambda: controls(7))]
    made.stop()
    assert all(np.array_equal(a, b) for a, b in zip(images, want))


def test_a_version_done_after_a_newer_one_shows_nothing_and_the_newest_is_committed(
    room, monkeypatch
):
    """Refresh 1 on the replica, held after its cut until refresh 2, cut after it on the client's
    card, is done: refresh 2's frames show, refresh 1 gives none, and the block committed is
    refresh 2's: the next block is that of a client cut on its controls."""
    want = reference(room, monkeypatch, 2)
    made = client(room, monkeypatch, cards=2)
    first_six(made)
    list(made.step(lambda: controls(6)))
    live, gate, ready = made._rollout, threading.Event(), threading.Event()
    denoise_cut = live.denoise_cut

    def denoising(block_cut, replica=None, on_step=None):
        if replica is not None:
            assert gate.wait(60.0)
        return denoise_cut(block_cut, replica, on_step)

    monkeypatch.setattr(live, "denoise_cut", denoising)
    slow = made.refresh(lambda: controls(6, 1), lambda seconds: START, card=1)
    fast = made.refresh(lambda: controls(6, 2), lambda seconds: START, ready.set, card=0)
    assert ready.wait(60.0)
    gate.set()
    shown = list(slow) + list(fast)
    assert [f.index for f in shown] == list(range(START, START + 16))
    assert {f.version for f in shown} == {2}
    records = made.records[-1].refreshes
    assert [(r.card, r.first_frame) for r in records] == [(1, -1), (0, START)]
    assert all(np.array_equal(f.image, w) for f, w in zip(shown, want[:16]))
    after = [f.image for f in made.step(lambda: controls(7))]
    made.stop()
    assert all(np.array_equal(a, b) for a, b in zip(after, want[16:]))


@pytest.mark.parametrize("cards", [1, 2])
def test_a_refresh_preview_comes_first_and_the_versions_own_frames_after_it(
    room, monkeypatch, cards
):
    """With ``preview_step`` 1 a refresh hands the x0 of its first denoising step to the decoder
    (held there until the preview's decode has begun): its frames, marked, come first from the
    switch on, then the version's own frames, those of the same refresh without a preview, byte
    for byte; a preview frame is not its version's frame."""
    made = client(room, monkeypatch, cards=cards)
    first_six(made)
    list(made.step(lambda: controls(6)))
    plain = list(made.refresh(lambda: controls(6, 1), lambda seconds: START + 4))
    live, previewing = made._rollout, threading.Event()

    def hold(on_step):
        def step(i, x0):
            on_step(i, x0)
            if i == 0:
                assert previewing.wait(60.0)

        return step

    if cards > 1:
        denoise_cut = live.denoise_cut
        monkeypatch.setattr(
            live,
            "denoise_cut",
            lambda cut, replica=None, on_step=None: denoise_cut(cut, replica, hold(on_step)),
        )
    else:
        refresh = live.refresh
        monkeypatch.setattr(
            live, "refresh", lambda f0, controls, on_step=None: refresh(f0, controls, hold(on_step))
        )

    def switch(seconds):
        previewing.set()
        return START + 4

    shown = list(made.refresh(lambda: controls(6, 1), switch, preview_step=1))
    previews = [f for f in shown if f.preview]
    own = [f for f in shown if not f.preview]
    assert len(previews) >= 4 and all(f.index >= START + 4 for f in previews)
    assert {f.version for f in shown} == {2}
    assert [f.index for f in own] == [f.index for f in plain]
    assert all(np.array_equal(a.image, b.image) for a, b in zip(own, plain))
    final = {f.index: f.image for f in own}
    assert not any(np.array_equal(f.image, final[f.index]) for f in previews)
    record = made.records[-1].refreshes[-1]
    assert record.preview_frame == START + 4 and 0 < record.t_preview <= record.t_latents
    made.stop()


def test_the_warm_up_replay_denoises_each_window_block_again_on_the_replica(room, monkeypatch):
    """A client with a replica replaying the recorded controls (the warm-up's replay) denoises
    each window block again on the replica and finds it equal."""
    made = client(room, monkeypatch, cards=2)
    made.stop()
    made.cfg = config(room, source="gt")
    made.start(load_round_index_row(made.cfg.paths.round_index, 0))
    while not made.finished:
        list(made.step())
    made.stop()
    windows = sum(record.f0 >= FIRST_TARGET for record in made.records)
    assert windows >= 2 and made.replicas_equal == [True] * windows


def test_the_cards_are_checked():
    """Each card by its index, once (the CPU stands in for cards in the tests), none the
    decoder's."""
    with pytest.raises(ValueError, match="at least one more"):
        ServingOptions(refresh_devices=("cuda:0",))
    for named in (("cuda:0", "gpu1"), ("cuda", "cuda:1"), ("cuda:0", "cuda")):
        with pytest.raises(ValueError, match="refresh_devices are cuda:<index>"):
            ServingOptions(refresh_devices=named)
    with pytest.raises(ValueError, match="names a card twice"):
        ServingOptions(refresh_devices=("cuda:0", "cuda:1", "cuda:1"))
    with pytest.raises(ValueError, match="cuda:1 is the decoder's card"):
        ServingOptions(decoder="wan", decoder_device="cuda:1", refresh_devices=("cuda:0", "cuda:1"))
    wan = ServingOptions(
        decoder="wan", decoder_device="cuda:1", refresh_devices=["cuda:0", "cuda:2"]
    )
    assert wan.refresh_devices == ("cuda:0", "cuda:2")
    assert ServingOptions(refresh_devices=["cpu", "cpu"]).refresh_devices == ("cpu", "cpu")


def test_compiled_replicas_keep_their_compiled_parts(monkeypatch):
    """With ``compile`` dynamo keeps the graphs of every card's DiT-block parts: its limit of
    graphs per code object grows with the cards."""
    import torch._dynamo.config as dynamo

    from worldcast.engine.inference.client import _raise_recompile_limit

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo, name):
            monkeypatch.setattr(dynamo, name, 8)
    _raise_recompile_limit(3)
    assert dynamo.cache_size_limit == 24


def test_a_job_on_the_replica_never_waits_for_the_prefill_queued_behind_it(room):
    """``prefilled()`` on the replica's own thread with its prefill still to come raises, where
    it would wait for ever."""
    replica = replica_of(room)
    replica._prefill = Future()  # a prefill that would run after the job asking for it
    with pytest.raises(RuntimeError, match="queued behind it"):
        replica.thread.submit(replica.prefilled).result(60.0)
    replica._prefill.set_result(None)
    replica.thread.submit(replica.prefilled).result(60.0)
    replica.thread.shutdown()


def test_a_refresh_asked_for_with_its_blocks_step_on_the_replica_is_cut_off_its_thread(
    room, monkeypatch
):
    """A refresh asked for right with the step of a block that is prepared in that step (the block
    after an open one) goes to the replica before the block's prefill: it is cut on the cuts'
    thread and denoised on the replica's behind the prefill, and gives the frames of a client
    without a replica (no wait for ever)."""
    want = {}
    for cards in (1, 2):
        made = client(room, monkeypatch, cards=cards)
        first_six(made)
        list(made.step(lambda: controls(6)))
        asked = made.step(lambda: controls(7))  # prepared in its step, after block 25's commit
        on = {"card": 1} if cards > 1 else {}
        again = made.refresh(lambda: controls(7, 1), lambda seconds: START + 16, **on)
        done, taken = threading.Event(), []

        def take(asked=asked, again=again, taken=taken, done=done):
            taken.extend(newest(list(asked) + list(again)).values())
            done.set()

        threading.Thread(target=take, daemon=True).start()
        assert done.wait(120.0), "the refresh waits for ever"
        want[cards] = [f.image for f in taken]
        made.stop()
    assert len(want[2]) == len(want[1]) == 16
    assert all(np.array_equal(a, b) for a, b in zip(want[1], want[2]))


def test_a_version_whose_turn_has_passed_stops_waiting():
    """A refresh waits for its turn to cut; one whose turn has gone by (another block's, or a later
    version's) stops with an error instead of waiting for ever."""
    running = object.__new__(_RunningRollout)
    running._cuts, running._turn, running._broken = threading.Condition(), (29, 1), False
    running.client = types.SimpleNamespace(stream=None)
    for f0, version in ((25, 2), (29, 0)):
        with pytest.raises(RuntimeError, match="missed its turn"):
            running._cut_version(f0, version, lambda: None, RefreshRecord())


def test_a_client_whose_replicas_are_let_go_refreshes_on_its_own_card(room, monkeypatch):
    """``disable_replicas`` (a replica the warm-up found otherwise): the client's next rollout
    refreshes on its card alone, as a client without replicas."""
    made = client(room, monkeypatch, cards=2)
    made.stop()
    made.disable_replicas()
    assert made.replicas == () and made.refresh_cards == 1
    made.start(load_round_index_row(made.cfg.paths.round_index, 0))
    sw.own_position_oracle(made._rollout)
    first_six(made)
    list(made.step(lambda: controls(6)))
    again = list(made.refresh(lambda: controls(6, 1), lambda seconds: START))
    assert [f.index for f in again] == list(range(START, START + 16))
    assert [r.card for r in made.records[-1].refreshes] == [0]
    made.stop()


def test_cards_need_live_refresh_and_start_with_the_clients(room, monkeypatch):
    sw.patch_ticks(monkeypatch, room["worlds"]["standing"]["tables"])
    models = room["models"].setdefault(
        "plain", ClientModels.load(config(room), ServingOptions(decoder="none"))
    )
    for settings, cards, error in (
        (None, ("cpu", "cpu"), "live.refresh true"),
        (REFRESH, ("cuda:0", "cpu"), "starts with the client's card"),
    ):
        serving = ServingOptions(decoder="none", cuda_graphs=False, refresh_devices=cards)
        with pytest.raises(ValueError, match=error):
            Client(config(room, settings=settings), serving, models=models)
