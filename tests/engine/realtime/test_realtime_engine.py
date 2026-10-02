"""CPU smoke tests of the realtime engine with a tiny random model on a synthetic round.

* The engine with the paper settings, and with every exact latency option (fast generator, early prefill,
  overlapped decode, no lock-step for a lone client), produces the release client's latents bit for bit.
* Streaming: frames arrive in order, latent 0 gives one frame and every block 16, encoded as JPEG.
* Controls: recorded controls passed explicitly change nothing; other controls change the block.
"""

import io

import numpy as np
import pytest
import torch

import tests.engine.realtime.synthetic_world as sw
from worldcast.data.index import load_round_index_row
from worldcast.engine.inference.client import Client
from worldcast.engine.realtime.config import RealtimeConfig
from worldcast.engine.realtime.controls import BlockControls, LiveControls
from worldcast.engine.realtime.engine import Engine, EngineModels, RoundSpec
from worldcast.engine.realtime.taehv import TinyDecoder
from worldcast.modeling.wan22.attention import sdpa_attention

MAX_BLOCKS = 2  # plain prefix (6 blocks) + 2 reconstituted blocks = 33 latents


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("rt")
    world = sw.make_world(tmp / "world", clients=(0,))
    weights = sw.make_weights(tmp / "weights")
    return dict(tmp=tmp, world=world, weights=weights)


@pytest.fixture
def reference(session, monkeypatch):
    sw.patch_ticks(monkeypatch, session["world"]["tables"])
    out = session["tmp"] / "client"
    if not (out / "latents.npy").exists():
        cfg = sw.config(
            session["world"],
            session["weights"],
            out,
            max_blocks=MAX_BLOCKS,
            pool=str(session["tmp"] / "client_pool"),
        )
        Client(cfg, attention=sdpa_attention).run()
    return np.load(out / "latents.npy")


def run_engine(session, rt: RealtimeConfig, *, controls=None, models=None):
    cfg = sw.config(
        session["world"], session["weights"], session["tmp"] / "engine", max_blocks=MAX_BLOCKS
    )
    models = models or EngineModels.load(cfg, rt, attention=sdpa_attention)
    engine = Engine(cfg, rt, models=models)
    info = engine.start(row=load_round_index_row(cfg.paths.round_index, 0))
    frames, k = [], 0
    while not engine.finished:
        frames.extend(engine.step(None if controls is None else controls(k)))
        k += 1
    latents = engine.latents[0].float().numpy()
    assert list(engine.step()) == []  # past the end: nothing more
    engine.stop()
    return info, latents, frames, engine


@pytest.mark.parametrize(
    "rt",
    [
        RealtimeConfig(decoder="none"),
        RealtimeConfig(generator="fast", early_prefill=True, decoder="none", lockstep=False),
        RealtimeConfig(generator="fast", early_prefill=True, decode_overlap=True, decoder="none"),
    ],
    ids=["paper", "fast-early", "fast-overlap"],
)
def test_engine_latents_equal_the_client(session, reference, rt):
    info, latents, frames, _ = run_engine(session, rt)
    assert info["latents"] == reference.shape[0] == 1 + 4 * (6 + MAX_BLOCKS)
    assert np.array_equal(latents, reference)
    assert len(frames) == reference.shape[0]  # decoder 'none': one latent per frame


def test_streaming_frames_and_jpeg(session, reference, monkeypatch):
    sw.patch_ticks(monkeypatch, session["world"]["tables"])
    rt = RealtimeConfig(
        generator="fast",
        early_prefill=True,
        decode_overlap=True,
        decoder="taehv",
        taehv_path="unused",
        encoder="jpeg",
        jpeg_device="cpu",
        lockstep=False,
    )
    cfg = sw.config(
        session["world"], session["weights"], session["tmp"] / "engine", max_blocks=MAX_BLOCKS
    )
    models = EngineModels.load(cfg, RealtimeConfig(decoder="none"), attention=sdpa_attention)
    torch.manual_seed(0)
    models.tiny = TinyDecoder().eval().requires_grad_(False)
    info, latents, frames, engine = run_engine(session, rt, models=models)
    assert np.array_equal(latents, reference)
    n = reference.shape[0]
    assert [f.index for f in frames] == list(range(1 + 4 * (n - 1)))
    assert [f.block for f in frames[:2]] == [0, 1] and frames[-1].block == n - 4
    from PIL import Image

    img = Image.open(io.BytesIO(frames[5].image))
    assert img.size == (672, 384)
    assert all(r.t_first_frame >= r.t_x0 > 0 for r in engine.records)


def test_controls(session, reference, monkeypatch):
    sw.patch_ticks(monkeypatch, session["world"]["tables"])
    rt = RealtimeConfig(generator="fast", early_prefill=True, decoder="none")
    cfg = sw.config(
        session["world"], session["weights"], session["tmp"] / "engine", max_blocks=MAX_BLOCKS
    )
    models = EngineModels.load(cfg, rt, attention=sdpa_attention)
    # the recorded controls, passed explicitly, reproduce the recorded run
    engine = Engine(cfg, rt, models=models)
    engine.start(row=load_round_index_row(cfg.paths.round_index, 0))
    batch = engine._session.round_batch

    def recorded(k):
        return BlockControls(
            buttons=batch["button_condition"][0, 1 + 16 * k : 17 + 16 * k].numpy(),
            camera=batch["camera_condition"][0, 1 + 16 * k : 17 + 16 * k].numpy(),
            weapon=batch["weapon_condition"][0, 1 + 16 * k : 17 + 16 * k].numpy(),
        )

    engine.stop()
    _, latents, _, _ = run_engine(session, rt, controls=recorded, models=models)
    assert np.array_equal(latents, reference)

    # live controls: a key and a turn change the block they are sampled into, and none before it
    def scripted(press_at):
        live = LiveControls()

        def controls(k):
            if k == press_at:
                live.press("attack")
                live.turn(0.0, 25.0)
            return live.sample()

        return controls

    _, idle, _, _ = run_engine(session, rt, controls=scripted(None), models=models)
    _, steered, _, _ = run_engine(session, rt, controls=scripted(7), models=models)
    assert not np.array_equal(idle[1:5], reference[1:5])  # no input is not the recorded input
    assert np.array_equal(steered[:29], idle[:29])  # blocks before the press: unchanged
    assert not np.array_equal(steered[29:33], idle[29:33])  # block 29 (k = 7) carries it


def test_session_options(session, reference, monkeypatch):
    """Per-session lock-step and seats nobody plays (dead and unseen; the recorded run is unchanged otherwise)."""
    sw.patch_ticks(monkeypatch, session["world"]["tables"])
    rt = RealtimeConfig(generator="fast", decoder="none", lockstep=False)
    cfg = sw.config(
        session["world"], session["weights"], session["tmp"] / "engine", max_blocks=MAX_BLOCKS
    )
    engine = Engine(cfg, rt, models=EngineModels.load(cfg, rt, attention=sdpa_attention))
    engine.start(
        row=load_round_index_row(cfg.paths.round_index, 0), only_clients=True, lockstep=True
    )
    batch = engine._session.round_batch
    assert float(batch["player_states"][0, 1:, :, 5].abs().sum()) == 0.0
    assert (
        float(batch["player_states"][0, 0, :, 5].sum()) > 0
        and float(batch["observer_visibility"][0, 1:].sum()) == 0
    )
    assert engine._session.peers == ()  # a lone client: nobody to wait for
    list(engine.step())
    first = engine.latents[0, 1:5].float().numpy()
    engine.stop()
    assert not np.array_equal(
        first, reference[1:5]
    )  # the other nine players are gone from the field


def test_round_spec_rows(session):
    from worldcast.data.media import MediaIndex

    spec = RoundSpec(match_id=sw.MATCH, map_name=sw.MAP, round=sw.ROUND, clients=(0, 3))
    row = spec.index_row(3, MediaIndex.load(session["world"]["media_index"]))
    assert row.media_id == sw.media_id(3) and row.lockstep_peers() == (sw.media_id(0),)
    assert row.latent_key == "win_000000"


@pytest.mark.parametrize("commit,target", [(2, 4), (1, 1)])
def test_sub_block_modes(session, reference, commit, target):
    """Sub-block generation: the prefix is the paper's; the rest runs every ``commit`` latents with a fresh window."""
    rt = RealtimeConfig(
        generator="fast",
        early_prefill=True,
        decoder="none",
        lockstep=False,
        commit_latents=commit,
        target_latents=target,
    )
    published = []
    cfg = sw.config(
        session["world"], session["weights"], session["tmp"] / "engine", max_blocks=MAX_BLOCKS
    )
    models = EngineModels.load(cfg, rt, attention=sdpa_attention)
    engine = Engine(cfg, rt, models=models, on_message=lambda m: published.append(type(m).__name__))
    engine.start(row=load_round_index_row(cfg.paths.round_index, 0))
    frames, steps = [], 0
    while not engine.finished:
        frames.extend(engine.step())
        steps += 1
    latents = engine.latents[0].float().numpy()
    engine.stop()
    assert np.array_equal(latents[:25], reference[:25])  # the plain prefix is untouched
    assert not np.array_equal(latents[25:29], reference[25:29])
    committed = {(2, 4): 33, (1, 1): 30}[(commit, target)]  # one frame per latent (no decoder)
    assert steps == {(2, 4): 9, (1, 1): 11}[(commit, target)] and len(frames) == committed
    assert published.count("SceneBlockMessage") == (committed - 1) // 4  # whole blocks only
    assert [f.index for f in frames] == list(range(len(frames)))


def test_closed_loop_equals_the_client(session, monkeypatch):
    """own_state='state_model' (player_state.source = predicted, a small random state model): the engine produces
    the release client's latents, with its paper settings and with the exact latency options."""
    from worldcast.config.inference import with_overrides

    tmp = session["tmp"]
    world = sw.make_world(
        tmp / "world_closed", clients=(0,), seconds=11.0
    )  # the state model reads 10 s windows
    sw.patch_ticks(monkeypatch, world["tables"])
    sw.patch_control_ticks(monkeypatch, world["tables"])
    closed = {"player_state.source": "predicted", **sw.write_state_model(tmp / "state_model.pt")}
    out = tmp / "client_closed"
    cfg = with_overrides(
        sw.config(
            world,
            session["weights"],
            out,
            max_blocks=MAX_BLOCKS,
            pool=str(tmp / "client_closed_pool"),
        ),
        closed,
    )
    Client(cfg, attention=sdpa_attention).run()
    reference = np.load(out / "latents.npy")
    models = EngineModels.load(cfg, RealtimeConfig(decoder="none"), attention=sdpa_attention)
    for k, rt in enumerate(
        (
            RealtimeConfig(decoder="none"),
            RealtimeConfig(
                generator="fast",
                early_prefill=True,
                decode_overlap=True,
                decoder="none",
                lockstep=False,
            ),
        )
    ):
        published = []
        engine = Engine(
            cfg, rt, models=models, on_message=lambda m: published.append(type(m).__name__)
        )
        assert engine.own_state_name == "state_model"
        engine.start(row=load_round_index_row(cfg.paths.round_index, 0))
        while not engine.finished:
            list(engine.step())
        latents = engine.latents[0].float().numpy()
        engine.stop()
        assert np.array_equal(latents, reference), f"setting {k}"
        assert published.count("PositionMessage") == 6 + MAX_BLOCKS
