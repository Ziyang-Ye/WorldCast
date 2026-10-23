"""The client of the generator trained without scene state (``model.scene_state: false``), on the
CPU with the tiny models.

* Its window is the ordinary window of its training: every generator call of a block sits at the
  window positions of the training window (the first frame at 0, the context written clean, block
  by block, the target frames last), and the first denoising step of a block predicts what the
  stage's training forward (teacher forcing on a clean context copy with the field on it) predicts
  for the block's frames in that window: before the window is full (the first blocks, at their
  positions in the round) and after (41 latent frames, the target frames at 37-40).
* Live: two clients play together past the round's slide, with refresh; a refresh on the same
  controls gives the block bit for bit; a client refreshed on other controls and back plays the
  rollout of a client that never refreshed; a round that slides generates what a longer round
  generates; nothing is published as a memory entry.
* Its turns are read ``clip``-encoded, its idle clip is as long as its recent context, its round is
  45 latent frames at the least, and a client refuses a checkpoint of the other generator.
"""

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.data.controls import quantize_camera_delta
from worldcast.data.latents import BLOCK, FIRST_TARGET, video_frame_count
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import run_client
from worldcast.engine.inference.field_rows import FieldRows
from worldcast.engine.inference.idle_context import IdleContext
from worldcast.engine.inference.live_rollout import live_round_latents
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.messages import MessagePart, MessageWorldState
from worldcast.engine.inference.serving import ServingOptions
from worldcast.modeling.wan22.attention import causal_blocks
from worldcast.modeling.wan22.training import TrainingForward
from worldcast.player_state import field_builder
from worldcast.sampling.sampler import CONTEXT_NOISE
from worldcast.sampling.schedulers import FlowMatchScheduler, renoise_frames
from worldcast.sampling.window import KV_CACHE_LATENTS, ORDINARY_RECENT, WindowLayout

NOSCENE = {"model.scene_state": False}
#: The blocks of the offline test: latent frames 1-60, the window full from latent frame 37 on.
OFFLINE_BLOCKS = 15
#: The blocks of the live tests: the round of 45 latent frames slides from the eleventh on.
LIVE_BLOCKS = 13


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    """Tiny weights of the generator trained without scene state, of the other one, a tiny state
    model, a recorded round whose players walk and turn (16 s, one client) and a standing round
    with the clients of slots 0 and 5 (12 s, a seat that shows nobody in its first frame)."""
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("noscene")
    weights = sw.make_weights(tmp / "weights", scene_state=False)
    live = {"player_state.source": "live", **sw.write_state_model(tmp / "state_model.pt")}
    standing = sw.standing_world(tmp / "standing", clients=(0, 5), seconds=16.0)
    sw.empty_first_frame(standing)
    worlds = {
        "moving": sw.make_world(tmp / "moving", clients=(0,), seconds=16.0),
        "standing": standing,
    }
    return dict(tmp=tmp, weights=weights, live=live, worlds=worlds, models={})


def config(room, *, world: str, settings: dict | None = None):
    cfg = sw.config(room["worlds"][world], room["weights"], room["tmp"] / "out", max_blocks=0)
    return cfg.with_overrides({**NOSCENE, **(settings or {})})


def models(room, cfg) -> ClientModels:
    key = cfg.player_state.source
    if key not in room["models"]:
        room["models"][key] = ClientModels.load(cfg, ServingOptions(decoder="none"))
    return room["models"][key]


def make(room, patch, *, world: str, settings: dict | None = None, row: int = 0, relay=None):
    """A rollout of the client of ``row`` of ``world``; under live controls on open ground, its
    state model reading its position where its keys put it."""
    cfg = config(room, world=world, settings=settings)
    sw.patch_ticks(patch, room["worlds"][world]["tables"])
    live = cfg.player_state.source == "live"
    if live:
        sw.patch_extent(patch)
    media = load_round_index_row(cfg.paths.round_index, row).media_id
    world_state = relay.join(media) if relay else MessageWorldState(client=media, poll_s=0.01)
    made = sw.make_rollout(cfg, world_state, models=models(room, cfg), row=row)
    if live:
        sw.own_position_oracle(made)
    return made


def spy_calls(rollout) -> list[dict]:
    """Every call of the generator of ``rollout``: its window position, latent frames, timestep,
    input and flow."""
    calls, generator = [], rollout.sampler.generator

    class Spy:
        input_dtype = generator.input_dtype

        def __call__(self, noisy, timestep, conditions, *, kv_cache, frame_offset):
            flow = generator(
                noisy, timestep, conditions, kv_cache=kv_cache, frame_offset=frame_offset
            )
            calls.append(
                dict(
                    at=(frame_offset, int(noisy.shape[1])),
                    t=float(timestep[0, 0]),
                    noisy=noisy.clone(),
                    flow=flow.clone(),
                )
            )
            return flow

    rollout.sampler.generator = Spy()
    return calls


# ------------------------------------------------------------------------- the ordinary window
def test_the_window_is_the_ordinary_window_of_the_training():
    """Up to block 37 the window is the training window's first latent frames; from there on it is
    the training window, 41 latent frames, written in the blocks of the training's attention."""
    assert ORDINARY_RECENT == 36 and KV_CACHE_LATENTS == 41
    for s in range(FIRST_TARGET, 81, BLOCK):
        layout = WindowLayout.ordinary(s)
        frames = min(s + BLOCK, KV_CACHE_LATENTS)
        assert not layout.with_memory and layout.num_frames == frames
        assert layout.target_positions == list(range(frames - BLOCK, frames))
        assert layout.context_ranges + [(frames - BLOCK, frames)] == causal_blocks(frames)
        assert layout.own_latents(s) == list(range(max(1, s - ORDINARY_RECENT), s + BLOCK))


def offline(room) -> tuple:
    """The offline client of the moving round (its recorded controls and states) for
    :data:`OFFLINE_BLOCKS` blocks: the rollout, the calls of each block's denoising and of its
    context's writes, and the window each block was prepared with."""
    if "offline" not in room:
        with pytest.MonkeyPatch.context() as patch:
            rollout = make(room, patch, world="moving")
            calls = spy_calls(rollout)
            denoised, written, prepared = {}, {}, {}
            for k in range(OFFLINE_BLOCKS):
                f0 = 1 + BLOCK * k
                prepared[f0], mark = rollout.prepared, len(calls)
                x0 = rollout.denoise(f0, lambda: None)
                denoised[f0], mark = calls[mark:], len(calls)
                rollout.commit(f0, x0)
                written[f0], mark = written.get(f0, []) + calls[mark:], len(calls)
                rollout.prepare_next(f0)
                written[f0 + BLOCK] = calls[mark:]
        room["offline"] = rollout, denoised, written, prepared
    return room["offline"]


def test_a_block_is_written_and_denoised_at_the_positions_of_the_training_window(room):
    """The first six blocks run at their positions in the round, each written into the cache clean
    after its denoising; every later block writes its window's context clean, range by range at
    the window positions of the training's blocks, and is denoised after it, at positions up to
    37-40 however long the session; no context write is noised."""
    rollout, denoised, written, _ = offline(room)
    assert rollout.sampler.context_noise == 0
    assert rollout.recording.spec.camera_encoding == "clip"
    steps = [round(float(t), 3) for t in rollout.sampler.denoising_timesteps]
    for k in range(OFFLINE_BLOCKS):
        f0 = 1 + BLOCK * k
        if f0 < FIRST_TARGET:  # its write follows its denoising
            at = f0
            assert [(c["at"], c["t"]) for c in written[f0]] == [((f0, BLOCK), 0.0)]
        else:
            layout = WindowLayout.ordinary(f0)
            at = layout.target_positions[0]
            ranges = [(start, end - start) for start, end in layout.context_ranges]
            assert [c["at"] for c in written[f0]] == ranges
            assert all(c["t"] == 0.0 for c in written[f0])
        assert [c["at"] for c in denoised[f0]] == [(at, BLOCK)] * len(steps)
        assert [round(c["t"], 3) for c in denoised[f0]] == steps
    assert at == KV_CACHE_LATENTS - BLOCK


def window_of(rollout, prepared, f0: int) -> tuple[torch.Tensor, dict]:
    """The clean latents ``[1, F - 4, ...]`` of the context block ``f0`` attended to and the
    conditions of its window: the round's for the first six blocks (their window is the round's
    first latent frames), the prepared window's for the others."""
    if f0 < FIRST_TARGET:
        return rollout.store[None, :f0].float(), rollout.round_conditions
    context = prepared[f0].context
    return prepared[f0].batch["latents"][:, :context].float(), prepared[f0].conditions


@pytest.mark.parametrize("f0", [5, 21, 25, 37, 41, 57])
def test_a_block_predicts_what_the_training_forward_predicts_in_its_window(room, f0):
    """The flow of a block's first denoising step is the stage's training forward on its window:
    teacher forcing, ``[clean context copy | noisy window]``, the context copy clean and the field
    added to it as well (stages 3 and 4 without scene state, as trained), RoPE and the control
    history at the window positions. The rules of the stages with scene state predict otherwise:
    the field off the context copy, or the context noised to 16."""
    rollout, denoised, _, prepared = offline(room)
    context, conditions = window_of(rollout, prepared, f0)
    first = denoised[f0][0]
    frames = context.shape[1] + BLOCK
    noisy = torch.cat([context, first["noisy"]], dim=1)
    timestep = torch.zeros(1, frames)
    timestep[:, -BLOCK:] = first["t"]
    clean = torch.cat([context, torch.zeros_like(first["noisy"])], dim=1)
    # the inputs cast as the client's generator call casts them (float32 on the CPU)
    forward = TrainingForward(rollout.models.generator, field_builder(), input_dtype=torch.float32)
    noised_t = torch.full((1, frames), float(CONTEXT_NOISE))
    noised = renoise_frames(
        clean, noised_t, FlowMatchScheduler(), rng=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        trained = forward(
            noisy, timestep, conditions, context_latents=clean, field_on_context=True
        )[:, -BLOCK:]
        field_off = forward(noisy, timestep, conditions, context_latents=clean)[:, -BLOCK:]
        context_noise = forward(
            torch.cat([noised[:, :-BLOCK], first["noisy"]], dim=1),
            timestep,
            conditions,
            context_latents=noised,
            context_timestep=noised_t,
            field_on_context=True,
        )[:, -BLOCK:]
    torch.testing.assert_close(first["flow"], trained, rtol=0, atol=1e-4)
    for other in (field_off, context_noise):
        assert float((other - first["flow"]).abs().max()) > 1e-3


def test_the_window_of_a_block_before_it_is_full_is_the_rounds_first_latent_frames(room):
    """Up to block 37 the window is the round's latent frames 0 .. s + 3 as they are, the controls
    and players of their video frames included; from block 41 on the first frame and the 36 latent
    frames before the block."""
    rollout, _, _, prepared = offline(room)
    for f0 in range(FIRST_TARGET, 1 + BLOCK * OFFLINE_BLOCKS, BLOCK):
        window = prepared[f0].batch
        latents = list(range(max(1, f0 - ORDINARY_RECENT), f0 + BLOCK))
        assert torch.equal(window["latents"][0, 1:-BLOCK], rollout.store[latents[:-BLOCK]])
        if f0 + BLOCK <= KV_CACHE_LATENTS:
            video = video_frame_count(f0 + BLOCK)
            for key in ("buttons", "view_deltas", "player_states"):
                axis = 2 if key == "player_states" else 1
                whole = rollout.round_batch[key].narrow(axis, 0, video)
                assert torch.equal(window[key], whole), (f0, key)


# ----------------------------------------------------------------------------------- live play
def controls(k: int, version: int = 0):
    """Block ``k``'s controls: walk forward, now and then strafing, turning left; the others strafe
    right turning right (1)."""
    if version == 1:
        return sw.live_controls(k, "move_right", yaw=-30.0, pitch=4.0)
    held = ("forward", "move_left") if k % 5 == 2 else ("forward",)
    return sw.live_controls(k, *held, yaw=12.0 * np.sin(k / 3.0), pitch=2.0 * np.sin(k / 5.0))


def play(room, *, versions: tuple[int, ...], refresh: bool) -> dict:
    """The client of slot 0 beside the client of slot 5, who strafes, cross-wired, for
    :data:`LIVE_BLOCKS` blocks. Each window block of slot 0 is cut on ``controls(k, versions[0])``
    and refreshed on each later version in turn; the first six blocks on the last version.
    Returns every block's x0 of both clients, slot 0's parts, what each block read, the shifts and
    the rows of slot 5 in slot 0's world."""
    settings = {**room["live"], "run.latent_frames": live_round_latents(False)}
    with pytest.MonkeyPatch.context() as patch:
        relay = sw.Relay()
        one = make(
            room,
            patch,
            world="standing",
            relay=relay,
            settings={**settings, "live.refresh": refresh},
        )
        two = make(room, patch, world="standing", relay=relay, row=1, settings=settings)
        sent, x0s, reads, shifts, seen = [], {}, [], [], []
        outbox = one.world_state.outbox
        one.world_state.outbox = lambda part: (sent.append(part.encode()), outbox(part))
        for k in range(LIVE_BLOCKS):
            f0 = 1 + BLOCK * k
            x0s["two", f0] = two.denoise(f0, lambda k=k: sw.live_controls(k, "move_left")).clone()
            two.commit(f0, x0s["two", f0])
            two.prepare_next(f0)
            relay.deliver(two.media_id)
            first = versions[0] if f0 >= FIRST_TARGET else versions[-1]
            x0 = one.denoise(f0, lambda k=k: controls(k, first))
            assert one.can_refresh(f0) == (refresh and f0 >= FIRST_TARGET)
            if one.can_refresh(f0):
                for version in versions[1:]:
                    x0 = one.refresh(f0, lambda k=k, version=version: controls(k, version))
            x0s["one", f0] = x0.clone()
            reads.append(one.commit(f0, x0))
            one.prepare_next(f0)
            shifts.append(one.shift)
            seen.append(one.field_rows(f0 + BLOCK - 1).source[5])
            relay.deliver(one.media_id)
    return dict(x0=x0s, sent=sent, reads=reads, shifts=shifts, seen=seen)


def test_two_clients_play_past_the_slide_and_publish_no_memory_entry(room):
    """The round of 45 latent frames slides after each block from the eleventh on; the windows grow
    to 41 latent frames; the other client is in the world; a client publishes its ``live`` part each
    block, and no memory entry or withdrawal."""
    plain = play(room, versions=(0,), refresh=False)
    assert plain["shifts"] == [0] * 10 + [4, 8, 12]
    frames = [read.window_frames for read in plain["reads"][6:]]
    assert frames == [29, 33, 37, 41, 41, 41, 41]
    assert all(read.entry is None and read.admitted == 0 for read in plain["reads"][6:])
    assert all(source == FieldRows.LIVE for source in plain["seen"])
    kinds = [MessagePart.decode(data).kind for data in plain["sent"]]
    assert set(kinds) == {"live"} and kinds.count("live") == LIVE_BLOCKS
    assert all(bool(torch.isfinite(x0).all()) for x0 in plain["x0"].values())


def test_a_refresh_on_the_same_controls_gives_the_block_bit_for_bit(room):
    """A window block denoised again on the controls of its first cut is its first x0, the RNG
    where the first denoising left it, on either side of the slide; on other controls another."""
    settings = {**room["live"], "run.latent_frames": live_round_latents(False)}
    with pytest.MonkeyPatch.context() as patch:
        live = make(room, patch, world="standing", settings={**settings, "live.refresh": True})
        for k in range(LIVE_BLOCKS):
            f0 = 1 + BLOCK * k
            first = live.denoise(f0, lambda k=k: controls(k)).clone()
            if f0 in (FIRST_TARGET, 45):
                after = torch.get_rng_state()
                assert torch.equal(live.refresh(f0, lambda k=k: controls(k)), first)
                assert torch.equal(torch.get_rng_state(), after)
                assert not torch.equal(live.refresh(f0, lambda k=k: controls(k, 1)), first)
                assert torch.equal(live.refresh(f0, lambda k=k: controls(k)), first)
            live.commit(f0, first)
            live.prepare_next(f0)
        assert live.shift == 12


def test_a_client_refreshed_on_other_controls_and_back_plays_as_one_that_never_refreshed(room):
    """Every block of both clients, and every part slot 0 sends, is the same bits."""
    plain = play(room, versions=(0,), refresh=False)
    refreshed = play(room, versions=(1, 0), refresh=True)
    for key, x0 in plain["x0"].items():
        assert torch.equal(refreshed["x0"][key], x0), key
    assert refreshed["sent"] == plain["sent"]


def test_a_round_that_slides_generates_what_a_longer_round_generates(room):
    """The client alone on a round of 45 latent frames, which slides from its eleventh block, and on
    a round of 61, which slides from its fifteenth: every block is the same bits."""
    x0s = {}
    for latents in (45, 61):
        settings = {**room["live"], "run.latent_frames": latents}
        with pytest.MonkeyPatch.context() as patch:
            live = make(room, patch, world="standing", settings=settings)
            for k in range(16):
                f0 = 1 + BLOCK * k
                x0s[latents, f0] = live.denoise(f0, lambda k=k: controls(k)).clone()
                live.commit(f0, x0s[latents, f0])
                live.prepare_next(f0)
            assert live.shift == {45: 24, 61: 8}[latents]
    for f0 in range(1, 1 + BLOCK * 16, BLOCK):
        assert torch.equal(x0s[45, f0], x0s[61, f0]), f0


def test_the_offline_client_runs_the_closed_loop_without_scene_state(room, tmp_path, monkeypatch):
    """Under predicted player states the client publishes its position and reads the others' before
    every block (none here: it is the round's one client), and writes no memory entry and no step
    record into the world state."""
    paths = {key: value for key, value in room["live"].items() if key.startswith("paths.")}
    cfg = config(room, world="moving").with_overrides(
        {
            **paths,
            "player_state.source": "predicted",
            "run.max_blocks": 2,
            "run.index_row": 0,
            "paths.world_state_dir": str(tmp_path / "world_state"),
            "paths.out_dir": str(tmp_path / "out"),
        }
    )
    sw.patch_ticks(monkeypatch, room["worlds"]["moving"]["tables"])
    summary = run_client(cfg)
    assert [block["window_frames"] for block in summary["blocks"]] == [29, 33]
    assert np.isfinite(np.load(tmp_path / "out" / "latents.npy")).all()
    written = {path.parent.name for path in (tmp_path / "world_state").rglob("*") if path.is_file()}
    assert "state" in written and "steps" not in written  # positions, no step record
    assert not any(name.startswith("win_") for name in written)  # no memory entry


# ------------------------------------------------------------------------- what else differs
def test_a_fast_turn_is_read_clip_encoded(room):
    """A turn of 30 degrees in a video frame is the code 1 of the ``clip`` encoding in the controls
    the generator reads, where the input path gave its ``noclip`` code; slower rows keep theirs."""
    settings = {**room["live"], "run.latent_frames": live_round_latents(False)}
    with pytest.MonkeyPatch.context() as patch:
        live = make(room, patch, world="standing", settings=settings)
        given = sw.live_controls(0, yaw=96.0)
        given.view_deltas[3, 1] = quantize_camera_delta([0.0, 30.0], clip=False)[1]
        assert given.view_deltas[3, 1] > 1.0
        live.denoise(1, lambda: given)
        rows = live.round_batch["view_deltas"][0, 1:17].numpy()
        assert rows[3, 1] == 1.0 and np.array_equal(
            np.delete(rows, 3, 0), np.delete(given.view_deltas, 3, 0)
        )


def test_the_idle_clip_is_as_long_as_the_recent_context_and_a_short_round_is_refused(room):
    settings = {**room["live"], "run.latent_frames": 45, "live.idle_context": "freeze"}
    with pytest.MonkeyPatch.context() as patch:
        live = make(room, patch, world="standing", settings=settings)
        assert isinstance(live.idle_context, IdleContext)
        assert live.idle_context.clip_blocks == ORDINARY_RECENT // BLOCK == 9
        with pytest.raises(ValueError, match="keeps a round of 45 latent frames at the least"):
            make(room, patch, world="standing", settings={**settings, "run.latent_frames": 41})
    assert live_round_latents(True) == 29 and live_round_latents(False) == 45


def test_a_client_refuses_a_checkpoint_of_the_other_generator(room, tmp_path):
    cfg = config(room, world="moving")
    released = sw.make_weights(tmp_path / "released")
    with pytest.raises(ValueError, match="tools/make_noscene_generator.py"):
        ClientModels.load(
            cfg.with_overrides({"paths.checkpoint": released["checkpoint"]}),
            ServingOptions(decoder="none"),
        )
    with pytest.raises(ValueError, match="model.scene_state false"):
        ClientModels.load(
            cfg.with_overrides({"model.scene_state": True}), ServingOptions(decoder="none")
        )
