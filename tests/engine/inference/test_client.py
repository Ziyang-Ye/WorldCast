"""The client on the CPU with a tiny random model on a synthetic round.

* A client on messages gives the latents of the offline client, which runs on the shared world
  state's directory: with lockstep, without (a lone client), and with a decoder streaming the
  frames.
* Streaming: frames arrive in order, latent 0 gives one frame and every block 16.
* Decoding: a frame is a latent frame of the output (the paper's rollouts) or of the block's x0
  rounded to the output's dtype (a rollout without a last block, which writes the output again).
* The rows the player state field read, taken on the caller's thread while the client runs, under
  the GT states and the closed loop.
* Controls: the recorded controls passed explicitly change nothing; other controls change their
  block.
* Two clients in lockstep, one process each: clients taking their controls through a callable, on
  world states their callers own, exchange blocks through the shared world state and produce the
  offline clients' latents and records; without lockstep they run on whatever has arrived."""

import multiprocessing
import queue
import threading
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.data.controls import CONTROL_BUTTONS, quantize_camera_delta
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import (
    Client,
    _RunningRollout,
    client_world_state,
    run_client,
)
from worldcast.engine.inference.controls import BlockControls
from worldcast.engine.inference.loading import ClientModels, load_window
from worldcast.engine.inference.messages import MessagePart
from worldcast.engine.inference.serving import ServingOptions
from worldcast.engine.inference.taehv import TinyDecoder

MAX_BLOCKS = sw.MAX_BLOCKS


def run_client_on_messages(synthetic, serving: ServingOptions, *, controls=None, models=None):
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "live", max_blocks=MAX_BLOCKS
    )
    client = Client(cfg, serving, models=models)
    client.start(load_round_index_row(cfg.paths.round_index, 0))
    frames, k = [], 0
    while not client.finished:
        frames.extend(client.step(None if controls is None else controls(k)))
        k += 1
    assert list(client.step()) == []  # past the end: nothing more
    client.stop()
    return client.latents[0].float().numpy(), frames, client


@pytest.mark.parametrize("lockstep", [True, False], ids=["lockstep", "latest"])
def test_latents_equal_the_offline_client(synthetic, reference, lockstep):
    latents, frames, client = run_client_on_messages(
        synthetic, ServingOptions(decoder="none", lockstep=lockstep)
    )
    assert reference.shape[0] == 1 + 4 * (6 + MAX_BLOCKS)
    assert np.array_equal(latents, reference)
    assert len(frames) == reference.shape[0]  # decoder 'none': one latent per frame
    # a record per block; the blocks after the first six read the scene state
    assert [record.f0 for record in client.records] == list(range(1, reference.shape[0], 4))
    assert [record.read is not None for record in client.records] == [False] * 6 + [True] * 2


def test_streaming_frames(synthetic, reference, monkeypatch):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    serving = ServingOptions(decoder="taehv", taehv_path="unused", lockstep=False)
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "live", max_blocks=MAX_BLOCKS
    )
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    torch.manual_seed(0)
    models.tiny = TinyDecoder().eval().requires_grad_(False)
    latents, frames, client = run_client_on_messages(synthetic, serving, models=models)
    assert np.array_equal(latents, reference)
    n = reference.shape[0]
    assert [f.index for f in frames] == list(range(1 + 4 * (n - 1)))
    assert [f.f0 for f in frames[:2]] == [0, 1] and frames[-1].f0 == n - 4
    assert frames[5].image.shape == (384, 672, 3) and frames[5].image.dtype == np.uint8
    assert all(r.t_first_frame >= r.t_controls > 0 for r in client.records)


@pytest.fixture
def closed(synthetic, monkeypatch):
    """The synthetic round under ``player_state.source = predicted`` (a small random state model),
    11 s long (the state model reads 10 s windows): its config and the offline client's latents;
    the round's tick tables are served for the rest of the test."""
    tmp = synthetic["tmp"]
    world = sw.make_world(tmp / "world_closed", clients=(0,), seconds=11.0)
    sw.patch_ticks(monkeypatch, world["tables"])
    overrides = {
        "player_state.source": "predicted",
        **sw.write_state_model(tmp / "state_model.pt"),
    }
    out = tmp / "client_closed"
    cfg = sw.config(
        world,
        synthetic["weights"],
        out,
        max_blocks=MAX_BLOCKS,
        world_state=str(tmp / "world_state_closed"),
    ).with_overrides(overrides)
    if not (out / "latents.npy").exists():
        run_client(cfg)
    return cfg, np.load(out / "latents.npy")


class RoundedOutput:
    """A rollout whose output is bfloat16: it hands out a block's x0 in float32 and keeps it rounded
    in the output, as a rollout on the GPU does."""

    latent_frames = 29

    def __init__(self, open_ended: bool) -> None:
        self.open_ended = open_ended
        self.output = torch.zeros((1, 29, 2, 3, 4), dtype=torch.bfloat16)

    def denoise(self, f0, controls):
        x0 = torch.randn((1, 4, 2, 3, 4), generator=torch.Generator().manual_seed(f0))
        self.output[:, f0 : f0 + 4] = x0
        return x0

    def commit(self, f0, x0):
        return None

    def prepare_next(self, f0):
        pass


@pytest.mark.parametrize("open_ended", [True, False], ids=["open_ended", "ends"])
def test_a_frame_is_the_output_or_the_block_rounded_to_the_outputs_dtype(open_ended):
    """A rollout without a last block rewrites its output with every block, so its frames are
    decoded from the block's x0 rounded to the output's dtype: the same bits the output held, a
    float32 x0 decoded as it is would differ. A rollout that ends has its frames read from the
    output, as the paper's clients do: nothing rides the delivery."""
    rollout = RoundedOutput(open_ended)
    running = object.__new__(_RunningRollout)
    running.client = types.SimpleNamespace(stream=None, records=[])
    running.rollout, running.decoder = rollout, None
    running.delivered, running.next_frame = queue.Queue(), 0
    running.jobs, running._lock = [], threading.Lock()
    running._block(1, None)
    (delivery,) = running.delivered.queue
    written = rollout.output[0, 1:5].float().numpy().copy()
    x0 = rollout.denoise(1, None)  # what the block gave, as float32 numbers
    assert not np.array_equal(x0[0].numpy(), written)  # the rounding changes them
    assert (delivery.latents is None) == (not open_ended)
    if open_ended:
        assert delivery.latents.dtype == torch.bfloat16
        rollout.output.zero_()  # a later block wrote the output again
    frames = list(running._drain(1))
    assert [f.index for f in frames] == [0, 1, 2, 3]
    expected = written if open_ended else rollout.output[0, 1:5].float().numpy()
    for i, frame in enumerate(frames):
        assert np.array_equal(frame.image, expected[i])


@pytest.mark.parametrize("source", ["gt", "predicted"])
def test_the_field_rows_are_read_while_the_client_runs(synthetic, source, request):
    """The page reads the rows of each frame on its own thread while the generation thread
    prepares the next block, which under ``predicted`` writes the rows of the blocks after it: the
    latents stay the offline client's, which reads none, and every row read then is the row at the
    end."""
    if source == "gt":
        reference = request.getfixturevalue("reference")
        cfg = sw.config(
            synthetic["world"],
            synthetic["weights"],
            synthetic["tmp"] / "live",
            max_blocks=MAX_BLOCKS,
        )
    else:
        cfg, reference = request.getfixturevalue("closed")
    row = load_round_index_row(cfg.paths.round_index, 0)
    client = Client(cfg, ServingOptions(decoder="none"))
    client.start(row)
    read = {}
    while not client.finished:
        for frame in client.step():
            read[frame.index] = client.field_rows(frame.index)  # decoder 'none': frame = latent
    client.stop()
    assert np.array_equal(client.latents[0].float().numpy(), reference)
    assert sorted(read) == list(range(33))
    for latent, rows in read.items():
        assert np.array_equal(rows.states, client.field_rows(latent).states)
        assert rows.source.tolist() == [0] + [2] * 9 and rows.status == source
    # the client's own positions: the recorded ones, or the closed loop's
    recorded = load_window(cfg, row).player_frames[0].states[0:129:4, :3]
    own = np.stack([read[f].states[0, :3] for f in range(33)])
    if source == "gt":
        assert np.array_equal(own, recorded)
    else:
        assert np.array_equal(own[0], recorded[0])
        assert not np.any(np.all(own[1:] == recorded[1:], axis=1))


def _controls(press_at=None):
    """Block ``k``'s controls: none; at ``press_at`` attack held and a 25 degree turn."""

    def controls(k):
        buttons, turn = np.zeros((16, 11), np.float32), np.zeros((16, 2), np.float32)
        if k == press_at:
            buttons[:, CONTROL_BUTTONS.index("attack")] = 1.0
            turn[:4, 1] = 25.0 / 4
        deltas = np.stack([quantize_camera_delta(t, clip=False) for t in turn])
        return BlockControls(buttons=buttons, view_deltas=deltas, weapon=np.zeros(16, np.int64))

    return controls


def test_controls(synthetic, reference, monkeypatch):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    serving = ServingOptions(decoder="none")
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "live", max_blocks=MAX_BLOCKS
    )
    models = ClientModels.load(cfg, serving)
    # the recorded controls, passed explicitly, reproduce the recorded run
    item = load_window(cfg, load_round_index_row(cfg.paths.round_index, 0)).item

    def recorded(k):
        return BlockControls(
            buttons=item.buttons[1 + 16 * k : 17 + 16 * k].numpy(),
            view_deltas=item.view_deltas[1 + 16 * k : 17 + 16 * k].numpy(),
            weapon=item.weapon[1 + 16 * k : 17 + 16 * k].numpy(),
        )

    latents, _, _ = run_client_on_messages(synthetic, serving, controls=recorded, models=models)
    assert np.array_equal(latents, reference)

    # live controls: a key and a turn change the block they are taken for, and none before it
    idle, _, _ = run_client_on_messages(synthetic, serving, controls=_controls(), models=models)
    steered, _, _ = run_client_on_messages(
        synthetic, serving, controls=_controls(press_at=7), models=models
    )
    assert not np.array_equal(idle[1:5], reference[1:5])  # no input is not the recorded input
    assert np.array_equal(steered[:29], idle[:29])  # blocks before the press: unchanged
    assert not np.array_equal(steered[29:33], idle[29:33])  # block 29 (k = 7) carries it

    # a block is 16 video frames: fewer rows of controls are refused, not broadcast
    client = Client(cfg, serving, models=models)
    client.start(load_round_index_row(cfg.paths.round_index, 0))
    short = BlockControls(
        buttons=np.zeros((1, 11)), view_deltas=np.zeros((1, 2)), weapon=np.zeros(1)
    )
    with pytest.raises(ValueError, match="16 rows"):
        list(client.step(short))
    client.stop()


def test_a_client_outside_a_rollout(synthetic, monkeypatch, tmp_path):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    cfg = sw.config(
        synthetic["world"],
        synthetic["weights"],
        tmp_path / "out",
        max_blocks=MAX_BLOCKS,
        world_state=str(tmp_path / "world_state"),
    )
    row = load_round_index_row(cfg.paths.round_index, 0)
    client = Client(cfg, ServingOptions(decoder="none"))
    part = MessagePart("another", "done", status="ok")
    assert client.finished
    with pytest.raises(RuntimeError, match="no rollout was started"):
        client.latents
    with pytest.raises(RuntimeError, match="no rollout was started"):
        client.follow_stats
    with pytest.raises(RuntimeError, match="no rollout was started"):
        client.field_rows(0)
    with pytest.raises(RuntimeError, match="no rollout was started"):
        client.camera_pitch(0)
    with pytest.raises(RuntimeError, match="no rollout is running: call start\\(\\) first"):
        client.step()
    with pytest.raises(RuntimeError, match="no rollout is running: call start\\(\\) first"):
        client.receive(part)
    # on the caller's world state the rollout takes no messages
    client.start(row, world_state=client_world_state(cfg, row.media_id))
    with pytest.raises(RuntimeError, match="it takes no messages"):
        client.receive(part)
    frames = list(client.step())
    with pytest.raises(
        ValueError, match="latent frame 5 has not been generated: latent frames 0-4"
    ):
        client.field_rows(5)
    with pytest.raises(
        ValueError, match="video frame 17 has not been generated: video frames 0-16 have"
    ):
        client.camera_pitch(17)
    client.stop()
    # the latents, the counters and the rows of the last rollout stay readable once it has stopped
    assert len(frames) == 5 and client.latents.shape == (1, 33, 48, 24, 42)
    assert client.follow_stats.steps_applied == 0
    assert client.field_rows(4).status == "gt"
    # the own camera's pitch at each video frame: under gt, the recorded one
    recorded = load_window(cfg, row).player_frames[0].states[:17, 4]
    assert [client.camera_pitch(k) for k in range(17)] == pytest.approx(recorded, abs=1e-6)
    assert len(set(recorded.tolist())) == 17  # it turns frame by frame
    assert bool(client.latents[0, 1:5].abs().sum() > 0) and client.finished


def test_closed_loop_equals_the_client(closed):
    """player_state.source = predicted (a small random state model): the client over messages,
    with lockstep and without, gives the offline client's latents."""
    cfg, reference = closed
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    for lockstep in (True, False):
        published = []
        client = Client(
            cfg,
            ServingOptions(decoder="none", lockstep=lockstep),
            models=models,
            on_message=lambda m: published.append(m.kind),
        )
        client.start(load_round_index_row(cfg.paths.round_index, 0))
        while not client.finished:
            list(client.step())
        latents = client.latents[0].float().numpy()
        client.stop()
        assert np.array_equal(latents, reference), f"lockstep={lockstep}"
        assert published.count("position") == 6 + MAX_BLOCKS


class _Patch:
    def setattr(self, module, name, value):
        setattr(module, name, value)


def _worker(job: dict) -> list:
    """One client: the offline ``run_client`` (``kind='offline'``) or a caller-driven
    :class:`Client` (``'live'``)."""
    torch.set_num_threads(1)
    world = sw.make_world(Path(job["world"]), clients=(0, 3), seed=job["seed"])
    sw.patch_ticks(_Patch(), world["tables"])
    cfg = sw.config(
        world,
        job["weights"],
        Path(job["out"]),
        max_blocks=MAX_BLOCKS,
        row=job["row"],
        world_state=job["world_state"],
    )
    if job["kind"] == "offline":
        run_client(cfg)
        return np.load(Path(job["out"]) / "latents.npy").tolist()
    serving = ServingOptions(decoder="none", lockstep=job.get("lockstep", True))
    row = load_round_index_row(cfg.paths.round_index, job["row"])
    world_state = client_world_state(cfg, row.media_id)
    client = Client(cfg, serving)
    client.start(row, world_state=world_state)
    while not client.finished:
        list(client.step(lambda: None))
    latents = client.latents[0].float().numpy()
    client.stop()
    world_state.mark_done()
    return latents.tolist()


def _session(tmp: Path, kind: str, weights: dict, **extra) -> list:
    name = kind + ("" if extra.get("lockstep", True) else "-latest")
    jobs = [
        dict(
            kind=kind,
            world=str(tmp / f"world-{name}"),
            seed=0,
            weights=weights,
            row=k,
            out=str(tmp / name / str(k)),
            world_state=str(tmp / name / "world_state"),
            **extra,
        )
        for k in range(2)
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as ex:
        return [
            np.asarray(f.result(timeout=900), dtype=np.float32)
            for f in [ex.submit(_worker, j) for j in jobs]
        ]


def test_live_clients_in_lockstep_equal_the_offline_clients(tmp_path):
    weights = sw.make_weights(tmp_path / "weights")
    offline = _session(tmp_path, "offline", weights)
    live = _session(tmp_path, "live", weights)
    for a, b in zip(offline, live):
        assert a.shape == b.shape and np.array_equal(a, b)

    def blocks(kind, k):
        world_state = tmp_path / kind / "world_state"
        return sorted(p.name for p in world_state.rglob("blk_*.npy") if f"p0{k}" in str(p))

    for k in (0, 3):
        assert blocks("offline", k) == blocks("live", k)


def test_clients_without_lockstep_run_on_whatever_has_arrived(tmp_path):
    """Without lockstep each client admits the other clients' blocks that have arrived."""
    latents = _session(tmp_path, "live", sw.make_weights(tmp_path / "weights"), lockstep=False)
    assert all(a.shape == (33, 48, 24, 42) and np.isfinite(a).all() for a in latents)
