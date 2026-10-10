"""The client on the CPU with a tiny random model on a synthetic round.

* A client the caller drives gives the latents of the offline client, which runs on the shared
  world state's directory: with lockstep, without (a lone client), and with a decoder streaming the
  frames.
* Streaming: frames arrive in order, latent 0 gives one frame and every block 16.
* Controls: the recorded controls passed explicitly change nothing; other controls change their
  block.
* Two clients in lockstep, one process each: clients taking their controls through a callable, on
  world states their callers own, exchange blocks through the shared world state and produce the
  offline clients' latents and records; without lockstep they run on whatever has arrived."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from worldcast.data.controls import CONTROL_BUTTONS, quantize_camera_delta
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import Client, client_world_state, run_client
from worldcast.engine.inference.controls import BlockControls
from worldcast.engine.inference.loading import ClientModels, load_window
from worldcast.engine.inference.serving import ServingOptions

MAX_BLOCKS = sw.MAX_BLOCKS


def run_driven_client(
    synthetic, serving: ServingOptions, root: Path, *, controls=None, models=None
):
    """One client stepped by the caller on a world state of its own under ``root``."""
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "driven", max_blocks=MAX_BLOCKS
    )
    row = load_round_index_row(cfg.paths.round_index, 0)
    client = Client(cfg, serving, models=models)
    client.start(row, world_state=sw.directory_world_state(root, row.media_id, poll_s=0.01))
    frames, k = [], 0
    while not client.finished:
        frames.extend(client.step(None if controls is None else controls(k)))
        k += 1
    assert list(client.step()) == []  # past the end: nothing more
    client.stop()
    return client.latents[0].float().numpy(), frames, client


@pytest.mark.parametrize("lockstep", [True, False], ids=["lockstep", "latest"])
def test_latents_equal_the_offline_client(synthetic, reference, lockstep, tmp_path):
    latents, frames, client = run_driven_client(
        synthetic, ServingOptions(decoder="none", lockstep=lockstep), tmp_path
    )
    assert reference.shape[0] == 1 + 4 * (6 + MAX_BLOCKS)
    assert np.array_equal(latents, reference)
    assert len(frames) == reference.shape[0]  # decoder 'none': one latent per frame
    # a record per block; the blocks after the first six read the scene state
    assert [record.f0 for record in client.records] == list(range(1, reference.shape[0], 4))
    assert [record.read is not None for record in client.records] == [False] * 6 + [True] * 2


def test_streaming_frames(synthetic, reference, monkeypatch, tiny_vae, tmp_path):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    serving = ServingOptions(decoder="wan", lockstep=False)
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "driven", max_blocks=MAX_BLOCKS
    )
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    models.wan_vae = tiny_vae
    latents, frames, client = run_driven_client(synthetic, serving, tmp_path, models=models)
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


def test_controls(synthetic, reference, monkeypatch, tmp_path):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    serving = ServingOptions(decoder="none")
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "driven", max_blocks=MAX_BLOCKS
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

    latents, _, _ = run_driven_client(
        synthetic, serving, tmp_path / "recorded", controls=recorded, models=models
    )
    assert np.array_equal(latents, reference)

    # other controls: a key and a turn change the block they are taken for, and none before it
    idle, _, _ = run_driven_client(
        synthetic, serving, tmp_path / "idle", controls=_controls(), models=models
    )
    steered, _, _ = run_driven_client(
        synthetic, serving, tmp_path / "steered", controls=_controls(press_at=7), models=models
    )
    assert not np.array_equal(idle[1:5], reference[1:5])  # no input is not the recorded input
    assert np.array_equal(steered[:29], idle[:29])  # blocks before the press: unchanged
    assert not np.array_equal(steered[29:33], idle[29:33])  # block 29 (k = 7) carries it

    # a block is 16 video frames: fewer rows of controls are refused, not broadcast
    client = Client(cfg, serving, models=models)
    row = load_round_index_row(cfg.paths.round_index, 0)
    client.start(row, world_state=sw.directory_world_state(tmp_path / "short", row.media_id))
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
    assert client.finished
    with pytest.raises(RuntimeError, match="no rollout was started"):
        client.latents
    with pytest.raises(RuntimeError, match="no rollout was started"):
        client.follow_stats
    with pytest.raises(RuntimeError, match="no rollout is running: call start\\(\\) first"):
        client.step()
    client.start(row, world_state=client_world_state(cfg, row.media_id))
    frames = list(client.step())
    client.stop()
    # the latents and the counters of the last rollout stay readable once it has stopped
    assert len(frames) == 5 and client.latents.shape == (1, 33, 48, 24, 42)
    assert client.follow_stats.steps_applied == 0
    assert bool(client.latents[0, 1:5].abs().sum() > 0) and client.finished


def test_closed_loop_equals_the_client(closed, tmp_path):
    """player_state.source = predicted (a small random state model): the client the caller
    drives, with lockstep and without, gives the offline client's latents."""
    cfg, reference = closed
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    row = load_round_index_row(cfg.paths.round_index, 0)
    for lockstep in (True, False):
        world_state = sw.directory_world_state(tmp_path / f"lockstep-{lockstep}", row.media_id)
        positions = []
        publish = world_state.publish_position
        world_state.publish_position = lambda *args, _publish=publish: (
            positions.append(args),
            _publish(*args),
        )[1]
        client = Client(cfg, ServingOptions(decoder="none", lockstep=lockstep), models=models)
        client.start(row, world_state=world_state)
        while not client.finished:
            list(client.step())
        latents = client.latents[0].float().numpy()
        client.stop()
        assert np.array_equal(latents, reference), f"lockstep={lockstep}"
        assert len(positions) == 6 + MAX_BLOCKS


class _Patch:
    def setattr(self, module, name, value):
        setattr(module, name, value)


def _worker(job: dict) -> list:
    """One client: the offline ``run_client`` (``kind='offline'``) or a caller-driven
    :class:`Client` (``'driven'``)."""
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


def test_driven_clients_in_lockstep_equal_the_offline_clients(tmp_path):
    weights = sw.make_weights(tmp_path / "weights")
    offline = _session(tmp_path, "offline", weights)
    driven = _session(tmp_path, "driven", weights)
    for a, b in zip(offline, driven):
        assert a.shape == b.shape and np.array_equal(a, b)

    def blocks(kind, k):
        world_state = tmp_path / kind / "world_state"
        return sorted(p.name for p in world_state.rglob("blk_*.npy") if f"p0{k}" in str(p))

    for k in (0, 3):
        assert blocks("offline", k) == blocks("driven", k)


def test_clients_without_lockstep_run_on_whatever_has_arrived(tmp_path):
    """Without lockstep each client admits the other clients' blocks that have arrived."""
    latents = _session(tmp_path, "driven", sw.make_weights(tmp_path / "weights"), lockstep=False)
    assert all(a.shape == (33, 48, 24, 42) and np.isfinite(a).all() for a in latents)
