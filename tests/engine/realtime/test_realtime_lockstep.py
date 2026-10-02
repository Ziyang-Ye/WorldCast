"""Two clients in lock-step, one process each (CPU, tiny random model, synthetic round): the engines exchange blocks
through the paper's pool directory and produce the release clients' latents and pool records."""

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MAX_BLOCKS = 2


class _Patch:
    def setattr(self, module, name, value):
        setattr(module, name, value)


def _worker(job: dict) -> list:
    """One client: the release ``Client`` (``kind='client'``) or the engine (``'engine'``)."""
    sys.path[:0] = [str(HERE), str(HERE.parents[2])]
    import torch

    torch.set_num_threads(1)
    import tests.engine.realtime.synthetic_world as sw
    from worldcast.data.index import load_round_index_row

    world = (
        sw.make_world(Path(job["world"]), clients=(0, 3), seed=job["seed"])
        if job.get("build")
        else job["w"]
    )
    sw.patch_ticks(_Patch(), world["tables"])
    cfg = sw.config(
        world,
        job["weights"],
        Path(job["out"]),
        max_blocks=MAX_BLOCKS,
        row=job["row"],
        pool=job["pool"],
    )
    if job["kind"] == "client":
        from worldcast.engine.inference.client import Client
        from worldcast.modeling.wan22.attention import sdpa_attention

        Client(cfg, attention=sdpa_attention).run()
        return np.load(Path(job["out"]) / "latents.npy").tolist()
    from worldcast.engine.realtime.config import RealtimeConfig
    from worldcast.engine.realtime.engine import Engine

    rt = RealtimeConfig(
        generator="fast",
        early_prefill=True,
        decode_overlap=True,
        decoder="none",
        lockstep=job.get("lockstep", True),
        poll_s=0.01,
    )
    engine = Engine(cfg, rt, pool_dir=job["pool"])
    engine.start(row=load_round_index_row(cfg.paths.round_index, job["row"]))
    while not engine.finished:
        list(engine.step(lambda: None))
    latents = engine.latents[0].float().numpy()
    engine.stop()
    return latents.tolist()


def _session(tmp: Path, kind: str, weights: dict, **extra) -> list:
    name = kind + ("" if extra.get("lockstep", True) else "-latest")
    jobs = [
        dict(
            kind=kind,
            world=str(tmp / f"world-{name}"),
            seed=0,
            build=True,
            weights=weights,
            row=k,
            out=str(tmp / name / str(k)),
            pool=str(tmp / name / "pool"),
            **extra,
        )
        for k in range(2)
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as ex:
        return [
            np.asarray(f.result(timeout=900), dtype=np.float32)
            for f in [ex.submit(_worker, j) for j in jobs]
        ]


def test_engines_in_lockstep_equal_the_clients(tmp_path):
    import tests.engine.realtime.synthetic_world as sw

    weights = sw.make_weights(tmp_path / "weights")
    clients = _session(tmp_path, "client", weights)
    engines = _session(tmp_path, "engine", weights)
    for a, b in zip(clients, engines):
        assert a.shape == b.shape and np.array_equal(a, b)

    def blocks(kind, k):
        return sorted(
            p.name for p in (tmp_path / kind / "pool").rglob("blk_*.npy") if f"p0{k}" in str(p)
        )

    for k in (0, 3):
        assert blocks("client", k) == blocks("engine", k)


def test_engines_without_lockstep_run_on_whatever_has_arrived(tmp_path):
    """Latest-message mode: each engine admits the peer blocks that have arrived (timing-dependent output)."""
    import tests.engine.realtime.synthetic_world as sw

    latents = _session(tmp_path, "engine", sw.make_weights(tmp_path / "weights"), lockstep=False)
    assert all(a.shape == (33, 48, 24, 42) and np.isfinite(a).all() for a in latents)
