"""GPU smoke tests of the training stages: a few steps per stage on the real
weights and data, an exact resume, the distillation update schedule, and a trained checkpoint in the
inference client.

Skipped unless CUDA is present and ``WORLDCAST_TRAIN_SMOKE`` names a YAML file::

    nproc: 8                          # GPUs per run (torchrun --nproc_per_node)
    common:                           # dotted overrides for every stage (data paths, prompt
                                      #   embedding, snapshot)
      data.bucket_dir: /data/buckets
      data.media_index: /data/media_index.jsonl
      data.dataset_root: /data/opencs2_dataset
      data.latent_cache_root: /data/latents
      data.visibility_label_root: /data/vislabels
      data.observer_signal_label_root: /data/obslabels
      data.prompt_embedding: /weights/fixed_prompt_umt5xxl_bf16.safetensors
      model.wan22_root: /weights/Wan2.2-TI2V-5B
    stages:                           # the stages to run and their own overrides (init chain)
      "2":  {checkpoint.init: /ckpt/stage1_long/checkpoint_model_006000/model.pt}
      "2s": {checkpoint.init: /ckpt/stage2/checkpoint_model_020000/model_ema.pt,
             data.collision_meshes: {de_dust2: /meshes/de_dust2_world_collision_complete.glb,
                                     ...}}
      "3":  {checkpoint.init: /ckpt/stage2s/checkpoint_model_005000/model_ema.pt,
             data.collision_meshes: {...}}
      "4":  {checkpoint.init: /ckpt/stage3/checkpoint_model_005000/model_ema.pt,
             distillation.teacher: /ckpt/stage2/checkpoint_model_025000/model_ema.pt,
             distillation.critic: /ckpt/stage2/checkpoint_model_025000/model_ema.pt,
             data.collision_meshes: {...}}
    inference_config: /path/to/inference.yaml   # optional: one round is run
    inference_round_of: 59                      #   (tools/run_session.py) on the trained
                                                #   checkpoint: --round-of 59 on --gpus 0,1,2,
                                                #   37 latents
    inference_gpus: "0,1,2"

Run::

    WORLDCAST_TRAIN_SMOKE=smoke.yaml python -m pytest tests/engine/training -q

Pass conditions: finite losses and gradient norms on every step; with ``stage 4``, a generator
update on the first and sixth step only and a critic update on every step (EMA start lowered to
step 2); a run resumed from the continuous run's step-1 checkpoint has its step-2 loss bit for bit
and its weights within the backward's nondeterminism; the trained checkpoint loads strictly into
``worldcast.modeling.build.load_generator`` and, with ``inference_config``, a client runs on it.
"""

import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from worldcast.config.training import STAGES

REPO = Path(__file__).resolve().parents[3]


def _settings():
    path = os.environ.get("WORLDCAST_TRAIN_SMOKE")
    if not path or not torch.cuda.is_available():
        pytest.skip(
            "GPU smoke tests need CUDA and $WORLDCAST_TRAIN_SMOKE (see the module docstring)"
        )
    import yaml

    return yaml.safe_load(Path(path).read_text())


@pytest.fixture(scope="module")
def smoke():
    return _settings()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _train(smoke, stage: str, output: Path, steps: int, extra=None, resume: str = "auto") -> str:
    import yaml

    sets = {
        **(smoke.get("common") or {}),
        **((smoke.get("stages") or {}).get(stage) or {}),
        "run.output_dir": str(output),
        "run.max_steps": steps,
        "checkpoint.interval": 1,
        **(extra or {}),
    }
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "127.0.0.1",
        "--master_port",
        str(_free_port()),
        "--nproc_per_node",
        str(int(smoke.get("nproc", 1))),
        str(REPO / "tools" / "train.py"),
        "--config",
        str(REPO / "configs" / "train" / f"stage{stage}.yaml"),
        "--resume",
        resume,
    ]
    for key, value in sets.items():
        text = yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("...").strip()
        cmd += ["--set", f"{key}={text}"]
    done = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-8000:]
    return done.stdout


def _metrics(output: Path):
    return [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _stages(smoke):
    return [s for s in STAGES if s in (smoke.get("stages") or {})]


def test_every_configured_stage_trains(smoke, tmp_path):
    stages = _stages(smoke)
    if not stages:
        pytest.skip("no stage configured")
    for stage in stages:
        output = tmp_path / f"stage{stage}"
        distillation = stage.startswith("4")
        # one checkpoint at the end (a 5B stage-4 checkpoint with both optimizers is about 160 GB)
        extra = {"ema.start_step": 2, "checkpoint.interval": 0}
        _train(smoke, stage, output, 6 if distillation else 3, extra=extra)
        rows = _metrics(output)
        assert [r["step"] for r in rows] == list(range(1, len(rows) + 1))
        for row in rows:
            assert all(math.isfinite(v) for k, v in row.items() if isinstance(v, float)), (
                stage,
                row,
            )
        times = {"data_wait_sec", "forward_backward_time_sec", "optimizer_time_sec"}
        assert all({"timestamp", "max_memory_allocated_gib", *times} <= set(r) for r in rows)
        if distillation:  # generator on steps 0 and 5 (logged as 1 and 6), critic every step
            trained = [r["trained_generator"] for r in rows]
            assert trained == [True, False, False, False, False, True]
            assert all("critic_loss" in r and "critic_score_timestep" in r for r in rows)
            assert "generator_loss" in rows[0] and "generator_loss" not in rows[1]
        else:
            assert all("loss" in r and "grad_norm" in r and "lr_backbone" in r for r in rows)
        assert (output / f"checkpoint_model_{len(rows):06d}" / "checkpoint.ready.json").is_file()
        for row in rows:
            values = " ".join(
                f"{k}={v:.6g}" for k, v in sorted(row.items()) if isinstance(v, float)
            )
            print(f"stage {stage}: {values}")
        shutil.rmtree(output)  # the next stage needs the disk


def test_resume_matches_the_continuous_run(smoke, tmp_path):
    """A run resumed from the continuous run's own step-1 checkpoint trains step 2 on the same
    weights, batch and
    noise: its step-2 loss is the continuous run's, bit for bit (the forward is deterministic). Its
    weights after step 2 are the continuous run's within the backward's nondeterminism
    (flash-attention and FSDP reductions are not bit-reproducible on the GPU; AdamW's first steps
    move a weight by about lr whatever its gradient's size, so a gradient near zero can flip it):
    reported, with a 5e-2 relative bound per tensor."""
    stages = _stages(smoke)
    if not stages:
        pytest.skip("no stage configured")
    stage = smoke.get("resume_stage", stages[0])
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    _train(smoke, stage, continuous, 2)
    resumed.mkdir()
    shutil.copytree(
        continuous / "checkpoint_model_000001",
        resumed / "checkpoint_model_000001",
        copy_function=os.link,
    )
    log = _train(smoke, stage, resumed, 2)
    assert "resuming" in log
    rows_a, rows_b = _metrics(continuous), _metrics(resumed)
    assert [r["step"] for r in rows_b] == [2]
    assert rows_b[0]["loss"] == rows_a[1]["loss"], (rows_a[1], rows_b[0])
    a = torch.load(
        str(continuous / "checkpoint_model_000002" / "model.pt"),
        map_location="cpu",
        weights_only=True,
    )["generator"]
    b = torch.load(
        str(resumed / "checkpoint_model_000002" / "model.pt"),
        map_location="cpu",
        weights_only=True,
    )["generator"]
    assert a.keys() == b.keys()
    worst = max(
        float((a[k].float() - b[k].float()).norm() / a[k].float().norm().clamp_min(1e-12))
        for k in a
    )
    print(
        f"stage {stage}: resumed step-2 loss {rows_b[0]['loss']!r}"
        f" = continuous {rows_a[1]['loss']!r};"
        f" weights after step 2, worst relative difference {worst:.3e}"
    )
    assert worst <= 5e-2


def test_trained_checkpoint_in_the_inference_client(smoke, tmp_path):
    stage = next((s for s in ("4", "3") if s in (smoke.get("stages") or {})), None)
    if stage is None:
        pytest.skip("configure stage 3 or 4 to make an inference-compatible checkpoint")
    output = tmp_path / f"stage{stage}"
    _train(smoke, stage, output, 1, extra={"ema.start_step": 0})
    payload = torch.load(
        str(output / "checkpoint_model_000001" / "model.pt"), map_location="cpu", weights_only=True
    )
    state = payload.get("generator_ema") or payload["generator"]
    checkpoint = tmp_path / "trained.pt"
    torch.save({"generator_ema": state, "step": 1}, str(checkpoint))

    from worldcast.modeling.build import load_generator
    from worldcast.modeling.wan22.attention import sdpa_attention
    from worldcast.modeling.wan22.model import GeneratorConfig

    generator = load_generator(
        checkpoint, GeneratorConfig(), device="cuda", attention=sdpa_attention
    )
    assert sum(p.numel() for p in generator.parameters()) == 5_098_162_792
    inference = smoke.get("inference_config")
    if inference:  # one round of the index (the clients run in lockstep), on the trained weights
        out = tmp_path / "session"
        row = int(smoke.get("inference_round_of", 59))
        gpus = str(smoke.get("inference_gpus", "0,1,2"))
        done = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "run_session.py"),
                "--config",
                str(inference),
                "--set",
                f"paths.checkpoint={checkpoint}",
                "--round-of",
                str(row),
                "--gpus",
                gpus,
                "--out-dir",
                str(out),
                "--set",
                "run.latent_frames=37",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        assert done.returncode == 0, done.stderr[-6000:]
        import numpy as np

        produced = sorted(out.glob("*/*/latents.npy"))
        assert produced
        for path in produced:
            latents = np.load(path)
            assert (
                latents.ndim == 4
                and latents.shape[1:] == (48, 24, 42)
                and np.isfinite(latents).all()
            )
