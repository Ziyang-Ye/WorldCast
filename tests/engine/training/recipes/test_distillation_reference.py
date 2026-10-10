"""The losses and gradients of one generator step and one critic step of stage 4, against reference
numbers.

The trainer of each stage-4 config is built as a run builds it (``build_distillation_parts``; one
process, no process group) on a 3-layer generator with the paper's window layout (41 latents of 48 x
24 x 42): the generator, and a teacher and a critic from two different checkpoints, so that the
distribution matching gradient is not 0. It steps on one synthetic window.

``distillation_reference.json`` holds the numbers: each loss as float64 hex and, per parameter, the
sha256 of its float32 gradient with the gradient's sum and norm. On the platform and torch version
that wrote the file (its ``written_with``) the comparison is exact; elsewhere CPU kernels may round
differently, and the losses, sums and norms are compared with a relative tolerance instead (a change
of the algorithm moves them by far more). After a change meant to move the numbers,
``python -m tests.engine.training.recipes.test_distillation_reference`` rewrites the file.
"""

import hashlib
import json
import platform
import tempfile
from pathlib import Path

import pytest
import torch

from tests.engine.training.support import (
    RUN_DIMS,
    ItemDataset,
    make_item,
    write_init_checkpoint,
    write_prompt_embedding,
)
from worldcast import distributed as D
from worldcast.config.training import paper_config
from worldcast.data.window import collate_windows
from worldcast.engine.training.build import build_distillation_parts
from worldcast.engine.training.recipes.distillation import DistillationTrainer

REFERENCE = Path(__file__).with_name("distillation_reference.json")
STAGES = ("4_noscene", "4")
#: Three layers: the player state field, injected after the second, passes through one more.
DIMS = {"model.dims": {**RUN_DIMS, "num_layers": 3}}
#: The seeds of the checkpoints, of the window and of the two steps' draws.
SEEDS = dict(generator=21, teacher=22, critic=23, window=8, generator_step=77, critic_step=78)
#: Relative tolerance off the platform that wrote the reference.
RTOL = 1e-3


@pytest.fixture(autouse=True)
def restore_deterministic_algorithms():
    """The stage-4 build switches deterministic algorithms on for the process; the rest of the test
    session keeps its own setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled)


def this_platform() -> dict:
    return {
        "torch": torch.__version__.split("+")[0],
        "system": platform.system(),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
    }


def build_trainer(stage: str, tmp: Path) -> DistillationTrainer:
    """The stage's trainer on tiny dimensions, built from checkpoints written under ``tmp``."""
    assert not D.DistInfo().initialized, "the reference is that of a run without a process group"
    checkpoints = {
        "checkpoint.init": (stage, SEEDS["generator"]),
        "distillation.teacher": ("2", SEEDS["teacher"]),
        "distillation.critic": ("2", SEEDS["critic"]),
    }
    settings = {
        **DIMS,
        "model.attention": "sdpa",
        "data.prompt_embedding": str(write_prompt_embedding(tmp / "prompt.safetensors")),
        "data.num_workers": 0,
    }
    for key, (source, seed) in checkpoints.items():
        path = tmp / f"{key}.pt"
        settings[key] = str(write_init_checkpoint(path, paper_config(source, DIMS), seed=seed))
    cfg = paper_config(stage, settings)
    parts = build_distillation_parts(cfg, D.DistInfo(), dataset=ItemDataset(1))
    return DistillationTrainer(cfg, parts)


def _sha256(gradient: torch.Tensor) -> str:
    return hashlib.sha256(gradient.float().contiguous().numpy().tobytes()).hexdigest()


def step_numbers(trainer: DistillationTrainer) -> dict:
    """The loss and the gradients of a generator step, then of a critic step, on one window."""
    scene_state = trainer.cfg.stage.scene_state
    window = make_item(SEEDS["window"], target_start=29 if scene_state else None)
    steps = {
        "generator": (trainer.generator, trainer.generator_loss),
        "critic": (trainer.critic, trainer.critic_loss),
    }
    numbers = {}
    for name, (model, loss_of) in steps.items():
        torch.manual_seed(SEEDS[f"{name}_step"])
        loss = loss_of(collate_windows([window]))
        loss.backward()
        gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        others = [other for other, _ in steps.values() if other is not model]
        assert not any(p.grad is not None for other in others for p in other.parameters())
        numbers[name] = {
            "loss": {
                "dtype": str(loss.dtype).removeprefix("torch."),
                "hex": float(loss.detach()).hex(),
            },
            "gradients": {
                n: {
                    "sha256": _sha256(g),
                    "sum": float(g.double().sum()),
                    "norm": float(g.double().norm()),
                }
                for n, g in sorted(gradients.items())
            },
        }
        model.zero_grad(set_to_none=True)
    return numbers


@pytest.mark.parametrize("stage", STAGES)
def test_a_distillation_step_gives_the_reference_losses_and_gradients(stage, tmp_path):
    reference = json.loads(REFERENCE.read_text())
    exact = reference["written_with"] == this_platform()
    numbers = step_numbers(build_trainer(stage, tmp_path))
    for name, want in reference["stages"][stage].items():
        got = numbers[name]
        # the generator's loss is float64, the critic's float32
        assert got["loss"]["dtype"] == want["loss"]["dtype"], name
        loss, expected = float.fromhex(got["loss"]["hex"]), float.fromhex(want["loss"]["hex"])
        assert expected > 0
        assert sorted(got["gradients"]) == sorted(want["gradients"]), name
        if exact:
            assert got == want, name
            continue
        assert loss == pytest.approx(expected, rel=RTOL), name
        total = sum(g["norm"] ** 2 for g in want["gradients"].values()) ** 0.5
        for parameter, gradient in want["gradients"].items():
            mine = got["gradients"][parameter]
            close = pytest.approx(gradient["norm"], rel=RTOL, abs=RTOL * 1e-3 * total)
            assert mine["norm"] == close, (name, parameter)
            close = pytest.approx(gradient["sum"], rel=RTOL, abs=RTOL * 1e-2 * total)
            assert mine["sum"] == close, (name, parameter)


def write_reference() -> None:
    """Rewrite ``distillation_reference.json`` from the trainer on this platform."""
    torch.set_num_threads(1)
    stages = {}
    for stage in STAGES:
        with tempfile.TemporaryDirectory() as tmp:
            stages[stage] = step_numbers(build_trainer(stage, Path(tmp)))
    reference = {"written_with": this_platform(), "stages": stages}
    REFERENCE.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    write_reference()
