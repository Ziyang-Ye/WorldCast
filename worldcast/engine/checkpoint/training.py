"""Training checkpoints: the Wan2.2 backbone stage 1 starts from, saving and resuming a run.

``<output_dir>/checkpoint_model_<step:06d>/`` holds

* ``model.pt``: ``{"generator": state}``, the online weights (stage 4: also ``"generator_ema"``);
* ``model_ema.pt``: ``{"generator_ema": state}`` (stages 2-3, once the EMA exists);
* ``critic.pt``: ``{"critic": state}`` (stage 4);
* ``rank_<r:05d>.pt``: each rank's optimizer shards, RNG states, data position and EMA shards;
* ``checkpoint.ready.json``: written last; a directory without it is never resumed.

Generator keys are the generator's own names under the ``model.`` root
(:func:`worldcast.modeling.build.read_generator` reads them), so a training checkpoint loads into
:func:`worldcast.modeling.build.load_generator`. Resume is exact and needs the checkpoint's world
size and per-GPU batch.
"""

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldcast.distributed.process_group import DistInfo, barrier
from worldcast.modeling.build import CHECKPOINT_ROOT
from worldcast.utils.weights import read_state_dict

__all__ = [
    "CRITIC_FILE",
    "CRITIC_KEY",
    "EMA_FILE",
    "MODEL_FILE",
    "checkpoint_dir",
    "find_latest_checkpoint",
    "generator_state",
    "is_resumable",
    "load_rank_state",
    "load_wan22_backbone",
    "prune_checkpoints",
    "prune_shards",
    "release_state",
    "save_checkpoint",
]

#: The weight files of a checkpoint directory: the generator, its EMA (stages 2-3) and the critic
#: (stage 4).
MODEL_FILE, EMA_FILE, CRITIC_FILE = "model.pt", "model_ema.pt", "critic.pt"
#: The payload entry of the critic's weights in :data:`CRITIC_FILE`.
CRITIC_KEY = "critic"
#: Written last: a directory without it is never resumed.
_READY = "checkpoint.ready.json"
#: The attribute a trainer's wrapped module keeps its generator under
#: (:class:`worldcast.modeling.wan22.training.TrainingForward`): the prefix of its state's keys.
_GENERATOR_PREFIX = "generator."


def _rank_file(rank: int) -> str:
    return f"rank_{rank:05d}.pt"


def generator_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A trainer module's ``generator.*`` state under the generator's own key names."""
    return {key.removeprefix(_GENERATOR_PREFIX): value for key, value in state.items()}


def release_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A trainer module's ``generator.*`` state -> checkpoint keys ``model.<generator key>``."""
    return {CHECKPOINT_ROOT + key: value for key, value in generator_state(state).items()}


def load_wan22_backbone(wan22_root: str | Path) -> dict[str, torch.Tensor]:
    """The Wan2.2-TI2V-5B backbone of a snapshot directory, the stage-1 initialisation.

    The snapshot's key names are the backbone's names in the generator; the controls are not in it.
    """
    root = Path(wan22_root)
    index = json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        out.update(read_state_dict(root / shard))
    return out


# ================================================================================ save and resume
def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    """The directory of the checkpoint after ``step`` optimizer steps."""
    return Path(output_dir) / f"checkpoint_model_{int(step):06d}"


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(tmp, path)


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    info: DistInfo,
    metadata: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    rank_state: Mapping[str, Any],
    keep: int = 0,
    keep_shards: int = 0,
) -> Path:
    """Write one checkpoint directory (module docstring), the ready marker last, then prune.

    Args:
        output_dir (str | Path): the run's output directory.
        step (int): optimizer steps done.
        info (DistInfo): this process.
        metadata (Mapping[str, Any]): stored in every file with the world size (``step``,
            ``base_seed``, ``stage``, ``run_name``).
        models (Mapping): ``{file name: {payload key: state}}``, rank 0's gathered states.
        rank_state (Mapping[str, Any]): this rank's optimizer shards, RNG, data position and EMA
            shards.
        keep (int): complete checkpoints kept (``checkpoint.keep``; 0: all).
        keep_shards (int): complete checkpoints that keep their ``rank_*.pt`` files
            (``checkpoint.keep_shards``; 0: all).

    Returns:
        Path: the checkpoint directory.
    """
    directory = checkpoint_dir(output_dir, step)
    metadata = {**metadata, "world_size": info.world_size}
    if info.is_main:
        directory.mkdir(parents=True, exist_ok=True)
    barrier()
    if info.is_main:
        for filename, payload in models.items():
            _atomic_save({**metadata, **payload}, directory / filename)
    _atomic_save({**metadata, "rank": info.rank, **rank_state}, directory / _rank_file(info.rank))
    barrier()
    if info.is_main:
        files = [*models, *(_rank_file(rank) for rank in range(info.world_size))]
        ready = {
            **{k: v for k, v in metadata.items() if isinstance(v, (int, float, str, bool))},
            "files": {name: (directory / name).stat().st_size for name in files},
        }
        tmp = directory / (_READY + ".tmp")
        tmp.write_text(json.dumps(ready, indent=2, sort_keys=True))
        os.replace(tmp, directory / _READY)
        prune_checkpoints(output_dir, keep)
        prune_shards(output_dir, keep_shards)
    barrier()
    return directory


def _complete_checkpoints(output_dir: str | Path) -> list[Path]:
    done = [p for p in Path(output_dir).glob("checkpoint_model_*") if (p / _READY).is_file()]
    return sorted(done, key=lambda p: int(p.name.rsplit("_", 1)[1]))


def prune_checkpoints(output_dir: str | Path, keep: int) -> list[Path]:
    """Delete the complete checkpoints older than the newest ``keep`` (``keep <= 0``: none).

    Incomplete directories are left alone. Returns the deleted directories.
    """
    if keep <= 0:
        return []
    old = _complete_checkpoints(output_dir)[:-keep]
    for directory in old:
        # the marker first: a half-deleted directory is never taken for a complete one
        (directory / _READY).unlink()
        shutil.rmtree(directory)
    return old


def prune_shards(output_dir: str | Path, keep_shards: int) -> list[Path]:
    """Delete the ``rank_*.pt`` files of the complete checkpoints older than the newest
    ``keep_shards`` (``keep_shards <= 0``: none); their weights stay. Returns the pruned
    directories."""
    if keep_shards <= 0:
        return []
    pruned = []
    for directory in _complete_checkpoints(output_dir)[:-keep_shards]:
        # rank 0's file first: without it the checkpoint is no longer resumable
        shards = sorted(directory.glob("rank_*.pt"))
        for shard in shards:
            shard.unlink()
        if shards:
            pruned.append(directory)
    return pruned


def is_resumable(directory: str | Path) -> bool:
    """Whether ``directory`` is a complete checkpoint that still holds its resume files."""
    directory = Path(directory)
    return (directory / _READY).is_file() and (directory / _rank_file(0)).is_file()


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """The newest checkpoint under ``output_dir`` a run can resume from (:func:`is_resumable`)."""
    done = [p for p in _complete_checkpoints(output_dir) if is_resumable(p)]
    return done[-1] if done else None


def load_rank_state(directory: str | Path, info: DistInfo) -> dict[str, Any]:
    """This rank's resume file; the world size must be the checkpoint's."""
    ready = json.loads((Path(directory) / _READY).read_text())
    if int(ready["world_size"]) != info.world_size:
        raise RuntimeError(
            f"the checkpoint ran on {ready['world_size']} ranks, this run on {info.world_size}:"
            " the data order and the EMA shards depend on it"
        )
    path = Path(directory) / _rank_file(info.rank)
    # this run's own file: it holds the python and numpy RNG states, which are no tensors
    return torch.load(str(path), map_location="cpu", weights_only=False)
