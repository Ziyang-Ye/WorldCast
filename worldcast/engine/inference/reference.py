"""The reference runs: the fingerprints they recorded, and a client's comparison with them.

A reference run recorded, per client, the fingerprint
(:func:`worldcast.utils.fingerprints.fingerprint`) of its entry noise (``entry_noise``), of its
first frame, latent 0 in bf16 (``first_frame``), of latents 0-24, the first frame and the six blocks
that read no scene state (``latents_0_24``), and of all its latents (``all_latents``). Bit equality
needs the stack of the reference runs (docs/inference.md, "Numerics").
"""

import json
import platform
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.latent_cache import load_first_latent
from worldcast.data.latents import BLOCK, FIRST_TARGET, LATENT_SHAPE
from worldcast.data.recordings import RoundIndexRow, read_round_index
from worldcast.modeling.wan22.attention import flash_attention_backend
from worldcast.sampling.schedulers import entry_noise
from worldcast.utils.fingerprints import fingerprint
from worldcast.utils.precision import GENERATOR_DTYPE

from .client import Client
from .directory import DirectoryWorldState
from .loading import load_window
from .serving import ServingOptions
from .session import rounds_of

__all__ = [
    "Check",
    "ClientCheck",
    "ReferenceRun",
    "check_inputs",
    "entry_noise_fingerprint",
    "environment",
    "first_frame_fingerprint",
    "latent_fingerprints",
    "read_reference_runs",
    "verify_client",
    "verify_latents",
]


@dataclass(frozen=True)
class ReferenceRun:
    """What a reference run recorded for one client: the run's ``latent_frames`` and its
    fingerprints by name (a fingerprint the run did not record is absent)."""

    latent_frames: int
    fingerprints: dict[str, str]


def read_reference_runs(*tables: str | Path) -> dict[str, list[ReferenceRun]]:
    """``{media_id: [run, ...]}`` of fingerprint tables (``examples/manifest.json``,
    ``examples/table3_fingerprints.json``)."""
    runs: dict[str, list[ReferenceRun]] = {}
    for table in tables:
        for case in json.loads(Path(table).read_text())["cases"].values():
            for media_id, recorded in case["fingerprints"].items():
                recorded = {name: value for name, value in recorded.items() if value is not None}
                runs.setdefault(media_id, []).append(ReferenceRun(case["latent_frames"], recorded))
    return runs


def entry_noise_fingerprint(seed: int, latent_frames: int) -> str:
    """The fingerprint of a client's entry noise for a run of ``latent_frames`` latent frames."""
    noise = entry_noise(
        seed, latent_frames, latent_frames, LATENT_SHAPE, device="cpu", dtype=torch.float32
    )
    return fingerprint(noise)


def first_frame_fingerprint(first_frame: torch.Tensor) -> str:
    """The fingerprint of a client's first frame: latent 0 as the generator reads it, in bf16."""
    return fingerprint(first_frame.to(GENERATOR_DTYPE))


def latent_fingerprints(latents: np.ndarray) -> dict[str, str]:
    """``latents_0_24`` and ``all_latents`` of a client's ``latents.npy`` ``[N, 48, 24, 42]``."""
    return {
        "latents_0_24": fingerprint(latents[:FIRST_TARGET]),
        "all_latents": fingerprint(latents),
    }


@dataclass(frozen=True)
class Check:
    """One fingerprint of a run beside the reference run's (``None``: it recorded none)."""

    name: str
    value: str
    reference: str | None

    @property
    def ok(self) -> bool:
        """The fingerprint equals the reference run's, or the run recorded none."""
        return self.reference is None or self.value == self.reference

    def __str__(self) -> str:
        if self.reference is None:
            return f"-      {self.name:24s} {self.value}  (the reference run recorded none)"
        if self.ok:
            return f"MATCH  {self.name:24s} {self.value}"
        return f"DIFF   {self.name:24s} {self.value}  (reference {self.reference})"


def environment() -> dict[str, Any]:
    """What decides the bits: torch, CUDA, the GPU, the attention kernel, TF32."""
    env: dict[str, Any] = dict(
        python=platform.python_version(),
        machine=platform.machine(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        attention=flash_attention_backend(),
        tf32=dict(
            matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32
        ),
    )
    for name in ("flash_attn", "flash_attn_interface"):
        try:
            env[name] = getattr(__import__(name), "__version__", "present")
        except ImportError:
            env[name] = None
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


@dataclass
class ClientCheck:
    """One client run through latent 24 beside its reference runs.

    Attributes:
        checks (list[Check]): the entry noise of every reference run that recorded it, the first
            frame and latents 0-24.
        blocks (dict[str, str]): the fingerprint of every block of latents 1-24 (``"1-4"``, ...):
            between two runs, the first block that differs localises a change.
        latents (np.ndarray): latents 0-24, ``[25, 48, 24, 42]`` float32.
        environment (dict[str, Any]): :func:`environment` as the client set it up.
        seconds (float): the time the six blocks took.
    """

    checks: list[Check]
    blocks: dict[str, str]
    latents: np.ndarray
    environment: dict[str, Any]
    seconds: float


def verify_client(
    cfg: InferenceConfig, row: RoundIndexRow, runs: Sequence[ReferenceRun]
) -> ClientCheck:
    """Run one client through latent 24 and compare it with its reference runs.

    The six blocks of latents 1-24 read no scene state, so the client runs alone: the generator and
    the sampler, on one GPU.

    Args:
        cfg (InferenceConfig): the client's config.
        row (RoundIndexRow): the client's row of the round index.
        runs (Sequence[ReferenceRun]): the client's reference runs, at least one.
    """
    recorded = _first(runs).fingerprints
    checks = [
        Check(
            f"entry_noise ({run.latent_frames} latents)",
            entry_noise_fingerprint(cfg.run.seed, run.latent_frames),
            run.fingerprints["entry_noise"],
        )
        for run in runs
        if "entry_noise" in run.fingerprints
    ]
    # the block the client prepares next must not wait for the round's other clients
    client = Client(cfg, ServingOptions(decoder="none", lockstep=False))
    env = environment()  # TF32 as the client switched it on
    with tempfile.TemporaryDirectory() as alone:  # a world state of its own: nobody else publishes
        world_state = DirectoryWorldState(alone, client=row.media_id, poll_s=cfg.world_state.poll_s)
        client.start(row, world_state=world_state)
        started = time.time()
        for _ in range((FIRST_TARGET - 1) // BLOCK):
            list(client.step())
        seconds = time.time() - started
        first_frame = first_frame_fingerprint(client.latents[0, :1])
        latents = client.latents[0, :FIRST_TARGET].float().cpu().numpy()
        client.stop()
    checks += [
        Check("first_frame", first_frame, recorded.get("first_frame")),
        Check("latents_0_24", fingerprint(latents), recorded.get("latents_0_24")),
    ]
    blocks = {
        f"{s}-{s + BLOCK - 1}": fingerprint(latents[s : s + BLOCK])
        for s in range(1, FIRST_TARGET, BLOCK)
    }
    return ClientCheck(checks, blocks, latents, env, seconds)


def verify_latents(path: str | Path, runs: Sequence[ReferenceRun]) -> list[Check]:
    """Compare a client's ``latents.npy`` with its reference runs: latents 0-24, and all latents
    with the reference run of the same length (none: only latents 0-24 are compared).

    Raises:
        ValueError: the file is not float32 ``[N, 48, 24, 42]``, or ``runs`` is empty.
    """
    recorded = _first(runs).fingerprints
    latents = np.load(path, mmap_mode="r")
    if latents.dtype != np.float32 or latents.shape[1:] != LATENT_SHAPE:
        raise ValueError(f"{path}: {latents.dtype} {latents.shape}, not float32 [N, 48, 24, 42]")
    got = latent_fingerprints(latents)
    same_length = [run.fingerprints for run in runs if run.latent_frames == latents.shape[0]]
    return [
        Check("latents_0_24", got["latents_0_24"], recorded.get("latents_0_24")),
        Check("all_latents", got["all_latents"], (same_length or [{}])[0].get("all_latents")),
    ]


def _first(runs: Sequence[ReferenceRun]) -> ReferenceRun:
    """The first reference run of a client, whose fingerprints of latents 0-24 every run of the
    client shares."""
    if not runs:
        raise ValueError("the client has no reference run")
    return runs[0]


def check_inputs(cfg: InferenceConfig, runs: Mapping[str, Sequence[ReferenceRun]]) -> list[str]:
    """Load the inputs of every client of a round as a client does, on the CPU, and check them.

    Args:
        cfg (InferenceConfig): the data paths and ``run.latent_frames``; its round index is one
            round.
        runs (Mapping[str, Sequence[ReferenceRun]]): the reference runs by media id.

    Returns:
        list[str]: one line per client: its latents, the players recorded, how often another player
        is in view, and that its first frame is the reference run's.

    Raises:
        ValueError: the index is not one round, a client's recording is shorter than
            ``run.latent_frames``, or a first frame is not its reference run's.
    """
    rows = read_round_index(cfg.paths.round_index)
    if len(rounds_of(rows)) != 1:
        raise ValueError(f"{cfg.paths.round_index} is not one round")
    lines = []
    for row in rows:
        window = load_window(cfg, row)
        n = window.spec.latent_frames
        if n != cfg.run.latent_frames:
            raise ValueError(
                f"{row.media_id} covers {n} latent frames, the round runs {cfg.run.latent_frames}"
            )
        recorded = _first(runs.get(row.media_id, ())).fingerprints["first_frame"]
        first_frame = load_first_latent(cfg.paths.latent_cache_root, row.media_id, row.start_frame)
        if first_frame_fingerprint(first_frame) != recorded:
            raise ValueError(f"the first frame of {row.media_id} is not the reference run's")
        visible = window.item.client_visibility.numpy()
        others = np.delete(visible, window.media.player_slot, axis=0)
        lines.append(
            f"{row.media_id}: {n} latents, {len(window.round_slots)} players recorded, "
            f"another player in view in {others.any(axis=0).mean():.0%} of the frames; "
            "the first frame is the reference run's"
        )
    return lines
