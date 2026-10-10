"""The WorldCast client (Sec. 3.1): one player's generator and state model, resident on its GPU.

:class:`Client` loads the models once. :meth:`Client.start` starts the rollout of one player of
one round; each :meth:`Client.step` generates the next block from its controls and yields the
block's frames as they are decoded. :func:`run_client` runs it on the recorded controls (the
offline client of Table 3). With the default settings, a round's recorded length
(``run.latent_frames``) and the default :class:`ServingOptions` it reproduces the paper's latents
bit for bit.

One generation thread does all model and scene-state work in block order
(:class:`~worldcast.engine.inference.rollout.Rollout`); the frames are decoded latent frame by
latent frame on the caller's thread as it takes them. One client per process, as the denoising
steps draw from the device's global RNG.
"""

import contextlib
import json
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.config.inference import InferenceConfig
from worldcast.data.latents import BLOCK
from worldcast.data.recordings import RoundIndexRow, load_round_index_row
from worldcast.data.window import DataPaths
from worldcast.modeling.wan22.attention import (
    AttentionFn,
    attention_kernel,
    fa3_attention,
    fa3_available,
    flash_attention,
)
from worldcast.player_state import field_builder
from worldcast.sampling.sampler import Sampler
from worldcast.sampling.window import KV_CACHE_LATENTS
from worldcast.scene_state import FollowStats
from worldcast.utils.precision import enable_tf32, generator_dtype

from .controls import BlockControls
from .decode import WanFrameDecoder, pixels_to_uint8_frames
from .directory import DirectoryWorldState
from .fast import FastGenerator
from .loading import ClientModels, load_window
from .rollout import BlockRead, Rollout
from .serving import ServingOptions
from .world_state import DONE_FAILED, DONE_OK, WorldState

__all__ = [
    "BlockRecord",
    "Client",
    "Frame",
    "StepControls",
    "client_world_state",
    "run_client",
]

log = logging.getLogger(__name__)

#: A block's controls as :meth:`Client.step` takes them: the controls, a callable that returns them
#: (called on the generation thread right before the block's denoising), or ``None`` for the
#: recorded ones.
StepControls = BlockControls | Callable[[], BlockControls | None] | None


@dataclass
class Frame:
    """One video frame of a rollout.

    Attributes:
        index (int): its index in the rollout's video; frame 0 is latent frame 0.
        f0 (int): the first latent frame of its block (0 for frame 0).
        image (np.ndarray): ``[H, W, 3]`` uint8; without a decoder the latent ``[48, 24, 42]``
            float32.
        t_ready (float): the ``time.monotonic()`` when it was decoded.
    """

    index: int
    f0: int
    image: np.ndarray
    t_ready: float


@dataclass
class BlockRecord:
    """One block of a rollout.

    Attributes:
        f0 (int): its first latent frame.
        t_controls (float): the ``time.monotonic()`` when its controls were taken, right before its
            denoising.
        t_first_frame (float): the ``time.monotonic()`` when its first frame was decoded.
        read (BlockRead | None): what it read from the shared world state; ``None`` for the first
            six blocks.
    """

    f0: int
    t_controls: float = 0.0
    t_first_frame: float = 0.0
    read: BlockRead | None = None


class Client:
    """A resident WorldCast client (see the module docstring).

    Args:
        cfg (InferenceConfig): ``paths`` and ``run`` (``run.index_row`` is not read);
            ``model.attention`` is the kernel, ``player_state.source`` chooses the GT states or
            the closed loop.
        serving (ServingOptions): how to decode and wait (default: the paper's).
        models (ClientModels | None): loaded weights to share; default: loaded here.

    Attributes:
        records (deque[BlockRecord]): the blocks of the rollout started last.
    """

    def __init__(
        self,
        cfg: InferenceConfig,
        serving: ServingOptions = ServingOptions(),
        *,
        models: ClientModels | None = None,
    ) -> None:
        self.cfg, self.serving = cfg, serving
        self.device = torch.device(cfg.run.device)
        cuda = self.device.type == "cuda"
        self.dtype = generator_dtype(self.device)
        enable_tf32()
        attention = self._attention()  # before the weights load: an unknown name fails here
        if models is None:
            models = ClientModels.load(cfg, serving)
        else:
            models.load_missing(cfg, serving)
        self.models = models
        self.generator = FastGenerator(
            models.generator,
            field_builder(),
            input_dtype=self.dtype,
            attention=attention,
            cuda_graphs=serving.cuda_graphs and cuda,
            compile=serving.compile,
        )
        # one KV cache for every rollout: the CUDA graphs captured in one replay into the next
        self.cache = models.generator.allocate_kv_cache(KV_CACHE_LATENTS)
        bind = {}  # the current CUDA device is per thread: the generation thread takes the client's
        if cuda:
            index = self.device.index
            bind = dict(initializer=torch.cuda.set_device, initargs=(index,))
            if index is None:
                bind["initargs"] = (torch.cuda.current_device(),)
        self.thread = ThreadPoolExecutor(1, "worldcast-gen", **bind)
        self.stream = torch.cuda.Stream(device=self.device) if cuda else None
        self.records: deque[BlockRecord] = deque()
        self._rollout: Rollout | None = None
        self._running: _RunningRollout | None = None

    def _attention(self) -> AttentionFn:
        """The kernel of ``model.attention`` on the client's device: SDPA off CUDA, and
        flash-attention 2 for a FlashAttention-3 that is not installed."""
        kernel = attention_kernel(self.cfg.model.attention, self.device)
        if kernel is not fa3_attention:
            return kernel
        if fa3_available():
            log.info("attention: FlashAttention-3")
            return kernel
        log.warning("FlashAttention-3 is not installed: attention falls back to flash-attention 2")
        return flash_attention

    def start(self, row: RoundIndexRow, *, world_state: WorldState) -> None:
        """Start a rollout: the client of a round-index ``row``.

        Seeds, writes latent frame 0 (the first frame) into the KV cache and decodes it: the first
        :meth:`step` yields frame 0, then the first block's frames.

        Args:
            row (RoundIndexRow): the client's row; its group is the lockstep group.
            world_state (WorldState): the shared world state, which the caller opens and marks
                done (a :class:`DirectoryWorldState`).
        """
        self.stop()
        self.records = deque()
        self._running = _RunningRollout(self, row, world_state)
        self._rollout = self._running.rollout

    def _require_running(self) -> "_RunningRollout":
        if self._running is None:
            raise RuntimeError("no rollout is running: call start() first")
        return self._running

    def _require_started(self) -> Rollout:
        if self._rollout is None:
            raise RuntimeError("no rollout was started: call start() first")
        return self._rollout

    def step(self, controls: StepControls = None) -> Iterator[Frame]:
        """Generate the next block; yield its frames as they are decoded (the first: frame 0 too).

        Args:
            controls (BlockControls | Callable | None): the block's controls; ``None``: the recorded
                ones; a callable is called on the generation thread right before the block's
                denoising.

        Returns:
            Iterator[Frame]: the block's frames, to be taken to the end; nothing once
            :attr:`finished`.
        """
        return self._require_running().step(controls)

    @property
    def finished(self) -> bool:
        """Every block of the rollout has been requested."""
        return self._running is None or self._running.finished

    @property
    def latents(self) -> torch.Tensor:
        """The latents ``[1, N, 48, 24, 42]`` of the rollout started last (``latents.npy`` holds
        them as float32); kept after :meth:`stop`."""
        return self._require_started().output

    @property
    def follow_stats(self) -> FollowStats:
        """How the scene state of the rollout started last followed the other clients'
        withdrawals; kept after :meth:`stop`."""
        return self._require_started().scene.follow_stats

    def stop(self) -> None:
        """End the rollout: wait for its work.

        Raises the error of a job that failed, unless :meth:`step` raised it already.
        """
        running, self._running = self._running, None
        if running is not None:
            running.close()


def client_world_state(cfg: InferenceConfig, media_id: str) -> DirectoryWorldState:
    """The session's world-state directory ``paths.world_state_dir``, as client ``media_id`` sees
    it."""
    return DirectoryWorldState(
        cfg.paths.world_state_dir, client=media_id, poll_s=cfg.world_state.poll_s
    )


def run_client(cfg: InferenceConfig) -> dict[str, Any]:
    """Run the client of ``run.index_row`` on its recorded controls, in lockstep with the round's
    other clients; writes ``latents.npy`` (the output buffer, float32) and ``client.json`` (what
    each block read, the withdrawals followed and the lockstep waits).

    A client that fails marks itself failed in the world state, so the other clients stop with an
    error.

    Args:
        cfg (InferenceConfig): ``paths`` (data, weights, ``world_state_dir``, ``out_dir``) and
            ``run.index_row``.

    Returns:
        dict[str, Any]: ``client.json``: ``media_id``, ``index_row``, ``seed``,
        ``latent_frames``, ``blocks`` (per block that read the scene state: its ``f0`` and its
        :class:`~worldcast.engine.inference.rollout.BlockRead`), ``follow``
        (:class:`~worldcast.scene_state.state.FollowStats`) and the lockstep ``wait``
        (:class:`~worldcast.engine.inference.world_state.WaitStats`).
    """
    cfg.paths.require("round_index", "world_state_dir", "out_dir", *DataPaths.names())
    if cfg.run.index_row is None:
        raise ValueError("run.index_row is not set: one client renders one row of the round index")
    row = load_round_index_row(cfg.paths.round_index, int(cfg.run.index_row))
    world_state = client_world_state(cfg, row.media_id)
    try:
        client = Client(cfg, ServingOptions(decoder="none"))
        client.start(row, world_state=world_state)
        while not client.finished:
            deque(client.step(), maxlen=0)
        client.stop()
    except BaseException as exc:
        world_state.mark_done(status=DONE_FAILED, note=_failure_note(exc))
        raise
    world_state.mark_done(status=DONE_OK)

    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents = client.latents[0].float().cpu().numpy()
    np.save(out_dir / "latents.npy", latents)
    wait = world_state.wait_stats
    summary = dict(
        media_id=row.media_id,
        index_row=cfg.run.index_row,
        seed=cfg.run.seed,
        latent_frames=len(latents),
        blocks=[dict(f0=r.f0, **asdict(r.read)) for r in client.records if r.read is not None],
        follow=asdict(client.follow_stats),
        wait=dict(n_waits=wait.n_waits, seconds_total=round(wait.seconds_total, 3)),
    )
    (out_dir / "client.json").write_text(json.dumps(summary, indent=1))
    return summary


def _failure_note(error: BaseException) -> str:
    """Why a client failed, as its done marker says it."""
    return f"{type(error).__name__}: {error}"


@dataclass
class _Delivery:
    """Latent frames ``first .. first + count - 1`` handed from the generation thread to the
    caller's, with their block's record and, on CUDA, the event that marks them written."""

    first: int
    count: int
    record: BlockRecord
    ready: Any = None


class _RunningRollout:
    """A rollout as the client runs it: its blocks generated on the client's generation thread, its
    frames decoded on the caller's thread."""

    def __init__(self, client: Client, row: RoundIndexRow, world_state: WorldState) -> None:
        cfg = client.cfg
        self.client = client
        self.rollout = Rollout(
            cfg,
            client.models,
            Sampler.create(client.generator, client.cache),
            load_window(cfg, row),
            world_state,
            lockstep=client.serving.lockstep,
            dtype=client.dtype,
        )
        self.decoder = None
        if client.serving.decoder == "wan":
            self.decoder = WanFrameDecoder(client.models.wan_vae)
        self.next_f0 = 1
        self.next_frame = 0
        self.delivered: queue.Queue[_Delivery] = queue.Queue()
        self.jobs: list[Future] = []
        self.failed: BaseException | None = None
        self._lock = threading.Lock()
        self._submit(self._open)

    @property
    def finished(self) -> bool:
        return self.next_f0 + BLOCK > self.rollout.latent_frames

    # ------------------------------------------------------------------- the generation thread
    def _open(self) -> None:
        with torch.no_grad():
            self.rollout.open()
        self.delivered.put(_Delivery(0, 1, BlockRecord(f0=0)))

    def _block(self, f0: int, controls: StepControls) -> None:
        record = BlockRecord(f0=f0)

        def take() -> BlockControls | None:
            record.t_controls = time.monotonic()
            return controls() if callable(controls) else controls

        with torch.no_grad():
            x0 = self.rollout.denoise(f0, take)
            ready = torch.cuda.Event() if self.client.stream is not None else None
            if ready is not None:
                ready.record()
            record.read = self.rollout.commit(f0, x0)
            self.client.records.append(record)
            self.delivered.put(_Delivery(f0, BLOCK, record, ready))
            self.rollout.prepare_next(f0)

    def _submit(self, job: Callable[..., None], *args: Any) -> None:
        future = self.client.thread.submit(job, *args)
        with self._lock:
            self.jobs.append(future)

    def _check(self) -> None:
        """Raise the error of a finished job, once."""
        with self._lock:
            done = [f for f in self.jobs if f.done()]
            self.jobs = [f for f in self.jobs if not f.done()]
        for future in done:
            if future.exception() is not None:
                self.failed = self.failed or future.exception()
                raise future.exception()

    # ----------------------------------------------------------------------- the caller's thread
    def step(self, controls: StepControls) -> Iterator[Frame]:
        if self.finished:
            return iter(())
        f0, self.next_f0 = self.next_f0, self.next_f0 + BLOCK
        self._submit(self._block, f0, controls)
        return self._drain(f0)

    def _drain(self, f0: int) -> Iterator[Frame]:
        """The frames of the blocks delivered up to block ``f0``, decoded on the caller's
        thread."""
        stream, output = self.client.stream, self.rollout.output
        while True:
            try:
                delivery = self.delivered.get(timeout=0.05)
            except queue.Empty:
                self._check()
                continue
            if delivery.ready is not None:
                stream.wait_event(delivery.ready)
            block = output[0, delivery.first :]
            for i in range(delivery.count):
                on_stream = (
                    contextlib.nullcontext() if stream is None else torch.cuda.stream(stream)
                )
                with on_stream, torch.no_grad():
                    if self.decoder is None:
                        images = [block[i].float().cpu().numpy()]
                    else:
                        pixels = self.decoder.decode(block[i : i + 1])
                        images = pixels_to_uint8_frames(pixels)
                t_ready = time.monotonic()
                record = delivery.record
                record.t_first_frame = record.t_first_frame or t_ready
                for image in images:
                    yield Frame(self.next_frame, delivery.first, image, t_ready)
                    self.next_frame += 1
            if delivery.first == f0:
                self._check()
                return

    def close(self) -> None:
        """Wait for the pending jobs; raise a new error."""
        with self._lock:
            jobs, self.jobs = self.jobs, []
        error = next((f.exception() for f in jobs if f.exception() is not None), None)
        if error is not None:
            raise error
