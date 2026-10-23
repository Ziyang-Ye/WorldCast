"""One client's latency in a paced session: input to photon, controls to first frame, frame rate
(docs/demo.md, "Latency"; ``tools/bench_realtime.py``, ``tools/bench_live.py``).

What is measured is the client: when each block's controls are taken and when its frames are ready.
The inputs are synthetic, evenly spread over the session (:func:`measure_latency`), or a script of
input samples played to a worker's session (:func:`measure_live`), and the path from the worker to
the screen is assumed: :data:`NETWORK_S` to the browser and :data:`DISPLAY_S` from arrival to light.
Of a script that turns, :func:`measure_live` also times how soon the view follows each turn: in the
controls of the frames shown and in their pixels (:func:`view_turn`, :func:`turn_response`).
"""

import asyncio
import functools
import math
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldcast.config.inference import InferenceConfig
from worldcast.data import VIDEO_FRAMES_PER_BLOCK, RoundIndexRow
from worldcast.data.controls import CONTROL_BUTTONS, encode_weapon
from worldcast.data.game import ASPECT_TAN_RATIO, HFOV_DEGREES
from worldcast.data.latents import BLOCK, FIRST_TARGET
from worldcast.engine.inference.client import Client, Frame
from worldcast.engine.inference.decode import MP4_WRITER_KWARGS, write_mp4
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.serving import LOW_LATENCY_ATTENTION, ServingOptions

from .config import DemoConfig, PlayConfig
from .inputs import InputBuffer
from .library import DEFAULT_LOADOUT, Library, RoundStart, Seat
from .pacing import BACKLOG_FRAMES, PACING_MARGIN_S, Pacing

__all__ = [
    "SCRIPTS",
    "SETTINGS",
    "UNMEASURED_BLOCKS",
    "asked_turns",
    "compare_settings",
    "display_times",
    "input_script",
    "input_to_photon",
    "measure_latency",
    "GpuBusy",
    "PagePlayout",
    "measure_live",
    "shown_by_page",
    "shown_by_pacing",
    "stall_details",
    "stats",
    "turn_response",
    "view_gain",
    "view_turn",
]

#: Outside the client (assumed, not measured): one way to the browser on a LAN, and from arrival to
#: light.
NETWORK_S, DISPLAY_S = 0.005, 0.010
#: Blocks after latent 24 left out of the statistics: a session's first window blocks, which may
#: still compile and capture CUDA graphs.
UNMEASURED_BLOCKS = 3
#: Input times the latency is averaged over, evenly spread over the session.
INPUT_SAMPLES = 20_000
#: The settings ``tools/bench_realtime.py`` compares: ``model.attention`` and the fields of
#: :meth:`ServingOptions.low_latency` it replaces. ``exact`` gives the paper's latents; ``fast`` is
#: the demo's default (``torch.compile`` and FlashAttention-3).
SETTINGS: dict[str, tuple[str, dict[str, Any]]] = {
    "exact": ("flash", {"compile": False}),
    "fast": (LOW_LATENCY_ATTENTION, {}),
}


#: The input scripts of :func:`measure_live`: ``walk`` (the movement keys change every 0.3-1.5 s,
#: :data:`WALK_KEYS`), ``orbit`` (a strafe held while the turn changes every 0.3-1.5 s, 20-60
#: degrees a second either way), ``look`` (flicks of the mouse, :data:`FLICK_YAW_DEGREES`, sent as
#: the page sends a mouse's turn: as it moves), ``idle`` (samples without a key or a turn).
SCRIPTS = ("walk", "orbit", "look", "idle")
#: Input samples a script sends per second: the page's on a display of 60 Hz.
SAMPLE_HZ = 60.0
#: How long a script holds one key set or one turn, s (uniform).
SEGMENT_S = (0.3, 1.5)
#: The key sets of ``walk``, each a change from the one before.
WALK_KEYS = (
    (),
    ("forward",),
    ("forward", "move_left"),
    ("forward", "move_right"),
    ("back",),
    ("move_left",),
    ("move_right",),
)
#: The turn rates of ``orbit``, degrees a second (uniform, either sign).
ORBIT_DEGREES_S = (20.0, 60.0)
#: ``look``: a flick of the mouse turns the yaw by FLICK_YAW_DEGREES and the pitch by
#: FLICK_PITCH_DEGREES (uniform, each either sign), the mouse moving it evenly over FLICK_FRAMES
#: frames of the page (0.1 s); a flick starts FLICK_GAP_S after the one before (uniform).
FLICK_YAW_DEGREES = (10.0, 60.0)
FLICK_PITCH_DEGREES = (5.0, 25.0)
FLICK_FRAMES = 6
FLICK_GAP_S = (0.4, 1.5)
#: ``look`` keeps the view within this many degrees of level: a pitch flick that would take the
#: pitch the flicks asked for farther turns the other way.
LOOK_PITCH_DEGREES = 45.0
#: An input event that asks for less turn than this, degrees, is not timed by
#: :func:`turn_response`.
RESPONSE_MIN_DEGREES = 1.0
#: The shares of an input event's turn :func:`turn_response` times the view to.
RESPONSE_LEVELS = (0.5, 0.9)
#: :func:`view_gain` reads an event's turn, or an axis of it, of at least this many degrees.
GAIN_MIN_DEGREES = 3.0
#: :func:`view_turn` reads the rows of a frame above its bottom WEAPON_SHARE (the weapon, which
#: turns with the view), on grids of RESPONSE_GRID_DEGREES a cell, RESPONSE_SIDE_DEGREES either
#: side of the view's middle.
WEAPON_SHARE = 0.35
RESPONSE_GRID_DEGREES = 0.2
RESPONSE_SIDE_DEGREES = 40.0
#: :func:`view_turn` whitens the cross power of the two frames by this power of its magnitude
#: (1: phase correlation, whose peak is too sharp to place between two cells).
WHITENING = 0.5
#: The JPEG quality of the frames :func:`measure_live` sends (``worker.jpeg_quality``).
JPEG_QUALITY = 85
#: A frame on screen this many frame periods or longer is a stall (a held frame).
STALL_PERIODS = 1.5
#: :func:`measure_live`'s steady state leaves out the input events before the first frame of the
#: measured block after this many (what is left of the start of a session).
STEADY_AFTER_BLOCKS = 2
#: The page's playout (``Playout`` in ``static/stream.js``, :class:`PagePlayout`): its period
#: within PAGE_STRETCH of the blocks' frame period; the measured block period and its jitter follow
#: each block's by PAGE_FOLLOW, a block counting at most PAGE_OUTLIER_FRAMES frame periods off; a
#: gap of PAGE_PAUSE_BLOCKS block periods starts over; the display's interval follows each frame's
#: by PAGE_SHOWN_FOLLOW; a frame due within PAGE_SLACK_S is shown at the animation frame.
PAGE_STRETCH = (0.8, 1.25)
PAGE_FOLLOW = 0.2
PAGE_OUTLIER_FRAMES = 2
PAGE_PAUSE_BLOCKS = 2
PAGE_SHOWN_FOLLOW = 0.2
PAGE_SLACK_S = 0.002
#: The page's display: animation frames a second.
DISPLAY_HZ = 60.0
#: :class:`GpuBusy` samples each card's utilization this often, s.
BUSY_SAMPLE_S = 0.1


def stats(values: np.ndarray) -> dict[str, float]:
    """``n``, ``p50`` and ``p95`` of a sample."""
    values = np.asarray(values, dtype=np.float64)
    p50, p95 = np.percentile(values, [50, 95])
    return dict(n=int(values.size), p50=float(p50), p95=float(p95))


def input_to_photon(taken: np.ndarray, photon: np.ndarray) -> dict[str, dict[str, float]]:
    """The latency of inputs at evenly spread times.

    An input reaches the first block whose controls are taken after it and shows when that block's
    first frame is displayed.

    Args:
        taken (np.ndarray): ``[n]`` when each block's controls were taken, s.
        photon (np.ndarray): ``[n]`` when each block's first frame is shown, s.

    Returns:
        dict[str, dict[str, float]]: :func:`stats` of the ``total`` latency and of its
        ``lookahead``, the wait for the next block's controls.
    """
    taken, photon = np.asarray(taken), np.asarray(photon)
    inputs = np.linspace(taken[0], taken[-1], INPUT_SAMPLES)
    block = np.searchsorted(taken, inputs, side="left")
    return dict(total=stats(photon[block] - inputs), lookahead=stats(taken[block] - inputs))


def measure_latency(
    cfg: InferenceConfig,
    models: ClientModels,
    serving: ServingOptions,
    row: RoundIndexRow,
    *,
    fps: float,
) -> dict:
    """One session of a client on its recorded controls, paced as a worker paces it; the measured
    blocks, those after the first :data:`UNMEASURED_BLOCKS` blocks after latent 24, start on a
    fresh display.

    With ``serving.decoder_device`` the next block starts while the frames of this one are
    decoded, as a worker does (at most one block ahead of the decoder); else each block's frames
    are decoded before the next block starts.

    Args:
        cfg (InferenceConfig): the client's config; ``run.max_blocks`` blocks follow latent 24.
        models (ClientModels): its loaded weights.
        serving (ServingOptions): how it is served.
        row (RoundIndexRow): the client's row of the round index.
        fps (float): the frames shown per second (the demo's ``play.fps``).

    Returns:
        dict: ``latents`` ``[N, 48, 24, 42]`` float32, ``seconds``, ``controls_to_first_frame_ms``
        (:func:`stats`), ``input_to_photon_s`` (:func:`input_to_photon`) and ``fps``, the frames
        generated per second, and the stages of a block in :func:`stats` of ms: ``dit_ms`` (its
        controls to its latents), ``first_frame_adder_ms`` (its latents to its first frame),
        ``decode_ms_per_block`` and ``decode_ms_per_latent_frame`` (the decoder's time on it,
        from when it had its latents and the block before was decoded), and ``block_period_ms``
        (between the controls of successive blocks); the statistics leave out the first
        :data:`UNMEASURED_BLOCKS` blocks after latent 24.

    Raises:
        ValueError: fewer than two blocks follow the unmeasured ones.
    """
    client = Client(cfg, serving, models=models)
    client.start(row)
    pacing, shown, t0 = Pacing(fps), {}, time.monotonic()
    block_s = VIDEO_FRAMES_PER_BLOCK / fps
    unmeasured = (FIRST_TARGET - 1) // BLOCK + UNMEASURED_BLOCKS  # with the blocks of latents 1-24
    flight: deque[Future] = deque()

    def show(frames: list[Frame]) -> None:
        block = next(r for r in reversed(list(client.records)) if r.f0 == frames[-1].f0)
        arrivals = [frame.t_ready + NETWORK_S for frame in frames]
        times = pacing.record(block.t_controls, arrivals)
        for frame, t in zip(frames, times):
            shown.setdefault(frame.f0, t)  # when the block's first frame is shown

    steps = 0
    with ThreadPoolExecutor(1) as frames_thread:  # takes a block's frames while the next starts
        while not client.finished:
            if steps == unmeasured:
                # the measured blocks start on a display of their own: the blocks before them, slow
                # and irregular (compilation, CUDA graph captures), leave them neither a backlog of
                # frames nor their critical delays
                while flight:
                    show(flight.popleft().result())
                pacing = Pacing(fps)
            start = pacing.next_start(len(flight) * block_s)
            if start is not None:
                time.sleep(max(0.0, start - time.monotonic()))
            steps += 1
            if serving.decoder_device is None:
                show(list(client.step()))
                continue
            flight.append(frames_thread.submit(list, client.step()))
            while len(flight) >= 2:
                show(flight.popleft().result())
        while flight:
            show(flight.popleft().result())
    seconds = time.monotonic() - t0
    latents = client.latents[0].float().cpu().numpy()
    ordered = list(client.records)
    client.stop()
    measured = [i for i, r in enumerate(ordered) if r.read is not None][UNMEASURED_BLOCKS:]
    if len(measured) < 2:
        raise ValueError(f"the session has {len(measured)} measured blocks; it needs two")
    blocks = [ordered[i] for i in measured]
    decoding = np.array(
        [
            r.t_last_frame - max(r.t_latents, ordered[i - 1].t_last_frame)
            for i, r in zip(measured, blocks)
        ]
    )
    taken = np.array([record.t_controls for record in blocks])
    first_frame = np.array([record.t_first_frame for record in blocks])
    photon = np.array([shown[record.f0] for record in blocks]) + DISPLAY_S
    return dict(
        latents=latents,
        seconds=seconds,
        controls_to_first_frame_ms=stats(1e3 * (first_frame - taken)),
        input_to_photon_s=input_to_photon(taken, photon),
        fps=VIDEO_FRAMES_PER_BLOCK / float(np.diff(first_frame).mean()),
        dit_ms=stats(1e3 * np.array([r.t_latents - r.t_controls for r in blocks])),
        first_frame_adder_ms=stats(1e3 * np.array([r.t_first_frame - r.t_latents for r in blocks])),
        decode_ms_per_block=stats(1e3 * decoding),
        decode_ms_per_latent_frame=stats(1e3 * decoding / BLOCK),
        block_period_ms=stats(1e3 * np.diff(taken)),
    )


def compare_settings(
    cfg: InferenceConfig,
    row: RoundIndexRow,
    taehv_path: str,
    names: Sequence[str] = tuple(SETTINGS),
    *,
    fps: float,
    decoder: str = "taehv",
    decoder_device: str | None = None,
    decoder_channels_last: bool = False,
) -> Iterator[tuple[str, dict]]:
    """Measure one client under each of the named :data:`SETTINGS`, on one set of loaded weights.

    Args:
        cfg (InferenceConfig): the client's config (:func:`measure_latency`).
        row (RoundIndexRow): the client's row of the round index.
        taehv_path (str | None): ``taew2_2.pth``, the tiny decoder's weights.
        names (Sequence[str]): the settings to run, in order.
        fps (float): the frames shown per second.
        decoder (str): ``taehv`` (the demo's) or ``wan`` (the Wan2.2 VAE, the paper's frames).
        decoder_device (str | None): where the decoder runs
            (:attr:`ServingOptions.decoder_device`).
        decoder_channels_last (bool): the Wan2.2 VAE in the channels-last layout
            (:attr:`ServingOptions.decoder_channels_last`).

    Yields:
        tuple[str, dict]: a setting's name and its :func:`measure_latency` result, where
        ``bit_identical_to_first`` (whether the latents equal those of the first setting) replaces
        ``latents``.
    """
    serving = {
        name: ServingOptions.low_latency(
            taehv_path,
            **SETTINGS[name][1],
            decoder=decoder,
            decoder_device=decoder_device,
            decoder_channels_last=decoder_channels_last,
        )
        for name in names
    }
    models = ClientModels.load(cfg, serving[names[0]])
    reference = None
    for name in names:
        setting = cfg.with_overrides({"model.attention": SETTINGS[name][0]})
        entry = measure_latency(setting, models, serving[name], row, fps=fps)
        latents = entry.pop("latents")
        reference = latents if reference is None else reference
        entry["bit_identical_to_first"] = bool(np.array_equal(latents, reference))
        yield name, entry


def percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """``n``, ``p50``, ``p90`` and ``p99`` of a sample (``None`` of an empty one)."""
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(n=0, p50=None, p90=None, p99=None)
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    return dict(n=int(values.size), p50=float(p50), p90=float(p90), p99=float(p99))


def input_script(
    name: str, seconds: float, seed: int
) -> list[tuple[float, int, float, float, bool]]:
    """The input samples of a script (:data:`SCRIPTS`), :data:`SAMPLE_HZ` a second for
    ``seconds``, drawn from ``seed``.

    Returns:
        list[tuple[float, int, float, float, bool]]: per sample, when it is sent (s from the
        start), its held buttons (bits of :data:`~worldcast.data.controls.CONTROL_BUTTONS`), the
        pitch and yaw turned since the sample before (degrees), and whether it is an input event:
        the first sample of a new key set or turn, or of a flick.

    Raises:
        ValueError: an unknown script.
    """
    return _scripted(name, seconds, seed)[0]


def asked_turns(name: str, seconds: float, seed: int) -> list[tuple[float, float]]:
    """Per input event of a script (:func:`input_script`), the turn the player asks for with it,
    ``(pitch, yaw)`` degrees: a flick's (``look``), else the turn of the samples from the event to
    the next one (``orbit``: the segment's; ``walk``: none).

    Raises:
        ValueError: an unknown script.
    """
    return _scripted(name, seconds, seed)[1]


def _scripted(
    name: str, seconds: float, seed: int
) -> tuple[list[tuple[float, int, float, float, bool]], list[tuple[float, float]]]:
    """:func:`input_script` and :func:`asked_turns` of a script, from one draw."""
    if name not in SCRIPTS:
        raise ValueError(f"no input script {name!r}; the scripts are {', '.join(SCRIPTS)}")
    rng = np.random.default_rng(seed)
    if name == "look":
        return _look(rng, int(seconds * SAMPLE_HZ))
    samples, keys, rate, ends = [], 0, (0.0, 0.0), 0.0
    for i in range(int(seconds * SAMPLE_HZ)):
        at, event = i / SAMPLE_HZ, False
        if name != "idle" and at >= ends:
            ends, event = at + rng.uniform(*SEGMENT_S), True
            if name == "walk":
                held = [k for k in WALK_KEYS if _bits(k) != keys]
                keys = _bits(held[rng.integers(len(held))])
            else:
                keys = _bits(("move_left",))
                yaw = rng.uniform(*ORBIT_DEGREES_S) * rng.choice((-1.0, 1.0))
                rate = (rng.uniform(-3.0, 3.0), yaw)
        samples.append((at, keys, rate[0] / SAMPLE_HZ, rate[1] / SAMPLE_HZ, event))
    asked: list[list[float]] = []
    for _, _, dpitch, dyaw, event in samples:
        if event:
            asked.append([0.0, 0.0])
        if asked:
            asked[-1][0] += dpitch
            asked[-1][1] += dyaw
    return samples, [(pitch, yaw) for pitch, yaw in asked]


def _look(
    rng: np.random.Generator, count: int
) -> tuple[list[tuple[float, int, float, float, bool]], list[tuple[float, float]]]:
    """``look``'s ``count`` samples: the page's frames at :data:`SAMPLE_HZ`, each sending the
    mouse's share of the flick under way, as the page sends a mouse's turn."""
    samples, asked = [], []
    pitch, start, moving = 0.0, 0.0, (0.0, 0.0, 0)
    for i in range(count):
        at, event = i / SAMPLE_HZ, False
        if at >= start:
            yaw = float(rng.uniform(*FLICK_YAW_DEGREES) * rng.choice((-1.0, 1.0)))
            flick = float(rng.uniform(*FLICK_PITCH_DEGREES) * rng.choice((-1.0, 1.0)))
            if abs(pitch + flick) > LOOK_PITCH_DEGREES:
                flick = -flick
            pitch += flick
            asked.append((flick, yaw))
            moving, event = (flick / FLICK_FRAMES, yaw / FLICK_FRAMES, FLICK_FRAMES), True
            start = at + rng.uniform(*FLICK_GAP_S)
        dp, dy, left = moving
        samples.append((at, 0, dp if left else 0.0, dy if left else 0.0, event))
        moving = (dp, dy, max(left - 1, 0))
    return samples, asked


def _bits(names: Sequence[str]) -> int:
    return sum(1 << CONTROL_BUTTONS.index(name) for name in names)


@functools.cache
def _view_rays(yaw: bool) -> np.ndarray:
    """The rays of a grid of :func:`view_turn`, ``[rows, columns, 3]`` unit vectors in the
    camera's right, up and forward axes. On the yaw's grid a cell is a longitude (the angle about
    the camera's up axis) and a latitude, so that a turn of the yaw moves the view along its rows
    by the angle; on the pitch's, the angle about the camera's right axis and the angle off the
    plane it turns in, so that a turn of the pitch moves the view along its columns by the angle.
    Each spans what the frame holds above the weapon over :data:`RESPONSE_SIDE_DEGREES` either
    side."""
    tan_up = math.tan(math.radians(HFOV_DEGREES / 2)) * ASPECT_TAN_RATIO
    low = 1 - 2 * (1 - WEAPON_SHARE)  # the weapon's top edge, -1..1 over the height, up
    side, cell = math.radians(RESPONSE_SIDE_DEGREES), math.radians(RESPONSE_GRID_DEGREES)
    spread = math.cos(side) if yaw else 1.0  # how far the frame's edges hold the grid's rows
    across, along = np.meshgrid(
        np.arange(-side, side, cell),
        np.arange(math.atan(tan_up * spread), math.atan(low * tan_up * spread), -cell),
    )
    if yaw:  # longitude, latitude
        return np.stack(
            [np.sin(across) * np.cos(along), np.sin(along), np.cos(across) * np.cos(along)], -1
        )
    return np.stack(  # off the plane of the pitch, about the right axis
        [np.sin(across), np.cos(across) * np.sin(along), np.cos(across) * np.cos(along)], -1
    )


def _turned_rays(rays: np.ndarray, pitch: float, yaw: float) -> np.ndarray:
    """Where ``rays`` ``[..., 3]`` of a camera point after it turned ``yaw`` and then ``pitch``
    degrees (as the controls turn): a turn left moves them right, a turn down moves them up."""
    p, y = math.radians(pitch), math.radians(yaw)
    turn_yaw = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    turn_pitch = np.array(
        [[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]]
    )
    return rays @ (turn_pitch @ turn_yaw).T


def _sampled(gray: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """``gray`` ``[H, W]`` where ``rays`` ``[rows, columns, 3]`` meet the frame (bilinear, the
    frame's edge beyond it)."""
    tan_across = math.tan(math.radians(HFOV_DEGREES / 2))
    tan_up = tan_across * ASPECT_TAN_RATIO
    height, width = gray.shape
    rows = (1 - rays[..., 1] / rays[..., 2] / tan_up) * height / 2 - 0.5
    columns = (1 + rays[..., 0] / rays[..., 2] / tan_across) * width / 2 - 0.5
    rows, columns = np.clip(rows, 0, height - 1), np.clip(columns, 0, width - 1)
    r0, c0 = np.floor(rows).astype(np.int64), np.floor(columns).astype(np.int64)
    fr, fc = rows - r0, columns - c0
    r1, c1 = np.minimum(r0 + 1, height - 1), np.minimum(c0 + 1, width - 1)
    top = gray[r0, c0] * (1 - fc) + gray[r0, c1] * fc
    bottom = gray[r1, c0] * (1 - fc) + gray[r1, c1] * fc
    return top * (1 - fr) + bottom * fr


def _shift(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """How far ``after`` lies moved from ``before`` (``[rows, columns]`` each), cells ``(down,
    right)``: the peak of their cross-correlation, whitened by :data:`WHITENING`, under a Hann
    window."""
    window = np.outer(np.hanning(before.shape[0]), np.hanning(before.shape[1]))
    spectra = [np.fft.fft2((grid - grid.mean()) * window) for grid in (before, after)]
    power = spectra[1] * np.conj(spectra[0])
    power /= np.abs(power) ** WHITENING + 1e-12
    correlation = np.fft.ifft2(power).real
    rows, columns = correlation.shape
    r, c = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak = correlation[r, c]
    down = _peak_offset(peak, correlation[(r - 1) % rows, c], correlation[(r + 1) % rows, c])
    right = _peak_offset(peak, correlation[r, (c - 1) % columns], correlation[r, (c + 1) % columns])
    return (r + rows // 2) % rows - rows // 2 + down, (
        c + columns // 2
    ) % columns - columns // 2 + right


def _peak_offset(centre: float, before: float, after: float) -> float:
    """Where a peak lies between its samples, cells from the centre one: a Gaussian through the
    three (a parabola where one of them is not positive)."""
    if min(centre, before, after) > 0:
        centre, before, after = math.log(centre), math.log(before), math.log(after)
    curve = before - 2 * centre + after
    return 0.0 if curve == 0 else 0.5 * (before - after) / curve


def view_turn(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """How far the view turned from one frame to the next, estimated from their pixels.

    Both frames are read above the weapon (:data:`WEAPON_SHARE`, which turns with the view) on
    two grids of :data:`RESPONSE_GRID_DEGREES` a cell over the camera's field of view
    (:data:`~worldcast.data.game.HFOV_DEGREES`): on one a turn of the yaw moves the view sideways
    by the angle, on the other a turn of the pitch moves it up or down by it (:func:`_view_rays`).
    Each turn is the shift of the peak of the cross-correlation on its grid (:func:`_shift`):
    the yaw's, then the pitch's with the next frame read turned back by that yaw, then the yaw's
    again with it read turned back by that pitch.

    Args:
        before (np.ndarray): ``[H, W, 3]`` uint8, the frame shown first.
        after (np.ndarray): ``[H, W, 3]`` uint8, the frame shown next.

    Returns:
        tuple[float, float]: ``(pitch, yaw)`` degrees, as the controls turn (pitch > 0 looks
        down: the picture moves up; yaw > 0 turns left: the picture moves right).
    """
    luma = np.array([0.299, 0.587, 0.114], np.float32)
    first, second = (frame.astype(np.float32) @ luma for frame in (before, after))
    across, down = _view_rays(yaw=True), _view_rays(yaw=False)
    seen = _sampled(first, across)
    yaw = RESPONSE_GRID_DEGREES * _shift(seen, _sampled(second, across))[1]
    back = _sampled(second, _turned_rays(down, 0.0, yaw))
    pitch = -RESPONSE_GRID_DEGREES * _shift(_sampled(first, down), back)[0]
    back = _sampled(second, _turned_rays(across, pitch, 0.0))
    yaw = RESPONSE_GRID_DEGREES * _shift(seen, back)[1]
    return float(pitch), float(yaw)


def turn_response(
    light: np.ndarray,
    turned: np.ndarray,
    events: Sequence[float],
    asked: Sequence[tuple[float, float]],
) -> dict:
    """How soon the frames shown follow each input event that turns.

    From an event on, the view's turn since the frame on screen at the event (the last one lit
    by then; the first one shown, if none was) is projected onto the turn the event asks for, and
    the event's time to each of :data:`RESPONSE_LEVELS` is the first frame that reaches that share
    of it, lit before the next event. An event that asks for less than
    :data:`RESPONSE_MIN_DEGREES` is not timed.

    Args:
        light (np.ndarray): ``[n]`` when each frame shown lights, s, increasing.
        turned (np.ndarray): ``[n, 2]`` how far the view of each has turned, ``(pitch, yaw)``
            degrees, from any origin.
        events (Sequence[float]): when each input event was taken, s, increasing.
        asked (Sequence[tuple[float, float]]): the turn each asks for, ``(pitch, yaw)`` degrees.

    Returns:
        dict: ``events`` (those timed), per level ``t50_s`` / ``t90_s`` (:func:`percentiles` of
        the times of the events that reached it) and ``reached_50`` / ``reached_90`` (the share
        of the events timed that reached it before the next event; ``None`` of none).
    """
    light = np.asarray(light, np.float64)
    turned = np.asarray(turned, np.float64).reshape(-1, 2)
    times: dict[float, list[float]] = {level: [] for level in RESPONSE_LEVELS}
    timed = 0
    for i, (at, goal) in enumerate(zip(events, asked)):
        goal = np.asarray(goal, np.float64)
        size = float(goal @ goal)
        if not len(light) or size < RESPONSE_MIN_DEGREES**2:
            continue
        timed += 1
        until = events[i + 1] if i + 1 < len(events) else math.inf
        on_screen = max(int(np.searchsorted(light, at, side="right")) - 1, 0)
        after = np.flatnonzero((light > at) & (light < until))
        progress = (turned[after] - turned[on_screen]) @ goal / size
        for level in RESPONSE_LEVELS:
            reached = np.flatnonzero(progress >= level)
            if len(reached):
                times[level].append(float(light[after[reached[0]]] - at))
    out: dict[str, Any] = {"events": timed}
    for level in RESPONSE_LEVELS:
        name = round(100 * level)
        out[f"t{name}_s"] = percentiles(times[level])
        out[f"reached_{name}"] = len(times[level]) / timed if timed else None
    return out


def view_gain(
    light: np.ndarray, viewed: np.ndarray, held: np.ndarray, events: Sequence[float]
) -> dict:
    """How far the view turns for the turn the controls of its frames hold: per input event, from
    the frame on screen at it (the last one lit by then) to the last one lit before the next event,
    the view's turn over the controls'.

    Args:
        light (np.ndarray): ``[n]`` when each frame shown lights, s, increasing.
        viewed (np.ndarray): ``[n, 2]`` how far the view of each has turned (its pixels),
            ``(pitch, yaw)`` degrees, from any origin.
        held (np.ndarray): ``[n, 2]`` how far its controls say the player has turned (the turn
            made: a gain the generator is fed is not in it), degrees.
        events (Sequence[float]): when each input event was taken, s, increasing.

    Returns:
        dict: :func:`percentiles` of the ratio of each event whose controls turned at least
        :data:`GAIN_MIN_DEGREES`: ``turn``, the view's turn projected on the controls'; ``pitch``
        and ``yaw``, of an axis that turned that much.
    """
    light = np.asarray(light, np.float64)
    viewed = np.asarray(viewed, np.float64).reshape(-1, 2)
    held = np.asarray(held, np.float64).reshape(-1, 2)
    ratios: dict[str, list[float]] = {"turn": [], "pitch": [], "yaw": []}
    for i, at in enumerate(events):
        until = events[i + 1] if i + 1 < len(events) else math.inf
        on_screen = int(np.searchsorted(light, at, side="right")) - 1
        last = int(np.searchsorted(light, until, side="left")) - 1
        if on_screen < 0 or last <= on_screen:
            continue
        view, command = viewed[last] - viewed[on_screen], held[last] - held[on_screen]
        size = float(command @ command)
        if size >= GAIN_MIN_DEGREES**2:
            ratios["turn"].append(float(view @ command / size))
        for axis, name in enumerate(("pitch", "yaw")):
            if abs(command[axis]) >= GAIN_MIN_DEGREES:
                ratios[name].append(float(view[axis] / command[axis]))
    return {name: percentiles(values) for name, values in ratios.items()}


def display_times(arrivals: Sequence[float], firsts: Sequence[bool], fps: float) -> np.ndarray:
    """When the pacing's display (:class:`~worldcast.demo.pacing.Pacing`) shows frames that arrive
    at ``arrivals``, in order, a block starting at each frame of ``firsts``."""
    pacing, shown, block = Pacing(fps), [], []
    for arrival, first in zip(arrivals, firsts):
        if first and block:
            shown += pacing.record(0.0, block)
            block = []
        block.append(arrival)
    if block:
        shown += pacing.record(0.0, block)
    return np.asarray(shown)


def _seam(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    """The mean absolute difference of two uint8 frames and their PSNR, dB (inf: the same)."""
    difference = before.astype(np.float32) - after.astype(np.float32)
    mse = float(np.mean(difference**2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    return float(np.mean(np.abs(difference))), psnr


@dataclass
class _Sent:
    """A frame the worker sent: its index, when it went, whether it is its block's first, the
    newest input sample and the cut of its version, how far its row says the player has turned, a
    preview's or not, its version, its pixels, whether it went again (``play.swap_sent``)."""

    index: int
    at: float
    first: bool
    seq: int
    taken: float
    turned: tuple[float, float]
    preview: bool
    version: int
    rgb: np.ndarray
    again: bool

    @property
    def rank(self) -> int:
        """Its rank among the versions of its frame (``rank`` in ``static/stream.js``)."""
        return 2 * self.version + (0 if self.preview else 1)


def shown_by_pacing(sent: Sequence[tuple], fps: float) -> tuple[list[int], np.ndarray]:
    """The frames the pacing's display shows (:func:`display_times`) of frames sent ``(index,
    when, its block's first, ...)`` in order, each arriving :data:`NETWORK_S` later: a frame shows
    in the slot of the first of its index sent, and a frame of an index sent again (a newer
    version, ``play.swap_sent``) takes its place if it arrives by then.

    Returns:
        tuple[list[int], np.ndarray]: per frame shown, in order, the one of ``sent`` shown, and
        when.
    """
    slots: dict[int, int] = {}
    for i, (k, *_) in enumerate(sent):
        slots.setdefault(k, i)
    order = sorted(slots.values())
    shown_at = display_times(
        [sent[i][1] + NETWORK_S for i in order], [sent[i][2] for i in order], fps
    )
    shown = {i: i for i in order}
    when = dict(zip(order, shown_at))
    for i, (k, at, *_) in enumerate(sent):
        if slots[k] != i and at + NETWORK_S <= when[slots[k]]:
            shown[slots[k]] = i
    return [shown[i] for i in order], np.asarray(shown_at)


class PagePlayout:
    """The page's playout, ``Playout`` in ``static/stream.js``, rule for rule, in seconds: when the
    page shows which frame it received (docs/demo.md, "Latency"). A frame at or before the frame
    shown is dropped; a frame received again, or sent again, takes the place of the one waiting if
    it ranks above it, and is dropped otherwise or with none waiting.

    Args:
        period (float): the frame period at ``play.fps``, s.
        per_block (int): the frames of a block.
        buffer (int): the frames held before the first is shown (``play.jitter_buffer_frames``).
        backlog (int): :data:`~worldcast.demo.pacing.BACKLOG_FRAMES`.
        margin (float): ``play.margin_frames``.
    """

    def __init__(
        self, period: float, per_block: int, buffer: int, backlog: int, margin: float
    ) -> None:
        self.per_block, self.buffer, self.backlog, self.margin = per_block, buffer, backlog, margin
        self.paced = period * per_block
        self.block_s, self.jitter = self.paced, 0.0
        self.arrived: dict[int, float] = {}
        self.newest = self.shown_k = -1
        self.queue: list[tuple[int, Any, int]] = []  # (frame, item, rank)
        self.playing, self.next, self.holding = False, 0.0, False
        self.holds: list[float] = []
        self.every, self.shown_at = period, None

    def add(
        self, k: int, first: bool, recv: float, item: Any, rank: int = 1, again: bool = False
    ) -> None:
        """Frame ``k`` (``item``) received at ``recv``; ``first``: its block's first; ``rank``:
        of its version (``rank`` in ``static/stream.js``); ``again``: sent again."""
        if k <= self.shown_k:
            return
        if k in self.arrived or again:
            self.queue = [
                (j, item, rank) if j == k and held < rank else (j, old, held)
                for j, old, held in self.queue
            ]
            return
        before = self.arrived.get(k - self.per_block)
        if before is not None:
            self._measure(recv - before)
        self.arrived[k] = recv
        self.newest = max(self.newest, k)
        for old in [j for j in self.arrived if j <= self.newest - 2 * self.per_block]:
            del self.arrived[old]
        i = len(self.queue)
        while i > 0 and self.queue[i - 1][0] > k:
            i -= 1
        self.queue.insert(i, (k, item, rank))
        if not first:
            return
        period = self.paced / self.per_block
        keeps = (
            self.backlog
            if self.every < PAGE_STRETCH[1] * period
            else math.floor(self.backlog * period / self.every)
        )
        del self.queue[: max(0, i - keeps)]

    def _measure(self, gap: float) -> None:
        if gap > PAGE_PAUSE_BLOCKS * self.block_s:
            self.block_s, self.jitter = self.paced, 0.0
            return
        share = PAGE_FOLLOW / self.per_block
        bound = max(PAGE_OUTLIER_FRAMES * self.block_s / self.per_block, 4 * self.jitter)
        deviation = min(max(gap - self.block_s, -bound), bound)
        self.jitter += share * (abs(deviation) - self.jitter)
        self.block_s = max(self.paced, self.block_s + share * deviation)

    def take(self, now: float) -> Any | None:
        """The frame to show at the animation frame ``now``, or ``None`` (none due, or a hold)."""
        if not self.playing:
            if len(self.queue) < self.buffer:
                return None
            self.playing, self.next = True, now
        if now < self.next - PAGE_SLACK_S:
            return None
        if not self.queue:
            if not self.holding:
                self.holds.append(now)
            self.holding = True
            return None
        self.shown_k, item, _ = self.queue.pop(0)
        late = self.holding or now - self.next >= self.block_s / self.per_block
        gap = None if self.shown_at is None else now - self.shown_at
        if not self.holding and gap is not None and gap <= PAGE_PAUSE_BLOCKS * self.block_s:
            self.every += PAGE_SHOWN_FOLLOW * (gap - self.every)
        self.shown_at, self.holding = now, False
        due = now if late else self.next
        self.next = due + self._period(due)
        return item

    def _period(self, due: float) -> float:
        base = self.block_s / self.per_block
        margin = max(self.margin * base, 2 * self.jitter)
        period, measured = PAGE_STRETCH[0] * base, False
        for i in range(self.newest + 1, self.newest + self.per_block + 1):
            before = self.arrived.get(i - self.per_block)
            if before is None:
                continue
            measured = True
            frames = len(self.queue) + i - self.newest
            period = max(period, (before + self.block_s + margin - due) / frames)
        return min(period, PAGE_STRETCH[1] * base) if measured else base


def shown_by_page(
    sent: Sequence[tuple[int, float, bool, int, bool]], play: PlayConfig, per_block: int
) -> tuple[list[int], np.ndarray, list[float]]:
    """The frames the page shows (:class:`PagePlayout` at ``play``'s settings, its animation
    frames :data:`DISPLAY_HZ` a second) of frames sent ``(index, when, its block's first, the
    rank of its version, whether it went again)``, each arriving :data:`NETWORK_S` later.

    Returns:
        tuple[list[int], np.ndarray, list[float]]: per frame shown, in order, the one of ``sent``
        shown, and when; and when each hold began.
    """
    page = PagePlayout(
        1.0 / play.fps, per_block, play.jitter_buffer_frames, BACKLOG_FRAMES, play.margin_frames
    )
    arrivals = sorted(range(len(sent)), key=lambda i: sent[i][1])
    shown, when = [], []
    if not arrivals:
        return shown, np.zeros(0), []
    tick, j = sent[arrivals[0]][1] + NETWORK_S, 0
    end = sent[arrivals[-1]][1] + NETWORK_S + 2 * per_block / play.fps
    while tick <= end:
        while j < len(arrivals) and sent[arrivals[j]][1] + NETWORK_S <= tick:
            i = arrivals[j]
            k, at, first, rank, again = sent[i]
            page.add(k, first, at + NETWORK_S, i, rank, again)
            j += 1
        item = page.take(tick)
        if item is not None:
            shown.append(item)
            when.append(tick)
        tick += 1.0 / DISPLAY_HZ
    return shown, np.asarray(when), page.holds


def measure_live(
    client: Any,
    row: RoundIndexRow,
    *,
    play: PlayConfig,
    script: str,
    seconds: float,
    seed: int,
    out_dir: str | Path,
) -> dict:
    """A worker's session of one live client (:class:`~worldcast.demo.worker.Worker`, its pacing,
    refreshes and send queue as it plays a browser's) on an input script, measured.

    The client starts on ``row`` (:meth:`WorldCastClient.start_row`); the script
    (:func:`input_script`) starts once the blocks before the measured ones have been sent: the
    first six and :data:`UNMEASURED_BLOCKS` more. Each frame the worker sends is taken to arrive
    :data:`NETWORK_S` later and to be shown by the pacing's display (:func:`shown_by_pacing`), and
    to light :data:`DISPLAY_S` after it is shown; and by the page's playout
    (:func:`shown_by_page`, ``page``). The displayed sequence (the pacing's) is written to
    ``out_dir/displayed.mp4``.

    Args:
        client (WorldCastClient): the client, warm (:meth:`WorldCastClient.warm_up`); with
            ``play.refresh`` it runs ``live.refresh``.
        row (RoundIndexRow): the client's row of the round index.
        play (PlayConfig): the worker's play settings: the display's frames per second, refresh,
            the input mapping, the gains, the refresh threshold, the send queue's lead and swap, the
            gather, the preview step, the page's playout margin and jitter buffer.
        script (str): one of :data:`SCRIPTS`.
        seconds (float): the script's length.
        seed (int): the script's draw.
        out_dir (str | Path): receives ``displayed.mp4``.

    Returns:
        dict: ``i2p_s`` (:func:`percentiles` of input to photon of the script's events: from an
        event's sample at the worker to the light of the first frame shown whose controls hold
        it) and ``i2p_steady_s`` (of the events after the first :data:`STEADY_AFTER_BLOCKS`
        measured blocks), ``dropped_frames`` (frames the send queue dropped, as the display does
        before a block that came early), ``fps`` (the frames shown per second),
        ``stalls_per_min`` (frames held :data:`STALL_PERIODS` frame periods or longer),
        ``refreshes_per_block`` and the share of
        refreshes that ``switched`` frames, ``per_refresh`` the ``host_ms`` of its cut, the
        ``denoise_ms`` of its denoising steps, ``decode_ms`` to its first frame and ``total_ms``
        from its start to its first frame (:func:`percentiles`), ``step_lead_s`` (from the start
        of a block's step, which commits the block before it, to its first frame shown) and
        ``paced_lead_s`` (the pacing's lead and margin at a block's first frame), ``seams``:
        per kind of consecutive frames shown (``switch``: a refresh's first frame shown after a
        frame of another version of its block; ``preview``: a preview's frame after another
        version's or none; ``final``: a version's own frame after its preview's; ``block``: a
        block's first frame; ``ordinary``) their mean absolute pixel difference and median PSNR,
        and how soon the view follows the events that turn (:func:`turn_response` of
        :func:`asked_turns`): ``control_response`` in the controls of the frames shown (how far
        each frame's row says the player has turned) and ``camera_response`` in their pixels
        (:func:`view_turn` between consecutive frames shown, summed), and ``view_gain``, the
        view's turn over the controls' per event (:func:`view_gain`: a ratio of 1 is a view that
        turns as far as the player); ``page``, the same of the frames the page's playout shows
        (``i2p_s``, ``i2p_steady_s``, ``fps``, ``holds_per_min``, ``control_response``,
        ``camera_response``); ``preview`` (with ``play.preview_step``): preview frames ``sent``,
        shown by the pacing's display and the page, ``on_screen_s`` (per refresh, the page's time
        on its preview frames) and the share of the page's time on preview frames, and
        ``vs_final`` (a preview frame against its version's own frame of the same index, where
        both were decoded: mean absolute difference and median PSNR); ``swap`` (with
        ``play.swap_sent``): the frames sent again and the page's showing them; ``per_card``,
        each card's refreshes.
    """
    from .wire import load_jpeg_encoder
    from .worker import SWITCH_INTERVAL_S, Session, Worker

    sys.setswitchinterval(SWITCH_INTERVAL_S)  # as the worker's process runs
    load_jpeg_encoder()  # as the worker does when it starts

    fps = play.fps
    block_s = VIDEO_FRAMES_PER_BLOCK / fps
    unmeasured = (FIRST_TARGET - 1) // BLOCK + UNMEASURED_BLOCKS
    start = 1 + VIDEO_FRAMES_PER_BLOCK * unmeasured  # the first measured block's first frame
    seat = Seat(slot=row.player_slot, team="T", spawn=(0.0,) * 5)
    round_start = RoundStart(
        "bench",
        row.map_name,
        "bench",
        row.start_frame,
        (seat,),
        match_id=row.match_id,
        round=row.round,
    )
    worker = Worker(DemoConfig(play=play), client, Library({"bench": round_start}))
    weapon = encode_weapon(DEFAULT_LOADOUT[seat.team][0])
    sends: list[_Sent] = []
    leads: list[float] = []
    records: dict[int, Any] = {}

    class Measured(Session):
        """The bench's session: its frames sent to the measurement instead of a browser."""

        def qualities(self) -> tuple[int, ...]:
            return (JPEG_QUALITY,)

        def frame_message(self, frame, jpeg, ready, encoded, first, taken) -> bytes:
            message = super().frame_message(frame, jpeg, ready, encoded, first, taken)
            made.setdefault(frame.index, {})[id(message)] = frame, taken
            key = (frame.index, frame.version)
            if frame.preview:
                previewed[key] = frame.rgb
            elif key in previewed:  # a preview frame and its version's own, both decoded
                pairs.append(pairing.submit(_seam, previewed.pop(key), frame.rgb))
            return message

        def send_frame(self, k, messages, first, again=False) -> None:
            frame, taken = made[k].pop(id(messages[JPEG_QUALITY]))
            for old in [j for j in made if j < k - 4 * VIDEO_FRAMES_PER_BLOCK]:
                del made[old]
            for old in [key for key in previewed if key[0] < k - 4 * VIDEO_FRAMES_PER_BLOCK]:
                del previewed[old]
            sends.append(
                _Sent(
                    k,
                    time.monotonic(),
                    first,
                    frame.input_seq,
                    taken,
                    frame.turned,
                    frame.preview,
                    frame.version,
                    frame.rgb,
                    again,
                )
            )
            if first:
                leads.append(self.pacing.lead + PACING_MARGIN_S)
            # the generation thread appends records: a copy in one step, without Python code
            for record in list(client.client.records):
                records[record.f0] = record

    made: dict[int, dict[int, tuple]] = {}
    # a preview frame decoded, until its version's own; the pairs' differences, on a thread
    previewed: dict[tuple[int, int], np.ndarray] = {}
    pairs: list[Future] = []
    pairing = ThreadPoolExecutor(1, thread_name_prefix="pairs")
    session = Measured(
        room="bench",
        round_start=round_start,
        seat=seat,
        token="",
        t0=time.monotonic(),
        inputs=InputBuffer(
            fps=fps,
            mapping=play.mapping,
            weapon=weapon,
            release_after_s=play.release_after_s,
            pitch_gain=play.gains[0],
            yaw_gain=play.gains[1],
            gather_s=play.gather_ms / 1000.0,
        ),
        pacing=Pacing(fps),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events: list[tuple[int, float]] = []
    samples, asked = _scripted(script, seconds, seed)
    busy = GpuBusy()

    async def run() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(worker.client_thread, client.start_row, row)
        worker.session = session
        session.ready.set()
        session.connected.set()
        playing = asyncio.create_task(worker._play(session))
        try:
            while not sends or sends[-1].index < start:
                if playing.done():
                    playing.result()
                await asyncio.sleep(0.005)
            began = time.monotonic()
            busy.start()
            for seq, (at, buttons, dpitch, dyaw, event) in enumerate(samples, 1):
                await asyncio.sleep(max(0.0, began + at - time.monotonic()))
                sample = dict(q=seq, b=buttons, dp=dpitch, dy=dyaw, w=weapon)
                session.add_input(sample, room_time=worker.room_time(session))
                if event:
                    events.append((seq, time.monotonic()))
            await asyncio.sleep(3 * block_s)  # the last inputs reach the screen
            busy.stop()
        finally:
            worker.session = None
            await asyncio.wait({playing}, timeout=4 * block_s)
            playing.cancel()
            await asyncio.gather(playing, return_exceptions=True)
            await loop.run_in_executor(worker.frame_thread, bool)
            await loop.run_in_executor(worker.client_thread, client.stop)

    try:
        asyncio.run(run())
    finally:
        busy.stop()
        pairing.shutdown()
        worker.client_thread.shutdown()
        if worker.frame_thread is not worker.client_thread:
            worker.frame_thread.shutdown()
    sent = [(s.index, s.at, s.first, s.rank, s.again) for s in sends]
    shown, _ = shown_by_pacing(sent, fps)
    # per frame shown: its slot's index, when its slot's frame went, whether it starts its block,
    # and the newest input sample and the cut of the version shown
    slots = {}
    for i, s in enumerate(sends):
        slots.setdefault(s.index, i)
    released = [
        (
            sends[i].index,
            sends[slots[sends[i].index]].at,
            sends[slots[sends[i].index]].first,
            sends[i].seq,
            sends[i].taken,
        )
        for i in shown
    ]
    images = [sends[i].rgb for i in shown]
    seams = [(math.nan, math.nan)] + [_seam(a, b) for a, b in zip(images, images[1:])]
    views = [(0.0, 0.0)] + [view_turn(a, b) for a, b in zip(images, images[1:])]
    write_mp4(iter(images), out / "displayed.mp4", writer=dict(MP4_WRITER_KWARGS, fps=fps))
    turns = [sends[i].turned for i in shown]
    previews = [sends[i].preview for i in shown]
    measured = _measured(
        released, seams, events, records, leads, start=start, fps=fps, previews=previews
    )
    measured.update(_responses(released, turns, views, events, asked, start=start, fps=fps))
    log = list(session.sender.log) if session.sender is not None else []
    trace = list(session.trace)
    measured["stalls"] = stall_details(
        released, log, trace, start=start, fps=fps, frames=VIDEO_FRAMES_PER_BLOCK
    )
    page_shown, page_at, holds = shown_by_page(sent, play, VIDEO_FRAMES_PER_BLOCK)
    firsts = {s.index: s.first for s in reversed(sends)}  # the first sent of each frame's
    measured["page"] = _page(
        sends, page_shown, page_at, holds, events, asked, firsts=firsts, start=start
    )
    measured["preview"] = _preview(
        sends, shown, page_shown, page_at, [pair.result() for pair in pairs], start=start
    )
    resent = list(session.sender.resent) if session.sender is not None else []
    measured["swap"] = dict(
        resent=len([k for k, _ in resent if k >= start]),
        shown_page=sum(
            1 for i in page_shown if sends[i].index >= start and slots[sends[i].index] != i
        ),
    )
    measured["per_card"] = _per_card(records, start)
    measured["gpu_busy"] = busy.result()
    origin = sends[0].at if sends else 0.0
    measured["trace"] = [_relative(entry, origin) for entry in trace]
    return measured


def _page(
    sends: list[_Sent],
    shown: list[int],
    at: np.ndarray,
    holds: list[float],
    events: list[tuple[int, float]],
    asked: list[tuple[float, float]],
    *,
    firsts: dict[int, bool],
    start: int,
) -> dict:
    """:func:`measure_live`'s ``page``: of the frames the page shows from frame ``start`` on
    (``firsts``: whether each index starts its block)."""
    keep = [j for j, i in enumerate(shown) if sends[i].index >= start]
    shown, light = [shown[j] for j in keep], at[keep] + DISPLAY_S
    if not shown:
        return {}
    span = float(light[-1] - light[0])
    seqs = np.array([sends[i].seq for i in shown])
    starts = [light[j] for j, i in enumerate(shown) if firsts[sends[i].index]]
    steady_from = starts[STEADY_AFTER_BLOCKS] if len(starts) > STEADY_AFTER_BLOCKS else np.inf
    i2p, steady = [], []
    for seq, when in events:
        holding = np.flatnonzero(seqs >= seq)
        if len(holding):
            i2p.append(light[holding[0]] - when)
            if when >= steady_from:
                steady.append(i2p[-1])
    images = [sends[i].rgb for i in shown]
    viewed = np.cumsum([(0.0, 0.0)] + [view_turn(a, b) for a, b in zip(images, images[1:])], 0)
    turned = np.asarray([sends[i].turned for i in shown], np.float64)
    times = [t for _, t in events]
    measured_holds = [t for t in holds if light[0] <= t + DISPLAY_S <= light[-1]]
    return dict(
        i2p_s=percentiles(i2p),
        i2p_steady_s=percentiles(steady),
        fps=(len(shown) - 1) / span if span else None,
        holds_per_min=60.0 * len(measured_holds) / span if span else None,
        control_response=turn_response(light, turned, times, asked),
        camera_response=turn_response(light, viewed, times, asked),
    )


def _preview(
    sends: list[_Sent],
    shown: list[int],
    page: list[int],
    page_at: np.ndarray,
    pairs: list[tuple[float, float]],
    *,
    start: int,
) -> dict:
    """:func:`measure_live`'s ``preview``: of the frames from frame ``start`` on; ``pairs``: the
    difference of each preview frame from its version's own frame of its index."""
    measured = [i for i, s in enumerate(sends) if s.index >= start]
    if not pairs and not any(sends[i].preview for i in measured):
        return {}
    on_screen: dict[tuple[int, int], float] = {}
    keep = [j for j, i in enumerate(page) if sends[i].index >= start]
    for a, b in zip(keep, keep[1:]):
        s = sends[page[a]]
        if s.preview:
            block = s.index - (s.index - 1) % VIDEO_FRAMES_PER_BLOCK
            key = (block, s.version)
            on_screen[key] = on_screen.get(key, 0.0) + float(page_at[b] - page_at[a])
    span = float(page_at[keep[-1]] - page_at[keep[0]]) if len(keep) > 1 else 0.0
    return dict(
        sent=sum(sends[i].preview for i in measured),
        shown=sum(sends[i].preview for i in shown if sends[i].index >= start),
        shown_page=sum(sends[page[j]].preview for j in keep),
        on_screen_s=percentiles(list(on_screen.values())),
        screen_share=sum(on_screen.values()) / span if span else None,
        vs_final=dict(
            n=len(pairs),
            mean_abs_diff=float(np.mean([m for m, _ in pairs])) if pairs else None,
            median_psnr_db=float(np.median([p for _, p in pairs])) if pairs else None,
        ),
    )


def _per_card(records: dict[int, Any], start: int) -> dict:
    """:func:`measure_live`'s ``per_card``: per card, the refreshes of the measured blocks and
    their denoising (:func:`percentiles` of ms)."""
    cards: dict[int, list[float]] = {}
    for f0, record in records.items():
        if 1 + 4 * (f0 - 1) < start:
            continue
        for refresh in record.refreshes:
            cards.setdefault(refresh.card, []).append(refresh.denoise_ms)
    return {
        str(card): dict(refreshes=len(ms), denoise_ms=percentiles(ms))
        for card, ms in sorted(cards.items())
    }


class GpuBusy:
    """The share of the time each visible CUDA card was busy (NVML's utilization: a kernel ran),
    sampled every :data:`BUSY_SAMPLE_S` from :meth:`start` to :meth:`stop` on a thread of its
    own; by ``torch.cuda.utilization`` (pynvml) or else ``nvidia-smi``. Without CUDA (or either)
    nothing is sampled."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import torch

        if not torch.cuda.is_available():
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def result(self) -> dict[str, float | None]:
        """The mean busy share per card, ``cuda:<index>`` (an empty dict without samples)."""
        return {card: float(np.mean(s)) / 100.0 for card, s in self._samples.items() if s}

    def _sample(self) -> None:
        import torch

        cards = range(torch.cuda.device_count())
        read = _nvml_utilization if _has_pynvml() else _smi_utilization
        while not self._stop.wait(BUSY_SAMPLE_S):
            try:
                busy = read(cards)
            except Exception:  # a read that fails (a timeout) gives no sample this time
                continue
            for card, percent in busy.items():
                self._samples.setdefault(f"cuda:{card}", []).append(percent)


def _has_pynvml() -> bool:
    try:
        import pynvml  # noqa: F401
    except ImportError:
        return False
    return True


def _nvml_utilization(cards: Sequence[int]) -> dict[int, float]:
    """The cards' utilization by NVML, of those that can be read now."""
    import torch

    out = {}
    for card in cards:
        try:
            out[card] = float(torch.cuda.utilization(card))
        except Exception:  # a card that cannot be read now gives no sample
            continue
    return out


@functools.cache
def _card_uuids(count: int) -> dict[str, int]:
    """The UUIDs of the visible CUDA cards, as ``nvidia-smi`` ends them."""
    import torch

    return {str(torch.cuda.get_device_properties(card).uuid): card for card in range(count)}


def _smi_utilization(cards: Sequence[int]) -> dict[int, float]:
    """The cards' utilization by ``nvidia-smi``, matched by their UUIDs."""
    import subprocess

    uuids = _card_uuids(len(cards))
    lines = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.splitlines()
    out = {}
    for line in lines:
        uuid, _, percent = (part.strip() for part in line.partition(","))
        card = next((card for known, card in uuids.items() if uuid.endswith(known)), None)
        if card is not None and percent.isdigit():  # a card that cannot be read: "[N/A]"
            out[card] = float(percent)
    return out


def _relative(entry: Any, origin: float) -> dict:
    """A :attr:`~worldcast.demo.worker.Session.trace` entry as a dict, its ``kind`` and its times
    (``_t``) from ``origin``, s."""
    out: dict[str, Any] = {"kind": entry.kind}
    for name, value in asdict(entry).items():
        timed = name.endswith("_t")
        # a time of 0 was not reached (a refresh that decoded nothing)
        out[name] = (None if not value else round(float(value) - origin, 4)) if timed else value
    return out


def stall_details(
    released: list[tuple[int, float, bool, int, float]],
    log: list[tuple[int, bool, float, float, float, float, int]],
    trace: list[Any],
    *,
    start: int,
    fps: float,
    frames: int,
) -> list[dict]:
    """What held each frame that stalled the display of :func:`measure_live` (a frame shown
    :data:`STALL_PERIODS` frame periods or more after the one before it), from frame ``start`` on.

    Args:
        released (list): per frame sent, as :func:`measure_live` keeps it.
        log (list): :attr:`~worldcast.demo.worker.SendQueue.log` (none without refresh).
        trace (list): :attr:`~worldcast.demo.worker.Session.trace`, its
            :class:`~worldcast.demo.worker.BlockTrace` and
            :class:`~worldcast.demo.worker.RefreshTrace` entries.
        start (int): the first measured frame.
        fps (float): the display's frame rate.
        frames (int): a block's frames.

    Returns:
        list[dict]: per stall, ``t`` (s from the first measured frame shown), ``frame`` and the
        first frame of its ``block``, ``gap_ms`` (on screen of the frame before it), ``late_ms``
        (how much later than its slot the frame was ready to go), ``sent_late_ms`` (than it was
        to go), ``step_late_ms`` (its block's step after its start was due), ``critical_ms`` (the
        step's start to its first frame), ``prepare_ms`` (of that, to the block's cut),
        ``refreshes`` (of its block or the block before, being generated when its step was asked
        for: the step waited for them), ``decoding`` (refreshes whose frames were decoded while
        the step ran, to its first frame), and ``cause``: ``late refresh`` (the block's step
        waited for a refresh), ``late start`` (the step began late), ``refresh decode`` (the
        block's first frame came late while a refresh's frames were decoded), ``late block``
        (the block's first frame came late: its step took longer than its lead), ``decoder`` (a
        later frame of a block came late), ``late release`` (ready in time, sent late) or
        ``display``.
    """
    period = 1.0 / fps
    index = np.array([r[0] for r in released])
    sent = np.array([r[1] for r in released])
    firsts = np.array([r[2] for r in released])
    display = display_times(sent + NETWORK_S, firsts, fps)
    queued = {entry[0]: entry for entry in log}
    blocks = [entry for entry in trace if entry.kind == "block"]
    refreshes = [entry for entry in trace if entry.kind == "refresh"]
    out = []
    measured = np.flatnonzero(index >= start)
    for i in measured[1:]:
        gap = display[i] - display[i - 1]
        if gap < STALL_PERIODS * period:
            continue
        k = int(index[i])
        _, _, put, go, _, went, _ = queued.get(k, (k, firsts[i], sent[i], sent[i], 0, sent[i], 0))
        slot = display[i - 1] + period
        block = next((b for b in blocks if b.first <= k < b.first + frames), None)
        detail = dict(
            t=round(float(display[i] - display[measured[0]]), 3),
            frame=k,
            block=None if block is None else block.first,
            gap_ms=round(1e3 * float(gap), 1),
            late_ms=round(1e3 * float(put + NETWORK_S - slot), 1),
            sent_late_ms=round(1e3 * float(went - go), 1),
        )
        if block is not None:
            first, started, first_put = block.first, block.started_t, block.first_out_t
            late = 0.0 if block.due_t is None else started - block.due_t
            detail.update(
                step_late_ms=round(1e3 * late, 1),
                critical_ms=round(1e3 * (first_put - started), 1),
                prepare_ms=round(1e3 * (block.cut_t - started), 1),
            )
            near = [r for r in refreshes if r.first in (first, first - frames)]
            # being generated when the step was asked for, and decoded while it ran
            running = [r for r in near if r.asked_t <= started < (r.generated_t or math.inf)]
            decoding = [
                r for r in near if r.switch_t and r.switch_t < first_put and r.last_out_t > started
            ]
            detail.update(refreshes=len(running), decoding=len(decoding))
            if put + NETWORK_S > slot + 1e-3:
                if running:
                    detail["cause"] = "late refresh"
                elif late > 1e-2:
                    detail["cause"] = "late start"
                elif (k == first or put <= first_put + 1e-3) and decoding:
                    detail["cause"] = "refresh decode"
                elif k == first or put <= first_put + 1e-3:
                    detail["cause"] = "late block"
                else:
                    detail["cause"] = "decoder"
        if "cause" not in detail:
            detail["cause"] = "late release" if went - go > 1e-2 else "display"
        out.append(detail)
    return out


def _responses(
    released: list[tuple[int, float, bool, int, float]],
    turns: list[tuple[float, float]],
    views: list[tuple[float, float]],
    events: list[tuple[int, float]],
    asked: list[tuple[float, float]],
    *,
    start: int,
    fps: float,
) -> dict:
    """:func:`measure_live`'s ``control_response`` and ``camera_response`` of the frames sent from
    frame ``start`` on."""
    index = np.array([r[0] for r in released])
    sent = np.array([r[1] for r in released])
    firsts = np.array([r[2] for r in released])
    light = display_times(sent + NETWORK_S, firsts, fps) + DISPLAY_S
    measured = index >= start
    at = [t for _, t in events]
    viewed = np.cumsum(np.asarray(views, np.float64).reshape(-1, 2), axis=0)
    held = np.asarray(turns, np.float64).reshape(-1, 2)
    return dict(
        control_response=turn_response(light[measured], held[measured], at, asked),
        camera_response=turn_response(light[measured], viewed[measured], at, asked),
        view_gain=view_gain(light[measured], viewed[measured], held[measured], at),
    )


def _measured(
    released: list[tuple[int, float, bool, int, float]],
    seams: list[tuple[float, float]],
    events: list[tuple[int, float]],
    records: dict[int, Any],
    leads: list[float],
    *,
    start: int,
    fps: float,
    previews: list[bool],
) -> dict:
    """:func:`measure_live`'s statistics of the frames sent from frame ``start`` on."""
    index = np.array([r[0] for r in released])
    sent = np.array([r[1] for r in released])
    firsts = np.array([r[2] for r in released])
    seqs = np.array([r[3] for r in released])
    takens = np.array([r[4] for r in released])
    display = display_times(sent + NETWORK_S, firsts, fps)
    measured = index >= start
    # the steady state: from the third measured block's first frame shown on
    starts = display[measured & firsts]
    steady_from = starts[STEADY_AFTER_BLOCKS] if len(starts) > STEADY_AFTER_BLOCKS else np.inf
    i2p, steady = [], []
    for seq, at in events:
        holds = np.flatnonzero(measured & (seqs >= seq))
        if len(holds):
            i2p.append(display[holds[0]] + DISPLAY_S - at)
            if at >= steady_from:
                steady.append(i2p[-1])
    shown = display[measured]
    span = float(shown[-1] - shown[0]) if len(shown) > 1 else 0.0
    stalls = int((np.diff(shown) >= STALL_PERIODS / fps).sum())
    kinds: dict[str, list[tuple[float, float]]] = {
        "switch": [],
        "preview": [],
        "final": [],
        "block": [],
        "ordinary": [],
    }
    for i in np.flatnonzero(measured)[1:]:
        if firsts[i]:
            kind = "block"
        elif previews[i] and not (previews[i - 1] and takens[i] == takens[i - 1]):
            kind = "preview"
        elif previews[i - 1] and not previews[i] and takens[i] == takens[i - 1]:
            kind = "final"
        else:
            kind = "switch" if takens[i] != takens[i - 1] else "ordinary"
        kinds[kind].append(seams[i])
    blocks = [r for f0, r in sorted(records.items()) if 1 + 4 * (f0 - 1) >= start]
    refreshes = [refresh for record in blocks for refresh in record.refreshes]
    first_shown = {k: t for k, t, first in zip(index, display, firsts) if first}
    leads_shown = [
        first_shown[1 + 4 * (r.f0 - 1)] - r.t_start
        for r in blocks
        if 1 + 4 * (r.f0 - 1) in first_shown
    ]
    return dict(
        i2p_s=percentiles(i2p),
        i2p_steady_s=percentiles(steady),
        i2p_each_s=[round(float(v), 4) for v in i2p],
        dropped_frames=(
            int(index[measured][-1] - start + 1 - measured.sum()) if measured.any() else 0
        ),
        events=len(events),
        frames=int(measured.sum()),
        fps=(len(shown) - 1) / span if span else None,
        stalls_per_min=60.0 * stalls / span if span else None,
        blocks=len(blocks),
        refreshes_per_block=len(refreshes) / len(blocks) if blocks else None,
        switched=(
            sum(r.first_frame >= 0 for r in refreshes) / len(refreshes) if refreshes else None
        ),
        per_refresh=dict(
            host_ms=percentiles([1e3 * (r.t_cut - r.t_controls) for r in refreshes]),
            denoise_ms=percentiles([r.denoise_ms for r in refreshes]),
            decode_ms=percentiles(
                [1e3 * (r.t_first_frame - r.t_latents) for r in refreshes if r.t_first_frame]
            ),
            total_ms=percentiles(
                [1e3 * (r.t_first_frame - r.t_start) for r in refreshes if r.t_first_frame]
            ),
        ),
        step_lead_s=percentiles(leads_shown),
        paced_lead_s=percentiles(leads),
        seams={
            kind: dict(
                n=len(pairs),
                mean_abs_diff=float(np.mean([m for m, _ in pairs])) if pairs else None,
                median_psnr_db=float(np.median([p for _, p in pairs])) if pairs else None,
            )
            for kind, pairs in kinds.items()
        },
    )
