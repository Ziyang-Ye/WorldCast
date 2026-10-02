"""Keypress-to-photon latency and frame rate of a served session, from the engine's timestamps.

A block's controls are taken once (``BlockRecord.t_sample``); a key pressed at time ``tau`` first shows in the
first block whose controls are taken after ``tau``, at the moment that block's first frame is displayed. Frames are
displayed at 16 fps as they arrive: frame ``i`` at ``D_i = max(R_i + network, D_{i-1} + 1/16)``, with ``R_i`` the
time it was ready (decoded and encoded). So::

    keypress -> photon = (t_sample_k - tau)            input lookahead: up to one block
                       + (D_first(k) - t_sample_k)       ladder, decode, encode, network, display queue
                       + display                         the browser shows it (half a refresh + decode)
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["FPS", "stats", "display_times", "LatencyModel", "keypress_to_photon", "throughput"]

#: Video frames per second of the stream.
FPS = 16.0


def stats(values: Sequence[float]) -> dict[str, float]:
    """``n``, ``p50``, ``p95``, ``mean`` of a sample (empty: zeros)."""
    v = np.asarray(list(values), dtype=np.float64)
    if v.size == 0:
        return dict(n=0, p50=0.0, p95=0.0, mean=0.0)
    return dict(
        n=int(v.size),
        p50=float(np.percentile(v, 50)),
        p95=float(np.percentile(v, 95)),
        mean=float(v.mean()),
    )


def display_times(
    ready: Sequence[float], *, network_s: float = 0.0, fps: float = FPS
) -> np.ndarray:
    """When each frame is shown: as soon as it has arrived, but never faster than ``fps``."""
    r = np.asarray(ready, dtype=np.float64) + float(network_s)
    out = np.empty_like(r)
    for i, t in enumerate(r):
        out[i] = t if i == 0 else max(t, out[i - 1] + 1.0 / fps)
    return out


@dataclass(frozen=True)
class LatencyModel:
    """The parts outside the engine (assumed, not measured here): ``network_s`` one way from the engine to the
    browser (LAN WebSocket, one ~40 kB JPEG), ``display_s`` from arrival to light (decode plus half a 60 Hz
    refresh)."""

    network_s: float = 0.005
    display_s: float = 0.010


def keypress_to_photon(
    samples: Sequence[float],
    first_frames: Sequence[int],
    ready: Sequence[float],
    *,
    model: LatencyModel = LatencyModel(),
    n: int = 20000,
    seed: int = 0,
    skip_blocks: int = 2,
) -> dict[str, object]:
    """Keypress-to-photon latency over uniformly random keypress times.

    Args:
        samples: per block (in order), the time its controls were taken.
        first_frames: per block, the stream index of its first frame.
        ready: per stream frame, the time it was ready for the network.
        skip_blocks: warm-up blocks left out (the keypresses fall after their sample times).

    Returns ``stats`` of the total and of its parts (``lookahead``, ``to_display``), in seconds.
    """
    a = np.asarray(samples, dtype=np.float64)
    shown = display_times(ready, network_s=model.network_s)
    photon = np.asarray([shown[i] for i in first_frames], dtype=np.float64) + model.display_s
    if len(a) <= skip_blocks + 1:
        raise ValueError("need more blocks than the warm-up")
    rng = np.random.default_rng(seed)
    tau = rng.uniform(a[skip_blocks], a[-1], size=int(n))
    k = np.searchsorted(
        a, tau, side="left"
    )  # first block whose controls were taken at or after tau
    total = photon[k] - tau
    return dict(total=stats(total), lookahead=stats(a[k] - tau), to_display=stats(photon[k] - a[k]))


def throughput(first_frames: Sequence[int], first_ready: Sequence[float]) -> dict[str, float]:
    """Sustained frames per second: video frames between the first frames of the first and last generation over
    the time between them (frames come in bursts, one per latent frame, so the burst starts are the clock), and the
    longest wait between two generations' first frames."""
    i = np.asarray(first_frames, dtype=np.float64)
    t = np.asarray(first_ready, dtype=np.float64)
    if i.size < 2 or t[-1] <= t[0]:
        return dict(fps=0.0, max_gap_s=0.0)
    return dict(fps=float((i[-1] - i[0]) / (t[-1] - t[0])), max_gap_s=float(np.max(np.diff(t))))
