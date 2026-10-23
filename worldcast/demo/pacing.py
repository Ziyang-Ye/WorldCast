"""When a live client starts its next block (docs/demo.md, "Architecture").

A block starts just in time for its first frame to be shown when the previous block's frames run
out: its controls are taken as late as possible, and a fast client does not run ahead of the
display. The demo's workers and ``tools/bench_realtime.py`` pace by this rule.
"""

from collections import deque
from collections.abc import Iterable, Sequence

__all__ = [
    "BACKLOG_FRAMES",
    "LEAD_PERCENTILE",
    "LEAD_WINDOW",
    "PACING_MARGIN_S",
    "Pacing",
    "percentile",
]

#: A block starts as early as the last this many blocks needed (their critical delays) ...
LEAD_WINDOW = 16
#: ... at this percentile of them, rounded down to one of them: the two slowest of 16 (a CUDA graph
#: capture, a busy host), of n the slowest n - 1 - (n - 1) * 90 // 100, do not make the blocks
#: after them start early.
LEAD_PERCENTILE = 90
#: Extra head start of a block over that, s (absorbs network jitter).
PACING_MARGIN_S = 0.04
#: The frames of earlier blocks that the browser keeps in front of a block's first frame when the
#: block has arrived, while its display keeps up with the frame period: the page drops the older
#: ones (``Playout.add`` in ``stream.js``). The first frame waits for the kept ones and for the
#: display's next frame time: about 3 frame periods, up to 3.75 where the display shows its frames
#: 1.25 periods apart, and a display tick. A slower display keeps fewer.
BACKLOG_FRAMES = 3


def percentile(values: Iterable[float], p: int = LEAD_PERCENTILE) -> float:
    """The ``p``-th percentile of ``values``, rounded down to one of them (0 of none)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[(len(ordered) - 1) * p // 100]


class Pacing:
    """The display's schedule and, from it, the start of each block.

    The display shows a frame when it arrives, never sooner than a frame period after the previous
    one, and a block's first frame at most :data:`BACKLOG_FRAMES` frame periods after it arrives:
    the page drops the frames of earlier blocks that wait beyond that, and so does the schedule,
    so that a block that came early (the one after a slow first block) leaves no backlog that the
    blocks after it would be timed from. This display runs at the frame period with no jitter
    margin, which the page's playout adds, and has no display ticks: the schedule's first-frame
    wait is the optimistic one, below the page's by about the margin, and within a frame period of
    it where a block comes early.

    A block's critical delay is how long before its first frame is due its controls must be taken
    for every one of its frames to arrive in time. It is measured from when they were taken, not
    from when the block was asked for: a wait in between (the client still preparing the block) is
    the client's, and starting earlier does not shorten it. The next block starts
    :data:`PACING_MARGIN_S` plus the :data:`LEAD_PERCENTILE`-th percentile of the critical delays
    of the last :data:`LEAD_WINDOW` blocks before its first frame is due. Where the decode is slower
    than the display, a block's frames arrive no faster than it decodes them and the display shows
    them as they arrive: ``shown`` and the critical delays follow the decode.

    Args:
        fps (float): the frame rate of the display.
    """

    def __init__(self, fps: float) -> None:
        self.period = 1.0 / float(fps)
        self.critical: deque[float] = deque(maxlen=LEAD_WINDOW)
        self.shown: float | None = None

    @property
    def lead(self) -> float:
        """The :data:`LEAD_PERCENTILE`-th percentile of the recent critical delays, rounded down to
        one of them, s (0 before the first block)."""
        return percentile(self.critical)

    def record(self, taken: float, arrivals: Sequence[float]) -> list[float]:
        """A block whose controls were taken at ``taken`` and whose frames arrived at
        ``arrivals``.

        Returns:
            list[float]: when the display shows each of the frames.
        """
        self.critical.append(max(t - taken - j * self.period for j, t in enumerate(arrivals)))
        shown = []
        for j, arrival in enumerate(arrivals):
            due = arrival if self.shown is None else self.shown + self.period
            if j == 0:
                due = min(due, arrival + BACKLOG_FRAMES * self.period)
            self.shown = max(arrival, due)
            shown.append(self.shown)
        return shown

    def next_start(self, in_flight: float = 0.0) -> float | None:
        """When to start the next block; ``None``: now (the display holds no frame to follow).

        Args:
            in_flight (float): the display time of the blocks started and not yet recorded, s
                (a worker whose decoder runs on its own device starts a block while the frames of
                the one before are still being decoded): they are taken to arrive in time for the
                display, so the next block follows them.
        """
        if self.shown is None:
            return None
        return self.shown + in_flight + self.period - self.lead - PACING_MARGIN_S

    def rebuffer(self) -> None:
        """The display starts over (a browser reconnected): it shows the next frame on arrival."""
        self.shown = None
