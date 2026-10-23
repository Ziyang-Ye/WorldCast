"""The pacing rule: the display shows a frame on arrival, a frame period after the previous one at
the earliest, and a block's first frame at most a backlog of frame periods after it arrives; a
block starts the 90th percentile of the recent critical delays, measured from when the controls
were taken, plus the margin, before its first frame is due."""

import pytest

from worldcast.demo.pacing import (
    BACKLOG_FRAMES,
    LEAD_PERCENTILE,
    LEAD_WINDOW,
    PACING_MARGIN_S,
    Pacing,
)


def steady(pacing: Pacing, blocks: int, critical: float, start: float = 0.0) -> float:
    """Record ``blocks`` blocks of one frame, a second apart from ``start``, each with the
    critical delay ``critical``; returns when the next would be taken."""
    for b in range(blocks):
        pacing.record(start + b, [start + b + critical])
    return start + blocks


def test_the_display_shows_frames_a_period_apart_and_late_ones_on_arrival():
    pacing = Pacing(fps=16.0)
    assert pacing.next_start() is None  # nothing shown yet: start now
    # the second frame is early and waits for its turn; the fourth is late and delays what follows
    shown = pacing.record(100.0, [100.5, 100.51, 100.625, 100.9])
    assert shown == pytest.approx([100.5, 100.5625, 100.625, 100.9])
    assert pacing.record(101.0, [101.0]) == pytest.approx([101.0])  # an idle display: at once
    pacing.rebuffer()
    assert pacing.next_start() is None


def test_a_blocks_critical_delay_is_that_of_its_latest_frame():
    pacing = Pacing(fps=16.0)
    # controls taken at 100 s; the fourth frame, due 3/16 s after the first, is the late one
    pacing.record(100.0, [100.5, 100.55, 100.6, 100.9])
    assert list(pacing.critical) == pytest.approx([0.9 - 3 / 16])
    assert pacing.lead == pytest.approx(0.9 - 3 / 16)
    assert pacing.next_start() == pytest.approx(100.9 + 1 / 16 - pacing.lead - PACING_MARGIN_S)


def test_a_block_in_flight_is_followed_by_the_next_block_at_its_display_time():
    pacing = Pacing(fps=16.0)
    assert pacing.next_start(in_flight=1.0) is None  # nothing shown yet: start now
    pacing.record(100.0, [100.5, 100.55, 100.6, 100.9])
    alone = pacing.next_start()
    assert alone == pacing.next_start(in_flight=0.0)
    # a block started and not recorded yet is shown before the next one: a second of display
    assert pacing.next_start(in_flight=1.0) == pytest.approx(alone + 1.0)
    assert pacing.next_start(in_flight=2.0) == pytest.approx(alone + 2.0)


def test_a_decode_slower_than_the_display_is_shown_as_it_comes_and_holds_the_blocks_back():
    """Four frames arrive each 0.4 s at a display of 0.25 s for four: the display shows them on
    arrival, the lead is the decode's, and the next block waits for the frames that follow."""
    pacing = Pacing(fps=16.0)
    arrivals = [0.4] * 4 + [0.8] * 4 + [1.2] * 4 + [1.6] * 4
    shown = pacing.record(0.0, arrivals)
    assert shown[3:5] == pytest.approx([0.4 + 3 / 16, 0.8])  # the display runs dry between groups
    assert shown[-1] == pytest.approx(1.6 + 3 / 16)
    assert pacing.lead == pytest.approx(max(a - j / 16 for j, a in enumerate(arrivals)))
    assert pacing.next_start() == pytest.approx(shown[-1] + 1 / 16 - pacing.lead - PACING_MARGIN_S)


def test_a_block_starts_the_90th_percentile_of_the_recent_critical_delays_early():
    assert (LEAD_WINDOW, LEAD_PERCENTILE, PACING_MARGIN_S) == (16, 90, 0.04)
    pacing = Pacing(fps=16.0)
    steady(pacing, 14, 0.1)
    pacing.record(20.0, [21.0])  # 1.0 s
    pacing.record(30.0, [32.0])  # 2.0 s
    # sixteen blocks: the 90th percentile, rounded down, is the third slowest; the two slowest are
    # left out
    assert pacing.lead == pytest.approx(0.1)
    assert pacing.next_start() == pytest.approx(32.0 + 1 / 16 - 0.1 - PACING_MARGIN_S)
    pacing.record(40.0, [40.5])  # a third slow block, the oldest one leaves the window
    assert pacing.lead == pytest.approx(0.5)
    steady(pacing, LEAD_WINDOW, 0.25, start=50.0)
    assert pacing.lead == pytest.approx(0.25)  # the slow blocks have left the window


def test_one_spike_moves_the_next_start_by_no_more_than_the_margin():
    pacing = Pacing(fps=16.0)
    taken = steady(pacing, LEAD_WINDOW, 0.3)
    head_start = pacing.shown + pacing.period - pacing.next_start()  # before the first frame is due
    pacing.record(taken, [taken + 2.0])
    assert pacing.lead == pytest.approx(0.3)
    later = pacing.shown + pacing.period - pacing.next_start()
    assert head_start == pytest.approx(0.3 + PACING_MARGIN_S) and later == pytest.approx(head_start)


def test_a_slow_first_block_sets_the_start_of_the_next_block_only():
    pacing = Pacing(fps=16.0)
    pacing.record(0.0, [10.0])  # 10 s: compilation, say
    assert pacing.lead == pytest.approx(10.0)  # nothing else is known yet
    pacing.record(10.0, [10.3])
    assert pacing.lead == pytest.approx(0.3)  # of two, the faster
    steady(pacing, 3, 0.3, start=11.0)
    assert pacing.lead == pytest.approx(0.3)


def test_a_block_that_comes_early_is_shown_a_backlog_of_frames_after_it_arrives():
    assert BACKLOG_FRAMES == 3
    pacing = Pacing(fps=16.0)
    assert pacing.record(0.0, [0.5] * 16) == pytest.approx([0.5 + j / 16 for j in range(16)])
    # the display holds 11 of these frames when the next block arrives at 0.8, and is dry at 1.5:
    # the page drops all of them but 3, so the block's first frame is shown 3 frame periods after
    # it arrived and the rest follow it
    shown = pacing.record(0.6, [0.8] * 16)
    assert shown == pytest.approx([0.8 + 3 / 16 + j / 16 for j in range(16)])


def test_a_block_that_comes_within_the_backlog_waits_for_the_display():
    pacing = Pacing(fps=16.0)
    pacing.record(0.0, [0.5] * 16)  # dry at 1.5
    shown = pacing.record(0.6, [1.4] * 16)  # 1.6 frame periods early: the frames before it all show
    assert shown == pytest.approx([1.5 + j / 16 for j in range(16)])


def test_the_block_after_an_early_one_starts_as_if_the_page_had_dropped_the_frames_before_it():
    pacing = Pacing(fps=16.0)
    pacing.record(0.0, [10.0] * 16)  # a slow first block, a critical delay of 10 s
    assert pacing.lead == pytest.approx(10.0)
    pacing.record(10.0, [10.5] * 16)  # the next starts at once and comes 0.5 s later
    assert pacing.lead == pytest.approx(0.5)
    # the display held the first block's frames until 11.0 and would have kept the next block
    # waiting for them; the page drops them but 3, so it shows the block from 10.5 + 3/16 on
    assert pacing.shown == pytest.approx(10.5 + 3 / 16 + 15 / 16)
    assert pacing.next_start() == pytest.approx(
        10.5 + 3 / 16 + 15 / 16 + 1 / 16 - 0.5 - PACING_MARGIN_S
    )
