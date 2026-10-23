"""The latency of a paced session: its statistics on known times, and a session on the CPU; the
``look`` script's flicks of the mouse, and how soon the view follows a turn: the view's turn read
off two frames, the times of a known response and the view's gain."""

import math
import re
import time

import numpy as np
import pytest
import torch

from tests.tools.support import synthetic_round
from worldcast.data import VIDEO_FRAMES_PER_BLOCK, load_round_index_row
from worldcast.data.game import ASPECT_TAN_RATIO, HFOV_DEGREES
from worldcast.data.latents import FRAME_SIZE
from worldcast.demo import latency
from worldcast.demo.config import PlayConfig
from worldcast.demo.coordinator import STATIC
from worldcast.demo.latency import (
    PagePlayout,
    asked_turns,
    input_script,
    input_to_photon,
    measure_latency,
    shown_by_pacing,
    shown_by_page,
    stats,
    turn_response,
    view_gain,
    view_turn,
)
from worldcast.demo.pacing import BACKLOG_FRAMES, Pacing
from worldcast.demo.worker import BlockTrace, RefreshTrace
from worldcast.engine.inference.client import Client, run_client
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.serving import ServingOptions
from worldcast.engine.inference.taehv import TinyDecoder


def test_stats():
    assert stats(np.arange(101.0)) == dict(n=101, p50=50.0, p95=95.0)


def test_an_input_waits_for_the_next_block_and_shows_with_its_first_frame():
    # a block a second; each block's first frame is shown 0.6 s after its controls are taken
    taken = np.arange(5.0)
    measured = input_to_photon(taken, taken + 0.6)
    # an input waits between 0 and 1 s for the next block's controls: 0.5 s at the median
    assert measured["lookahead"]["p50"] == pytest.approx(0.5, abs=1e-3)
    assert measured["lookahead"]["p95"] == pytest.approx(0.95, abs=1e-3)
    assert measured["total"]["p50"] == pytest.approx(1.1, abs=1e-3)


def test_the_two_settings_are_the_exact_client_and_the_demos_default():
    assert latency.SETTINGS == {"exact": ("flash", {"compile": False}), "fast": ("fa3", {})}


def test_a_paced_session_gives_the_clients_latents(tmp_path, monkeypatch):
    _, cfg = synthetic_round(
        tmp_path, monkeypatch, max_blocks=2, world_state=str(tmp_path / "world_state")
    )
    run_client(cfg)
    serving = ServingOptions(decoder="none", lockstep=False)
    row = load_round_index_row(cfg.paths.round_index, 0)
    models = ClientModels.load(cfg, serving)
    monkeypatch.setattr(latency, "UNMEASURED_BLOCKS", 0)  # two blocks follow latent 24: both count
    result = measure_latency(cfg, models, serving, row, fps=16.0)
    assert np.array_equal(result["latents"], np.load(tmp_path / "run" / "latents.npy"))
    assert result["fps"] > 0 and result["controls_to_first_frame_ms"]["n"] == 2
    # an input shows after the next block's controls are taken
    measured = result["input_to_photon_s"]
    assert measured["total"]["p50"] > measured["lookahead"]["p50"] > 0
    monkeypatch.setattr(latency, "UNMEASURED_BLOCKS", 1)  # one block left
    with pytest.raises(ValueError, match="the session has 1 measured blocks; it needs two"):
        measure_latency(cfg, models, serving, row, fps=16.0)


def test_a_decoder_on_its_own_device_has_the_next_block_start_while_this_one_decodes(
    tmp_path, monkeypatch
):
    """The session gives the latents and the stages of a block; the next block is timed to follow
    the one still in flight (``Pacing.next_start(in_flight)``) only where the decoder has a
    device of its own."""
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=2)
    row = load_round_index_row(cfg.paths.round_index, 0)
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    torch.manual_seed(0)
    models.tiny = TinyDecoder().eval().requires_grad_(False)
    monkeypatch.setattr(latency, "UNMEASURED_BLOCKS", 0)
    asked, results = [], {}
    next_start = Pacing.next_start

    def spy(pacing, in_flight=0.0):
        asked.append(in_flight)
        return next_start(pacing, in_flight)

    monkeypatch.setattr(Pacing, "next_start", spy)
    for device in (None, "cpu"):
        serving = ServingOptions(
            decoder="taehv", taehv_path="unused", lockstep=False, decoder_device=device
        )
        asked.clear()
        results[device] = measure_latency(cfg, models, serving, row, fps=16.0), list(asked)
    (alone, asked_alone), (apart, asked_apart) = results[None], results["cpu"]
    assert np.array_equal(alone["latents"], apart["latents"])
    assert set(asked_alone) == {0.0} and max(asked_apart) == VIDEO_FRAMES_PER_BLOCK / 16.0
    for entry in (alone, apart):
        stages = ("dit_ms", "decode_ms_per_block", "decode_ms_per_latent_frame", "block_period_ms")
        assert all(entry[stage]["p50"] > 0 for stage in stages)
        assert entry["decode_ms_per_latent_frame"]["p50"] == pytest.approx(
            entry["decode_ms_per_block"]["p50"] / 4, rel=1e-6
        )
        # the first frame comes after the latents, by the decode of one latent frame at least
        assert entry["first_frame_adder_ms"]["p50"] > 0
        assert entry["controls_to_first_frame_ms"]["n"] == 2


def test_the_measured_blocks_start_on_a_fresh_display(tmp_path, monkeypatch):
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=2)
    serving = ServingOptions(decoder="none", lockstep=False)
    row = load_round_index_row(cfg.paths.round_index, 0)
    clients, displays = [], []

    class RecordedClient(Client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    class Display(Pacing):
        """A display that keeps when each block's controls were taken, as it was told."""

        def __init__(self, fps):
            super().__init__(fps)
            self.taken = []
            displays.append(self)

        def record(self, taken, arrivals):
            self.taken.append(taken)
            return super().record(taken, arrivals)

    monkeypatch.setattr(latency, "Client", RecordedClient)
    monkeypatch.setattr(latency, "Pacing", Display)
    monkeypatch.setattr(latency, "UNMEASURED_BLOCKS", 0)
    result = measure_latency(cfg, ClientModels.load(cfg, serving), serving, row, fps=16.0)
    assert result["controls_to_first_frame_ms"]["n"] == 2
    # the six blocks of latents 1-24 on one display, the two measured blocks on a fresh one; each
    # block's critical delay counts from when its controls were taken
    assert [len(display.taken) for display in displays] == [6, 2]
    (client,) = clients
    assert [t for display in displays for t in display.taken] == [
        record.t_controls for record in client.records
    ]


def stalled(late: dict[int, float], *, started: float = 1.0, due: float = 1.0, refreshes=()):
    """:func:`~worldcast.demo.latency.stall_details` of two blocks of 16 frames at 10 fps, the
    second's step started at ``started`` (due at ``due``), cut 0.3 s later; its first frame sent
    at 1.6 s, the others a frame period apart, those in ``late`` that many seconds later."""
    sent = {k: 0.1 * k + late.get(k, 0.0) for k in range(32)}
    released = [(k, t, k % 16 == 0, -1, 0.0) for k, t in sent.items()]
    log = [(k, k % 16 == 0, t, t, t, t, 0) for k, t in sent.items()]
    trace = [
        BlockTrace(0, None, -0.5, -0.4, sent[0], sent[15]),
        BlockTrace(16, due, started, started + 0.3, sent[16], sent[31]),
        *refreshes,
    ]
    return latency.stall_details(released, log, trace, start=0, fps=10.0, frames=16)


def test_each_stall_says_what_held_its_frame_back():
    """A stall's frame, its block, how late, and the cause: the block's first frame late (its step
    took longer than its lead), with a refresh of the block before decoded meanwhile, or a refresh
    being generated as the step was asked for (it waited); a step that began late; a later frame
    of a block late (the decoder)."""
    (stall,) = stalled({k: 0.3 for k in range(16, 32)})
    assert stall["frame"] == stall["block"] == 16 and stall["cause"] == "late block"
    assert stall["critical_ms"] == pytest.approx(900.0) and stall["prepare_ms"] == 300.0
    assert stall["refreshes"] == stall["decoding"] == 0 and stall["late_ms"] > 0
    # a refresh of the block before decoded from 1.2 s to 1.5 s, while the step ran
    decoded = RefreshTrace(0, 0.7, 0.75, 0.9, 1.2, 1.25, 1.5, replaced=12)
    (stall,) = stalled({k: 0.3 for k in range(16, 32)}, refreshes=[decoded])
    assert stall["cause"] == "refresh decode" and stall["decoding"] == 1
    # one asked for at 0.95 s and generated at 1.1 s: the step waited for it
    generating = RefreshTrace(0, 0.95, 0.96, 1.1)
    (stall,) = stalled({k: 0.3 for k in range(16, 32)}, refreshes=[generating])
    assert stall["cause"] == "late refresh" and stall["refreshes"] == 1
    (stall,) = stalled({k: 0.3 for k in range(16, 32)}, started=1.2)
    assert stall["cause"] == "late start" and stall["step_late_ms"] == pytest.approx(200.0)
    (stall,) = stalled({k: 0.3 for k in range(20, 32)})
    assert stall["frame"] == 20 and stall["block"] == 16 and stall["cause"] == "decoder"
    assert stalled({}) == []


# ------------------------------------------------------------------------------------------ look
FRAME_S = 1 / 60


def test_the_look_script_flicks_the_mouse_and_sends_its_turn_as_it_moves():
    """Flicks of 10-60 degrees of yaw and 5-25 of pitch, of either sign, 0.4-1.5 s apart, the
    view kept within 45 degrees of level, each an event; each flick's turn goes in the samples of
    the six page frames the mouse moves over, a sixth in each, all of it before the next flick (no
    cap, no ramp: as the page sends a mouse's turn); the same seed gives the same script."""
    samples = input_script("look", 60.0, seed=3)
    asked = np.array(asked_turns("look", 60.0, seed=3))
    assert samples == input_script("look", 60.0, seed=3) and samples != input_script(
        "look", 60.0, 4
    )
    times = np.array([at for at, *_ in samples])
    events = np.flatnonzero([event for *_, event in samples])
    assert len(events) == len(asked) > 40 and events[0] == 0
    gaps = np.diff(times[events])
    assert gaps.min() >= 0.4 - FRAME_S and gaps.max() <= 1.5 + FRAME_S
    assert 10.0 <= np.abs(asked[:, 1]).min() and np.abs(asked[:, 1]).max() <= 60.0
    assert 5.0 <= np.abs(asked[:, 0]).min() and np.abs(asked[:, 0]).max() <= 25.0
    assert (asked > 0).any(axis=0).all() and (asked < 0).any(axis=0).all()
    assert np.abs(np.cumsum(asked[:, 0])).max() <= 45.0
    assert not any(buttons for _, buttons, *_ in samples)
    turn = np.array([(dp, dy) for _, _, dp, dy, _ in samples])
    for i, (start, end) in enumerate(zip(events, [*events[1:], len(samples)])):
        frames = latency.FLICK_FRAMES
        np.testing.assert_allclose(turn[start : start + frames], [asked[i] / frames] * frames)
        assert not turn[start + frames : end].any()
    assert np.abs(turn[:, 1] / FRAME_S).max() > 500  # a 60 degree flick in 0.1 s: 600 a second


def test_the_other_scripts_ask_for_the_turn_of_their_segments():
    samples = input_script("orbit", 20.0, seed=3)
    asked = asked_turns("orbit", 20.0, seed=3)
    events = [i for i, sample in enumerate(samples) if sample[-1]] + [len(samples)]
    for (start, end), turn in zip(zip(events, events[1:]), asked):
        segment = np.array([(dp, dy) for _, _, dp, dy, _ in samples[start:end]]).sum(axis=0)
        np.testing.assert_allclose(segment, turn, atol=1e-9)
    assert all(turn == (0.0, 0.0) for turn in asked_turns("walk", 20.0, seed=3))
    assert asked_turns("idle", 5.0, seed=3) == []


# --------------------------------------------------------------------------- the view's turn
HEIGHT, WIDTH = FRAME_SIZE
TAN_ACROSS = math.tan(math.radians(HFOV_DEGREES / 2))


def view(yaw: float, pitch: float) -> np.ndarray:
    """``[H, W, 3]`` uint8: a camera at ``yaw`` and ``pitch`` degrees (yaw counter-clockwise from
    +x, pitch > 0 looks down) inside a sphere painted with waves of longitude and latitude, the
    weapon a grey block in the bottom right third that turns with the view."""
    rng = np.random.default_rng(0)
    waves = 30
    lon_k, lat_k = rng.integers(4, 80, waves), rng.uniform(-80, 80, waves)
    phase, amplitude = rng.uniform(0, 2 * np.pi, waves), rng.uniform(0.5, 1.0, waves)
    across = ((np.arange(WIDTH) + 0.5) / WIDTH * 2 - 1) * TAN_ACROSS
    up = (1 - (np.arange(HEIGHT) + 0.5) / HEIGHT * 2) * TAN_ACROSS * ASPECT_TAN_RATIO
    x, y = np.meshgrid(across, up)
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    axes = np.array(
        [
            [math.sin(yaw), -math.cos(yaw), 0.0],
            [math.cos(yaw) * math.sin(pitch), math.sin(yaw) * math.sin(pitch), math.cos(pitch)],
            [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)],
        ]
    )
    rays = x[..., None] * axes[0] + y[..., None] * axes[1] + axes[2]
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lon, lat = np.arctan2(rays[..., 1], rays[..., 0]), np.arcsin(rays[..., 2])
    paint = sum(
        a * np.sin(k * lon + m * lat + p) for k, m, p, a in zip(lon_k, lat_k, phase, amplitude)
    )
    image = np.clip(128 + 20 * paint, 0, 255)
    image[int(HEIGHT * 2 / 3) :, WIDTH // 2 :] = 60
    return np.repeat(image[..., None], 3, axis=2).astype(np.uint8)


@pytest.mark.parametrize(
    ("pitch", "yaw"),
    [
        (0.0, 0.3),
        (0.3, 0.0),
        (-0.5, 1.0),
        (0.0, -3.0),
        (3.0, 0.0),
        (2.0, -6.5),
        (0.0, 10.0),
        (-10.0, 0.0),
    ],
)
def test_the_views_turn_is_read_off_two_frames_with_the_controls_signs(pitch, yaw):
    """A camera turned left (yaw > 0: the picture moves right) and down (pitch > 0: it moves up)
    by up to 10 degrees is read to within 0.1 degree, the weapon, which does not move, left out."""
    estimate = view_turn(view(0.0, 0.0), view(yaw, pitch))
    np.testing.assert_allclose(estimate, (pitch, yaw), atol=0.1)


def test_a_pitched_cameras_yaw_is_read_too():
    """A camera that looks 20 degrees down turns about the world's up axis, which is not its own:
    the yaw is read to within a few hundredths of it, with a few hundredths of it as pitch."""
    pitch, yaw = view_turn(view(0.0, 20.0), view(8.0, 20.0))
    assert yaw == pytest.approx(8.0, rel=0.03) and abs(pitch) < 0.05 * 8.0


# ----------------------------------------------------------------------------- the response
def test_the_response_times_the_first_frame_lit_that_reaches_each_share_of_the_turn():
    """Frames lit 15 a second from 0; the view turns 30 degrees of yaw over 0.15 s from 1.3 s
    on, 0.3 s after the event at 1.0 s that asked for it: half of it lights with the frame at 1.4
    s, nine tenths at 1.4667 s. An event asking for half a degree is not timed; one whose next
    event comes before the view gets there counts as not reached."""
    light = np.arange(60) / 15
    yaw = 30 * np.clip((light - 1.3) / 0.15, 0, 1) - 15 * np.clip((light - 3.0) / 0.15, 0, 1)
    turned = np.stack([np.zeros_like(yaw), yaw], axis=1)
    out = turn_response(light, turned, [1.0, 2.0, 2.9, 3.05], [(0, 30), (0, 0.5), (0, -15), (0, 1)])
    assert out["events"] == 3 and out["reached_50"] == pytest.approx(1 / 3)
    assert out["t50_s"]["n"] == 1 and out["t50_s"]["p50"] == pytest.approx(1.4 - 1.0)
    assert out["t90_s"]["p50"] == pytest.approx(22 / 15 - 1.0)
    assert turn_response(light, turned, [], [])["reached_90"] is None


def test_the_views_gain_is_its_turn_over_the_turn_its_controls_hold():
    """Per event, from the frame on screen at it to the last one lit before the next: the view
    turns 24 degrees of yaw where the controls hold 30 (0.8), then 20 of pitch and 15 of yaw where
    they hold 20 and 12 (1 and 1.25 by axis, 1.08 along the turn); an event whose controls turn
    under 3 degrees, and one with no frame after it, count for nothing."""
    light = np.arange(40) / 10.0
    held = np.zeros((40, 2))
    held[12:, 1] = 30.0
    held[22:, 0], held[22:, 1] = 20.0, 42.0
    viewed = np.zeros((40, 2))
    viewed[13:, 1] = 24.0
    viewed[23:, 0], viewed[23:, 1] = 20.0, 39.0
    out = view_gain(light, viewed, held, [1.0, 2.0, 3.0, 3.95])
    assert out["yaw"]["n"] == 2
    assert out["yaw"]["p50"] == pytest.approx((0.8 + 1.25) / 2)
    assert out["pitch"]["n"] == 1 and out["pitch"]["p50"] == pytest.approx(1.0)
    along = (20 * 20 + 15 * 12) / (20 * 20 + 12 * 12)
    assert out["turn"]["n"] == 2
    assert out["turn"]["p50"] == pytest.approx((0.8 + along) / 2)
    assert view_gain(light, viewed, held, [])["turn"]["n"] == 0


# ------------------------------------------------------------------------ the page's playout model
def test_the_page_model_is_the_pages_playout():
    """The bench's model of the page (:class:`PagePlayout`) keeps the numbers of ``Playout`` in
    ``static/stream.js``."""
    stream = (STATIC / "stream.js").read_text(encoding="utf-8")

    def js(name: str) -> list[float]:
        value = re.search(rf"^const {name} = (.+);$", stream, flags=re.M).group(1)
        return [float(number) for number in re.findall(r"[\d.]+", value)]

    assert js("STRETCH") == list(latency.PAGE_STRETCH)
    assert js("FOLLOW") == [latency.PAGE_FOLLOW]
    assert js("OUTLIER_FRAMES") == [latency.PAGE_OUTLIER_FRAMES]
    assert js("PAUSE_BLOCKS") == [latency.PAGE_PAUSE_BLOCKS]
    assert js("SHOWN_FOLLOW") == [latency.PAGE_SHOWN_FOLLOW]
    assert js("SLACK_MS") == [1000 * latency.PAGE_SLACK_S]
    assert "if (this.arrived.has(k) || frame.header.r) {" in stream
    assert "const rank = (header) => 2 * (header.v ?? 0) + (header.pv ? 0 : 1);" in stream


def paced(blocks: int, fps: float, per_block: int) -> list[tuple[int, float, bool]]:
    """Frames sent a frame period apart, ``(index, when, its block's first, rank, sent again)``."""
    return [(k, k / fps, k % per_block == 0, 1, False) for k in range(blocks * per_block)]


@pytest.mark.parametrize("margin", [0.5, 1.0, 1.5])
def test_the_page_shows_paced_frames_about_its_margin_after_they_arrive(margin):
    """Frames that arrive a frame period apart are shown each about ``margin`` frame periods after
    it arrives, once the playout has measured the blocks, at the frame rate and without a hold
    (but after the last)."""
    fps, per_block = 15.0, 16
    sent = paced(12, fps, per_block)
    play = PlayConfig(fps=fps, margin_frames=margin)
    shown, at, holds = shown_by_page(sent, play, per_block)
    assert shown == list(range(len(sent))) and all(hold > at[-1] for hold in holds)
    waited = at - np.array([t for _, t, *_ in sent]) - latency.NETWORK_S
    steady = waited[4 * per_block :]
    assert np.median(steady) == pytest.approx(margin / fps, abs=0.5 / fps)
    assert np.median(np.diff(at[4 * per_block :])) == pytest.approx(1 / fps, abs=1 / 60)


def test_a_frame_received_again_takes_the_place_of_the_one_waiting_if_it_is_newer():
    """A newer version (rank) of a frame waiting takes its place; an older or the same one, a
    frame sent again that never came and one at or before the frame shown are dropped."""
    page = PagePlayout(0.1, 4, 1, BACKLOG_FRAMES, 1.5)
    for k in range(3):
        page.add(k, k == 0, 0.0, f"{k}")
    assert page.take(0.0) == "0"
    page.add(1, False, 0.01, "1 again", rank=3, again=True)
    page.add(1, False, 0.02, "1 late", rank=1, again=True)  # older than the one waiting
    page.add(0, False, 0.01, "0 again", rank=3, again=True)  # shown already
    page.add(5, False, 0.01, "5 again", rank=3, again=True)  # never received
    assert [item for _, item, _ in page.queue] == ["1 again", "2"]
    assert page.take(0.2) == "1 again"
    page.add(1, False, 0.3, "1 after", rank=5, again=True)  # shown already
    assert [item for _, item, _ in page.queue] == ["2"]


def test_a_frame_sent_again_shows_where_it_arrives_before_its_slot():
    """The pacing's display shows a frame sent again (``play.swap_sent``) in its slot, if it arrives
    by then; the slots are those of the frames sent first."""
    sent = [(0, 0.0, True), (1, 0.0, False), (2, 0.0, False), (1, 0.05, False), (2, 0.5, False)]
    shown, at = shown_by_pacing(sent, 10.0)
    assert shown == [0, 3, 2]
    np.testing.assert_allclose(at, np.array([0.0, 0.1, 0.2]) + latency.NETWORK_S)


def test_the_cards_busy_share_is_sampled_on_past_a_read_that_fails(monkeypatch):
    """``GpuBusy`` samples every card it can read, a read that fails (an ``nvidia-smi`` that
    timed out) giving no sample that time and the sampling going on."""
    reads = []

    def read(cards):
        reads.append(len(reads))
        if len(reads) == 1:
            raise TimeoutError("nvidia-smi")
        return {0: 50.0, 1: 100.0} if len(reads) % 2 else {0: 30.0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(latency, "_has_pynvml", lambda: False)
    monkeypatch.setattr(latency, "_smi_utilization", read)
    monkeypatch.setattr(latency, "BUSY_SAMPLE_S", 0.005)
    busy = latency.GpuBusy()
    busy.start()
    while len(reads) < 6:
        time.sleep(0.005)
    busy.stop()
    result = busy.result()
    assert set(result) == {"cuda:0", "cuda:1"} and result["cuda:1"] == 1.0
    assert 0.3 <= result["cuda:0"] <= 0.5
