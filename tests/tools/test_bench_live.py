"""``tools/bench_live.py`` on the CPU: the demo's worker and its WorldCast client (tiny random
models) played by a script, with and without refresh, and the ``bench.json`` and the mp4 it
writes; with a replica of the generator on a second card of the CPU, previews, the send queue's
swap and the shorter margins; and the demo of the generator trained without scene state."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
from tests.tools.support import load_tool, synthetic_round, write_config
from worldcast.demo import latency
from worldcast.demo.latency import SCRIPTS, display_times, input_script
from worldcast.engine.inference.decode import mp4_decoded_frames
from worldcast.engine.inference.taehv import TinyDecoder

DEMO = Path(__file__).resolve().parents[2] / "configs" / "demo" / "worldcast.yaml"
NOSCENE_DEMO = DEMO.with_name("worldcast_noscene.yaml")


def test_the_scripts_change_the_input_every_few_tenths_of_a_second():
    """A key set or a turn every 0.3-1.5 s, a flick every 0.4-1.5 s (``look``, whose samples
    change with what the governor commands: ``tests/demo/test_latency.py``)."""
    assert SCRIPTS == ("walk", "orbit", "look", "idle")
    for name in SCRIPTS:
        samples = input_script(name, 20.0, seed=3)
        assert len(samples) == 1200 and samples == input_script(name, 20.0, seed=3)
        times = np.array([at for at, *_ in samples])
        events = times[[event for *_, event in samples]]
        if name == "idle":
            assert not len(events) and not any(b or dp or dy for _, b, dp, dy, _ in samples)
            continue
        gaps = np.diff(events)
        shortest = 0.4 if name == "look" else 0.3
        assert gaps.min() >= shortest - 1 / 60 and gaps.max() <= 1.5 + 1 / 60
        if name == "look":
            continue
        changed = [samples[i][1:4] != samples[i - 1][1:4] for i in range(1, len(samples))]
        assert sum(changed) == len(events) - 1  # a sample changes only at an event
    assert input_script("walk", 20.0, seed=3) != input_script("walk", 20.0, seed=4)
    with pytest.raises(ValueError, match="no input script 'run'"):
        input_script("run", 1.0, seed=0)


def test_the_display_shows_a_frame_on_arrival_a_period_after_the_one_before_at_the_earliest():
    shown = display_times(
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.52], [True, False, False, True, False, False], 10.0
    )
    np.testing.assert_allclose(shown, [0.0, 0.1, 0.2, 0.5, 0.6, 0.7])


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """The synthetic round's client of slot 0 under live controls, the tiny decoder's weights, and
    the tool's arguments; the measured blocks start right after the first six."""
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=3)
    sw.patch_extent(monkeypatch)
    live = sw.write_state_model(tmp_path / "state_model.pt")
    config = write_config(cfg.with_overrides(live), tmp_path / "config.yaml")
    torch.manual_seed(0)
    torch.save(TinyDecoder().state_dict(), tmp_path / "taew2_2.pth")
    monkeypatch.setattr(latency, "UNMEASURED_BLOCKS", 0)
    arguments = ["--config", config, "--index-row", "0", "--demo", str(DEMO)]
    arguments += ["--taehv", str(tmp_path / "taew2_2.pth"), "--settings", "exact", "--no-warm-up"]
    return arguments + ["--seed", "1"]


@pytest.mark.parametrize(
    ("refresh", "fps", "seconds", "script"),
    # the tiny models take about 2 s a block on one CPU thread: a display at 4 fps (4 s a block)
    # leaves the room for refreshes that a GPU leaves at 15
    [("off", 16.0, 5.0, "walk"), ("on", 4.0, 8.0, "walk"), ("on", 4.0, 8.0, "look")],
)
def test_the_bench_measures_a_scripted_player_with_and_without_refresh(
    bench, tmp_path, refresh, fps, seconds, script, capsys
):
    tool, out = load_tool("bench_live"), tmp_path / refresh
    run = ["--refresh", refresh, "--fps", str(fps), "--seconds", str(seconds), "--no-swap-sent"]
    assert tool.main([*bench, *run, "--script", script, "--out-dir", str(out)]) == 0
    result = json.loads((out / "bench.json").read_text())
    assert (result["refresh"], result["script"], result["display_fps"]) == (refresh, script, fps)
    # the demo's mapping and gains: front and the yaw 1.45 times with refresh, latest and none
    # without
    assert result["serving"]["compile"] is False
    assert result["mapping"] == ("front" if refresh == "on" else "latest")
    assert result["gains"] == ([1.0, 1.45] if refresh == "on" else [1.0, 1.0])
    assert result["events"] > 3 and result["i2p_s"]["n"] > 0 and result["i2p_s"]["p50"] > 0
    assert result["i2p_steady_s"]["n"] <= result["i2p_s"]["n"] and result["dropped_frames"] >= 0
    assert result["frames"] > 16 and result["fps"] > 0 and result["stalls_per_min"] >= 0
    assert result["blocks"] >= 2 and result["step_lead_s"]["n"] >= 2
    assert set(result["seams"]) == {"switch", "preview", "final", "block", "ordinary"}
    assert result["seams"]["preview"]["n"] == result["seams"]["final"]["n"] == 0
    assert result["seams"]["ordinary"]["n"] > 0 and result["seams"]["block"]["n"] > 0
    if refresh == "off":
        assert result["refreshes_per_block"] == 0 and result["seams"]["switch"]["n"] == 0
    else:
        assert result["refreshes_per_block"] > 0
        timed = result["per_refresh"]
        assert timed["total_ms"]["n"] > 0 and timed["host_ms"]["p50"] > 0
    decoded, error = mp4_decoded_frames(out / "displayed.mp4")
    assert not error and decoded > result["frames"]
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert printed["fps"] == result["fps"]
    # the page's playout shows the same frames, at its own times
    page = result["page"]
    assert page["i2p_s"]["n"] == result["i2p_s"]["n"] and page["i2p_s"]["p50"] > 0
    assert page["fps"] > 0 and page["holds_per_min"] >= 0
    assert printed["page_i2p"] == [page["i2p_s"]["p50"], page["i2p_s"]["p90"]]
    # --no-swap-sent in place of the demo config's swap: no frame goes twice
    assert result["preview"] == {} and result["swap_sent"] is False
    assert result["swap"]["resent"] == 0
    assert result["gpu_busy"] == {} and result["replicas_equal"] == []
    # how soon the view follows the turns: the walk asks for none; each flick is timed in the
    # controls of the frames shown (the tiny models' pixels follow nothing)
    control, camera = result["control_response"], result["camera_response"]
    assert control["events"] == camera["events"] == (0 if script == "walk" else result["events"])
    if script == "look":
        assert control["t50_s"]["n"] > 0 and control["t50_s"]["p50"] > 0
        # the flicks reaching 90% are a subset of those reaching 50%: their medians need not order
        assert control["t90_s"]["n"] <= control["t50_s"]["n"]
        assert 0 < control["reached_50"] <= 1
        assert printed["control"][0] == control["t50_s"]["p50"]
        # the view's turn over the controls': read, whatever the tiny models' pixels do
        assert result["view_gain"]["turn"]["n"] > 0
        assert printed["view_gain"] == result["view_gain"]["turn"]["p50"]


def test_the_bench_refreshes_on_two_cards_with_previews_and_the_swap(bench, tmp_path, capsys):
    """The client with a replica on a second card (of the CPU), previews of the first denoising
    step, newer versions of frames sent going again, a shorter lead, no gather and a margin of a
    frame: refreshes run on both cards, the previews come before their versions' frames and differ
    from them, and the settings are in bench.json."""
    tool, out = load_tool("bench_live"), tmp_path / "cards"
    run = ["--refresh", "on", "--fps", "4", "--seconds", "10", "--script", "look"]
    run += ["--refresh-devices", "cpu", "cpu", "--preview-step", "1", "--swap-sent"]
    run += ["--send-lead-ms", "20", "--gather-ms", "0", "--margin-frames", "1.0"]
    assert tool.main([*bench, *run, "--out-dir", str(out)]) == 0
    result = json.loads((out / "bench.json").read_text())
    assert result["serving"]["refresh_devices"] == ["cpu", "cpu"]
    assert (result["send_lead_ms"], result["gather_ms"], result["margin_frames"]) == (20, 0, 1)
    assert result["preview_step"] == 1 and result["swap_sent"] is True
    assert result["refreshes_per_block"] > 0 and set(result["per_card"]) <= {"0", "1"}
    assert result["per_card"]["1"]["refreshes"] > 0  # the replica's card is free first
    preview = result["preview"]
    assert preview["sent"] > 0 and preview["vs_final"]["n"] > 0
    assert preview["vs_final"]["median_psnr_db"] < 100  # a preview is not its version's frame
    assert 0 <= preview["screen_share"] < 1
    assert result["swap"]["resent"] >= 0 and result["page"]["i2p_s"]["n"] > 0
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert printed["per_card"] == {card: c["refreshes"] for card, c in result["per_card"].items()}


def test_the_bench_runs_the_demo_of_the_generator_trained_without_scene_state(
    tmp_path, monkeypatch, capsys
):
    """``configs/demo/worldcast_noscene.yaml`` with the tiny generator without scene state, on a
    round long enough for its round of 45 latent frames: the client refreshes its window blocks."""
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=0, seconds=12.0)
    sw.patch_extent(monkeypatch)
    weights = sw.make_weights(tmp_path / "noscene", scene_state=False)
    live = sw.write_state_model(tmp_path / "state_model.pt")
    live.update((f"paths.{key}", path) for key, path in weights.items())
    config = write_config(cfg.with_overrides(live), tmp_path / "config.yaml")
    torch.manual_seed(0)
    torch.save(TinyDecoder().state_dict(), tmp_path / "taew2_2.pth")
    monkeypatch.setattr(latency, "UNMEASURED_BLOCKS", 0)
    arguments = ["--config", config, "--index-row", "0", "--demo", str(NOSCENE_DEMO)]
    arguments += ["--taehv", str(tmp_path / "taew2_2.pth"), "--settings", "exact", "--no-warm-up"]
    arguments += ["--fps", "4", "--seconds", "8", "--seed", "1", "--out-dir", str(tmp_path / "b")]
    assert load_tool("bench_live").main(arguments) == 0
    result = json.loads((tmp_path / "b" / "bench.json").read_text())
    assert result["refresh"] == "on" and result["blocks"] >= 2
    assert result["refreshes_per_block"] > 0 and result["i2p_s"]["n"] > 0
    capsys.readouterr()


def test_the_bench_asks_for_what_it_needs(bench, tmp_path, capsys):
    tool = load_tool("bench_live")
    for refused, message in (
        ([*bench, "--out-dir", str(tmp_path), "--script", "run"], "invalid choice: 'run'"),
        ([*bench, "--out-dir", str(tmp_path), "--seconds", "0"], "--seconds: positive"),
        (
            [*bench[:6], "--out-dir", str(tmp_path), "--decoder", "wan", "--decoder-device", "gpu"],
            "decoder_device must be",
        ),
        ([*bench, "--out-dir", str(tmp_path), "--yaw-gain", "0"], "play.yaw_gain must be positive"),
        ([*bench, "--out-dir", str(tmp_path), "--preview-step", "4"], "play.preview_step"),
        (
            [
                *bench,
                "--out-dir",
                str(tmp_path),
                "--refresh",
                "off",
                "--refresh-devices",
                "cpu,cpu",
            ],
            "--refresh-devices: the cards of the refreshes, with --refresh on",
        ),
        (
            [*bench, "--out-dir", str(tmp_path), "--refresh-devices", "cpu"],
            "at least one more",
        ),
    ):
        with pytest.raises(SystemExit) as stopped:
            tool.main(refused)
        assert stopped.value.code == 2 and message in capsys.readouterr().err
