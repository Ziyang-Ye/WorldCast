"""The demo config: its defaults, the files that merge over them, the settings it refuses."""

from dataclasses import replace
from pathlib import Path

import pytest

from worldcast.config.inference import InferenceConfig
from worldcast.config.loader import build, config_to_dict
from worldcast.demo.config import REFRESH_YAW_GAIN, DemoConfig, load_config
from worldcast.demo.inputs import GATHER_S
from worldcast.demo.pacing import PACING_MARGIN_S
from worldcast.demo.worldcast_client import inference_config
from worldcast.engine.inference.serving import ServingOptions

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "demo"


def test_a_live_session_has_its_own_bound_and_display_rate():
    config = load_config()
    assert config == DemoConfig() and config.client.kind == "mock"
    # a smaller memory bank than the paper's and a display slower than the recording's 16 fps
    assert config.scene_state.bound == 32 < InferenceConfig().scene_state.bound
    assert config.play.fps == 16.0
    assert load_config((), {"scene_state.bound": 48}).scene_state.bound == 48


def test_a_block_is_cut_by_front_with_refresh_and_by_latest_without_unless_asked_otherwise():
    """By default a refreshed block's turn is done in its first frames (``front``, from a
    refresh's switch on) and a block without refresh spreads its turn over the block
    (``latest``); a mapping given is the mapping (``tests/demo/test_inputs.py``)."""
    assert load_config().play.input_mapping is None and load_config().play.mapping == "latest"
    assert load_config((), {"play.refresh": True}).play.mapping == "front"
    refreshed = {"play.refresh": True, "play.input_mapping": "latest"}
    assert load_config((), refreshed).play.mapping == "latest"
    for mapping in ("latest", "front", "hybrid", "realtime"):
        assert load_config((), {"play.input_mapping": mapping}).play.mapping == mapping


def test_with_refresh_the_generator_is_fed_the_yaw_45_per_cent_more_unless_asked_otherwise():
    """The view a refresh turns follows 0.6-0.8 of the yaw it is fed: by default the yaw is fed
    1.45 times with refresh, the pitch as made, and both as made without refresh; a gain given is
    the gain."""
    assert REFRESH_YAW_GAIN == 1.45
    assert load_config().play.gains == (1.0, 1.0)
    assert load_config((), {"play.refresh": True}).play.gains == (1.0, 1.45)
    given = {"play.refresh": True, "play.yaw_gain": 1.1, "play.pitch_gain": 1.2}
    assert load_config((), given).play.gains == (1.2, 1.1)
    assert load_config((), {"play.yaw_gain": 1.25}).play.gains == (1.0, 1.25)


def test_the_page_starts_on_one_frame():
    """``jitter_buffer_frames``, the frames the page holds before it shows the first one, which
    the welcome gives it."""
    assert load_config().play.jitter_buffer_frames == 1
    assert load_config((), {"play.jitter_buffer_frames": 2}).play.jitter_buffer_frames == 2


def test_the_playout_keeps_a_frame_and_a_half_of_margin():
    """``margin_frames``, the frame periods before it is due that the page's playout has every
    coming frame arrive, which the welcome gives it (``Playout`` in ``static/stream.js``): a
    setting, so that a shorter margin can be tried against the holds it costs."""
    assert load_config().play.margin_frames == 1.5
    assert load_config((), {"play.margin_frames": 0.5}).play.margin_frames == 0.5
    with pytest.raises(ValueError, match="play.margin_frames"):
        load_config((), {"play.margin_frames": -1})


def test_a_frame_goes_forty_ms_before_it_is_shown_and_a_flick_waits_twenty_for_more_of_it():
    """``send_lead_ms`` (how long before the pacing's display shows it a frame goes, with refresh)
    and ``gather_ms`` (how long a refresh for a turn that starts waits for more of it): the 40 ms of
    the pacing's margin and the 20 ms of ``GATHER_S`` by default, settings that the bench tries
    shorter."""
    play = load_config().play
    assert (
        (play.send_lead_ms, play.gather_ms)
        == (1000 * PACING_MARGIN_S, 1000 * GATHER_S)
        == (
            40.0,
            20.0,
        )
    )
    play = load_config((), {"play.send_lead_ms": 15, "play.gather_ms": 0}).play
    assert (play.send_lead_ms, play.gather_ms) == (15.0, 0.0)
    for key in ("send_lead_ms", "gather_ms"):
        with pytest.raises(ValueError, match=f"play.{key}"):
            load_config((), {f"play.{key}": -1})


def test_the_demo_refreshes_and_swaps_the_frames_sent_and_keeps_the_other_margins():
    """``configs/demo/worldcast.yaml`` refreshes the block on screen and sends a newer version of a
    frame that went while the page has not shown it (``play.swap_sent``); the previews, the lead,
    the gather and the page's margin are the defaults, and so is every play setting off the demo
    config."""
    play = load_config([CONFIGS / "worldcast.yaml"]).play
    assert play.refresh and play.swap_sent and play.preview_step == 0
    assert (play.send_lead_ms, play.gather_ms, play.margin_frames) == (40.0, 20.0, 1.5)
    default = load_config().play
    assert not default.swap_sent and default.preview_step == 0


def test_held_keys_outlast_a_congested_link():
    """A cut releases the held buttons a second after the newest input sample: longer than the
    gaps of a congested link (``tests/demo/test_inputs.py``)."""
    assert load_config().play.release_after_s == 1.0


def test_a_seat_is_released_after_a_quarter_of_an_hour_away_not_for_playing():
    """A page hidden, or a player without a key or a mouse movement, for 15 minutes: not a limit on
    play, a player who plays is never released (``tests/demo/test_worker.py``)."""
    play = load_config().play
    assert (play.away_s, play.idle_s) == (900.0, 900.0)


def test_a_frame_may_take_a_quarter_second_on_a_browsers_link():
    """Behind the frames in flight (``tests/demo/test_worker.py``, flow control)."""
    worker = load_config().worker
    assert (worker.max_inflight_ms, worker.jpeg_quality) == (250.0, 85)


def test_files_merge_over_the_defaults():
    assert load_config([CONFIGS / "mock_block16.yaml"]).client.mock.frames_per_block == 16
    exact = {"compile": False}
    client = load_config([CONFIGS / "worldcast.yaml"], {"client.worldcast.serving": exact}).client
    assert client.kind == "worldcast"
    assert client.worldcast.configs == ["weights/paths.yaml", "data/paths.yaml"]
    # the demo plays live, keeps the context of an idle player and stops at the walls
    assert client.worldcast.overrides == {
        "player_state.source": "live",
        "live.idle_context": "freeze",
        "live.idle_with_peers": True,
        "live.wall_keys": "guard",
        "live.dead_zone_u": 4.0,
        "live.hop": "spread",
    }
    served = ServingOptions.low_latency("weights/taew2_2.pth", **exact)
    assert client.worldcast.serving_options() == served


def test_the_demos_overrides_give_the_client_the_idle_context_freeze_and_no_other_config_does():
    """The demo's inference overrides are settings of the inference config; every other config keeps
    ``none``, the block loop the paper's players are run with."""
    overrides = load_config([CONFIGS / "worldcast.yaml"]).client.worldcast.overrides
    live = InferenceConfig().with_overrides(overrides)
    assert (live.player_state.source, live.live.idle_context) == ("live", "freeze")
    assert live.live.idle_with_peers is True
    assert (live.live.wall_keys, live.live.dead_zone_u, live.live.hop) == ("guard", 4.0, "spread")
    assert InferenceConfig().live.idle_context == "none"
    assert InferenceConfig().live.idle_with_peers is False
    mock = load_config([CONFIGS / "mock_block16.yaml"]).client
    assert mock.kind == "mock" and mock.worldcast.overrides == {}
    root = CONFIGS.parents[1]
    named = [
        path.relative_to(root).as_posix()
        for folder in ("configs", "examples")
        for path in sorted((root / folder).rglob("*.yaml"))
        if "idle_context" in path.read_text()
    ]
    assert named == ["configs/demo/worldcast.yaml"]


def test_the_demo_of_the_generator_without_scene_state_is_the_demos_with_its_own_keys():
    """``worldcast_noscene.yaml`` inherits the demo's file: its client runs the generator trained
    without scene state (with ``weights/noscene.yaml`` among its inference YAMLs) on a round of 45
    latent frames, with refresh on as the demo's; nothing else changes."""
    demo = load_config([CONFIGS / "worldcast.yaml"])
    noscene = load_config([CONFIGS / "worldcast_noscene.yaml"])
    worker, other = demo.client.worldcast, noscene.client.worldcast
    assert other.overrides == {**worker.overrides, "model.scene_state": False}
    assert other.configs == [*worker.configs, "weights/noscene.yaml"]
    assert noscene.play.refresh and demo.play.refresh
    assert replace(noscene, client=demo.client, play=demo.play) == demo
    assert noscene.play == demo.play
    assert replace(other, configs=worker.configs, overrides=worker.overrides) == worker
    cfg = inference_config(replace(other, configs=[]), noscene.scene_state, refresh=True)
    assert (cfg.model.scene_state, cfg.run.latent_frames, cfg.live.refresh) == (False, 45, True)
    cfg = inference_config(replace(worker, configs=[]), demo.scene_state)
    assert (cfg.model.scene_state, cfg.run.latent_frames, cfg.live.refresh) == (True, 29, False)
    given = replace(other, configs=[], overrides={**other.overrides, "run.latent_frames": 61})
    assert inference_config(given, noscene.scene_state).run.latent_frames == 61


def test_the_original_vae_on_a_second_card_is_a_setting_of_the_clients_serving():
    """``client.worldcast.serving`` replaces fields of ``ServingOptions.low_latency``; the printed
    config loads back."""
    second_card = {"decoder": "wan", "decoder_device": "cuda:1"}
    config = load_config([CONFIGS / "worldcast.yaml"], {"client.worldcast.serving": second_card})
    served = config.client.worldcast.serving_options()
    assert served == ServingOptions.low_latency("weights/taew2_2.pth", **second_card)
    assert (served.decoder, served.decoder_device) == ("wan", "cuda:1") and served.compile
    # the file that sets it, merged over the demo's: nothing else of the client changes
    files = [CONFIGS / "worldcast.yaml", CONFIGS / "worldcast_wan_vae.yaml"]
    merged = load_config(files).client.worldcast
    assert merged.serving_options() == served and merged.serving == second_card
    assert merged.configs == config.client.worldcast.configs
    assert merged.overrides == config.client.worldcast.overrides
    default = load_config([CONFIGS / "worldcast.yaml"]).client.worldcast.serving_options()
    assert (default.decoder, default.decoder_device) == ("taehv", None)
    assert not default.decoder_channels_last and not served.decoder_channels_last
    # the channels-last decoder: its own file over the demo's, or a key of the mapping
    files = [CONFIGS / "worldcast.yaml", CONFIGS / "worldcast_wan_vae_channels_last.yaml"]
    fast = load_config(files).client.worldcast.serving_options()
    assert (fast.decoder, fast.decoder_device, fast.decoder_channels_last) == (
        "wan",
        "cuda:1",
        True,
    )
    mapping = {**second_card, "decoder_channels_last": True}
    by_key = load_config([CONFIGS / "worldcast.yaml"], {"client.worldcast.serving": mapping})
    assert by_key.client.worldcast.serving_options() == fast
    # the tiny decoder's weights are not needed where the original VAE decodes
    no_tiny = {"client.worldcast.taehv_path": None, "client.worldcast.serving": second_card}
    assert load_config([CONFIGS / "worldcast.yaml"], no_tiny).client.worldcast.taehv_path is None
    assert build(DemoConfig, config_to_dict(config)) == config


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"client.kind": "gpu"}, ValueError),
        ({"client.worldcast.taehv_path": None}, ValueError),  # the tiny decoder needs its weights
        ({"client.worldcast.serving": {"decoder": "vae"}}, ValueError),
        ({"client.worldcast.serving": {"decoder": "wan", "decoder_device": "gpu1"}}, ValueError),
        (
            {"client.worldcast.serving": {"decoder_channels_last": True}},
            ValueError,
        ),  # the tiny decoder
        ({"client.worldcast.serving": {"decoder": "none", "decoder_device": "cuda:1"}}, ValueError),
        ({"client.worldcast.serving": {"lockstep": True}}, ValueError),  # never in live play
        ({"client.worldcast.serving": {"speed": 2}}, ValueError),
        ({"client.worldcast.device": "cuda"}, ValueError),  # no such key
        # the bound is the demo's setting: an override of the inference config would lose it
        ({"client.worldcast.overrides": {"scene_state.bound": 64}}, ValueError),
        ({"scene_state.bound": 0}, ValueError),
        ({"play.fps": 0}, ValueError),
        ({"play.release_after_s": 0}, ValueError),
        ({"play.away_s": 0}, ValueError),
        ({"play.idle_s": -1}, ValueError),
        ({"play.input_mapping": "first"}, ValueError),
        ({"play.yaw_gain": 0}, ValueError),
        ({"play.pitch_gain": -1.0}, ValueError),
        # the mock stands in for live, or for gt to show the drift
        ({"client.mock.player_state_source": "predicted"}, ValueError),
        ({"worker.max_inflight_ms": 0}, ValueError),
        ({"worker.port": "8101"}, ValueError),
    ],
)
def test_invalid_settings_refused(overrides, error):
    with pytest.raises(error):
        load_config([CONFIGS / "worldcast.yaml"], overrides)
