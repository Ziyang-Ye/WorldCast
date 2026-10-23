"""``tools/bench_realtime.py``: what it asks for, and the ``bench.json`` it writes."""

import json

import pytest

from tests.tools.support import load_tool, synthetic_round, write_config


def test_the_bench_writes_each_setting(tmp_path, monkeypatch, capsys):
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=5)
    config = write_config(cfg, tmp_path / "config.yaml")
    tool, measured = load_tool("bench_realtime"), []

    def compare_settings(
        cfg, row, taehv_path, names, *, fps, decoder, decoder_device, decoder_channels_last
    ):
        measured.append(
            (
                cfg.run.max_blocks,
                row.media_id,
                taehv_path,
                tuple(names),
                fps,
                decoder,
                decoder_device,
            )
        )
        yield "exact", {"fps": 15.6, "bit_identical_to_first": True}

    monkeypatch.setattr(tool, "compare_settings", compare_settings)
    out = tmp_path / "bench"
    arguments = ["--config", config, "--index-row", "0", "--taehv", "taew2_2.pth"]
    assert tool.main([*arguments, "--out-dir", str(out), "--blocks", "5", "--fps", "16"]) == 0
    media_id = cfg_media_id(cfg)
    assert measured == [(5, media_id, "taew2_2.pth", ("exact", "fast"), 16.0, "taehv", None)]
    result = json.loads((out / "bench.json").read_text())
    assert result["exact"] == {"fps": 15.6, "bit_identical_to_first": True}
    assert (result["index_row"], result["blocks"], result["display_fps"]) == (0, 5, 16.0)
    assert (result["decoder"], result["decoder_device"]) == ("taehv", None)
    capsys.readouterr()
    for refused, message in (
        (
            [*arguments, "--out-dir", str(out), "--blocks", "4"],
            "--blocks: at least 5 (the first 3 are not measured)",
        ),
        ([*arguments, "--out-dir", str(out), "--settings", "slow"], "invalid choice: 'slow'"),
        (["--index-row", "0", "--taehv", "t", "--out-dir", str(out)], "not set: paths.round_index"),
    ):
        with pytest.raises(SystemExit) as stopped:
            tool.main(refused)
        assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_the_bench_measures_the_wan_vae_on_a_second_card(tmp_path, monkeypatch, capsys):
    _, cfg = synthetic_round(tmp_path, monkeypatch, max_blocks=5)
    config = write_config(cfg, tmp_path / "config.yaml")
    tool, measured = load_tool("bench_realtime"), []

    def compare_settings(
        cfg, row, taehv_path, names, *, fps, decoder, decoder_device, decoder_channels_last
    ):
        measured.append((taehv_path, fps, decoder, decoder_device, decoder_channels_last))
        yield "fast", {"fps": 11.2}

    monkeypatch.setattr(tool, "compare_settings", compare_settings)
    out = tmp_path / "bench"
    arguments = ["--config", config, "--index-row", "0", "--out-dir", str(out), "--blocks", "5"]
    wan = ["--decoder", "wan", "--decoder-device", "cuda:1", "--fps", "11"]
    assert tool.main([*arguments, *wan]) == 0  # the tiny decoder's weights are not needed
    assert measured == [(None, 11.0, "wan", "cuda:1", False)]
    result = json.loads((out / "bench.json").read_text())
    assert (result["decoder"], result["decoder_device"]) == ("wan", "cuda:1")
    assert tool.main([*arguments, *wan, "--decoder-channels-last"]) == 0
    assert measured[-1] == (None, 11.0, "wan", "cuda:1", True)
    assert json.loads((out / "bench.json").read_text())["decoder_channels_last"] is True
    capsys.readouterr()
    for refused, message in (
        (arguments, "--taehv: the tiny decoder's weights"),
        ([*arguments, "--decoder", "wan", "--decoder-device", "gpu1"], "decoder_device must be"),
        (
            [*arguments, "--taehv", "t", "--decoder-channels-last"],
            "decoder_channels_last is the Wan2.2 VAE's",
        ),
    ):
        with pytest.raises(SystemExit) as stopped:
            tool.main(refused)
        assert stopped.value.code == 2 and message in capsys.readouterr().err


def cfg_media_id(cfg) -> str:
    """The media id of the first row of the config's round index."""
    with open(cfg.paths.round_index) as index:
        return json.loads(index.readline())["media_id"]
