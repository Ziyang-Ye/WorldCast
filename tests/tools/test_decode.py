"""``tools/decode.py`` on the CPU: a client's latents to ``video.mp4`` next to them, with a tiny VAE
in place of the Wan2.2 one."""

import io
import sys

import numpy as np
import pytest
import torch
import yaml

from tests.tools.support import load_tool


def test_latents_decode_next_to_themselves(tmp_path, tiny_vae, monkeypatch, capsys):
    pytest.importorskip("imageio_ffmpeg")
    tool = load_tool("decode")
    loaded = []

    def load_vae(wan22_root, device):
        loaded.append((wan22_root, device))
        return tiny_vae

    monkeypatch.setattr(tool, "load_vae", load_vae)
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"paths": {"wan22_root": str(tmp_path / "wan22")}}))
    torch.manual_seed(0)
    latents = tmp_path / "client" / "latents.npy"
    latents.parent.mkdir()
    np.save(latents, torch.randn(5, 48, 4, 6).numpy())
    arguments = ["--config", str(config), "--device", "cpu", "--latents", str(latents)]
    assert tool.main(arguments) == 0
    # the snapshot's VAE on the config's device; 5 latents are 1 + 4 x 4 frames
    assert loaded == [(str(tmp_path / "wan22"), "cpu")]
    assert (tmp_path / "client" / "video.mp4").stat().st_size > 0
    assert "17 frames" in capsys.readouterr().out
    for refused, message in (
        ([*arguments, str(latents), "--out", str(tmp_path / "a.mp4")], "--out names one file"),
        (["--latents", str(latents)], "not set: paths.wan22_root"),
    ):
        with pytest.raises(SystemExit) as stopped:
            tool.main(refused)
        assert stopped.value.code == 2 and message in capsys.readouterr().err


class Pipe(io.StringIO):
    """Stands in for a pipe, as ``examples/run.sh`` runs the decodes: what has gone through it
    (what was written before the last flush)."""

    through = ""

    def flush(self) -> None:
        self.through = self.getvalue()


def test_a_client_s_line_goes_out_before_the_next_decode(tmp_path, tiny_vae, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    tool = load_tool("decode")
    monkeypatch.setattr(tool, "load_vae", lambda wan22_root, device: tiny_vae)
    pipe, through = Pipe(), []
    decode_to_mp4 = tool.decode_to_mp4

    def decode(*args, **kwargs):
        through.append(pipe.through)  # when the decode starts
        return decode_to_mp4(*args, **kwargs)

    monkeypatch.setattr(tool, "decode_to_mp4", decode)
    monkeypatch.setattr(sys, "stdout", pipe)
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"paths": {"wan22_root": str(tmp_path / "wan22")}}))
    latents = [tmp_path / client / "latents.npy" for client in ("a", "b")]
    for path in latents:
        path.parent.mkdir()
        np.save(path, np.zeros((1, 48, 4, 6), np.float32))
    arguments = ["--config", str(config), "--device", "cpu", "--latents", *map(str, latents)]
    assert tool.main(arguments) == 0
    first = f"{latents[0]} -> {latents[0].with_name('video.mp4')}: 1 frames"
    assert through[0] == "" and through[1].startswith(first) and through[1].count("\n") == 1


def test_a_missing_vae_is_a_usage_error(tmp_path, capsys):
    tool = load_tool("decode")
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"paths": {"wan22_root": str(tmp_path / "wan22")}}))
    with pytest.raises(SystemExit) as stopped:
        tool.main(["--config", str(config), "--device", "cpu", "--latents", "latents.npy"])
    error = capsys.readouterr().err
    assert stopped.value.code == 2 and "usage:" in error
    assert f"No such file or directory: '{tmp_path / 'wan22' / 'Wan2.2_VAE.pth'}'" in error
