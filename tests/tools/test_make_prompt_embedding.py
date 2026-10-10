"""``tools/make_prompt_embedding.py``: the encoder of ``paths.wan22_root`` encodes the prompt on
``run.device``, and the embedding is written."""

import pytest
import torch
import yaml

from tests.tools.support import load_tool
from worldcast.modeling.wan22.text_encoder import FIXED_PROMPT, load_prompt_embedding


def test_the_prompt_is_encoded_and_written(tmp_path, monkeypatch, capsys):
    pytest.importorskip("safetensors")
    tool, built = load_tool("make_prompt_embedding"), []
    embeds = torch.zeros(1, 512, 8, dtype=torch.bfloat16)
    embeds[:, :10] = 1.0  # the fixed prompt's ten tokens

    class Encoder:
        """Stands in for the umT5 encoder."""

        def encode_prompt(self):
            return embeds

    def load_text_encoder(wan22_root, *, device):
        built.append((wan22_root, device))
        return Encoder()

    monkeypatch.setattr(tool, "load_text_encoder", load_text_encoder)
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"paths": {"wan22_root": "weights/Wan2.2-TI2V-5B"}}))
    out = tmp_path / "prompt.safetensors"
    arguments = ["--out", str(out), "--device", "cpu"]
    assert tool.main(["--config", str(config), *arguments]) == 0
    assert built == [("weights/Wan2.2-TI2V-5B", "cpu")]
    saved = load_prompt_embedding(out)
    assert saved.dtype == torch.bfloat16 and torch.equal(saved, embeds)
    printed = capsys.readouterr().out
    assert f"{FIXED_PROMPT!r}: embeds (1, 512, 8) torch.bfloat16" in printed
    with pytest.raises(SystemExit) as stopped:
        tool.main(arguments)  # no Wan2.2 snapshot
    assert stopped.value.code == 2 and "not set: paths.wan22_root" in capsys.readouterr().err
