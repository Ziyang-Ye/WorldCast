"""``tools/verify_reference.py`` on the CPU: a tiny random model on a synthetic round, compared with
the fingerprints of its own offline run."""

import json

import numpy as np

from tests.engine.inference.support import offline_run, patch_ticks
from tests.tools.support import load_tool, write_config


def test_client_and_check(tmp_path, monkeypatch, capsys):
    offline = offline_run(tmp_path / "offline")
    patch_ticks(monkeypatch, offline["world"]["tables"])
    tool = load_tool("verify_reference")
    monkeypatch.setattr(tool, "TABLES", (offline["table"],))
    config = write_config(offline["cfg"], tmp_path / "config.yaml")
    out = tmp_path / "client"
    # the entry noise of the longer reference run is another draw: the client mode reports it
    assert tool.main(["client", "--config", config, "--index-row", "0", "--out-dir", str(out)]) == 1
    report = json.loads((out / "report.json").read_text())
    assert not report["match"] and len(report["blocks"]) == 6
    assert report["fingerprints"]["latents_0_24"] == report["reference"]["latents_0_24"]
    assert np.array_equal(np.load(out / "latents_0_24.npy"), offline["latents"][:25])

    run = offline["tmp"] / "run"
    assert tool.main(["check", str(run)]) == 0
    assert "1/1 clients match their reference run bit for bit" in capsys.readouterr().out
    np.save(tmp_path / "latents.npy", offline["latents"] + 1.0)
    (tmp_path / "client.json").write_text((run / "client.json").read_text())
    assert tool.main(["check", str(tmp_path)]) == 1
