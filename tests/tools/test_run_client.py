"""``tools/run_client.py`` on the CPU: one client of the synthetic round, and what the tool asks
of its config."""

import numpy as np
import pytest

from tests.tools.support import load_tool


def test_a_client_writes_its_latents_and_reports(synthetic, tmp_path, capsys):
    config, _, media_id, latents = synthetic
    out = tmp_path / "client"
    arguments = ["--index-row", "0", "--world-state-dir", str(tmp_path / "world_state")]
    assert (
        load_tool("run_client").main(["--config", config, *arguments, "--out-dir", str(out)]) == 0
    )
    assert np.array_equal(np.load(out / "latents.npy"), latents)
    assert (out / "client.json").is_file()
    report = capsys.readouterr().out
    assert report.startswith(f"{media_id}: 29 latents -> {out} (1 blocks read the scene state, ")


def test_a_client_names_what_its_config_lacks(synthetic, capsys):
    config, _, _, _ = synthetic
    tool = load_tool("run_client")
    for arguments, message in (
        (["--index-row", "0"], "not set: paths.world_state_dir, paths.out_dir"),
        (["--world-state-dir", "w", "--out-dir", "o"], "run.index_row is not set"),
        (["--set", "run.sead=1"], "unknown config keys in run: ['sead']"),
        (["--config", "nowhere.yaml"], "nowhere.yaml"),
    ):
        with pytest.raises(SystemExit) as refused:
            tool.main(["--config", config, *arguments])
        assert refused.value.code == 2 and message in capsys.readouterr().err
