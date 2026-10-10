"""The comparison with the reference runs, on the CPU: a tiny random model on a synthetic round,
compared with the fingerprints of its own offline run."""

from dataclasses import replace

import numpy as np
import pytest

import tests.engine.inference.support as sw
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference import reference
from worldcast.utils.fingerprints import fingerprint


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    return sw.offline_run(tmp_path_factory.mktemp("reference"))


def test_the_fingerprints_of_a_run(offline):
    latents = offline["latents"]
    prints = reference.latent_fingerprints(latents)
    assert prints == dict(latents_0_24=fingerprint(latents[:25]), all_latents=fingerprint(latents))
    runs = reference.read_reference_runs(offline["table"])[offline["media_id"]]
    assert [run.latent_frames for run in runs] == [len(latents), len(latents) + 4]
    assert "all_latents" not in runs[1].fingerprints  # not recorded: absent


def test_a_client_through_latent_24_matches_its_reference_run(offline, monkeypatch):
    sw.patch_ticks(monkeypatch, offline["world"]["tables"])
    cfg = offline["cfg"]
    runs = reference.read_reference_runs(offline["table"])[offline["media_id"]]
    result = reference.verify_client(cfg, load_round_index_row(cfg.paths.round_index, 0), runs)
    assert [check.name for check in result.checks] == [
        f"entry_noise ({len(offline['latents'])} latents)",
        f"entry_noise ({len(offline['latents']) + 4} latents)",
        "first_frame",
        "latents_0_24",
    ]
    # the entry noise of the longer run is another draw; everything of the same run matches
    assert [check.ok for check in result.checks] == [True, False, True, True]
    assert np.array_equal(result.latents, offline["latents"][:25])
    assert list(result.blocks) == ["1-4", "5-8", "9-12", "13-16", "17-20", "21-24"]
    assert result.blocks["5-8"] == fingerprint(offline["latents"][5:9])
    assert result.environment["attention"] and result.seconds > 0


def test_a_finished_run_is_compared_on_what_its_reference_recorded(offline, tmp_path):
    runs = reference.read_reference_runs(offline["table"])[offline["media_id"]]
    path = offline["tmp"] / "run" / "latents.npy"
    same = reference.verify_latents(path, runs)
    assert [(check.name, check.ok) for check in same] == [
        ("latents_0_24", True),
        ("all_latents", True),
    ]
    assert str(same[0]).startswith("MATCH  latents_0_24")
    # against a reference run of another length only latents 0-24 are compared
    partial = reference.verify_latents(path, runs[1:])
    assert partial[1].reference is None and partial[1].ok and str(partial[1]).startswith("-")
    np.save(tmp_path / "latents.npy", offline["latents"] + 1.0)
    changed = reference.verify_latents(tmp_path / "latents.npy", runs)
    assert not any(check.ok for check in changed) and str(changed[0]).startswith("DIFF")
    np.save(tmp_path / "latents.npy", offline["latents"].astype(np.float64))
    with pytest.raises(ValueError, match="not float32"):
        reference.verify_latents(tmp_path / "latents.npy", runs)
    with pytest.raises(ValueError, match="the client has no reference run"):
        reference.verify_latents(path, [])


def test_the_inputs_of_a_round_are_checked_as_a_client_loads_them(offline, monkeypatch):
    sw.patch_ticks(monkeypatch, offline["world"]["tables"])
    cfg = offline["cfg"]
    runs = reference.read_reference_runs(offline["table"])
    with pytest.raises(ValueError, match="covers 29 latent frames, the round runs 441"):
        reference.check_inputs(cfg, runs)  # run.latent_frames is more than the recording holds
    cfg = replace(cfg, run=replace(cfg.run, latent_frames=29))
    (line,) = reference.check_inputs(cfg, runs)
    assert line.startswith(f"{offline['media_id']}: 29 latents, 10 players recorded")
    assert line.endswith(" of the frames; the first frame is the reference run's")
    other = {offline["media_id"]: [reference.ReferenceRun(29, dict(first_frame="0" * 32))]}
    with pytest.raises(ValueError, match="first frame"):
        reference.check_inputs(cfg, other)
