"""A session: the rounds of an index, a round's length as the paper ran it, the clients of a round
as processes, on the visible GPUs in turn and on the memory of each."""

import json
import subprocess

import pytest

from worldcast.data import read_round_index
from worldcast.engine.inference import session


def _row(match: int, slot: int, group: list[int], **extra) -> dict:
    media = [f"{match}-de_nuke-r03-p{s:02d}" for s in group]
    return dict(
        media_id=f"{match}-de_nuke-r03-p{slot:02d}",
        start_frame=0,
        match_id=match,
        round=3,
        map_name="de_nuke",
        latent_key="win_000000",
        player_slot=slot,
        group_media=media,
        group_slots=group,
        **extra,
    )


def _index(tmp_path, records: list[dict]):
    path = tmp_path / "round_index.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return read_round_index(path)


def test_the_rounds_of_an_index(tmp_path):
    records = [_row(7, s, [0, 1, 2], round_seconds=87) for s in (0, 1, 2)] + [_row(8, 4, [4])]
    rows = _index(tmp_path, records)
    assert session.rounds_of(rows) == [[0, 1, 2], [3]]
    assert [row.round_seconds for row in rows] == [87, 87, 87, None]
    assert session.round_name(rows[3]) == "8-de_nuke-r03-s000000"
    with pytest.raises(ValueError, match="do not cover their round"):
        session.rounds_of(rows[:2])  # the round's third client has no row


def test_a_round_runs_as_long_as_the_paper_ran_it():
    # the recorded length, rounded down to ten seconds and capped at 120 s; four latents a second
    assert session.round_latents(87) == 1 + 4 * 80
    assert session.round_latents(133) == 1 + 4 * 120
    assert session.round_latents(133, max_seconds=60) == 1 + 4 * 60


class Process:
    """Stands in for ``subprocess.Popen``: records the command; the client of GPU ``1`` fails."""

    started: list["Process"] = []

    def __init__(self, command, env, stdout, stderr):
        self.command, self.gpu = command, env["CUDA_VISIBLE_DEVICES"]
        self.threads = env["OMP_NUM_THREADS"]
        stdout.write(f"client on GPU {self.gpu}\n")
        Process.started.append(self)

    def wait(self) -> int:
        return int(self.gpu == "1")


def test_a_round_starts_a_process_per_client_on_one_world_state(tmp_path, monkeypatch):
    monkeypatch.setattr(session.subprocess, "Popen", Process)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")  # the caller's environment wins
    Process.started = []
    rows = _index(tmp_path, [_row(7, s, [0, 1, 2]) for s in (0, 1, 2)])
    out = tmp_path / "round"
    failed = session.run_round(["client"], [0, 1, 2], rows, gpus=["0", "1", "5"], out_dir=out)
    assert failed == ["7-de_nuke-r03-p01"]
    world_state = str(out / "world_state")
    for i, process in enumerate(Process.started):
        client_dir = out / rows[i].media_id
        assert process.command == [
            "client",
            "--index-row",
            str(i),
            "--world-state-dir",
            world_state,
            "--out-dir",
            str(client_dir),
        ]
        assert (client_dir / "client.log").read_text() == f"client on GPU {process.gpu}\n"
    assert [(p.gpu, p.threads) for p in Process.started] == [("0", "2"), ("1", "2"), ("5", "2")]
    # a session never reuses a world state; a client needs a GPU
    (out / "world_state" / "block").write_text("")
    with pytest.raises(ValueError, match="fresh"):
        session.run_round(["client"], [0], rows, gpus=["0"], out_dir=out)
    with pytest.raises(
        ValueError, match="^3 clients run at once and need a GPU id each, got 1: 0$"
    ):
        session.run_round(["client"], [0, 1, 2], rows, gpus=["0"], out_dir=tmp_path / "other")


def test_the_clients_take_the_visible_gpus_in_turn(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    eight = ["0", "1", "2", "3", "4", "5", "6", "7"]
    assert session.visible_gpus() == eight
    assert session.place_clients(10, eight) == [*eight, "0", "1"]
    assert session.place_clients(3, eight) == ["0", "1", "2"]
    # as the caller's CUDA_VISIBLE_DEVICES names them, as many as CUDA finds of them
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4, GPU-8f2c,7,9")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert session.visible_gpus() == ["4", "GPU-8f2c", "7"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert session.visible_gpus() == []


def test_a_gpu_without_the_memory_for_its_clients_is_named(monkeypatch):
    """nvidia-smi's memory of each GPU by index and by UUID; 15 GiB a client."""
    listed = "0, GPU-a1, 97871\n1, GPU-b2, 24564\n2, MIG-c3, [N/A]\n"

    def run(command, capture_output, text, check):
        assert command[0] == "nvidia-smi" and "--query-gpu=index,uuid,memory.total" in command
        return subprocess.CompletedProcess(command, 0, stdout=listed)

    monkeypatch.setattr(session.subprocess, "run", run)
    memory = session.gpu_memory()
    assert memory == pytest.approx({"0": 95.58, "GPU-a1": 95.58, "1": 23.99, "GPU-b2": 23.99}, 1e-3)
    session.check_gpu_memory(["0"] * 6 + ["1", "7"], memory)  # 90 GiB of 95.6; 7: not listed
    with pytest.raises(ValueError) as short:
        session.check_gpu_memory(["0"] * 7 + ["GPU-b2", "GPU-b2", "1"], memory)
    assert str(short.value) == (
        "GPU 0 has 96 GiB for 7 clients; GPU GPU-b2 has 24 GiB for 2 clients: a client takes about"
        " 15 GiB, place them on more GPUs (--gpus, CUDA_VISIBLE_DEVICES)"
    )

    def missing(command, capture_output, text, check):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(session.subprocess, "run", missing)
    assert session.gpu_memory() == {}
