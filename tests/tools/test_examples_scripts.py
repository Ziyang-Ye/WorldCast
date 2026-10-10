"""``examples/grid.sh`` and ``examples/run.sh`` on the CPU: the grid's layouts, what grid.sh says
when it finds no ffmpeg, a grid written in one pass, one that lost frames and one of no frame, and
run.sh end to end on the synthetic round with tiny random weights: the decode on the session's GPUs
and device, the frames of its videos out of the session's, a failed decode, a grid that lost frames,
one of a client's video that is short, and the closed loop, whose latents it does not compare."""

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import tests.engine.inference.support as sw
from tests.tools.support import load_tool, mp4_boxes
from worldcast.data.latent_cache import load_first_latent
from worldcast.engine.inference.reference import first_frame_fingerprint

REPO = Path(__file__).resolve().parents[2]
#: The stand-ins of the tools' processes in the run.sh tests (their ``sitecustomize``).
STAND_INS = Path(__file__).resolve().parent / "stand_ins"
#: The two clients of the run.sh tests.
MEDIA = [sw.media_id(0), sw.media_id(1)]


@pytest.fixture
def bash() -> str:
    found = shutil.which("bash")
    if found is None:
        pytest.skip("no bash")
    return found


def write_video(path: Path, level: int) -> None:
    """Two flat 32 x 16 frames at the grey ``level``."""
    write_frames(path, [np.full((16, 32, 3), level, np.uint8)] * 2)


def write_ramps(path: Path) -> None:
    """40 frames of 32 x 16 colour ramps that move from frame to frame; none takes 4 KiB of a grid
    of them, so that a lost page of the grid holds the start of a frame."""
    y, x = np.mgrid[0:16, 0:32]
    ramps = [
        np.stack([(x * 8 + 3 * i) % 256, (y * 16 + 5 * i) % 256, np.full_like(x, 7 * i % 256)], -1)
        for i in range(40)
    ]
    write_frames(path, [ramp.astype(np.uint8) for ramp in ramps])


def write_frames(path: Path, frames: list[np.ndarray]) -> None:
    """uint8 ``[16, 32, 3]`` frames as an mp4."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    writer = imageio_ffmpeg.write_frames(str(path), (32, 16))
    writer.send(None)
    for frame in frames:
        writer.send(frame.tobytes())
    writer.close()


@pytest.mark.parametrize(
    "clients, layout", [(1, (1, 1)), (3, (3, 1)), (4, (2, 2)), (7, (5, 2)), (10, (5, 2))]
)
def test_grid_tiles_the_clients_in_order(bash, tmp_path, clients, layout):
    levels = [20 + 20 * i for i in range(clients)]
    for i, level in enumerate(levels):
        write_video(tmp_path / f"p{i}.mp4", level)
    out = tmp_path / "grid.mp4"
    inputs = [str(tmp_path / f"p{i}.mp4") for i in range(clients)]
    env = dict(os.environ, PYTHON=sys.executable)
    script = [bash, str(REPO / "examples" / "grid.sh"), str(out), *inputs]
    done = subprocess.run(script, env=env, capture_output=True, text=True, check=True)
    columns, rows = layout
    assert done.stdout == f"{out} decodes to 2 frames: {clients} clients ({columns}x{rows})\n"
    tool = load_tool("make_showcase")
    assert tool.grid_layout(clients) == layout  # the showcase cuts a grid as grid.sh tiles it
    frames = list(tool.video_frames(out))
    assert len(frames) == 2 and frames[0].shape == (384 * rows, 672 * columns, 3)
    tiles = range(columns * rows)
    centres = [frames[0][192 + 384 * (i // columns), 336 + 672 * (i % columns)] for i in tiles]
    expected = levels + [0] * (columns * rows - clients)  # black tiles end the last row
    assert np.abs(np.array(centres, int).T - expected).max() <= 6


def test_the_grid_is_written_in_one_pass_and_named_once_it_decodes(bash, tmp_path):
    """Its index follows its frames: ffmpeg writes the file in one pass and never reads it back (as
    moving the index to the front does); no hidden partial is left."""
    write_video(tmp_path / "p0.mp4", 20)
    out = tmp_path / "grid.mp4"
    env = dict(os.environ, PYTHON=sys.executable)
    script = [bash, str(REPO / "examples" / "grid.sh"), str(out), str(tmp_path / "p0.mp4")]
    subprocess.run(script, env=env, capture_output=True, text=True, check=True)
    assert [kind for kind, _, _ in mp4_boxes(out)] == ["ftyp", "free", "mdat", "moov"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["grid.mp4", "p0.mp4"]


@pytest.mark.parametrize("loss", ["frames", "tail", "page", "bytes"])
def test_a_grid_that_lost_frames_is_not_written(bash, tmp_path, lossy_ffmpeg, loss):
    """Written by an ffmpeg that loses its frames (all of them, those from the page at the middle
    on, that page alone, or 256 bytes in a frame, which decodes with errors), the grid's decode
    stops at the first error: grid.sh says so with the decoder's, fails and leaves no grid."""
    videos = tmp_path / "videos"
    videos.mkdir()
    write_ramps(videos / "p0.mp4")
    out = videos / "grid.mp4"
    script = [bash, str(REPO / "examples" / "grid.sh"), str(out), str(videos / "p0.mp4")]
    env = dict(os.environ, FFMPEG=lossy_ffmpeg(loss))
    done = subprocess.run(script, env=env, capture_output=True, text=True)
    assert done.returncode == 1 and done.stdout == ""
    said = re.fullmatch(
        rf"grid\.sh: {re.escape(str(out))} not written: it decodes to (\d+) of its 40 frames;"
        r" ffmpeg: (\[h264 @ 0x[0-9a-f]+\] .+) \(the video file is incomplete\)\n",
        done.stderr,
    )
    assert said, done.stderr
    decoded, error = int(said[1]), said[2]
    # the frames before the first error: none of lost frames, some before a loss at the middle,
    # any before the largest frame (which lost bytes)
    assert decoded in {"frames": [0], "bytes": range(40)}.get(loss, range(1, 40))
    assert (loss == "bytes") != bool(re.search(r"Invalid NAL unit size \(0 > \d+\)\.$", error))
    assert sorted(path.name for path in videos.iterdir()) == ["p0.mp4"]


def test_a_grid_of_no_frame_is_not_written(bash, tmp_path):
    """An ffmpeg that says nothing, encodes and decodes nothing."""
    write_video(tmp_path / "p0.mp4", 20)
    out = tmp_path / "grid.mp4"
    script = [bash, str(REPO / "examples" / "grid.sh"), str(out), str(tmp_path / "p0.mp4")]
    env = dict(os.environ, FFMPEG=shutil.which("true"))
    done = subprocess.run(script, env=env, capture_output=True, text=True)
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr == (
        f"grid.sh: {out} not written: it decodes to 0 of its 0 frames (the video file is"
        " incomplete)\n"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["p0.mp4"]


def test_grid_names_what_it_misses(bash, tmp_path):
    out, video = tmp_path / "grid.mp4", tmp_path / "p0.mp4"
    env = {"PATH": str(tmp_path / "bin")}  # neither python nor ffmpeg
    script = [bash, str(REPO / "examples" / "grid.sh"), str(out), str(video)]
    done = subprocess.run(script, env=env, capture_output=True, text=True)
    assert done.returncode == 1
    assert done.stderr == (
        "grid.sh: no python to find imageio-ffmpeg with: set PYTHON to the package's"
        " interpreter, or FFMPEG\n"
    )
    false = shutil.which("false")
    done = subprocess.run(script, env=dict(env, PYTHON=false), capture_output=True, text=True)
    assert done.returncode == 1
    assert done.stderr == (
        f"grid.sh: no ffmpeg: {false} has no imageio-ffmpeg (pip install imageio-ffmpeg) and none"
        " is on PATH; or set FFMPEG\n"
    )


def build_case(bash, tmp_path, *, seconds=None, state_model=False):
    """Two clients of the synthetic round (``seconds`` long) with tiny random weights, for
    ``examples/run.sh`` under the stand-ins, and with a small random state model if asked:
    ``run(arguments, **variables)`` runs ``bash examples/run.sh synthetic <arguments>`` with these
    more environment variables; also the output directory and the round's in it. The weights'
    config names no device: the session runs on the CPU only by ``--device cpu``, and so does the
    decode."""
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("PIL")
    world = sw.make_world(
        tmp_path / "world", clients=(0, 1), **({"seconds": seconds} if seconds else {})
    )
    weights = sw.make_weights(tmp_path / "weights")
    if state_model:
        weights |= {
            key.removeprefix("paths."): value
            for key, value in sw.write_state_model(tmp_path / "state_model.pt").items()
        }
    paths = {key: value for key, value in world.items() if key != "tables"}
    data = tmp_path / "examples" / "data" / "synthetic"
    data.mkdir(parents=True)
    (data / "config.yaml").write_text(
        yaml.safe_dump({"paths": paths, "run": {"latent_frames": 29}})
    )
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({"paths": weights}))
    with open(tmp_path / "ticks.pkl", "wb") as file:
        pickle.dump(world["tables"], file)
    recorded = {
        media_id: dict(
            first_frame=first_frame_fingerprint(
                load_first_latent(world["latent_cache_root"], media_id, 0)
            ),
            latents_0_24="0" * 16,  # not this run's: the comparison fails, the case does not
        )
        for media_id in MEDIA
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(dict(cases=dict(synthetic=dict(latent_frames=29, fingerprints=recorded))))
    )
    env = dict(
        os.environ,
        PYTHON=sys.executable,
        DATA=str(tmp_path / "examples"),
        WEIGHTS=str(tmp_path / "paths.yaml"),
        OUT=str(tmp_path / "out"),
        PYTHONPATH=os.pathsep.join([str(STAND_INS), str(REPO)]),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        WORLDCAST_TEST_TICKS=str(tmp_path / "ticks.pkl"),
        WORLDCAST_TEST_MANIFEST=str(manifest),
    )

    def run(arguments: list[str], **variables) -> subprocess.CompletedProcess:
        script = [bash, str(REPO / "examples" / "run.sh"), "synthetic", *arguments]
        return subprocess.run(
            script, env={**env, **variables}, capture_output=True, text=True, timeout=600
        )

    return run, tmp_path / "out", tmp_path / "out" / f"{sw.MATCH}-{sw.MAP}-r{sw.ROUND:02d}-s000000"


@pytest.fixture
def case(bash, tmp_path):
    """The synthetic round of :func:`build_case`."""
    return build_case(bash, tmp_path)


@pytest.fixture
def closed_loop_case(bash, tmp_path):
    """The synthetic round of :func:`build_case`, 11 s long (the state model reads 10 s windows),
    with a small random state model."""
    return build_case(bash, tmp_path, seconds=11.0, state_model=True)


def polls_every_50_ms(round_dir: Path) -> bool:
    """Whether each client polled the world state every 50 ms."""
    return all(
        f"{media_id} polls the world state: poll_s=0.05\n"
        in (round_dir / media_id / "client.log").read_text()
        for media_id in MEDIA
    )


def test_run_renders_a_case(case):
    """Both clients on "GPU" 3 (the CPU), spelled with ``=`` and a space, with the showcase."""
    run, out, round_dir = case
    done = run(["--gpus=3, 3", "--device=cpu"], SHOWCASE="1")
    assert done.returncode == 0, done.stdout + done.stderr

    assert sorted(path.parent.name for path in round_dir.glob("*/video.mp4")) == MEDIA
    assert polls_every_50_ms(round_dir)
    # one decode on the GPU of both clients, on the session's device
    assert done.stdout.count("the tiny VAE") == 1
    assert "the tiny VAE on cpu, CUDA_VISIBLE_DEVICES=3\n" in done.stdout
    for media_id in MEDIA:
        assert json.loads((round_dir / media_id / "client.json").read_text())["latent_frames"] == 29
    tool = load_tool("make_showcase")
    grid = list(tool.video_frames(out / "grid.mp4"))
    showcase = list(tool.video_frames(out / "showcase.mp4"))
    assert len(grid) == len(showcase) == 113
    assert grid[0].shape == (384, 2 * 672, 3) and showcase[0].shape == (384, 2 * 672 + 384, 3)
    assert sorted(path.name for path in out.iterdir() if path.is_file()) == [
        "grid.mp4",
        "showcase.mp4",
    ]
    summary = done.stdout.split("\n\n")[-1]
    assert summary.startswith(
        f"synthetic: each client's latents.npy, video.mp4 and client.json in"
        f" {out}/<round>/<media_id>/\n{out}/grid.mp4 decodes to 113 of the session's 113 frames: 2"
        f" clients (2x1)\n{out}/showcase.mp4 decodes to 113 of the session's 113 frames: 2 views"
        " (2x1) and the map\n"
    )
    assert summary.count("DIFF   latents_0_24") == 2
    assert summary.endswith(
        "0/2 clients match their reference run bit for bit; 2 on latents 0-24 only (their"
        " reference run had another length)\n(the reference stack: docs/reproduce.md,"
        ' "Bit-exact reproduction")\n'
    )


def test_the_closed_loop_does_not_compare_the_latents(closed_loop_case):
    """``--set player_state.source=predicted``: the clients take their positions from the state
    model, so run.sh says the latents are not compared with the reference runs' (GT player states)
    and does not check them."""
    run, out, round_dir = closed_loop_case
    done = run(["--gpus", "3,3", "--device", "cpu", "--set", "player_state.source=predicted"])
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(path.parent.name for path in round_dir.glob("*/video.mp4")) == MEDIA
    summary = done.stdout.split("\n\n")[-1]
    assert summary.startswith(
        f"synthetic: each client's latents.npy, video.mp4 and client.json in"
        f" {out}/<round>/<media_id>/\n{out}/grid.mp4 decodes to 113 of the session's 113 frames: 2"
        " clients (2x1)\n"
    )
    assert summary.endswith(
        "player_state.source=predicted: the latents are not compared with the reference runs'"
        " (GT player states)\n"
    )
    assert "DIFF" not in done.stdout and "match their reference run" not in done.stdout


def test_without_gpus_the_clients_take_the_visible_gpus_in_turn(case):
    """One visible "GPU", 5, for both clients, as run_session.py places them: one decode."""
    run, _, round_dir = case
    done = run(["--device", "cpu"], CUDA_VISIBLE_DEVICES="5", WORLDCAST_TEST_GPUS="1")
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(path.parent.name for path in round_dir.glob("*/video.mp4")) == MEDIA
    assert done.stdout.count("the tiny VAE") == 1
    assert "the tiny VAE on cpu, CUDA_VISIBLE_DEVICES=5\n" in done.stdout


def test_on_the_cpu_without_a_gpu_one_decode_takes_every_client(case):
    run, _, round_dir = case
    done = run(["--device", "cpu"])
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(path.parent.name for path in round_dir.glob("*/video.mp4")) == MEDIA
    assert done.stdout.count("the tiny VAE") == 1
    assert "the tiny VAE on cpu, CUDA_VISIBLE_DEVICES=\n" in done.stdout


def test_a_failed_decode_fails_the_run_after_every_decode(case):
    """The clients on "GPUs" 3 and 5, each decoded on its own; the decode on 5 fails."""
    run, out, round_dir = case
    done = run(["--gpus", "3,5", "--device", "cpu"], WORLDCAST_TEST_FAILING_GPU="5")
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stderr.endswith("a client's decode failed: see above\n")
    assert polls_every_50_ms(round_dir)
    assert "the tiny VAE on cpu, CUDA_VISIBLE_DEVICES=3\n" in done.stdout
    assert "the tiny VAE on cpu, CUDA_VISIBLE_DEVICES=5\n" in done.stdout
    assert "RuntimeError: CUDA out of memory on GPU 5" in done.stderr
    # the other decode finished; nothing was tiled
    assert sorted(path.parent.name for path in round_dir.glob("*/video.mp4")) == MEDIA[:1]
    assert not (out / "grid.mp4").exists()


def test_a_grid_that_lost_frames_fails_the_run(case, lossy_ffmpeg):
    """The grid's ffmpeg loses its frames from the middle on: the run stops with grid.sh's reason,
    before the showcase and the summary, and leaves no grid."""
    run, out, round_dir = case
    done = run(["--device", "cpu"], SHOWCASE="1", FFMPEG=lossy_ffmpeg("tail"))
    assert done.returncode == 1, done.stdout + done.stderr
    assert sorted(path.parent.name for path in round_dir.glob("*/video.mp4")) == MEDIA
    said = re.search(
        rf"grid\.sh: {re.escape(str(out))}/grid\.mp4 not written: it decodes to (\d+) of its 113"
        r" frames; ffmpeg: .+ \(the video file is incomplete\)\n\Z",
        done.stderr,
    )
    assert said and 0 < int(said[1]) < 113, done.stderr
    assert "synthetic: each client's" not in done.stdout
    assert not [path for path in out.iterdir() if path.is_file()]


def test_a_short_client_video_fails_the_run(case, lossy_ffmpeg):
    """A client's video.mp4 that holds its first 3 s only, as the grid's ffmpeg reads it: the grid
    decodes to every frame it encoded, 48, and the run stops on 48 of the session's 113 frames,
    before the showcase and the summary."""
    run, out, _ = case
    done = run(["--device", "cpu"], SHOWCASE="1", FFMPEG=lossy_ffmpeg("input"))
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stderr.endswith(
        f"run.sh: {out}/grid.mp4 decodes to 48 of the session's 113 frames (e.g. a client's"
        " video.mp4 is short)\n"
    )
    assert "synthetic: each client's" not in done.stdout
    assert sorted(path.name for path in out.iterdir() if path.is_file()) == ["grid.mp4"]
