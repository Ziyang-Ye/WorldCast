"""``tools/make_showcase.py`` on the CPU: a box is the field's projection of the recorded rows, none
in a view that shows the scope and none of a player beside the view; the map faces each client its
way and keeps its players clear of its time and legend; a name covers no label and no nearer
player's name or box, and the colours stand out on dark and bright scenes; the showcase of a
session or of a reference run's grid lays the views out as ``examples/grid.sh`` does, beside the
map; it is written in one pass, and not at all when it lost its frames."""

import json
import re
import sys

import numpy as np
import pytest
import torch
import yaml

import tests.engine.inference.support as sw
from tests.tools.support import load_tool, mp4_boxes
from worldcast.config.inference import load_config
from worldcast.data import read_round_index
from worldcast.data.latents import FRAME_SIZE
from worldcast.engine.inference.loading import load_window
from worldcast.player_state.projection import project_players

CLIENTS = (0, 3, 7)


def colour(pixel) -> str:
    return "#{:02X}{:02X}{:02X}".format(*pixel)


def luminance(color) -> float:
    """The relative luminance of an sRGB colour, ``#RRGGBB`` or a tuple (WCAG 2)."""
    rgb = np.array(list(bytes.fromhex(color[1:])) if isinstance(color, str) else color) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722])


def contrast(a, b) -> float:
    """The contrast ratio of two colours (WCAG 2; 3 for a graphic that has to be seen)."""
    high, low = sorted([luminance(a), luminance(b)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def recorded_round(tool, frames: int, players: int, **fields):
    """A round of ``players`` players standing at the origin, slot 0 the one client, its view
    without boxes; ``fields`` replace these."""
    states = np.zeros((players, frames, 6), np.float32)
    states[..., 5] = 1.0
    round_fields = dict(
        media_ids=("p0",),
        clients=(0,),
        states=states,
        team_ids=np.zeros(players, np.int64),
        boxes=np.zeros((1, frames, players, 4), np.float32),
        shown=np.zeros((1, frames, players), bool),
        in_view=np.zeros((1, frames, players), bool),
    )
    return tool.RecordedRound(**{**round_fields, **fields})


def test_the_grid_layout_is_that_of_grid_sh():
    tool = load_tool("make_showcase")
    layouts = [tool.grid_layout(views) for views in range(1, 11)]
    assert layouts == [(1, 1), (2, 1), (3, 1), (2, 2), (5, 1), *[(5, 2)] * 5]


def test_the_map_faces_a_client_its_way_and_trails_the_players():
    pytest.importorskip("PIL")
    tool = load_tool("make_showcase")
    frames = 60
    recorded = recorded_round(tool, frames, players=3)
    states = recorded.states
    states[0, :, 3] = 90.0  # the client (slot 0) at the origin faces +y, north on the map
    states[1, :, 0] = np.linspace(-600.0, 600.0, frames)  # slot 1 walks east, 400 u south of it
    states[1, :, 1] = -400.0
    states[2, :, :2] = [0.0, 600.0]
    showcase = tool.Showcase(recorded)
    image = np.asarray(showcase.map_at(frames - 1))
    reach = tool.VIEW_LENGTH * showcase.side
    x, y = showcase.on_map(0.0, 0.0)
    assert image[round(y - reach / 2), round(x)].tolist() != list(tool.MAP_GROUND)  # its cone
    assert image[round(y + reach / 2), round(x)].tolist() == list(tool.MAP_GROUND)
    # slot 1's trail: the last 3 s (48 frames) in its colour, before them its faint path
    x, y = showcase.on_map(0.0, -400.0)  # frame 29.5
    assert colour(image[round(y), round(x)]) == showcase.colors[1]
    x, y = showcase.on_map(-500.0, -400.0)  # frame 4.9
    assert image[round(y), round(x)].tolist() == list(tool.MAP_PATHS)


def test_the_map_keeps_its_players_clear_of_its_time_and_legend():
    """The players at the round's corners, the client (the largest dot) among them."""
    pytest.importorskip("PIL")
    tool = load_tool("make_showcase")
    recorded = recorded_round(tool, 2, players=4)
    corners = [[-2000.0, 2400.0], [2000.0, 2400.0], [-2000.0, -2400.0], [2000.0, -2400.0]]
    recorded.states[:, :, :2] = np.array(corners)[:, None]
    showcase = tool.Showcase(recorded)
    radius = tool.CLIENT_RADIUS * showcase.side
    time_band = tool.CHIP_GAP + tool.CHIP_HEIGHT  # the time's chip ends there
    legend_band = showcase.side - tool.CHIP_GAP - tool.CHIP_HEIGHT  # the legend's begins
    for x, y in corners:
        px, py = showcase.on_map(x, y)
        assert time_band < py - radius and py + radius < legend_band
        assert 0 < px - radius and px + radius < showcase.side
    # as large as the bands leave the map of this tall round: its players reach them
    tops = [showcase.on_map(x, y)[1] for x, y in corners]
    margin = tool.MAP_MARGIN_U * showcase.scale
    assert min(tops) - radius - margin == pytest.approx(time_band)
    assert max(tops) + radius + margin == pytest.approx(legend_band)


def has_colour(view: np.ndarray, color: str) -> bool:
    """Whether a pixel of ``view`` (or a part of it) has the colour ``color``."""
    return any(colour(pixel) == color for pixel in view.reshape(-1, 3))


def test_a_name_covers_no_label_nor_a_nearer_players_name_or_box():
    """P2 near the camera; P3 far behind it, its name where P2's is; P4 far, its name inside P2's
    box; P5 far, its name under the view's label; P6 far and clear of the others: only P2's and
    P6's names are drawn, in their colours on a black edge, as is P2's box."""
    pytest.importorskip("PIL")
    tool = load_tool("make_showcase")
    boxes = np.zeros((1, 1, 6, 4), np.float32)
    boxes[0, 0, 1] = [300.0, 100.0, 340.0, 300.0]  # P2: 200 px tall
    boxes[0, 0, 2] = [312.0, 104.0, 328.0, 120.0]  # P3: 16 px
    boxes[0, 0, 3] = [312.0, 240.0, 328.0, 256.0]  # P4: its name at y 222-240, in P2's box
    boxes[0, 0, 4] = [10.0, 20.0, 26.0, 36.0]  # P5: its name at y 2-20, on the label
    boxes[0, 0, 5] = [500.0, 200.0, 516.0, 216.0]  # P6
    shown = np.array([[[False, True, True, True, True, True]]])
    recorded = recorded_round(tool, 1, players=6, boxes=boxes, shown=shown, in_view=shown)
    sand = np.full((384, 672, 3), (200, 170, 125), np.uint8)  # the sand of Dust2
    showcase = tool.Showcase(recorded)
    view = np.asarray(showcase.view(sand, 0, 0))
    p2, p3, p4, p5, p6 = showcase.colors[1:6]
    assert has_colour(view[80:100, 300:340], p2) and has_colour(view[180:200, 500:516], p6)
    assert not has_colour(view[30:100], p3)  # P2's name is there
    assert not has_colour(view[222:240, 312:328], p4)  # P2's box is there
    assert not has_colour(view[:20], p5) and has_colour(
        view[20:37, 10:27], p5
    )  # the label is there
    assert has_colour(view[104:121, 312:329], p3)  # the boxes are drawn
    # P2's box, bold: three pixels of its colour between two black ones
    black, ground = "#000000", "#C8AA7D"
    row = [colour(pixel) for pixel in view[200, 296:306]]
    assert row == [ground] * 3 + [black, *[p2] * 3, black] + [ground] * 2


def scope(level: int) -> np.ndarray:
    """A view through the scope: black beyond 175 px of its centre, a disc at ``level``."""
    y, x = np.mgrid[:384, :672]
    disc = np.hypot(y - 191.5, x - 335.5) < 175
    return np.where(disc[..., None], np.uint8(level), np.uint8(0)).repeat(3, axis=-1)


def test_a_view_that_shows_the_scope_has_no_boxes_and_says_so():
    """The view shows the scope whatever the client's scope label says; a dark tunnel and a black
    view do not show it. A player beside the view has no box there."""
    pytest.importorskip("PIL")
    tool = load_tool("make_showcase")
    rng = np.random.default_rng(0)
    tunnel = rng.integers(0, 30, (384, 672, 3), dtype=np.uint8)  # as dark as Ancient's darkest
    black = np.zeros((384, 672, 3), np.uint8)
    assert tool.shows_scope(scope(80))
    assert not any(map(tool.shows_scope, [tunnel, black, scope(20), np.full_like(black, 120)]))
    boxes = np.zeros((1, 1, 3, 4), np.float32)
    boxes[0, 0, 1] = [300.0, 100.0, 340.0, 300.0]  # P2, in the disc
    boxes[0, 0, 2] = [-150.0, 100.0, 20.0, 450.0]  # P3, near and beside the view: its side in it
    shown = np.array([[[False, True, True]]])
    recorded = recorded_round(tool, 1, players=3, boxes=boxes, shown=shown, in_view=shown)
    showcase = tool.Showcase(recorded)
    p2, p3 = showcase.colors[1:3]
    scoped = np.asarray(showcase.view(scope(80), 0, 0))
    assert not has_colour(scoped[30:], p2) and has_colour(scoped[:30], tool.NOTE_COLOR)
    seen = np.asarray(showcase.view(tunnel, 0, 0))
    assert has_colour(seen[100:301, 299:342], p2) and not has_colour(seen[30:], p3)
    assert not has_colour(seen[:30], tool.NOTE_COLOR)


def test_the_colours_stand_out_on_dark_and_bright_scenes():
    pytest.importorskip("PIL")
    tool = load_tool("make_showcase")
    colors = tool.Showcase(recorded_round(tool, 1, players=10)).colors
    assert colors == list(tool.PLAYER_COLORS)
    dark, sand = (48, 48, 48), (200, 170, 125)  # a tunnel of Ancient, the sand of Dust2
    assert min(contrast(color, dark) for color in colors) >= 3.0
    assert contrast("#5E5CE6", dark) < 3.0  # a darker P9 is lost in the tunnel
    # on sand the black edge stands out where a colour does not
    assert contrast(tool.EDGE, sand) >= 3.0 > contrast(colors[7], sand)


def make_video(path, frames: int, levels) -> None:
    """``frames`` frames of 672 x 384 tiles side by side, each flat at its grey level."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    frame_h, frame_w = FRAME_SIZE
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = [np.full((frame_h, frame_w, 3), level, np.uint8) for level in levels]
    writer = imageio_ffmpeg.write_frames(str(path), (frame_w * len(levels), frame_h))
    writer.send(None)
    for _ in range(frames):
        writer.send(np.concatenate(tiles, axis=1).tobytes())
    writer.close()


@pytest.fixture
def case(tmp_path, monkeypatch):
    """The synthetic round with three clients, its config file and the config."""
    pytest.importorskip("PIL")
    world = sw.make_world(tmp_path / "world", clients=CLIENTS)
    sw.patch_ticks(monkeypatch, world["tables"])
    paths = {key: value for key, value in world.items() if key != "tables"}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": paths, "run": {"latent_frames": 29}}))
    return str(config), load_config([config])


def test_the_showcase_of_a_session(case, tmp_path, capsys):
    config, cfg = case
    tool = load_tool("make_showcase")
    session = tmp_path / "session"
    for row, slot in enumerate(CLIENTS):
        client = session / "round" / sw.media_id(slot)
        make_video(client / "video.mp4", frames=6, levels=[60])
        (client / "client.json").write_text(
            json.dumps({"media_id": sw.media_id(slot), "index_row": row})
        )
    out = tmp_path / "showcase.mp4"
    assert tool.main(["--config", config, "--session", str(session), "--out", str(out)]) == 0
    assert capsys.readouterr().out == f"{out} decodes to 6 frames: 3 views (3x1) and the map\n"
    frames = list(tool.video_frames(out))
    assert len(frames) == 6 and frames[0].shape == (384, 3 * 672 + 384, 3)

    rows = read_round_index(cfg.paths.round_index)
    recorded = tool.load_round(cfg, rows)
    assert recorded.clients == CLIENTS and recorded.frame_count == 113  # 29 latent frames
    # a box's foot is the field's projection of the recorded rows into the client's camera
    window = load_window(cfg, rows[1])
    states = window.item.player_states.transpose(0, 1)[None]
    client = states[:, :, CLIENTS[1]]
    uv, _, in_front, _ = project_players(
        states[..., :3], client[..., :3], client[..., 3], client[..., 4], grid_h=384, grid_w=672
    )
    boxes = torch.from_numpy(recorded.boxes[1])
    assert torch.allclose((boxes[..., 0] + boxes[..., 2]) / 2, uv[0, ..., 0], atol=1e-3)
    assert torch.allclose(boxes[..., 3], uv[0, ..., 1], atol=1e-3)
    # shown: never the client, never one behind its camera; bold: shown where its GT label of the
    # video frame has the player in view; both kinds occur
    shown = recorded.shown[1]
    labels = (window.item.client_visibility.T > 0.5) & window.item.client_visibility_valid.T
    assert not shown[:, CLIENTS[1]].any() and not (shown & ~in_front[0].numpy()).any()
    assert np.array_equal(recorded.in_view[1], shown & labels.numpy())
    assert recorded.in_view[1].any() and (shown & ~recorded.in_view[1]).any()

    # drawn where it stands: a box's left edge in its player's colour, the client's dot on the map
    showcase = tool.Showcase(recorded)
    grey = np.full((384, 672, 3), 60, np.uint8)
    drawn = list(showcase.frames(iter([[grey] * 3] * recorded.frame_count)))
    inside = (recorded.boxes[..., 0] > 4) & (recorded.boxes[..., 2] < 668)
    inside &= (recorded.boxes[..., 1] > 24) & (recorded.boxes[..., 3] < 380)
    c, t, slot = np.argwhere(recorded.shown & inside)[0]
    x0, y0, _, y1 = recorded.boxes[c, t, slot]
    edge = drawn[t][int((y0 + y1) / 2), c * 672 + int(x0) - 1 : c * 672 + int(x0) + 2]
    assert showcase.colors[slot] in [colour(pixel) for pixel in edge]
    x, y = showcase.on_map(*recorded.states[CLIENTS[0], 0, :2])
    assert colour(drawn[0][int(y), 3 * 672 + int(x)]) == showcase.colors[CLIENTS[0]]


def test_the_showcase_of_a_reference_run(case, tmp_path, capsys):
    config, _ = case
    tool = load_tool("make_showcase")
    grid = tmp_path / "expected" / "synthetic.mp4"
    make_video(grid, frames=4, levels=[40, 120, 200])
    out = tmp_path / "showcase.mp4"
    assert tool.main(["--config", config, "--expected", str(grid), "--out", str(out)]) == 0
    assert capsys.readouterr().out == f"{out} decodes to 4 frames: 3 views (3x1) and the map\n"
    # its index follows its frames: ffmpeg writes the file in one pass and never reads it back
    assert [kind for kind, _, _ in mp4_boxes(out)] == ["ftyp", "free", "mdat", "moov"]
    frames = list(tool.video_frames(out))
    assert len(frames) == 4
    # each view where the grid has it, the map on the right (two lossy encodings away)
    corners = frames[0][370, [660, 672 + 660, 2 * 672 + 660]].astype(int)
    assert np.abs(corners - np.array([40, 120, 200])[:, None]).max() <= 6
    assert frames[0][370, 3 * 672 + 380].tolist() == pytest.approx([11, 11, 13], abs=6)


def test_what_does_not_fit_is_a_usage_error(case, tmp_path, capsys):
    config, _ = case
    tool = load_tool("make_showcase")
    session = tmp_path / "session" / sw.media_id(0)
    session.mkdir(parents=True)
    narrow = tmp_path / "narrow.mp4"
    make_video(narrow, frames=1, levels=[0])
    for arguments, message in (
        (["--session", str(tmp_path / "nowhere")], "no client.json under"),
        (["--expected", str(narrow)], f"{narrow} is 672 x 384, not 2016 x 384"),
        (["--expected", str(tmp_path / "missing.mp4")], "missing.mp4 not found"),
    ):
        with pytest.raises(SystemExit) as refused:
            tool.main(["--config", config, *arguments, "--out", str(tmp_path / "out.mp4")])
        assert refused.value.code == 2 and message in capsys.readouterr().err
    (session / "client.json").write_text(json.dumps({"media_id": sw.media_id(0), "index_row": 0}))
    with pytest.raises(SystemExit) as refused:
        tool.main(["--config", config, "--session", str(tmp_path / "session"), "--out", "x.mp4"])
    assert "video.mp4 not found: decode the session first" in capsys.readouterr().err
    (session / "client.json").write_text(json.dumps({"media_id": sw.media_id(3), "index_row": 0}))
    with pytest.raises(SystemExit) as refused:
        tool.main(["--config", config, "--session", str(tmp_path / "session"), "--out", "x.mp4"])
    assert f"{sw.media_id(3)} is not row 0 of the config's round index" in capsys.readouterr().err
    assert not (tmp_path / "out.mp4").exists()


def test_a_session_of_several_rounds_is_refused_by_its_round_folders(case, tmp_path, capsys):
    config, cfg = case
    tool = load_tool("make_showcase")
    index = tmp_path / "round_index.jsonl"  # the third client starts another round
    lines = open(cfg.paths.round_index).read().splitlines()
    lines[2] = json.dumps({**json.loads(lines[2]), "round": sw.ROUND + 1})
    index.write_text("\n".join(lines) + "\n")
    session = tmp_path / "session"
    for folder, row in (("first", 0), ("second", 2)):
        client = session / folder / sw.media_id(CLIENTS[row])
        client.mkdir(parents=True)
        summary = {"media_id": sw.media_id(CLIENTS[row]), "index_row": row}
        (client / "client.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit) as refused:
        tool.main(
            ["--config", config, "--set", f"paths.round_index={index}"]
            + ["--session", str(session), "--out", str(tmp_path / "out.mp4")]
        )
    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert f"{session} holds 2 rounds: give one round's folder, e.g. {session / 'first'}" in error


def test_a_missing_dependency_is_a_usage_error(case, tmp_path, monkeypatch, capsys):
    config, _ = case
    tool = load_tool("make_showcase")
    arguments = ["--config", config, "--expected", str(tmp_path / "grid.mp4"), "--out", "x.mp4"]

    def load_round(cfg, rows):
        raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")

    monkeypatch.setattr(tool, "load_round", load_round)
    with pytest.raises(SystemExit) as refused:
        tool.main(arguments)
    assert refused.value.code == 2 and (
        "No module named 'pyarrow': install the package with its dependencies, pip install -e ."
        in capsys.readouterr().err
    )

    def broken(cfg, rows):  # an import of the package's own, not a missing dependency
        raise ImportError("cannot import name 'player_boxes'")

    monkeypatch.setattr(tool, "load_round", broken)
    with pytest.raises(ImportError, match="cannot import name 'player_boxes'"):
        tool.main(arguments)
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(SystemExit) as refused:
        tool.main(arguments)
    assert refused.value.code == 2 and (
        'the showcase is drawn with Pillow: pip install -e ".[showcase]"' in capsys.readouterr().err
    )


def test_a_showcase_that_lost_frames_is_not_written(case, tmp_path, monkeypatch, lossy_ffmpeg):
    """Written by an ffmpeg that loses its frames, the showcase decodes to none of them: it is
    removed, and the error says so with the decoder's."""
    config, _ = case
    tool = load_tool("make_showcase")
    grid = tmp_path / "expected.mp4"
    make_video(grid, frames=4, levels=[40, 120, 200])
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", lossy_ffmpeg("frames"))
    out = tmp_path / "videos" / "showcase.mp4"
    with pytest.raises(RuntimeError) as lost:
        tool.main(["--config", config, "--expected", str(grid), "--out", str(out)])
    assert re.fullmatch(
        rf"{re.escape(str(out))} not written: it decodes to 0 of its 4 frames; ffmpeg: .+ \(the"
        r" video file is incomplete\)",
        str(lost.value),
    ), str(lost.value)
    assert list(out.parent.iterdir()) == []
