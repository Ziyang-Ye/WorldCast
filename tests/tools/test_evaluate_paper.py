"""``tools/evaluate_paper.py`` on the CPU: the steps and what they ask for; the entity criterion on
three views of the paper's ten-second windows (their stored detections, the labels given to the
generator and the paper's counts), with recorded and with given player states, from another
rollout start and on the partner alone; the intervals; the depth rows with a stand-in depth head
(the units in the order of their names, the windows' videos, the recorded depth of a round from
the latent cache's windows in the order of its members); the cells of depth agreement rows and of
matched pairs against the modules; the VBench bins; the cut of a round into windows; the tick
tables of a round; and the detection pass with a stand-in detector."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from tests.data.support import tick_table
from tests.tools.support import load_tool
from worldcast.data.latents import window_key
from worldcast.data.recordings import MediaIndex
from worldcast.engine.evaluation import depth_agreement, detector, matching, stats, vbench
from worldcast.engine.evaluation.entity import window_of
from worldcast.hub import METRIC_WEIGHTS

#: Two views with GT states and one with predicted states, with the paper's per-view counts.
FIXTURE = Path(__file__).resolve().parents[1] / "engine/evaluation/fixtures/entity/views.json.gz"
STEPS = (
    "cut",
    "detect",
    "entity",
    "entity-ci",
    "depth",
    "depth-cells",
    "match",
    "match-cells",
    "fvd",
    "vbench",
)


@pytest.fixture(scope="module")
def tool():
    return load_tool("evaluate_paper")


def run(tool, *arguments) -> int:
    return tool.main([str(a) for a in arguments])


def usage_error(tool, capsys, *arguments) -> str:
    """The message of a step that refuses its arguments."""
    with pytest.raises(SystemExit) as stopped:
        run(tool, *arguments)
    assert stopped.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("step", STEPS)
def test_every_step_has_its_help(tool, step, capsys):
    with pytest.raises(SystemExit) as stopped:
        run(tool, step, "--help")
    assert stopped.value.code == 0
    assert f" {step} [-h]" in capsys.readouterr().out


def test_the_steps_ask_for_what_they_need(tool, tmp_path, capsys):
    assert "--dets" in usage_error(tool, capsys, "entity", "--out", tmp_path / "x")
    assert "--arm" in usage_error(tool, capsys, "detect", "--eval-dir", tmp_path, "--out", "d")
    assert "I/N" in usage_error(tool, capsys, "detect", "--shard", "2/2", "--out", "d")
    message = usage_error(tool, capsys, "depth", "--rounds", "index.jsonl", "--out", "rows.npz")
    assert "--rounds takes --session" in message
    message = usage_error(tool, capsys, "depth", "--windows", "w.jsonl", "--recorded", "--out", "r")
    assert "--windows takes --videos" in message
    assert "--recorded-rows" in usage_error(tool, capsys, "match", "--recorded", "--out", "p.npz")
    assert "--eval-dir" in usage_error(tool, capsys, "fvd", "--out", tmp_path / "f.json")
    message = usage_error(tool, capsys, "fvd", "--eval-dir", "e", "--rounds", "r", "--out", "f")
    assert "not allowed with argument" in message
    message = usage_error(tool, capsys, "fvd", "--eval-dir", "e", "--sessions", "a=s", "--out", "f")
    assert "--eval-dir takes [--real-cache]" in message
    message = usage_error(tool, capsys, "fvd", "--rounds", "r", "--out", tmp_path / "f.json")
    assert "--rounds takes --sessions NAME=DIR" in message
    message = usage_error(tool, capsys, "vbench", "--out", tmp_path / "v.json")
    assert "give --hour, --whole-rounds or --sensitivity" in message
    message = usage_error(tool, capsys, "entity-ci", "--scores", "s.json", "--difference", "a")
    assert "expected A,B" in message
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------------- entity
def labels_of(item) -> np.ndarray:
    return np.array([[c == "1" for c in row] for row in item["labels"]])


@pytest.fixture
def paper_views(tmp_path):
    """The fixture's GT views as a detection dump and their visibility labels as label files (the
    labels at the views' source frames, all valid), with a config naming them."""
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        fixture = json.load(handle)
    labels = tmp_path / "vislabels"
    labels.mkdir()
    for item in fixture["gt_views"]:
        media_id, start = item["window"]["media_id"].rsplit("@", 1)
        visible = np.zeros((10, int(start) + 321), dtype=bool)
        visible[:, int(start) + 2 * np.arange(161)] = labels_of(item)
        np.savez(
            labels / f"{media_id}.npz",
            _binary_visible=visible,
            _binary_eval_valid=np.ones_like(visible),
        )
    detector.write_detections(
        tmp_path / "dets.json.gz", [i["window"] for i in fixture["gt_views"]], {}
    )
    config = tmp_path / "paths.yaml"
    config.write_text(f"paths:\n  visibility_label_root: {labels}\n")
    return fixture, config


def tsv(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines()[1:]]


def test_entity_counts_the_paper_views_as_the_paper(tool, tmp_path, paper_views, capsys):
    fixture, config = paper_views
    out = tmp_path / "out" / "table1"
    dets = tmp_path / "dets.json.gz"
    assert run(tool, "entity", "--config", config, "--dets", dets, "--out", out, "--perframe") == 0
    expected = {
        (arm, item["window"]["media_id"]): counts
        for item in fixture["gt_views"]
        for arm, counts in item["expected"].items()
        if arm != "real"
    }
    found = {
        (arm, view): [int(n) for n in counts]
        for arm, view, *counts in tsv(out.parent / "table1_perview.tsv")
    }
    assert found == expected
    scores = json.loads((out.parent / "table1.json").read_text())
    for (arm, view), (pairs, hits, dets_, unmatched) in expected.items():
        assert scores["rendered"][arm][view] == hits / pairs  # every view has 8 pairs or more
        assert scores["unmatched"][arm].get(view) == (unmatched / dets_ if dets_ >= 8 else None)
    summary = scores["summary"]["worldcast"]
    assert summary["rendered"] == np.median([35 / 43, 15 / 28]) and summary["rendered_views"] == 2
    assert "worldcast: Rendered 0.675 (2), Unmatched 0.047 (2)" in capsys.readouterr().out
    # every frame once per arm, and the crowding pools them
    frames = tsv(out.parent / "table1_perframe.tsv")
    assert len(frames) == 2 * 2 * 161
    crowding = scores["crowding"]["worldcast"]
    assert sum(b["pairs"] for b in crowding.values()) == 43 + 28
    assert sum(b["hits"] for b in crowding.values()) == 35 + 15
    # the recorded video against its own detections; an arm of no view is refused
    assert (
        run(tool, "entity", "--config", config, "--dets", dets, "--arm", "real", "--out", out) == 0
    )
    assert [counts for arm, view, *counts in tsv(out.parent / "table1_perview.tsv")] == [
        ["28", "28", "39", "0"],
        ["43", "43", "44", "0"],
    ]
    message = usage_error(tool, capsys, "entity", "--dets", dets, "--arm", "solaris", "--out", out)
    assert "--arm solaris: not in 2 views" in message


def paper_counts(items) -> dict[tuple[str, str], list[int]]:
    """``(arm, view) -> [n_pairs, hits, n_dets, n_unmatched]``, the paper's, of generated arms."""
    return {
        (arm, item["window"]["media_id"]): counts
        for item in items
        for arm, counts in item["expected"].items()
        if arm != "real"
    }


def perview_counts(path: Path) -> dict[tuple[str, str], list[int]]:
    return {(arm, view): [int(n) for n in counts] for arm, view, *counts in tsv(path)}


def test_entity_from_another_rollout_start(tool, tmp_path, paper_views, capsys):
    fixture, config = paper_views
    item = fixture["gt_views"][1]  # the view at source frame 2093
    dets = tmp_path / "one.json.gz"
    detector.write_detections(dets, [item["window"]], {})
    command = ["entity", "--config", config, "--dets", dets, "--out", tmp_path / "from2093"]
    assert run(tool, *command, "--rollout-start", "2093") == 0  # the view's own start
    assert perview_counts(tmp_path / "from2093_perview.tsv") == paper_counts([item])
    scores = json.loads((tmp_path / "from2093.json").read_text())
    assert scores["settings"]["rollout_start"] == 2093
    message = usage_error(tool, capsys, *command, "--rollout-start", "0")
    assert "the view at 2093 is not a frame of the rollout at 0" in message
    message = usage_error(tool, capsys, *command, "--rollout-start", "round")
    assert "expected window or a source frame" in message


def test_entity_on_the_partner_alone(tool, tmp_path, paper_views, capsys):
    """The partner is the window's other view player (``configs/eval/window_views.jsonl``). Every
    scored pair of these two views is the partner's, as the paper's partner-only counts of them
    say; with another player as the partner none is scored, and Unmatched does not change."""
    fixture, config = paper_views
    dets = tmp_path / "dets.json.gz"
    command = ["entity", "--config", config, "--dets", dets, "--scope", "partner"]
    assert run(tool, *command, "--out", tmp_path / "partner") == 0
    assert perview_counts(tmp_path / "partner_perview.tsv") == paper_counts(fixture["gt_views"])
    scores = json.loads((tmp_path / "partner.json").read_text())
    assert scores["settings"]["scope"] == "partner"
    rows = []
    for item in fixture["gt_views"]:
        media_id, start = item["window"]["media_id"].rsplit("@", 1)
        slot = item["window"]["obs_slot"]
        rows.append(
            {
                "media_id": media_id,
                "start_frame": int(start),
                "player_slot": slot,
                "view_player_slots": [slot, 9],
            }
        )
    views = tmp_path / "views.jsonl"
    views.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert run(tool, *command, "--views", views, "--out", tmp_path / "other") == 0
    assert [row[2:4] for row in tsv(tmp_path / "other_perview.tsv")] == [["0", "0"]] * 4
    other = json.loads((tmp_path / "other.json").read_text())
    assert other["rendered"] == {"frame_concat": {}, "worldcast": {}}
    assert other["unmatched"] == scores["unmatched"]
    views.write_text(json.dumps(rows[0]) + "\n")
    message = usage_error(tool, capsys, *command, "--views", views, "--out", tmp_path / "other")
    assert "--views: no row of 1 views" in message


def test_entity_with_the_states_given_to_the_generator(tool, tmp_path):
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        item = json.load(handle)["predicted_view"]
    window = item["window"]
    media_id, start = window["media_id"].rsplit("@", 1)
    states = tmp_path / "states"
    states.mkdir()
    np.savez(
        states / f"{media_id}__start{int(start):06d}.npz",
        states=np.array(item["states"]),
        labels=labels_of(item),
        start_frame=int(start),
    )
    other = dict(window, media_id=f"{media_id}@{int(start) + 320}")  # no table: left out
    detector.write_detections(tmp_path / "dets.json.gz", [window, other], {})
    out = tmp_path / "predicted"
    assert (
        run(
            tool,
            "entity",
            "--predicted-states",
            states,
            "--dets",
            tmp_path / "dets.json.gz",
            "--out",
            out,
        )
        == 0
    )
    assert tsv(tmp_path / "predicted_perview.tsv") == [
        ["predicted", window["media_id"], *map(str, item["expected"]["predicted"])]
    ]
    scores = json.loads((tmp_path / "predicted.json").read_text())
    assert (scores["views"], scores["dropped_views"]) == (1, 1)
    assert scores["settings"]["states"] == "given"


def test_entity_at_a_higher_confidence_filters_the_dump(tool, tmp_path, paper_views):
    _, config = paper_views
    dets = tmp_path / "dets.json.gz"
    out = tmp_path / "strict"
    assert run(tool, "entity", "--config", config, "--dets", dets, "--conf", "2", "--out", out) == 0
    scores = json.loads((tmp_path / "strict.json").read_text())
    assert scores["settings"]["conf"] == 2.0
    assert scores["summary"]["worldcast"]["rendered_views"] == 0  # nothing is confirmed
    assert tsv(tmp_path / "strict_perview.tsv") == []


def test_the_intervals_of_entity_scores(tool, tmp_path, capsys):
    rng = np.random.default_rng(0)
    views = [f"1-de_dust2-r{w:02d}-p0{s}@{320 * w}" for w in range(12) for s in (1, 2)]
    a = {view: float(rng.random()) for view in views}
    b = {view: float(rng.random()) for view in views[2:]}
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"rendered": {"a": a, "b": b}, "unmatched": {"a": a, "b": {}}}))
    out = tmp_path / "ci.json"
    assert run(tool, "entity-ci", "--scores", scores, "--difference", "a,b", "--out", out) == 0
    result = json.loads(out.read_text())
    ordered = sorted(a)
    low, high = stats.cluster_bootstrap([a[v] for v in ordered], [window_of(v) for v in ordered])
    assert result["arms"]["a"]["rendered"] == {
        "value": float(np.median(list(a.values()))),
        "low": low,
        "high": high,
        "views": 24,
    }
    assert result["arms"]["b"]["unmatched"] is None
    assert result["differences"]["a - b"]["unmatched"] is None
    difference = stats.paired_difference(a, b, window_of)
    assert result["differences"]["a - b"]["rendered"] == {
        "value": difference.point,
        "low": difference.low,
        "high": difference.high,
        "views": 22,
        "windows": 11,
    }
    assert "a - b rendered" in capsys.readouterr().out
    message = usage_error(
        tool, capsys, "entity-ci", "--scores", scores, "--difference", "a,c", "--out", out
    )
    assert "--difference a,c: no arm c in --scores (a, b)" in message


# ------------------------------------------------------------------------ depth agreement
class StubDepth:
    """The depth head: for latents marked ``m`` (their first value) the log depth ``1 + m / 1024``
    everywhere, in float16 as the paper's numerics; it keeps the marks in the order it saw them."""

    def __init__(self):
        self.marks: list[int] = []

    def log_depth(self, latents, *, paper_numerics: bool = False) -> np.ndarray:
        assert paper_numerics
        mark = int(float(latents[0, 0, 0, 0]))
        self.marks.append(mark)
        return np.full((len(latents), 4, 24, 42), 1 + mark / 1024, np.float16)


def depth_of(mark: int) -> float:
    """The axial depth of :class:`StubDepth`'s latents marked ``mark``."""
    return float(np.exp(np.float32(np.float16(1 + mark / 1024))))


@pytest.fixture
def depth_inputs(tool, tmp_path, monkeypatch):
    """The depth step with the stand-in depth head, the same tick table for every player and the
    depths each unit scores kept. Returns ``write``, which writes the media index of the media it
    is given and a config naming it (and the latent cache, the Wan2.2 snapshot) and returns the
    config; the depth head; and the depths of each unit, per client."""
    from worldcast.modeling import depth_head

    head = StubDepth()
    monkeypatch.setattr(depth_head, "load_depth_predictor", lambda *paths, device: head)
    monkeypatch.setattr(tool, "read_player_ticks", lambda *args, **kwargs: tick_table(40.0))
    scored = []

    def pair_rows(depths, cameras, **kwargs):
        scored.append([np.array(depth) for depth in depths])
        return original(depths, cameras, **kwargs)

    original = depth_agreement.pair_rows
    monkeypatch.setattr(depth_agreement, "pair_rows", pair_rows)

    def write(media: list[str], video_frames: int = 4000) -> Path:
        index = tmp_path / "media_index.jsonl"
        rows = []
        for media_id in media:
            match_id, map_name, round_, slot = media_id.split("-")
            row = {"media_id": media_id, "match_id": int(match_id), "map_name": map_name}
            row |= {"round": int(round_[1:]), "player_slot": int(slot[1:]), "fps": 32.0}
            row |= {"video_frames": video_frames, "ticks_path": "t", "ticks_rows": 1}
            row |= {"ticks_file_size": 1}
            rows.append(json.dumps(row) + "\n")
        index.write_text("".join(rows))
        config = tmp_path / "paths.yaml"
        names = ("depth_head", "depth_readout", "latent_cache_root", "wan22_root")
        settings = "".join(f"  {name}: {tmp_path / name}\n" for name in names)
        config.write_text(f"paths:\n{settings}  media_index: {index}\n  dataset_root: {tmp_path}\n")
        return config

    return write, head, scored


def session_latents(path: Path, count: int) -> None:
    """A client's ``latents.npy`` of ``count`` latent frames, all zero (marked 0)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(path, "w+", np.float16, (count, 48, 24, 42)).flush()


def test_depth_of_the_windows_encodes_the_generated_or_the_recorded_videos(
    tool, tmp_path, monkeypatch, depth_inputs, capsys
):
    """Table 10: every view's mp4 is encoded, ``generation/`` or with ``--recorded`` ``gt/``; the
    windows are numbered in the order of their names, each with its views sorted."""
    from worldcast.modeling.wan22 import vae as wan22_vae

    write, _, scored = depth_inputs
    windows = [  # not in the order of their names, the views of the first not sorted
        ("mv2covis-2392598-de_dust2-r05-f000640-p02p01", 640, ["p02", "p01"]),
        ("mv2covis-2392598-de_dust2-r02-f000320-p03p04", 320, ["p03", "p04"]),
    ]
    rounds = {640: "2392598-de_dust2-r05", 320: "2392598-de_dust2-r02"}
    lines = []
    for group_id, start, views in windows:
        media = [f"{rounds[start]}-{view}" for view in views]
        row = {"group_id": group_id, "start_frame": start, "view_media_ids": media}
        lines.append(json.dumps(row) + "\n")
    (tmp_path / "windows.jsonl").write_text("".join(lines))
    stems = [
        f"{rounds[start]}-{view}__start{start:06d}"
        for start, views in ((320, ("p03", "p04")), (640, ("p01", "p02")))
        for view in views
    ]
    config = write([stem.split("__")[0] for stem in stems])
    marks = {}
    for folder in ("generation", "gt"):
        for stem in stems:
            marks[f"{folder}/{stem}.mp4"] = len(marks) + 1
            (tmp_path / "eval" / folder).mkdir(parents=True, exist_ok=True)
            (tmp_path / "eval" / folder / f"{stem}.mp4").write_bytes(b"")
    encoded = []

    def encode_window(vae, video):
        assert vae == "vae"
        encoded.append(Path(video).relative_to(tmp_path / "eval").as_posix())
        return np.full((41, 1, 1, 1), marks[encoded[-1]], np.float32)

    monkeypatch.setattr(wan22_vae, "load_wan22_vae", lambda path, device, dtype: "vae")
    monkeypatch.setattr(tool, "_encode_window", encode_window)
    command = ["depth", "--config", config, "--windows", tmp_path / "windows.jsonl"]
    command += ["--videos", tmp_path / "eval", "--device", "cpu"]
    for folder, flags in (("generation", []), ("gt", ["--recorded"])):
        encoded.clear()
        scored.clear()
        out = tmp_path / f"{folder}.npz"
        assert run(tool, *command, *flags, "--out", out) == 0
        assert encoded == [f"{folder}/{stem}.mp4" for stem in stems]
        assert [[depth[0, 0, 0] for depth in unit] for unit in scored] == [
            [depth_of(marks[f"{folder}/{stem}.mp4"]) for stem in stems[i : i + 2]] for i in (0, 2)
        ]
        rows = tool._load_rows(out)
        assert rows.units == [group_id for group_id, _, _ in reversed(windows)]
        assert rows.clients == [[stem.split("__")[0] for stem in stems[i : i + 2]] for i in (0, 2)]
        assert rows.start_frames.tolist() == [320, 640] and rows.unit == "window"
    message = usage_error(tool, capsys, *command[:5], "--recorded", "--out", tmp_path / "r.npz")
    assert "--windows takes --videos" in message


def test_depth_numbers_the_rounds_in_the_order_of_their_names(tool, tmp_path, depth_inputs):
    write, head, _ = depth_inputs
    rounds = {"2392598-de_dust2-r05": ("p02", "p01"), "2392598-de_dust2-r02": ("p03", "p04")}
    lines = []
    for name, slots in rounds.items():
        match_id, map_name, round_ = name.split("-")
        media = [f"{name}-{slot}" for slot in slots]
        for media_id in media:
            row = {"media_id": media_id, "start_frame": 0, "match_id": int(match_id)}
            row |= {"round": int(round_[1:]), "map_name": map_name, "group_media": media}
            lines.append(json.dumps(row) + "\n")
            session_latents(tmp_path / "session" / f"{name}-s000000" / media_id / "latents.npy", 9)
    (tmp_path / "rounds.jsonl").write_text("".join(lines))
    config = write([m for name, slots in rounds.items() for m in (f"{name}-{s}" for s in slots)])
    out = tmp_path / "rows.npz"
    command = ["depth", "--config", config, "--rounds", tmp_path / "rounds.jsonl"]
    assert (
        run(tool, *command, "--session", tmp_path / "session", "--device", "cpu", "--out", out) == 0
    )
    rows = tool._load_rows(out)
    assert rows.units == ["2392598-de_dust2-r02-s000000", "2392598-de_dust2-r05-s000000"]
    assert rows.clients == [
        ["2392598-de_dust2-r02-p03", "2392598-de_dust2-r02-p04"],
        ["2392598-de_dust2-r05-p01", "2392598-de_dust2-r05-p02"],
    ]
    assert rows.latent_frames.tolist() == [[9, 9], [9, 9]] and rows.unit == "round"
    assert len(head.marks) == 4 * 3  # the first frame, then each block alone


def cache_archive(path: Path, starts: list[int]) -> None:
    """A latent-cache archive of the windows at ``starts`` in that order, each marked with its
    start."""
    members = {}
    for start in starts:
        window = np.zeros((1, 41, 48, 24, 42), np.float16)
        window[0, 0, 0, 0, 0] = start
        members[window_key(start)] = window
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **members)


def test_the_recorded_depth_of_a_round_reads_its_windows_in_the_papers_order(
    tool, tmp_path, depth_inputs, capsys
):
    """A recording of 1000 source frames has the windows 0, 161, 322, 483, 644, then 80, 241, 402,
    563, read in that order whatever the archive's order; another cached window is not read.
    Latent frame 85 (source frame 680) is the second video frame of a latent of window 644 and the
    third of one of window 402, equally far from their middles: it takes 644, the earlier in that
    order. The round's last latent frames lie past every window and stay NaN; a missing window
    stops the step before the depth head runs and is listed for the latent cache."""
    write, head, scored = depth_inputs
    name = "2392598-de_dust2-r03"
    media = [f"{name}-p01", f"{name}-p02"]
    lines = []
    for media_id in media:
        row = {"media_id": media_id, "start_frame": 0, "match_id": 2392598, "round": 3}
        row |= {"map_name": "de_dust2", "group_media": media}
        lines.append(json.dumps(row) + "\n")
        session_latents(tmp_path / "session" / f"{name}-s000000" / media_id / "latents.npy", 125)
    (tmp_path / "rounds.jsonl").write_text("".join(lines))
    config = write(media, video_frames=1000)
    grid = [0, 161, 322, 483, 644, 80, 241, 402, 563]
    for media_id in media:
        cache_archive(tmp_path / "latent_cache_root" / f"{media_id}.npz", sorted(grid + [40]))
    command = ["depth", "--config", config, "--rounds", tmp_path / "rounds.jsonl", "--recorded"]
    command += ["--session", tmp_path / "session", "--device", "cpu"]
    assert run(tool, *command, "--out", tmp_path / "recorded.npz") == 0
    assert head.marks == grid * 2
    assert len(scored) == 1
    first = scored[0][0]
    assert first[85, 0, 0] == depth_of(644) != depth_of(402)
    assert first[81, 0, 0] == depth_of(402) and first[120, 0, 0] == depth_of(644)
    assert np.isnan(first[0]).all() and not np.isnan(first[1:121]).any()
    assert np.isnan(first[121:]).all()  # source frames 968-992: past window 644's last frame

    head.marks.clear()
    cache_archive(tmp_path / "latent_cache_root" / f"{media[1]}.npz", [s for s in grid if s != 644])
    message = usage_error(tool, capsys, *command, "--out", tmp_path / "short.npz")
    assert (
        f"lacks 1 of the windows the recorded depth reads, e.g. {media[1]} from source frame 644"
        in (message)
    )
    assert head.marks == []
    listed = (tmp_path / "short.windows.jsonl").read_text()
    assert listed == json.dumps({"media_id": media[1], "start_frame": 644}) + "\n"


def rows_file(path: Path, rows: np.ndarray, units: int, unit: str = "round") -> Path:
    """A file of the depth step: ``rows`` of ``units`` units of two clients from frame 0."""
    np.savez(
        path,
        rows=rows,
        unit=np.array(unit),
        units=np.array([f"unit{u}" for u in range(units)]),
        clients=np.array([["m-p01", "m-p02"]] * units),
        latent_frames=np.full((units, 2), 241, dtype=np.int64),
        start_frames=np.zeros(units, dtype=np.int64),
    )
    return path


def synthetic_rows(rng, units: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of every kind with random counts, and the recorded video's rows of the same keys."""
    keys = []
    for unit in range(units):
        for a, b in ((0, 1), (1, 0)):
            keys += [(unit, a, b, k, kind) for kind in (0, 2) for k in range(1, 240, 3)]
        keys += [(unit, a, a, k, 1) for a in (0, 1) for k in range(1, 239, 3)]
    keys = np.array(keys)

    def counts():
        reached = rng.integers(0, 1008, len(keys))
        agree = rng.integers(0, reached + 1)
        return np.column_stack([reached, agree, agree, reached - agree, 0 * reached])

    return (
        np.column_stack([keys, counts()]).astype(np.int32),
        np.column_stack([keys, counts()]).astype(np.int32),
    )


def test_depth_cells_are_the_modules_cells(tool, tmp_path, capsys):
    rng = np.random.default_rng(1)
    worldcast, recorded = synthetic_rows(rng, 5)
    off = worldcast.copy()
    off[:, 6] = rng.integers(0, off[:, 5] + 1)
    files = {
        "worldcast": rows_file(tmp_path / "worldcast.npz", worldcast, 5),
        "off": rows_file(tmp_path / "off.npz", off, 5),
        "recorded": rows_file(tmp_path / "recorded.npz", recorded, 5),
    }
    out = tmp_path / "cells.json"
    command = ["depth-cells", *(a for n, f in files.items() for a in ("--rows", f"{n}={f}"))]
    command += ["--mask", files["recorded"], "--difference", "worldcast,off", "--out", out]
    assert run(tool, *command) == 0
    cells = json.loads(out.read_text())
    assert cells["segments"] == ["first 30 s", "after 30 s"] and cells["masked"]

    def table(rows, kind):
        if kind != depth_agreement.OWN_NEXT_FRAME:
            rows = rows[depth_agreement.covisibility_mask(rows, recorded)]
        return depth_agreement.segment_ratios(rows, kind, 5, depth_agreement.SCENE_SPLIT_SECONDS)

    for name, kind in (("between_clients", 0), ("own_next_frame", 1), ("chance", 2)):
        mine = table(worldcast, kind)
        assert cells["arms"]["worldcast"][name]["value"] == stats.mean_over_units(mine).tolist()
        difference = mine - table(off, kind)
        low, high = stats.unit_interval(difference)
        cell = cells["differences"]["worldcast - off"][name]
        assert (cell["low"], cell["high"]) == (low.tolist(), high.tolist())
        assert cell["units"] == [5, 5]
    other = rows_file(tmp_path / "other.npz", worldcast[worldcast[:, 0] < 4], 4)
    message = usage_error(
        tool,
        capsys,
        "depth-cells",
        "--rows",
        f"a={files['off']}",
        "--rows",
        f"b={other}",
        "--out",
        out,
    )
    assert "--rows b: other units than a" in message
    command[-3] = "worldcast,on"
    message = usage_error(tool, capsys, *command)
    assert "--difference worldcast,on: no arm on in --rows (worldcast, off, recorded)" in message


def test_matched_pair_cells_are_the_modules_cells(tool, tmp_path, capsys):
    rng = np.random.default_rng(2)
    _, recorded = synthetic_rows(rng, 3)
    recorded_file = rows_file(tmp_path / "recorded.npz", recorded, 3)
    keys = {}
    for matcher, stride in (("gim", matching.GIM_STRIDE), ("loftr", matching.LOFTR_STRIDE)):
        per_unit = []
        for unit in range(3):
            mine = matching.match_keys(recorded[recorded[:, 0] == unit], [241, 241], stride, (0, 2))
            per_unit.append(np.column_stack([np.full(len(mine), unit), mine]))
        keys[matcher] = np.concatenate(per_unit)

    def arm(path):
        values = {
            field: (rng.random(len(keys[m])) * 100).astype(np.float32)
            for m, fields in tool.MATCH_FIELDS.items()
            for field in fields
        }
        units = np.array([f"unit{u}" for u in range(3)])
        np.savez(path, units=units, gim_keys=keys["gim"], loftr_keys=keys["loftr"], **values)
        return path, values

    (on, on_values), (off, off_values) = arm(tmp_path / "on.npz"), arm(tmp_path / "off.npz")
    out = tmp_path / "cells.json"
    command = ["match-cells", "--pairs", f"on={on}", "--pairs", f"off={off}"]
    command += ["--recorded-rows", recorded_file, "--difference", "on,off", "--self-check", "on"]
    assert run(tool, *command, "--out", out) == 0
    cells = json.loads(out.read_text())

    def per_round(values, field, measure):
        matcher = "gim" if field.startswith("matched") else "loftr"
        found = []
        for unit in range(3):
            mine = keys[matcher][:, 0] == unit
            unit_keys = keys[matcher][mine, 1:]
            seen = matching.covisible(unit_keys, recorded[recorded[:, 0] == unit])
            found.append(measure(unit_keys, values[field][mine], seen, matching.STATISTICS[field]))
        return np.array(found, dtype=np.float64)

    for field in matching.STATISTICS:
        on_table = per_round(on_values, field, matching.round_statistics)
        assert cells["arms"]["on"][field] == {
            "value": stats.mean_over_units(on_table).tolist(),
            "units": np.isfinite(on_table).sum(0).tolist(),
        }
        difference = on_table - per_round(off_values, field, matching.round_statistics)
        low, _ = stats.unit_interval(difference)
        assert cells["differences"]["on - off"][field]["low"] == low.tolist()
        check = per_round(on_values, field, matching.self_check)
        assert cells["self_check"][field]["value"] == stats.mean_over_units(check).tolist()
    command[-1] = "recorded"
    message = usage_error(tool, capsys, *command, "--out", out)
    assert "--self-check: no arm recorded in --pairs (on, off)" in message


# ----------------------------------------------------------------------------------- VBench
def test_vbench_bins_the_shards(tool, tmp_path, capsys):
    dims = vbench.DIMENSIONS
    shard = tmp_path / "rounds.json"
    clips = {
        f"m-p0{p}__start{320 * k:06d}.mp4": {d: 0.5 + 0.01 * k + 0.001 * p for d in dims}
        for k in range(7)
        for p in (1, 2)
    }
    shard.write_text(json.dumps({"dims": list(dims), "clips": clips}))
    hour = tmp_path / "hour.json"
    hour.write_text(
        json.dumps(
            {"clips": {f"r__s{s:05d}.mp4": {dims[0]: s / 3600} for s in (0, 1199, 1200, 3590)}}
        )
    )
    damage = tmp_path / "damage.json"
    damage.write_text(
        json.dumps({"clips": {"orig__a.mp4": {dims[0]: 0.8}, "blur3__a.mp4": {dims[0]: 0.6}}})
    )
    out = tmp_path / "vbench.json"
    command = ["vbench", "--whole-rounds", shard, "--hour", hour, "--sensitivity", damage]
    assert run(tool, *command, "--out", out) == 0
    result = json.loads(out.read_text())
    rounds = vbench.round_bins(vbench.read_shards([shard]))
    assert result["whole_rounds"] == {str(k): v for k, v in rounds.items()}
    assert sorted(result["whole_rounds"]) == ["0", "1", "2", "3", "4", "5"]
    assert {k: v["clips"] for k, v in result["hour"].items()} == {"0": 2, "1": 1, "2": 1}
    change = result["sensitivity"]["blur3"]
    assert change["damage"] == "Blur, sigma=2 px" and change[dims[0]] == pytest.approx(-25.0)
    printed = capsys.readouterr().out
    assert "0-10 s (2 clips)" in printed and "0-20 min (2 clips)" in printed


# -------------------------------------------------------------------------------------- cut
def test_a_round_is_cut_into_windows_that_share_a_frame(tool, tmp_path):
    import imageio.v2 as imageio

    from worldcast.engine.inference.decode import write_mp4

    video = tmp_path / "session" / "1-de_dust2-r01-s000640" / "1-de_dust2-r01-p03" / "video.mp4"
    shade = [16 * i % 240 for i in range(1 + 160 * 2 + 40)]  # neighbours 16 grey levels apart
    frames = [np.full((32, 48, 3), value, dtype=np.uint8) for value in shade]
    write_mp4(frames, video)
    out = tmp_path / "windows"
    assert run(tool, "cut", "--videos", video, "--out-dir", out) == 0
    names = sorted(path.name for path in out.iterdir())
    assert names == [
        "1-de_dust2-r01-p03__start000640.mp4",
        "1-de_dust2-r01-p03__start000960.mp4",
    ]  # the third window is incomplete
    for name, first in zip(names, (0, 160)):
        window = [frame.mean() for frame in imageio.get_reader(str(out / name), format="ffmpeg")]
        assert len(window) == 161
        assert abs(window[0] - shade[first]) < 6 and abs(window[-1] - shade[first + 160]) < 6


def test_cut_refuses_a_video_outside_a_session(tool, tmp_path, capsys):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    assert "not a <round>/<media_id>/video.mp4" in usage_error(
        tool, capsys, "cut", "--videos", video, "--out-dir", tmp_path / "w"
    )


# ------------------------------------------------------------------------------- detection
def write_tick_table(path: Path, x: float) -> int:
    """A three-tick table of a player at ``x``, alive but in the last tick; returns its size."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    def floats(value: float):
        return pa.array(np.full(3, value, dtype=np.float32))

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "tick": pa.array(np.arange(100, 103, dtype=np.int32)),
                "t": pa.array(np.arange(3, dtype=np.float32) / 64),
                "x": floats(x),
                "y": floats(-2.5),
                "z": floats(0.25),
                "yaw": floats(90.0),
                "pitch": floats(-5.0),
                "is_alive": pa.array([True, True, False]),
                "active": pa.array([[], ["forward"], []], pa.list_(pa.string())),
                "delta_pitch": floats(0.0),
                "delta_yaw": floats(0.0),
                "input_weapon": pa.array(["weapon_ak47"] * 3),
                "team_num": pa.array([2] * 3, pa.int8()),
            }
        ),
        path,
    )
    return path.stat().st_size


def test_a_round_takes_every_tick_table_of_its_directory(tool, tmp_path):
    """Player 3 has no media row (no recorded video) but a tick table in the round's directory: it
    is a player of the round; the tables the media index lists are checked against it."""
    directory = Path("rounds/match_id=2392598/map_name=de_dust2/round=05")
    rows = []
    for slot in (1, 2, 3):
        path = directory / f"player={slot:02d}" / "ticks.parquet"
        size = write_tick_table(tmp_path / path, x=100.0 * slot)
        media_id = f"2392598-de_dust2-r05-p{slot:02d}"
        rows.append(
            {
                "media_id": media_id,
                "match_id": 2392598,
                "map_name": "de_dust2",
                "round": 5,
                "player_slot": slot,
                "fps": 32.0,
                "video_frames": 4000,
                "ticks_path": str(path),
                "ticks_rows": 3,
                "ticks_file_size": size,
            }
        )
    index_path = tmp_path / "media_index.jsonl"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:2]))
    index = MediaIndex.load(index_path)
    ticks = tool._round_ticks(index, str(tmp_path), index.media(rows[0]["media_id"]))
    assert sorted(ticks) == [1, 2, 3]
    times, states = ticks[3]
    assert times.tolist() == [0.0, 1 / 64, 2 / 64]
    assert states.tolist() == [[300.0, -2.5, 0.25, 90.0, -5.0, alive] for alive in (1, 1, 0)]
    for slot in (1, 2):
        assert np.array_equal(ticks[slot][0], times)
        assert np.array_equal(ticks[slot][1], states + [[100.0 * (slot - 3), 0, 0, 0, 0, 0]])
    (tmp_path / directory / "player=02" / "ticks.parquet").write_bytes(b"another table")
    with pytest.raises(RuntimeError, match="bytes"):
        tool._round_ticks(index, str(tmp_path), index.media(rows[0]["media_id"]))


def test_the_detection_pass_with_a_stand_in_detector(tool, tmp_path, monkeypatch, capsys):
    """Every view of the evaluation directory in the shard, every arm's video, a dump sorted by
    view with the detector's digest; an arm whose recorded video is another is refused."""
    media = [f"2392598-de_dust2-r01-p0{slot}" for slot in (1, 2)]
    index = tmp_path / "media_index.jsonl"
    index.write_text(
        "".join(
            json.dumps(
                {
                    "media_id": m,
                    "match_id": 2392598,
                    "map_name": "de_dust2",
                    "round": 1,
                    "player_slot": slot,
                    "fps": 32.0,
                    "video_frames": 4000,
                    "ticks_path": f"{m}.parquet",
                    "ticks_rows": 1,
                    "ticks_file_size": 1,
                }
            )
            + "\n"
            for slot, m in zip((1, 2), media)
        )
    )
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"detector")
    config = tmp_path / "paths.yaml"
    config.write_text(
        f"paths:\n  detector: {weights}\n  media_index: {index}\n  dataset_root: {tmp_path}\n"
    )
    stems = [f"{m}__start{start:06d}" for m in media for start in (0, 320)]
    for directory in ("ref/gt", "arm/generation", "arm/gt"):
        (tmp_path / directory).mkdir(parents=True)
        for stem in stems:
            (tmp_path / directory / f"{stem}.mp4").write_bytes(stem.encode())
    called = []

    def detect_window(model, recording, generated, ticks, **view):
        called.append((Path(recording).name, {k: Path(v).name for k, v in generated.items()}))
        assert model == "model" and ticks == {"round": 1}
        return {**view, "media_id": f"{view['media_id']}@{view['start_frame']}"}

    monkeypatch.setattr(detector, "load_detector", lambda path, device: "model")
    monkeypatch.setattr(detector, "detect_window", detect_window)
    monkeypatch.setattr(tool, "_round_ticks", lambda index, root, record: {"round": 1})
    out = tmp_path / "dets.json.gz"
    command = ["detect", "--config", config, "--eval-dir", tmp_path / "ref"]
    command += ["--arm", f"worldcast={tmp_path / 'arm'}", "--device", "cpu"]
    assert run(tool, *command, "--shard", "1/2", "--out", out) == 0
    assert called == [
        (f"{stem}.mp4", {"worldcast": f"{stem}.mp4"}) for stem in (stems[1], stems[3])
    ]
    dump = detector.read_detections(out)
    assert [w["media_id"] for w in dump["windows"]] == [f"{media[0]}@320", f"{media[1]}@320"]
    assert dump["windows"][1]["observer_slot"] == 2 and dump["windows"][1]["map_name"] == "de_dust2"
    assert dump["stamp"]["arms"] == ["worldcast"] and dump["stamp"]["conf"] == 0.05
    assert dump["stamp"]["detector_sha256"] == METRIC_WEIGHTS["detector"].sha256
    (tmp_path / "arm/gt" / f"{stems[1]}.mp4").write_bytes(b"another recording")
    message = usage_error(tool, capsys, *command, "--shard", "1/2", "--out", out)
    assert "is not the recorded video" in message
