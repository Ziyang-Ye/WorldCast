"""The bucket step of tools/prepare_data.py: the match split, a recording's windows, the visible
pairs from the labels, the evaluation reserve, the validation windows, the weights of the map slots,
and the files it writes, read back through the loader."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.tools.support import load_tool
from worldcast.config.training import TrainConfig
from worldcast.data import training as D
from worldcast.data.labels import visibility_rows
from worldcast.data.latents import WINDOW_SOURCE_FRAMES
from worldcast.data.preparation import buckets as S
from worldcast.data.recordings import MediaIndex, MediaRecord
from worldcast.utils.files import json_line, sha256_file

REPO = Path(__file__).resolve().parents[3]
PREPARE = load_tool("prepare_data")

#: A world of two maps: (match, map, round, video frames of its recordings). Each round has the
#: recordings of slots 0 (labels of a known pattern), 1 (every other player always visible) and
#: 2 (nobody visible).
ROUNDS = [
    (1, "de_a", 1, 1000),
    (1, "de_a", 2, 1000),
    (2, "de_a", 1, 1000),
    (2, "de_b", 1, 1000),
    (3, "de_b", 1, 1000),
    (3, "de_b", 2, 1000),
    (4, "de_a", 1, 500),
    (4, "de_b", 1, 500),
]
#: The starts of the windows of a 1000-frame recording and of a 500-frame one.
STARTS_1000 = [0, 80, 161, 241, 322, 402, 483, 563, 644]
STARTS_500 = [0, 80, 161]
#: Visible pairs of slot 0's windows by start: another player is seen in frames 0-9 and one in
#: frames 100-119 (a third in frames 200-219, never valid; the player's own row is marked
#: everywhere); the other windows see nobody.
PATTERN_PAIRS = {0: 15, 80: 10}
#: Visible pairs of slot 1's windows: 9 players in each of the 161 video frames.
DENSE_PAIRS = 9 * 161


def media_row(match: int, map_name: str, round_: int, slot: int, frames: int) -> dict:
    return {
        "media_id": f"{match}-{map_name}-r{round_:02d}-p{slot:02d}",
        "match_id": match,
        "map_name": map_name,
        "round": round_,
        "player_slot": slot,
        "fps": 32.0,
        "video_frames": frames,
        "ticks_path": f"{map_name}-{match}-{slot}.parquet",
        "ticks_rows": 2 * frames,
        "ticks_file_size": 1,
    }


def write_lines(path: Path, lines: list) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines))
    return path


def write_labels(root: Path, row: dict) -> None:
    """The label file of a recording, by its slot: the pattern, everybody, nobody."""
    frames, slot = row["video_frames"], row["player_slot"]
    visible, valid = np.zeros((10, frames), bool), np.zeros((10, frames), bool)
    if slot == 0:
        a, b, c = 1, 2, 3
        visible[a, 0:10] = valid[a, 0:10] = True
        visible[b, 100:120] = valid[b, 100:120] = True
        visible[c, 200:220] = True
        visible[slot] = valid[slot] = True
    elif slot == 1:
        visible[:] = valid[:] = True
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / f"{row['media_id']}.npz", _binary_visible=visible, _binary_eval_valid=valid)


@pytest.fixture
def world(tmp_path):
    """The media index, the labels of its recordings and the held-out matches {3}.

    Match 5 adds a recording shorter than a window, one without labels and one that the exclusion
    list names."""
    rows = [
        media_row(match, map_name, round_, slot, frames)
        for match, map_name, round_, frames in ROUNDS
        for slot in range(3)
    ]
    rows += [media_row(5, "de_a", 1, 0, 300), media_row(5, "de_a", 1, 1, 1000)]
    rows += [media_row(5, "de_a", 1, 2, 1000)]
    for row in rows:
        if row["media_id"] != "5-de_a-r01-p01":
            write_labels(tmp_path / "labels", row)
    return {
        "dir": tmp_path,
        "index": write_lines(tmp_path / "media_index.jsonl", map(json.dumps, rows)),
        "labels": tmp_path / "labels",
        "held_out": write_lines(tmp_path / "held_out.txt", [3]),
        "excluded": write_lines(tmp_path / "excluded.txt", ["5-de_a-r01-p02"]),
        "rows": rows,
    }


def argv(command: str, **flags) -> list[str]:
    """``buckets <command> --flag value ...``: a list value is a list of values."""
    out = ["buckets", command]
    for flag, value in flags.items():
        values = value if isinstance(value, list) else [value]
        out += [f"--{flag.replace('_', '-')}", *map(str, values)]
    return out


def windows_argv(world, out_dir, **flags) -> list[str]:
    return argv(
        "windows",
        **{
            "media_index": world["index"],
            "held_out": world["held_out"],
            "visibility_label_root": world["labels"],
            "maps": ["de_a", "de_b"],
            "out_dir": out_dir,
            **flags,
        },
    )


def run(capsys, command: list[str]) -> dict:
    """The JSON summary that ``tools/prepare_data.py <command>`` prints."""
    assert PREPARE.main(command) == 0
    return json.loads(capsys.readouterr().out)


def usage_error(capsys, command: list[str]) -> str:
    """What ``tools/prepare_data.py <command>`` says when it ends with exit status 2."""
    with pytest.raises(SystemExit) as stopped:
        PREPARE.main(command)
    assert stopped.value.code == 2
    return capsys.readouterr().err


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def bucket_rows(directory: Path) -> list[dict]:
    return [r for name in D.BUCKET_NAMES for r in read_rows(directory / f"train_{name}.jsonl")]


def rank(*parts) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def test_a_draw_orders_by_the_sha256_of_its_parts_joined_by_nul():
    # the orders that reproduce the paper's reserve and validation windows
    expected = {
        ("seed", 123): "b861ee0bfc9af4b19cd317f4371ebe664778feae2d024817e1109461d9babd88",
        (S.VALIDATION_SEED, "de_dust2", "m1", 161): (
            "807458b18671dfedee2e3ddc91d211236a5c788d6666322026117073e5bbb5d4"
        ),
        (S.RESERVE_SEED, "match", 123): (
            "4579853db7a5b5e26a8e45ad6bd3b1e05270f4ba8f6e31102c8a9153caccdf9e"
        ),
        (S.RESERVE_SEED, "round", 123, 4): (
            "a92df6f97ed890d46aa23d176b094239a7415f33c934787ef03cab35105e7092"
        ),
    }
    for parts, digest in expected.items():
        assert rank(*parts) == S._rank(*parts) == digest


# ======================================================================================= windows
@pytest.mark.parametrize(
    "frames, starts",
    [
        (320, []),
        (321, [0]),
        (400, [0]),
        (401, [0, 80]),
        (482, [0, 80, 161]),
        (500, STARTS_500),
        (1000, STARTS_1000),
        (1100, [*STARTS_1000, 724]),
    ],
)
def test_the_windows_of_a_recording_are_the_grid_and_its_midpoints_that_fit_the_video(
    frames, starts
):
    assert S.window_starts(frames) == starts


@pytest.mark.parametrize(
    "frames, starts",
    [(401, [0]), (500, STARTS_500), (1000, STARTS_1000), (1100, STARTS_1000)],
)
def test_checked_windows_have_a_midpoint_between_each_two_neighbours_only(frames, starts):
    assert S.window_starts(frames, gate=lambda start: True) == starts


def test_a_midpoint_needs_both_neighbours_to_be_kept():
    kept = {0, 483, 644}
    assert S.window_starts(1000, gate=kept.__contains__) == [0, 483, 563, 644]


def ticks(seconds: float, *, hz: int = 64, gap: tuple[float, float] | None = None) -> np.ndarray:
    """Tick times of ``seconds`` at ``hz``, without those in the half-open interval ``gap``."""
    t = np.arange(int(seconds * hz)) / hz
    return t if gap is None else t[(t < gap[0]) | (t >= gap[1])]


def test_a_window_must_not_cross_a_gap_of_the_ticks():
    # the grid window at start 161 k spans [5.03 k, 5.03 k + 10] s: 161 and 322 meet the gap
    t = ticks(40, gap=(10.0, 10.5))
    gate = S.tick_gate(t, [True] * len(t), 32.0)
    assert [start for start in range(0, 960, 161) if gate(start)] == [0, 483, 644, 805]
    assert S.window_starts(1280, gate) == [0, 483, 563, 644, 724, 805]


def test_a_window_must_lie_within_the_life_of_the_player():
    t = ticks(40)
    expected = [0, 80, 161, 241, 322]  # the player dies at 22 s: the window at 483 spans 15-25 s
    assert S.window_starts(1280, S.tick_gate(t, t < 22.0, 32.0)) == expected
    alive = [True if second < 22.0 else None for second in t]  # a tick without a value is dead
    assert S.window_starts(1280, S.tick_gate(t, alive, 32.0)) == expected


def test_the_tick_gap_the_gate_tolerates_is_a_parameter():
    t = ticks(40, hz=32)  # a tick per frame: gaps of 1 / 32 s
    assert S.window_starts(1280, S.tick_gate(t, [True] * len(t), 32.0)) == []
    gate = S.tick_gate(t, [True] * len(t), 32.0, max_tick_gap_seconds=2.5 / 64)
    assert S.window_starts(1280, gate) == [0, 80, 161, 241, 322, 402, 483, 563, 644, 724, 805]


# ======================================================================= visible pairs of a window
def test_the_visible_pairs_of_a_window_are_those_of_the_other_players_over_its_video_frames(
    tmp_path,
):
    row = media_row(1, "de_a", 1, 0, 1000)
    write_labels(tmp_path, row)
    visible, _ = visibility_rows(tmp_path, row["media_id"], np.arange(1000))
    # the video frames of start 0 are the even frames; the ones of 161 are odd
    assert S.window_pair_counts(visible, 0, STARTS_1000) == {
        start: PATTERN_PAIRS.get(start, 0) for start in STARTS_1000
    }
    # the row of the player at slot 0 is marked everywhere, and left out only as the own row
    assert S.window_pair_counts(visible, 3, [0]) == {0: 15 + 161}


def test_the_row_of_a_window_has_the_fields_of_the_bucket_index():
    record = MediaRecord.from_row(media_row(7, "de_a", 3, 4, 1000))
    assert json_line(S.window_row(record, 161, 12)) == (
        '{"latent_key":"win_000161","map_name":"de_a","match_id":7,"media_id":"7-de_a-r03-p04",'
        '"round":3,"start_frame":161,"vis_available":true,"vis_pair_count":12}\n'
    )


#: Windows whose visible pairs the paper's labels gave (rows of its validation index), and the
#: example rounds' labels they come from.
REAL_LABELS = [
    ("ancient_r02", "2392968-de_ancient-r02-p04", 1932, 100),
    ("ancient_r18", "2393400-de_ancient-r18-p01", 2415, 141),
    ("ancient_r02", "2392968-de_ancient-r02-p01", 885, 161),
    ("ancient_r18", "2393400-de_ancient-r18-p00", 241, 89),
    ("dust2_r09", "2392774-de_dust2-r09-p05", 885, 302),
    ("dust2_r09", "2392774-de_dust2-r09-p08", 80, 149),
]


@pytest.mark.parametrize("case, media_id, start, pairs", REAL_LABELS)
def test_the_visible_pairs_of_the_example_labels_are_those_of_the_papers_index(
    case, media_id, start, pairs
):
    root = REPO / "examples" / "data" / case / "vislabels"
    if not (root / f"{media_id}.npz").is_file():
        pytest.skip("the example labels are not fetched (tools/download_examples.py)")
    visible, _ = visibility_rows(root, media_id, np.arange(start + WINDOW_SOURCE_FRAMES))
    assert S.window_pair_counts(visible, int(media_id[-2:]), [start]) == {start: pairs}


# ===================================================================================== the split
def test_the_split_counts_matches_match_maps_rounds_and_recordings(world):
    records = [MediaRecord.from_row(row) for row in world["rows"]]
    assert S.count_split(records, {3}) == {
        "train": {"matches": 4, "match_maps": 6, "rounds": 7, "media": 21},
        "held_out": {"matches": 1, "match_maps": 1, "rounds": 2, "media": 6},
    }
    with pytest.raises(ValueError, match="not in the media index"):
        S.count_split(records, {3, 99})


def test_the_grid_windows_of_a_match_are_counted_per_map(world):
    records = [MediaRecord.from_row(row) for row in world["rows"]]
    assert S.grid_window_counts(records) == {
        1: {"de_a": 6 * 9},
        2: {"de_a": 3 * 9, "de_b": 3 * 9},
        3: {"de_b": 6 * 9},
        4: {"de_a": 3 * 3, "de_b": 3 * 3},
        5: {"de_a": 2 * 9},
    }


def test_the_match_drawn_when_one_fills_a_share_is_the_first_by_its_hash():
    windows = {match: {"de_a": 10} for match in (1, 2, 3, 4)}
    held = S.draw_held_out(windows, fraction=0.25, seed="seed")
    assert held == [min(windows, key=lambda match: rank("seed", match))]


def test_a_match_is_taken_only_while_a_map_below_its_share_has_windows_in_it():
    windows = {1: {"a": 10}, 2: {"a": 10}, 3: {"b": 10}, 4: {"b": 10}}
    for seed in map(str, range(8)):
        held = S.draw_held_out(windows, fraction=0.5, seed=seed)
        assert len(held) == 2 and {match in (1, 2) for match in held} == {True, False}


def test_the_draw_is_deterministic_seeded_and_independent_of_the_order_of_the_matches():
    windows = {m: {"a": 7 + m % 3, "b": 5 + m % 2} for m in range(1, 41)}
    first = S.draw_held_out(windows, fraction=0.2, seed="s")
    assert first == S.draw_held_out(dict(reversed(windows.items())), fraction=0.2, seed="s")
    assert len({tuple(S.draw_held_out(windows, fraction=0.2, seed=str(s))) for s in range(6)}) > 1
    got = Counter()
    for match in first:
        got.update(windows[match])
    total = sum(sum(per_map.values()) for per_map in windows.values())
    assert all(got[name] >= 0.2 * total / 2 for name in "ab")


def test_kept_matches_stay_and_excluded_matches_are_never_drawn():
    windows = {m: {"a": 10} for m in range(1, 11)}
    held = S.draw_held_out(windows, fraction=0.3, seed="s", keep=[4, 5], exclude=[1, 2, 3])
    assert {4, 5} <= set(held) and not {1, 2, 3} & set(held) and len(held) == 3
    assert S.draw_held_out(windows, fraction=0.3, seed="s", keep=[4, 5, 6]) == [4, 5, 6]


def test_a_draw_that_cannot_give_every_map_its_share_is_refused():
    windows = {1: {"a": 10}, 2: {"a": 10}, 3: {"b": 10}}
    with pytest.raises(ValueError, match="cannot give b its share"):
        S.draw_held_out(windows, fraction=0.5, seed="s", exclude=[3])
    with pytest.raises(ValueError, match="not in the media index"):
        S.draw_held_out(windows, fraction=0.5, seed="s", keep=[9])
    for fraction in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            S.draw_held_out(windows, fraction=fraction, seed="s")


# ======================================================================================= reserve
def reserve_rows(rounds: dict, starts=(0, 161, 322, 483), pairs=120) -> list[dict]:
    """A window row of every start for each recording of the rounds ``{(match, map, round): media
    ids}``, and a midpoint with many pairs that must not count."""
    return [
        {
            "media_id": media_id,
            "match_id": match,
            "map_name": map_name,
            "round": round_,
            "start_frame": start,
            "vis_pair_count": 500 if start == 80 else pairs,
        }
        for (match, map_name, round_), media in rounds.items()
        for media_id in media
        for start in (*starts, 80)
    ]


def reference_reserve(rows, target: int, seed: str) -> list[str]:
    """The reserve written out plainly: pass ``d`` gives each match its ``d``-th round."""
    media_of, dense = {}, {}
    for row in rows:
        key = (row["match_id"], row["map_name"], row["round"])
        media_of.setdefault(key, set()).add(row["media_id"])
        if row["start_frame"] % 161 == 0 and row["vis_pair_count"] >= 100:
            dense.setdefault(row["media_id"], []).append(row["start_frame"])
    capacity, rounds_of_match = {}, {}
    for key, media in media_of.items():
        capacity[key] = 0
        for media_id in media:
            last = None
            for start in sorted(dense.get(media_id, [])):
                if last is None or start - last >= 322:
                    capacity[key], last = capacity[key] + 1, start
        if capacity[key]:
            rounds_of_match.setdefault(key[0], []).append(key)
    for rounds in rounds_of_match.values():
        rounds.sort(key=lambda key: (rank(seed, "round", key[0], key[2]), key))
    order = sorted(rounds_of_match, key=lambda match: (rank(seed, "match", match), match))
    picked, reached = [], 0
    for depth in range(4):
        for match in order:
            if reached < target and depth < len(rounds_of_match[match]):
                picked.append(rounds_of_match[match][depth])
                reached += capacity[picked[-1]]
    assert reached >= target
    return sorted(media for key in picked for media in media_of[key])


#: Four matches of three rounds, each of two recordings: 2 windows of capacity each, 4 a round.
ROUNDS_OF_MEDIA = {
    (match, "de_a", round_): [f"{match}-r{round_}-p{slot}" for slot in range(2)]
    for match in (1, 2, 3, 4)
    for round_ in (1, 2, 3)
}


def test_the_reserve_is_whole_rounds_taken_breadth_first_over_the_matches():
    rows = reserve_rows(ROUNDS_OF_MEDIA)
    for target in (4, 13, 16, 17, 30, 48):
        reserve = S.select_reserve(rows, target_capacity=target)
        assert reserve == reference_reserve(rows, target, S.RESERVE_SEED)
        rounds = {tuple(media.split("-")[:2]) for media in reserve}
        assert len(reserve) == 2 * len(rounds)
    one_each = S.select_reserve(rows, target_capacity=16)
    assert Counter(media.split("-")[0] for media in one_each) == {"1": 2, "2": 2, "3": 2, "4": 2}
    first = min(range(1, 4), key=lambda round_: rank(S.RESERVE_SEED, "round", 1, round_))
    assert f"1-r{first}-p0" in one_each


def test_the_reserve_takes_a_second_round_of_the_match_first_in_the_order_of_the_matches():
    reserve = S.select_reserve(reserve_rows(ROUNDS_OF_MEDIA), target_capacity=17)
    first = min(range(1, 5), key=lambda match: rank(S.RESERVE_SEED, "match", match))
    assert Counter(media.split("-")[0] for media in reserve) == {
        str(match): 4 if match == first else 2 for match in range(1, 5)
    }


def test_the_reserve_follows_the_seed():
    rows = reserve_rows(ROUNDS_OF_MEDIA)
    reserves = {tuple(S.select_reserve(rows, target_capacity=20, seed=str(s))) for s in range(6)}
    assert len(reserves) > 1
    assert S.select_reserve(rows, target_capacity=20, seed="s") == reference_reserve(rows, 20, "s")


def test_a_grid_window_counts_to_the_capacity_from_100_pairs_and_two_must_be_322_apart():
    def reserve(starts, pairs):
        rows = reserve_rows({(1, "de_a", 1): ["m"]}, starts=starts, pairs=pairs)
        return S.select_reserve(rows, target_capacity=2)

    assert reserve((0, 322), 100) == ["m"]
    with pytest.raises(ValueError, match="reach a capacity of 1, not 2"):
        reserve((0, 161), 100)
    with pytest.raises(ValueError, match="reach a capacity of 0, not 2"):
        reserve((0, 322), 99)


def test_a_round_without_capacity_is_never_taken_and_a_target_out_of_reach_is_refused():
    rows = reserve_rows(ROUNDS_OF_MEDIA) + reserve_rows({(9, "de_a", 1): ["9-r1-p0"]}, pairs=99)
    reserve = S.select_reserve(rows, target_capacity=48)
    assert len(reserve) == 24 and not any(media.startswith("9-") for media in reserve)
    with pytest.raises(ValueError, match="reach a capacity of 48, not 49"):
        S.select_reserve(rows, target_capacity=49)


# ==================================================================================== validation
def validation_rows(map_name: str, pairs: list[int], media: list[str]) -> list[dict]:
    """A window of each pair count, the ``i``-th of media ``media[i]``, at start ``161 i``."""
    return [
        {
            "media_id": m,
            "match_id": 1,
            "map_name": map_name,
            "round": 1,
            "start_frame": 161 * i,
            "latent_key": f"win_{161 * i:06d}",
            "vis_available": True,
            "vis_pair_count": p,
        }
        for i, (p, m) in enumerate(zip(pairs, media))
    ]


def test_the_validation_windows_have_the_pairs_of_the_upper_quartile():
    # sorted 0 0 10 20 30 40 50 60: the 0.75 quantile is 40 + 0.25 x 10 = 42.5
    rows = validation_rows("de_a", [0, 0, 10, 20, 30, 40, 50, 60], [f"m{i}" for i in range(8)])
    chosen = S.select_validation(rows, maps=["de_a"], per_map=2)
    assert [row["vis_pair_count"] for row in chosen] == [50, 60]
    with pytest.raises(ValueError, match="2 recordings among the 2 windows of at least 42.5"):
        S.select_validation(rows, maps=["de_a"], per_map=3)


def test_a_recording_is_drawn_once_and_the_draw_follows_its_hash():
    rows = []
    for i in range(12):
        recording = validation_rows("de_a", [50, 50, 50], [f"m{i:02d}"] * 3)
        rows += [dict(row, start_frame=100 * i + row["start_frame"]) for row in recording]
    chosen = S.select_validation(rows, maps=["de_a"], per_map=4, seed="s")
    ordered = sorted(
        rows, key=lambda r: (rank("s", "de_a", r["media_id"], r["start_frame"]), r["media_id"])
    )
    first = {}
    for row in ordered:
        first.setdefault(row["media_id"], row)
    expected = list(first.values())[:4]
    assert chosen == sorted(expected, key=lambda r: (r["map_name"], r["media_id"]))
    assert S.select_validation(rows, maps=["de_a"], per_map=4, seed="t") != chosen


def test_windows_without_labels_and_maps_that_are_not_asked_for_are_left_out():
    rows = validation_rows("de_a", [10, 20, 30, 40], list("wxyz"))
    rows += validation_rows("de_b", [99, 99], list("pq"))
    rows[0]["vis_available"] = rows[3]["vis_available"] = False
    # the labelled windows are 20 and 30: the 0.75 quantile is 27.5
    assert [r["media_id"] for r in S.select_validation(rows, maps=["de_a"], per_map=1)] == ["y"]
    with pytest.raises(ValueError, match="no labelled windows of de_c"):
        S.select_validation(rows, maps=["de_c"], per_map=1)


def test_the_validation_windows_come_by_map_media_and_start():
    rows = validation_rows("de_b", [5] * 4, list("dcba")) + validation_rows(
        "de_a", [5] * 4, list("dcba")
    )
    chosen = S.select_validation(rows, maps=["de_b", "de_a"], per_map=4)
    keys = [(r["map_name"], r["media_id"], r["start_frame"]) for r in chosen]
    assert keys == sorted(keys) and len(keys) == 8


# ======================================================================================== weights
def stats_of(rows: list[list[int]]) -> list[S.SlotStats]:
    slots = []
    for pairs in rows:
        slots.append(S.SlotStats())
        for pair_count in pairs:
            slots[-1].add(pair_count)
    return slots


def loader_shares(rows: list[list[int]], weights: list[float]) -> list[float]:
    """The share of each slot of the loader's weights of the windows (:func:`D.read_bucket_index`
    weights a window by its file's weight and ``(1 + pairs) ** 0.5``, then floors the zero-pair
    windows)."""
    flat = [(slot, p) for slot, pairs in enumerate(rows) for p in pairs]
    plain = [weights[slot] * (1.0 + p) ** D.DENSITY_EXPONENT for slot, p in flat]
    floored = D.apply_zero_pair_floor(plain, [p for _, p in flat])
    mass = [sum(w for w, (slot, _) in zip(floored, flat) if slot == s) for s in range(len(rows))]
    return [m / sum(mass) for m in mass]


@pytest.mark.parametrize(
    "rows",
    [
        [[3, 3], [0, 8]],
        [[0, 0, 0, 5], [7, 9, 11, 13, 15, 0]],
        [[4, 4, 9], [1, 1], [20, 0, 0, 0, 0, 0, 0, 0], []],
        [[2, 3], [5, 6, 7]],
        [[0] * 20 + [1], [0] * 20 + [1, 8]],
    ],
)
def test_the_solved_weights_give_every_map_the_same_draws_under_the_loaders_floor(rows):
    weights = S.solve_slot_weights(stats_of(rows))
    live = [i for i, pairs in enumerate(rows) if pairs]
    assert max(weights[i] for i in live) == 1.0
    assert all(weights[i] == 1.0 for i in range(len(rows)) if i not in live)
    shares = loader_shares(rows, weights)
    assert [shares[i] for i in live] == pytest.approx([1 / len(live)] * len(live), abs=1e-12)
    assert S.draw_shares(stats_of(rows), weights) == pytest.approx(shares, abs=1e-12)


def test_the_statistics_of_the_papers_files_solve_to_the_weights_of_the_loader():
    # rows, rows without pairs and the sum of (1 + pairs) ** 0.5 of each of the five files
    papers = [
        (651666, 191681, 3811882.8830338567),
        (579532, 196608, 2901034.2610991336),
        (488562, 170241, 2509839.0809359313),
        (504945, 165916, 2747987.3385433615),
        (0, 0, 0.0),
    ]
    slots = [S.SlotStats(*row) for row in papers]
    assert [round(w, 9) for w in S.solve_slot_weights(slots)] == list(D.BUCKET_WEIGHTS)
    assert S.draw_shares(slots, D.BUCKET_WEIGHTS) == pytest.approx([0.25] * 4 + [0.0], abs=1e-9)
    # equal weights draw the maps unequally
    flat = S.draw_shares(slots, [1.0] * 5)
    assert flat[:4] == pytest.approx([0.310357, 0.246701, 0.213464, 0.229479], abs=1e-6)


def test_a_slot_alone_weighs_one_and_no_windows_cannot_be_weighed():
    assert S.solve_slot_weights(stats_of([[1, 2], []])) == [1.0, 1.0]
    with pytest.raises(ValueError, match="no windows"):
        S.solve_slot_weights(stats_of([[], []]))


# ================================================================================ the CLI
def test_the_bucket_index_holds_the_windows_of_each_map_in_its_file(world, capsys):
    out = world["dir"] / "buckets"
    summary = run(capsys, windows_argv(world, out, exclude_media=world["excluded"]))
    assert sorted(path.name for path in out.iterdir()) == [
        f"train_{name}.jsonl" for name in D.BUCKET_NAMES
    ]
    assert summary["split"] == {
        "train": {"matches": 4, "match_maps": 6, "rounds": 7, "media": 21},
        "held_out": {"matches": 1, "match_maps": 1, "rounds": 2, "media": 6},
    }
    assert summary["left_out_media"] == {"excluded": 1, "without a window": 1, "without labels": 1}
    assert {name: (f["map"], f["rows"]) for name, f in summary["files"].items()} == {
        "train_q00-01.jsonl": ("de_a", 6 * 9 + 3 * 9 + 3 * 3),
        "train_q01-03.jsonl": ("de_b", 3 * 9 + 3 * 3),
        "train_q03-05.jsonl": (None, 0),
        "train_q05-07.jsonl": (None, 0),
        "train_q07-10.jsonl": (None, 0),
    }
    for name, entry in summary["files"].items():
        assert entry["sha256"] == sha256_file(out / name)

    rows = read_rows(out / "train_q00-01.jsonl")
    assert {row["map_name"] for row in rows} == {"de_a"}
    assert 3 not in {row["match_id"] for row in bucket_rows(out)}
    keys = [(r["media_id"], r["start_frame"]) for r in rows]
    assert keys == sorted(set(keys))
    mine = [r for r in rows if r["media_id"] == "1-de_a-r01-p00"]
    assert [(r["start_frame"], r["vis_pair_count"]) for r in mine] == [
        (start, PATTERN_PAIRS.get(start, 0)) for start in STARTS_1000
    ]
    dense = [r for r in rows if r["media_id"] == "1-de_a-r01-p01"]
    assert {r["vis_pair_count"] for r in dense} == {DENSE_PAIRS}
    assert {r["start_frame"] for r in rows if r["media_id"] == "4-de_a-r01-p00"} == set(STARTS_500)
    assert not {r["media_id"] for r in rows if r["match_id"] == 5}


def test_a_row_is_written_with_sorted_keys_and_compact_separators(world, capsys):
    out = world["dir"] / "buckets"
    run(capsys, windows_argv(world, out))
    assert (out / "train_q00-01.jsonl").read_text().splitlines()[0] == (
        '{"latent_key":"win_000000","map_name":"de_a","match_id":1,"media_id":"1-de_a-r01-p00",'
        '"round":1,"start_frame":0,"vis_available":true,"vis_pair_count":15}'
    )


def test_the_bucket_index_is_the_same_on_every_run_and_the_loader_reads_it(world, capsys):
    first, second = world["dir"] / "first", world["dir"] / "second"
    run(capsys, windows_argv(world, first, exclude_media=world["excluded"]))
    run(capsys, windows_argv(world, second, exclude_media=world["excluded"]))
    for name in D.BUCKET_NAMES:
        path = f"train_{name}.jsonl"
        assert sha256_file(first / path) == sha256_file(second / path)
    windows = D.read_bucket_index(first)
    assert len(windows) == 90 + 36
    index = MediaIndex.load(world["index"])
    for w in windows:
        media = index.media(w.media_id)
        assert (w.match_id, w.round) == (media.match_id, media.round)
    assert all(w.weight > 0 for w in windows)


def test_the_loader_refuses_a_window_that_is_written_twice(world, capsys):
    out = world["dir"] / "buckets"
    run(capsys, windows_argv(world, out))
    path = out / "train_q00-01.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines, lines[0]]) + "\n")
    with pytest.raises(ValueError, match="repeats window 1-de_a-r01-p00@0"):
        D.read_bucket_index(out)


def test_the_reserve_leaves_the_loader_the_windows_that_are_not_reserved(world, capsys):
    out, reserve = world["dir"] / "buckets", world["dir"] / "reserve.txt"
    run(capsys, windows_argv(world, out))
    summary = run(capsys, argv("reserve", bucket_dir=out, out=reserve, target_capacity=5))
    media = reserve.read_text().splitlines()
    rows = bucket_rows(out)
    assert summary == {"media": len(media), "sha256": sha256_file(reserve)}
    assert media == sorted(set(media)) == reference_reserve(rows, 5, S.RESERVE_SEED)
    # every recording of a reserved round is reserved, and the loader finds each id in the buckets
    rounds = {(r["match_id"], r["map_name"], r["round"]) for r in rows if r["media_id"] in media}
    in_rounds = {
        r["media_id"] for r in rows if (r["match_id"], r["map_name"], r["round"]) in rounds
    }
    assert in_rounds == set(media)
    kept = D.read_bucket_index(out, D.read_media_exclusion(reserve))
    assert len(kept) == len(rows) - sum(r["media_id"] in media for r in rows)
    assert not {w.media_id for w in kept} & set(media)
    with pytest.raises(ValueError, match="not in the buckets"):
        D.read_bucket_index(out, frozenset([*media, "9-de_a-r01-p00"]))


def test_the_reserve_can_be_drawn_from_some_maps_only(world, capsys):
    out, reserve = world["dir"] / "buckets", world["dir"] / "reserve.txt"
    run(capsys, windows_argv(world, out))
    run(capsys, argv("reserve", bucket_dir=out, out=reserve, target_capacity=3, maps=["de_b"]))
    rows = [r for r in bucket_rows(out) if r["map_name"] == "de_b"]
    assert reserve.read_text().splitlines() == reference_reserve(rows, 3, S.RESERVE_SEED)
    assert {media.split("-")[1] for media in reserve.read_text().split()} == {"de_b"}


def test_the_validation_index_has_windows_of_the_held_out_matches_only(world, capsys):
    out = world["dir"] / "validation.jsonl"
    flags = {
        "media_index": world["index"],
        "held_out": world["held_out"],
        "visibility_label_root": world["labels"],
        "maps": "de_b",
        "out": out,
    }
    summary = run(capsys, argv("validation", per_map=2, **flags))
    assert summary.pop("split")["held_out"]["matches"] == 1
    assert summary == {"windows": 54, "left_out_media": {}, "rows": 2, "sha256": sha256_file(out)}
    # the upper quartile of the 54 windows are those of the two dense recordings
    assert [(r["media_id"], r["vis_pair_count"]) for r in read_rows(out)] == [
        ("3-de_b-r01-p01", DENSE_PAIRS),
        ("3-de_b-r02-p01", DENSE_PAIRS),
    ]
    assert "2 recordings among the 18 windows" in usage_error(
        capsys, argv("validation", per_map=3, **flags)
    )


def test_the_split_writes_the_matches_the_windows_step_reads(world, capsys):
    drawn = world["dir"] / "drawn.txt"
    flags = {"media_index": world["index"], "out": drawn, "seed": "s", "fraction": 0.2}
    summary = run(capsys, argv("split", exclude_matches=world["held_out"], **flags))
    matches = [int(line) for line in drawn.read_text().split()]
    assert matches == sorted(matches) and 3 not in matches
    assert summary["split"]["held_out"]["matches"] == len(matches)
    assert summary["sha256"] == sha256_file(drawn)
    windows = S.grid_window_counts(MediaRecord.from_row(row) for row in world["rows"])
    share = 0.2 * sum(sum(per_map.values()) for per_map in windows.values()) / 2
    assert min(summary["held_out_windows"].values()) >= share
    again = world["dir"] / "again.txt"
    run(capsys, argv("split", exclude_matches=world["held_out"], **{**flags, "out": again}))
    assert again.read_text() == drawn.read_text()
    world["held_out"] = drawn
    run(capsys, windows_argv(world, world["dir"] / "buckets"))
    assert not {r["match_id"] for r in bucket_rows(world["dir"] / "buckets")} & set(matches)


def test_a_map_is_checked_against_its_ticks_when_the_recordings_are_given(
    tmp_path, capsys, monkeypatch
):
    t = ticks(40, gap=(10.0, 10.5))
    recorded = [media_row(1, "de_g", 1, 0, 1280), media_row(1, "de_p", 1, 0, 1280)]
    held_out = media_row(2, "de_p", 1, 0, 400)
    for row in [*recorded, held_out]:
        write_labels(tmp_path / "labels", {**row, "player_slot": 2})
    for row in recorded:
        row["ticks_rows"] = len(t)
    read = []

    def read_tick_columns(path, columns, rows):
        read.append((Path(path).name, tuple(columns), rows))
        return {"t": list(t), "is_alive": [True] * len(t)}

    monkeypatch.setattr("worldcast.data.preparation.buckets.read_tick_columns", read_tick_columns)
    flags = {
        "media_index": write_lines(
            tmp_path / "index.jsonl", map(json.dumps, [*recorded, held_out])
        ),
        "held_out": write_lines(tmp_path / "held_out.txt", [2]),
        "visibility_label_root": tmp_path / "labels",
        "maps": ["de_g", "de_p"],
    }

    def starts(name: str, file: str) -> list[int]:
        return [r["start_frame"] for r in read_rows(tmp_path / name / file)]

    run(capsys, argv("windows", out_dir=tmp_path / "unchecked", **flags))
    every = [0, 80, 161, 241, 322, 402, 483, 563, 644, 724, 805, 885]
    assert starts("unchecked", "train_q00-01.jsonl") == every
    assert starts("unchecked", "train_q01-03.jsonl") == every
    assert read == []
    flags["dataset_root"] = tmp_path / "dataset"
    run(capsys, argv("windows", out_dir=tmp_path / "all", **flags))
    kept = [0, 483, 563, 644, 724, 805]
    assert starts("all", "train_q00-01.jsonl") == starts("all", "train_q01-03.jsonl") == kept
    columns = ("t", "is_alive")
    assert read == [("de_g-1-0.parquet", columns, len(t)), ("de_p-1-0.parquet", columns, len(t))]
    read.clear()
    run(capsys, argv("windows", out_dir=tmp_path / "one", tick_gate_maps="de_g", **flags))
    assert starts("one", "train_q00-01.jsonl") == kept
    assert starts("one", "train_q01-03.jsonl") == every
    assert read == [("de_g-1-0.parquet", columns, len(t))]


def test_the_weights_of_a_bucket_index_are_solved_from_its_files(world, capsys):
    out = world["dir"] / "buckets"
    built = run(capsys, windows_argv(world, out, exclude_media=world["excluded"]))
    summary = run(capsys, argv("weights", bucket_dir=out))
    assert summary["slot_weights"] == built["slot_weights"]
    assert summary["rows"] == [90, 36, 0, 0, 0] and summary["matches_loader_weights"] is False
    weights = summary["slot_weights"]
    assert max(weights[:2]) == 1.0 and weights[2:] == [1.0, 1.0, 1.0]
    slots = [
        [r["vis_pair_count"] for r in read_rows(out / f"train_{n}.jsonl")] for n in D.BUCKET_NAMES
    ]
    assert loader_shares(slots[:2], weights[:2]) == pytest.approx([0.5, 0.5], abs=1e-6)


def test_the_weights_step_prints_the_config_that_draws_every_map_equally(world, capsys):
    out = world["dir"] / "buckets"
    run(capsys, windows_argv(world, out, exclude_media=world["excluded"]))
    assert PREPARE.main(argv("weights", bucket_dir=out)) == 0
    printed = capsys.readouterr()
    weights = json.loads(printed.out)["slot_weights"]
    assert printed.err == f"data:\n  bucket_weights: {json.dumps(weights)}\n"
    # the printed lines are a config of their own: they set the loader's weights
    cfg = TrainConfig.from_dict({"run": {"stage": "2"}, **yaml.safe_load(printed.err)})
    assert cfg.data.bucket_weights == tuple(weights)
    files = [len(read_rows(out / f"train_{name}.jsonl")) for name in D.BUCKET_NAMES]
    for given, equal in ((cfg.data.bucket_weights, True), (None, False)):
        drawn = [w.weight for w in D.read_bucket_index(out, bucket_weights=given)]
        first = sum(drawn[: files[0]]) / sum(drawn)
        assert (first == pytest.approx(0.5, abs=1e-6)) is equal


def test_the_validation_step_prints_the_digest_the_evaluation_checks(world, capsys):
    out = world["dir"] / "validation.jsonl"
    flags = {
        "media_index": world["index"],
        "held_out": world["held_out"],
        "visibility_label_root": world["labels"],
        "maps": "de_b",
        "per_map": 2,
        "out": out,
    }
    assert PREPARE.main(argv("validation", **flags)) == 0
    printed = capsys.readouterr()
    digest = sha256_file(out)
    assert json.loads(printed.out)["sha256"] == digest
    lines = f"validation:\n  index_sha256: {digest}\n"
    assert printed.err == f"de_b: 54 windows\n{lines}"
    cfg = TrainConfig.from_dict({"run": {"stage": "2"}, **yaml.safe_load(lines)})
    assert cfg.validation.index_sha256 == digest


def test_the_weights_of_files_without_windows_or_labels_are_refused(tmp_path, capsys):
    for name in D.BUCKET_NAMES:
        (tmp_path / f"train_{name}.jsonl").write_text("")
    assert "no windows to weight" in usage_error(capsys, argv("weights", bucket_dir=tmp_path))
    (tmp_path / "train_q00-01.jsonl").write_text('{"vis_available":false}\n')
    assert "train_q00-01.jsonl:1 has no visibility labels" in usage_error(
        capsys, argv("weights", bucket_dir=tmp_path)
    )


# ==================================================================================== usage errors
@pytest.mark.parametrize(
    "flags, message",
    [
        ({"held_out": "missing.txt"}, "held-out matches does not exist"),
        ({"maps": ["de_a", "de_z"]}, "no recordings of de_z"),
        ({"tick_gate_maps": ["de_a"]}, "--tick-gate-maps needs --dataset-root"),
        ({"exclude_media": "nobody.txt"}, "excluded media does not exist"),
    ],
)
def test_a_windows_step_that_cannot_run_is_a_usage_error(world, capsys, flags, message):
    assert message in usage_error(capsys, windows_argv(world, world["dir"] / "out", **flags))


def test_matches_and_media_that_the_index_does_not_have_are_refused(world, capsys):
    out = world["dir"] / "out"
    held_out = write_lines(world["dir"] / "other.txt", [3, 99])
    assert "1 held-out matches are not in the media index" in usage_error(
        capsys, windows_argv(world, out, held_out=held_out)
    )
    nobody = write_lines(world["dir"] / "nobody.txt", ["9-de_a-r01-p00"])
    assert "1 excluded media are not in the media index" in usage_error(
        capsys, windows_argv(world, out, exclude_media=nobody)
    )
    twice = write_lines(world["dir"] / "twice.txt", [3, 3])
    assert "lists a match twice" in usage_error(capsys, windows_argv(world, out, held_out=twice))
    words = write_lines(world["dir"] / "words.txt", ["three"])
    assert "one match id per line" in usage_error(capsys, windows_argv(world, out, held_out=words))


def test_checked_maps_must_be_among_the_maps(world, capsys):
    error = usage_error(
        capsys,
        windows_argv(world, world["dir"] / "out", dataset_root=world["dir"], tick_gate_maps="de_c"),
    )
    assert "--tick-gate-maps must be among --maps" in error


def test_more_maps_than_bucket_files_are_refused(tmp_path, capsys):
    maps = [f"de_{name}" for name in "abcdef"]
    rows = [media_row(i, name, 1, 0, 1000) for i, name in enumerate(maps)]
    flags = {
        "media_index": write_lines(tmp_path / "index.jsonl", map(json.dumps, rows)),
        "held_out": write_lines(tmp_path / "held_out.txt", [0]),
        "visibility_label_root": tmp_path / "labels",
        "maps": maps,
        "out_dir": tmp_path / "out",
    }
    assert "at most 5 maps fit the bucket files" in usage_error(capsys, argv("windows", **flags))


def test_a_run_without_windows_is_refused(world, capsys):
    excluded = write_lines(
        world["dir"] / "all.txt", [row["media_id"] for row in world["rows"] if row["match_id"] != 3]
    )
    error = usage_error(capsys, windows_argv(world, world["dir"] / "out", exclude_media=excluded))
    assert "no windows to write; left out: {'excluded': 21}" in error


@pytest.mark.parametrize("command", ["split", "windows", "reserve", "validation", "weights"])
def test_every_step_has_help(command, capsys):
    with pytest.raises(SystemExit) as stopped:
        PREPARE.main(["buckets", command, "--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
