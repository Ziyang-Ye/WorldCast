"""The protocols and their sealed window selection."""

import dataclasses
import hashlib
import json

import pytest

from worldcast.engine.evaluation.protocols import (
    FOUR_STEP,
    PROTOCOLS,
    UNIPC,
    check_selection,
    load_windows,
    read_index,
    select_windows,
    selection_sha256,
    window_noise_seed,
)


def test_a_windows_noise_seed_follows_from_the_seed_and_the_window():
    assert window_noise_seed(20260829, "2393033-de_dust2-r02-p00", 80) == 7706051063969448305
    assert window_noise_seed(20260829, "2393233-de_dust2-r03-p00", 241) == 3923914349952882614


def test_the_selection_does_not_depend_on_the_order_of_the_index(index_rows):
    protocol = dataclasses.replace(UNIPC, count=8)
    windows = select_windows(index_rows + index_rows[:3], protocol)  # a repeated row counts once
    assert [w.index for w in windows] == list(range(8))
    assert len({(w.media_id, w.start_frame) for w in windows}) == 8
    reordered = select_windows(index_rows[::-1], protocol)
    assert [(w.media_id, w.start_frame, w.noise_seed) for w in reordered] == [
        (w.media_id, w.start_frame, w.noise_seed) for w in windows
    ]
    first = windows[0]
    assert first.noise_seed == window_noise_seed(20260829, first.media_id, first.start_frame)
    assert first.map_name == first.record["map_name"]
    with pytest.raises(ValueError, match="the index holds 5 windows, the protocol needs 8"):
        select_windows(index_rows[:5], protocol)


def test_the_stratum_of_a_window_is_its_visible_players_else_its_quality(index_rows):
    rows = [
        dict(index_rows[0], num_visible_players=2, quality=0.5),
        dict(index_rows[1], quality=0.25),
        index_rows[2],
    ]
    windows = select_windows(rows, dataclasses.replace(UNIPC, count=3))
    strata = {w.media_id: w.stratum for w in windows}
    assert [strata[row["media_id"]] for row in rows] == ["vis2", "vis0.25", "vis?"]


def test_a_selection_and_an_index_are_checked_against_their_sha256(tmp_path, index_rows):
    protocol = dataclasses.replace(UNIPC, count=8)
    windows = select_windows(index_rows, protocol)
    with pytest.raises(ValueError, match="are not those of worldcast-maps4-eval64-v1"):
        check_selection(windows, protocol)
    sealed = dataclasses.replace(protocol, selection_sha256=selection_sha256(windows))
    check_selection(windows, sealed)
    with pytest.raises(ValueError, match="are not those of"):
        check_selection(windows[::-1], sealed)

    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in index_rows))
    with pytest.raises(ValueError, match="is not the index of worldcast-maps4-eval64-v1"):
        read_index(index, UNIPC)
    own = dataclasses.replace(UNIPC, index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    assert read_index(index, own) == index_rows


def test_the_protocols_are_named_by_their_sampler():
    assert sorted(PROTOCOLS) == ["four_step", "unipc"]
    assert PROTOCOLS["unipc"] is UNIPC and PROTOCOLS["four_step"] is FOUR_STEP
    row = FOUR_STEP.fields()
    assert list(row) == [
        "window_set",
        "sampler",
        "denoising_steps",
        "index_sha256",
        "selection_sha256",
        "seed",
    ]
    assert (row["window_set"], row["sampler"]) == ("worldcast-maps4-eval64-v1", "four_step")
    # both protocols score the same windows with the same noise
    shared = ("window_set", "index_sha256", "selection_sha256", "seed")
    assert [UNIPC.fields()[key] for key in shared] == [row[key] for key in shared]


def _written(tmp_path, rows):
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return index, hashlib.sha256(index.read_bytes()).hexdigest()


def test_a_rebuilt_index_is_read_by_the_digest_it_is_given(tmp_path, index_rows):
    protocol = dataclasses.replace(UNIPC, count=8)
    index, digest = _written(tmp_path, index_rows)
    with pytest.raises(ValueError, match="is not the index of worldcast-maps4-eval64-v1"):
        load_windows(index, protocol)
    with pytest.raises(ValueError, match="is not the index of"):
        load_windows(index, protocol, "0" * 64)

    scored, windows = load_windows(index, protocol, digest)
    assert windows == select_windows(index_rows, protocol)
    assert scored == dataclasses.replace(
        protocol, index_sha256=digest, selection_sha256=selection_sha256(windows)
    )
    assert read_index(index, scored) == index_rows
    row = scored.fields()
    assert (row["index_sha256"], row["selection_sha256"]) == (digest, selection_sha256(windows))
    # the rest of the protocol is the paper's
    assert (scored.window_set, scored.seed, scored.sampler) == (UNIPC.window_set, 20260829, "unipc")


def test_without_a_digest_the_pinned_index_and_selection_are_checked(tmp_path, index_rows):
    protocol = dataclasses.replace(UNIPC, count=8)
    index, digest = _written(tmp_path, index_rows)
    sealed = dataclasses.replace(
        protocol,
        index_sha256=digest,
        selection_sha256=selection_sha256(select_windows(index_rows, protocol)),
    )
    for given in (None, digest):  # the protocol's own digest is the same as none
        scored, windows = load_windows(index, sealed, given)
        assert scored is sealed and len(windows) == 8
    other = dataclasses.replace(sealed, selection_sha256="0" * 64)
    with pytest.raises(ValueError, match="are not those of worldcast-maps4-eval64-v1"):
        load_windows(index, other)
