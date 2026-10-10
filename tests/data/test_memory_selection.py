"""Choosing the memory frames of a training window: the teammate blocks, the best block, the
acceptance and the target draw."""

import numpy as np
import pytest

from tests.data import synthetic_round
from tests.data.support import unscoped_signals
from tests.data.synthetic_round import TEAMMATE
from worldcast.data import memory_selection as M
from worldcast.data.controls import OPENCS2_WEAPONS
from worldcast.data.memory_frames import MemoryBlock
from worldcast.data.window import WindowSpec, load_client_window

SPEC = WindowSpec(41)
#: m_k of each target frame on the synthetic round: the part of the wall behind the client that
#: lies within 420 u of the teammate and off the HUD of both frames (rows 3-8, columns 5-15),
#: without the tokens that the teammate and the enemy stand in front of (right of column 7, from
#: their heads down).
MEMORY_MASK = [
    "000000000000000000000",
    "000000000000000000000",
    "000000000000000000000",
    "000001111111111100000",
    "000001111111111100000",
    "000001110000010000000",
    "000001110000000000000",
    "000001100000000000000",
    "000001100000000000000",
    "000000000000000000000",
    "000000000000000000000",
    "000000000000000000000",
]


def block_frames(name, f0=1, window_start=0, slot=0):
    """The four frames of one teammate block (no cameras or players: these tests do not read
    them)."""
    block = MemoryBlock.at(name, slot, window_start, f0, None)
    return [M.TeammateFrame(block, j, window_start + 8 * (f0 + j), None, None) for j in range(4)]


def _block(name, coverage):
    return block_frames(name), [np.array(coverage, dtype=bool)] * 4


def test_disjoint_teammates_cannot_be_combined_into_one_block():
    a, am = _block("a", [1, 1, 0, 0])
    b, bm = _block("b", [0, 0, 1, 1])
    frames = a + b
    selected, covered = M.best_block(am + bm, frames, np.ones(4, bool), lambda _: np.ones(4, bool))
    assert covered.tolist() == [True, True, False, False]
    assert [frames[i].block.media_id for i in selected] == ["a"] * 4  # the first on a tie
    assert [frames[i].j for i in selected] == [0, 1, 2, 3]


def test_obscured_geometric_winner_loses_to_measured_clear_source():
    a, am = _block("a", [1, 1, 1, 1])
    b, bm = _block("b", [1, 1, 1, 0])
    selected, covered = M.best_block(
        am + bm, a + b, np.ones(4, bool), lambda i: np.zeros(4, bool) if i < 4 else np.ones(4, bool)
    )
    assert selected == [4, 5, 6, 7] and covered.tolist() == [True, True, True, False]


def test_incomplete_block_is_not_padded_with_duplicate_frames():
    frames, masks = _block("a", [1, 1, 1, 1])
    selected, covered = M.best_block(
        masks[:3], frames[:3], np.ones(4, bool), lambda _: np.ones(4, bool)
    )
    assert selected == [] and not covered.any()
    assert list(M.complete_blocks(frames)) == [("a", 0, 1)] and M.complete_blocks(frames[:3]) == {}


def test_surface_the_context_saw_is_not_counted():
    frames, masks = _block("a", [1, 1, 1, 1])
    selected, covered = M.best_block(
        masks, frames, np.array([0, 0, 1, 0], bool), lambda _: np.ones(4, bool)
    )
    assert selected == [0, 1, 2, 3] and covered.tolist() == [False, False, True, False]


def test_least_covering_block_shares_no_frame_with_the_memory_frames():
    a, am = _block("a", [1, 1, 1, 1])
    b, bm = _block("b", [1, 0, 0, 0])
    frames = a + b
    unseen = np.ones(4, bool)
    assert M.least_covering_block(am + bm, frames, unseen, exclude=[]) == [4, 5, 6, 7]
    exclude = [("b", f.frame) for f in b]
    assert M.least_covering_block(am + bm, frames, unseen, exclude=exclude) == [0, 1, 2, 3]
    exclude += [("a", f.frame) for f in a]
    assert M.least_covering_block(am + bm, frames, unseen, exclude=exclude) is None


def test_a_latent_frame_is_known_clear_with_known_labels_no_flash_and_a_known_zoom():
    signals = unscoped_signals(6)
    signals["obs_flash_flag"][1] = 1  # flashed
    signals["obs_flash_valid"][2] = 0  # flash label unknown
    signals["obs_scope_valid"][3] = 0  # scope label unknown
    signals["obs_scope_on"][4:] = 1  # scoped ...
    signals["obs_scope_level"][4:] = [1, 0]  # ... at a zoom level, and at none
    awp = np.full(21, OPENCS2_WEAPONS.index("awp"))
    assert M.known_clear(signals, range(6), awp).tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    ak47 = np.full(21, OPENCS2_WEAPONS.index("ak47"))  # no zoom entry: a scoped frame is unknown
    assert M.known_clear(signals, [0, 4], ak47).tolist() == [True, False]


def _source(recorded_round, config=M.BLOCK_CAUSAL) -> M.MemoryFrameSource:
    return M.MemoryFrameSource(
        dataset_root=recorded_round.paths.dataset_root,
        latent_cache_root=recorded_round.paths.latent_cache_root,
        observer_signal_label_root=recorded_round.paths.observer_signal_label_root,
        meshes=recorded_round.meshes,
        spec=SPEC,
        config=config,
    )


def _window(recorded_round):
    return load_client_window(
        recorded_round.row(), recorded_round.media_index, recorded_round.paths, SPEC
    )


def test_the_candidates_are_the_blocks_of_the_living_teammates(recorded_round):
    blocks = M.teammate_blocks(_source(recorded_round), _window(recorded_round))
    # every block of the teammate's one cached window; none of the client, none of the enemy
    assert [(b.media_id, b.slot, b.window_start, b.f0) for b in blocks] == [
        (TEAMMATE, 1, 0, f0) for f0 in range(1, 41, 4)
    ]
    assert (blocks[3].t_first, blocks[3].t_last) == (104, 128)
    # the teammate looks along -x during block 13 only: its camera's forward axis
    assert [round(float(b.c2w[0, 0, 2])) for b in blocks] == [1, 1, 1, -1, 1, 1, 1, 1, 1, 1]


def test_the_memory_frames_of_a_window_show_what_its_context_has_not_seen(recorded_round):
    """The client looks along +x until it turns to the wall behind it at latent frame 29; the
    teammate looked at that wall during its block 13."""
    selection = M.select_memory_frames(_source(recorded_round), _window(recorded_round))
    # block 25 looks where the first frame looked; blocks 33 and 37 have block 29 in their context
    assert [target.start for target in selection.accepted] == [29]
    target = selection.accepted[0]
    assert (target.block.media_id, target.block.f0) == (TEAMMATE, 13)
    assert [frame.frame for frame in target.frames] == [104, 112, 120, 128]
    assert target.memory_mask.shape == (4, 12, 21) and target.memory_mask.dtype == bool
    for frame_mask in target.memory_mask:
        assert ["".join(str(int(bit)) for bit in row) for row in frame_mask] == MEMORY_MASK
    assert 0.9 * target.memory_mask.sum() <= target.matching_tokens <= target.memory_mask.sum()
    assert selection.window_c2w.shape == (41, 4, 4) and selection.window_tans.shape == (41, 2)


def test_the_bidirectional_stage_judges_the_unseen_surface_against_the_first_frame_alone(
    recorded_round,
):
    source = _source(recorded_round, M.BIDIRECTIONAL)
    selection = M.select_memory_frames(source, _window(recorded_round))
    # its target block is the window's last, and its 32 recent frames are noised with it
    assert [target.start for target in selection.accepted] == [37]
    assert (selection.accepted[0].block.media_id, selection.accepted[0].block.f0) == (TEAMMATE, 13)


def test_a_flashed_teammate_frame_makes_its_block_incomplete(tmp_path):
    pytest.importorskip("trimesh.ray.ray_pyembree")
    pytest.importorskip("cv2")
    with pytest.MonkeyPatch.context() as monkeypatch:
        # source frame 110 is a video frame of the teammate's latent frame 14, in block 13
        flashed = synthetic_round.write_round(tmp_path, flashed_teammate_frames=(110,))
        synthetic_round.patch_readers(monkeypatch, flashed)
        selection = M.select_memory_frames(_source(flashed), _window(flashed))
    assert selection.accepted == []


def test_the_target_draw_is_proportional_to_m_k():
    targets = [M.TargetBlock(start, np.ones(n, bool), []) for start, n in ((25, 10), (29, 30))]
    selection = M.MemoryFrameSelection(targets, None, None)
    draws = [M.draw_target(selection, dataset_index=i, start_frame=0).start for i in range(2000)]
    assert 0.2 < draws.count(25) / len(draws) < 0.3
    again = [M.draw_target(selection, dataset_index=i, start_frame=0).start for i in range(2000)]
    assert draws == again  # keyed by the window, not by a global RNG
    empty = M.MemoryFrameSelection([], None, None)
    assert M.draw_target(empty, dataset_index=0, start_frame=0) is None


def test_the_item_of_a_windows_memory_frames(recorded_round):
    source, window = _source(recorded_round), _window(recorded_round)
    selection = M.select_memory_frames(source, window)
    item = M.memory_frame_item(source, window, selection, selection.accepted[0])
    assert sorted(item) == sorted(
        ["window_target_start", "window_c2w", "window_tans", "window_memory_mask"]
        + [
            "memory_frames_" + name
            for name in (
                "latents",
                "c2w",
                "states",
                "buttons",
                "view_deltas",
                "weapon_ids",
                "control_substeps",
                "control_substep_valid",
                "tans",
                "obs_flash_flag",
                "obs_flash_valid",
                "obs_scope_on",
                "obs_scope_level",
                "obs_scope_valid",
            )
        ]
    )
    assert int(item["window_target_start"]) == 29
    assert item["window_memory_mask"].shape == (4, 12, 21)
    with np.load(recorded_round.paths.latent_cache_root / f"{TEAMMATE}.npz") as cache:
        cached = cache["win_000000"][0, 13:17].astype(np.float32)
    assert np.array_equal(item["memory_frames_latents"].numpy(), cached)
    assert item["memory_frames_states"].shape == (16, 6)
    assert item["memory_frames_states"][:, 0].tolist() == [-100.0] * 16  # the teammate's place
    assert item["memory_frames_c2w"][:, 0, 3].tolist() == [-100.0] * 4
