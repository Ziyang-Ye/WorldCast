"""One client's window: a player's frames, the item, the data paths and the collate."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data.dataloader import default_collate

from tests.data.support import media_record, tick_table
from tests.data.synthetic_round import CLIENT, ENEMY, TEAMMATE
from worldcast.data.labels import OBSERVER_SIGNAL_KEYS
from worldcast.data.window import (
    DataPaths,
    PlayerFrames,
    WindowRefused,
    WindowSpec,
    collate_windows,
    covered_frames,
    load_client_window,
)

SPEC = WindowSpec(latent_frames=9)


def test_a_window_of_n_latent_frames_has_1_plus_4_n_minus_1_video_frames():
    assert (SPEC.video_frames, WindowSpec(41).video_frames, WindowSpec(441).video_frames) == (
        33,
        161,
        1761,
    )
    # video frame k is source frame start + 2 k of the 32 fps recording
    assert WindowSpec(2).source_frames(16).tolist() == [16, 18, 20, 22, 24]


def test_a_players_frames_sample_the_last_tick_at_each_video_frame():
    table = tick_table()
    frames = PlayerFrames.from_ticks(table, media_record(table), 16, SPEC)
    assert frames.states.shape == (33, 6) and frames.states.dtype == np.float32
    assert frames.covered == 33 and frames.team_id == 2
    # video frame k is at (16 + 2 k) / 32 s; the player walks 250 u/s and turns 30 degrees/s
    assert frames.states[[0, 1, 32], 0].tolist() == [125.0, 140.625, 625.0]
    assert frames.states[[0, 1, 32], 3].tolist() == [15.0, 16.875, 75.0]
    assert (frames.states[:, 5] == 1.0).all()
    assert frames.control_substeps.shape == (33, 4, 13) and frames.control_substep_valid.all()
    assert frames.buttons.sum(0).tolist() == [33.0] + [0.0] * 10  # forward, every frame
    assert frames.weapon_ids.tolist() == [2] * 33  # the ak47


def test_frames_past_the_ticks_read_as_dead():
    table = tick_table(seconds=1.5)
    media = media_record(table)
    # the last tick is at 95 / 64 s; with the slack of 2.5 ticks, video frames 0 .. 24 are covered
    assert covered_frames(table, media, 0, SPEC) == 25
    frames = PlayerFrames.from_ticks(table, media, 0, SPEC)
    assert frames.covered == 25 and frames.states[:25, 5].all() and not frames.states[25:, 5].any()
    assert frames.states[25:, 0].tolist() == [frames.states[24, 0]] * 8  # the last tick's pose
    assert not frames.buttons[25:].any() and not frames.control_substep_valid[25:].any()
    absent = PlayerFrames.absent(SPEC)
    assert (absent.covered, absent.team_id) == (0, 0) and not absent.states.any()


def test_a_tick_rate_the_recording_does_not_divide_is_refused():
    table = tick_table()
    media = media_record(table)
    with pytest.raises(ValueError, match="not a multiple of fps 30.0"):
        PlayerFrames.from_ticks(table, SimpleNamespace(fps=30.0, media_id="m0"), 0, SPEC)
    with pytest.raises(ValueError, match="fps 0.0 of m0 is not positive"):
        PlayerFrames.from_ticks(table, SimpleNamespace(fps=0.0, media_id="m0"), 0, SPEC)
    assert PlayerFrames.from_ticks(table, media, 0, SPEC).covered == 33


def test_a_clients_window_holds_every_players_frames_and_the_item(recorded_round):
    spec = WindowSpec(41)
    window = load_client_window(
        recorded_round.row(), recorded_round.media_index, recorded_round.paths, spec
    )
    assert window.media.media_id == CLIENT and window.client_frames.covered == 161
    assert sorted(window.tick_tables) == [0, 1, 5]
    assert {slot: media.media_id for slot, media in window.round_slots.items()} == {
        0: CLIENT,
        1: TEAMMATE,
        5: ENEMY,
    }
    item = window.item.batch_dict()
    assert sorted(item) == sorted(
        [
            "buttons",
            "view_deltas",
            "weapon",
            "player_states",
            "player_weapon_ids",
            "player_control_substeps",
            "player_control_substep_valid",
            "player_team_ids",
            "client_slot",
            "client_visibility",
            "client_visibility_valid",
            "metadata",
            *OBSERVER_SIGNAL_KEYS,
        ]
    )
    assert item["metadata"] == {"media_id": CLIENT, "start_frame": 0}
    assert item["player_states"].shape == (10, 161, 6) and item["client_slot"].item() == 0
    assert item["player_team_ids"].tolist() == [2, 2, 0, 0, 0, 3, 0, 0, 0, 0]
    # the players stand at their places; the client turns after 7.03 s (video frame 113 on)
    assert item["player_states"][[0, 1, 5], 0, :2].tolist() == [[0, 0], [-100, 0], [-200, 100]]
    assert item["player_states"][0, [112, 113], 3].tolist() == [0.0, 180.0]
    assert not item["player_states"][[2, 3, 4, 6, 7, 8, 9]].any()  # slots without a recording
    assert item["client_visibility"].dtype == torch.float32
    assert item["client_visibility_valid"].dtype == torch.bool
    assert item["obs_flash_valid"].tolist() == [1] * 41 and not item["obs_flash_flag"].any()


def test_a_window_the_clients_ticks_do_not_cover_is_refused(recorded_round):
    row, media_index, paths = recorded_round.row(), recorded_round.media_index, recorded_round.paths
    # the ticks end after 10.5 s, 169 video frames; 45 latent frames are 177
    with pytest.raises(WindowRefused, match=r"\(169/177 video frames\)"):
        load_client_window(row, media_index, paths, WindowSpec(45))
    # up to four missing video frames are served (the round ended inside the last block)
    window = load_client_window(row, media_index, paths, WindowSpec(44))
    assert (window.client_frames.covered, window.spec.video_frames) == (169, 173)
    assert not window.item.player_states[0, 169:, 5].any()
    with pytest.raises(ValueError, match="names slot 3, media row 0"):
        load_client_window(dataclasses.replace(row, player_slot=3), media_index, paths, SPEC)


def test_data_paths_come_from_a_config_section():
    section = SimpleNamespace(**{name: f"/data/{name}" for name in DataPaths.names()})
    paths = DataPaths.from_config(section, "paths")
    assert paths.latent_cache_root == Path("/data/latent_cache_root")
    section.media_index = None
    with pytest.raises(ValueError, match="not set: paths.media_index"):
        DataPaths.from_config(section, "paths")


def test_collate_stacks_tensors_and_lists_the_rest():
    rng = torch.Generator().manual_seed(0)
    items = [
        {
            "latents": torch.randn(3, 2, generator=rng),
            "client_slot": torch.tensor(i),
            "player_states": torch.randn(10, 5, 6, generator=rng),
            "metadata": {"media_id": f"m{i}", "start_frame": 8 * i},
            "window_memory_mask": torch.rand(4, 12, 21, generator=rng) > 0.5,
        }
        for i in range(3)
    ]
    new, ref = collate_windows(items), default_collate(items)
    for key, value in new.items():
        if isinstance(value, torch.Tensor):
            assert value.dtype == ref[key].dtype and torch.equal(value, ref[key]), key
    assert new["client_slot"].tolist() == [0, 1, 2] and new["latents"].shape == (3, 3, 2)
    assert new["metadata"] == [it["metadata"] for it in items]
