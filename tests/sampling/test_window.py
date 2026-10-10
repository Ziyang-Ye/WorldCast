"""The window of a block: its layout, the context ranges, the gather of a round batch and the
cameras the ray embedding reads.

The round batch of the gather tests holds, in every per-frame entry, the index of its latent frame
or video frame, so a gathered window lists where each of its frames came from; the memory entry's
inputs count from 100 (latent frames) and 200 (video frames).
"""

import pytest
import torch

from worldcast.modeling.ray_embedding import RayConditions
from worldcast.sampling import window as W
from worldcast.sampling.window import WindowLayout, gather_window

FRAMES, PLAYERS, CLIENT, TARGET = 29, 2, 1, 25
VIDEO_FRAMES = 1 + 4 * (FRAMES - 1)
#: Where the frames of the gathered window come from: first frame | recent context | target frames.
OWN_LATENTS = [0.0] + [float(f) for f in range(13, 29)]
OWN_VIDEO_FRAMES = [0.0] + [float(t) for t in range(49, 113)]
MEMORY_LATENTS = [100.0, 101.0, 102.0, 103.0]
MEMORY_VIDEO_FRAMES = [200.0 + t for t in range(16)]
WITH_MEMORY, WITHOUT_MEMORY = WindowLayout(12, with_memory=True), WindowLayout(
    12, with_memory=False
)


def _index(n: int, *shape: int, offset: float = 0.0, axis: int = 1) -> torch.Tensor:
    """``[1, *shape]`` whose entries are ``offset +`` their index along ``axis``."""
    view = [1] * (1 + len(shape))
    view[axis] = n
    return (offset + torch.arange(n, dtype=torch.float32)).view(view).expand(1, *shape).clone()


def _round_batch() -> dict:
    states = _index(VIDEO_FRAMES, PLAYERS, VIDEO_FRAMES, 6, axis=2)
    states[..., 5] = 1.0  # alive
    batch = {
        "latents": _index(FRAMES, FRAMES, 1, 1, 1),
        "obs_flash_flag": _index(FRAMES, FRAMES),
        "buttons": _index(VIDEO_FRAMES, VIDEO_FRAMES, 11),
        "view_deltas": _index(VIDEO_FRAMES, VIDEO_FRAMES, 2),
        "weapon": _index(VIDEO_FRAMES, VIDEO_FRAMES).long(),
        "player_states": states,
        "player_weapon_ids": _index(VIDEO_FRAMES, PLAYERS, VIDEO_FRAMES, axis=2).long(),
        "player_control_substeps": _index(VIDEO_FRAMES, PLAYERS, VIDEO_FRAMES, 4, 13, axis=2),
        "player_control_substep_valid": torch.ones(1, PLAYERS, VIDEO_FRAMES, 4, dtype=torch.bool),
        "client_visibility": torch.ones(1, PLAYERS, VIDEO_FRAMES, dtype=torch.bool),
        "client_visibility_valid": torch.ones(1, PLAYERS, VIDEO_FRAMES, dtype=torch.bool),
        "client_slot": torch.tensor([CLIENT]),
        "player_team_ids": torch.tensor([[0, 1]]),
        "media_id": ["m0"],
        "window_target_start": torch.tensor([TARGET]),
        "window_c2w": _index(FRAMES, FRAMES, 4, 4),
        "window_tans": _index(FRAMES, FRAMES, 2, offset=1.0),
        W.ROUND_CONTINUOUS_COLUMNS_KEY: _index(FRAMES, FRAMES, PLAYERS, 7),
    }
    memory_states = _index(16, 16, 6, offset=200.0)
    memory_states[..., 5] = 1.0
    memory = {
        "latents": _index(4, 4, 1, 1, 1, offset=100.0),
        "obs_flash_flag": _index(4, 4, offset=100.0),
        "continuous_columns": _index(4, 4, PLAYERS, 7, offset=100.0),
        "buttons": _index(16, 16, 11, offset=200.0),
        "view_deltas": _index(16, 16, 2, offset=200.0),
        "weapon_ids": _index(16, 16, offset=200.0).long(),
        "states": memory_states,
        "control_substeps": _index(16, 16, 4, 13, offset=200.0),
        "control_substep_valid": torch.ones(1, 16, 4, dtype=torch.bool),
        "c2w": _index(4, 4, 4, 4, offset=100.0),
        "tans": _index(4, 4, 2, offset=101.0),
    }
    return {**batch, **{f"memory_frames_{name}": value for name, value in memory.items()}}


# -------------------------------------------------------------------------------------- the layout
def test_window_layout():
    """First frame | memory frames (4) | recent context (12) | target frames (4)."""
    layout = WindowLayout()  # the paper's window, with memory frames
    assert layout == WITH_MEMORY
    assert (layout.memory_positions, layout.target_positions) == ([1, 2, 3, 4], [17, 18, 19, 20])
    assert layout.num_frames == 21 and layout.own_latents(25) == list(range(13, 29))
    assert layout.context_ranges == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17)]
    without = WITHOUT_MEMORY
    assert (without.memory_positions, without.target_positions) == ([], [13, 14, 15, 16])
    assert without.num_frames == 17 and without.own_latents(25) == layout.own_latents(25)
    assert without.context_ranges == [(0, 1), (1, 5), (5, 9), (9, 13)]
    with pytest.raises(ValueError, match="positive multiple of 4"):
        WindowLayout(6)
    with pytest.raises(ValueError, match="leaves no room"):
        layout.own_latents(12)


# -------------------------------------------------------------------------------------- the gather
def test_a_window_entry_refuses_what_the_gather_cannot_place():
    """Only a per-player entry can be zero on the memory frames, and only one with a memory input
    says what the other players hold there."""
    assert W.WindowEntry("latent", "memory_frames_latents").axis == 1
    assert W.WindowEntry("player_video").axis == 2
    with pytest.raises(ValueError, match="frames must be one of"):
        W.WindowEntry("pixel", "memory_frames_latents")
    for frames in ("latent", "video"):
        with pytest.raises(ValueError, match="only a per-player entry can be zero"):
            W.WindowEntry(frames)
        with pytest.raises(ValueError, match=r"other_players must be one of \(None,\)"):
            W.WindowEntry(frames, "memory_frames_buttons", "zero")
    with pytest.raises(ValueError, match=r"other_players must be one of \(None,\)"):
        W.WindowEntry("player_video", other_players="zero")
    for other_players in (None, "last_frame"):
        with pytest.raises(ValueError, match=r"must be one of \('first_frame', 'zero'\)"):
            W.WindowEntry("player_video", "memory_frames_states", other_players)


def test_gather_puts_the_memory_frames_between_the_first_frame_and_the_recent_context():
    window, rays = gather_window(_round_batch(), WITH_MEMORY)
    in_window = OWN_LATENTS[:1] + MEMORY_LATENTS + OWN_LATENTS[1:]
    video_frames = OWN_VIDEO_FRAMES[:1] + MEMORY_VIDEO_FRAMES + OWN_VIDEO_FRAMES[1:]
    assert window["latents"].shape == (1, 21, 1, 1, 1)
    assert window["latents"].flatten().tolist() == in_window
    assert window["obs_flash_flag"][0].tolist() == in_window
    assert window["buttons"].shape == (1, 81, 11)
    assert window["buttons"][0, :, 0].tolist() == video_frames
    assert window["weapon"].dtype == torch.long
    assert window["weapon"][0].tolist() == video_frames

    assert isinstance(rays, RayConditions)
    assert rays.frame_c2w[0, :, 0, 0].tolist() == in_window
    assert rays.frame_tans[0, :, 0].tolist() == [v + 1.0 for v in in_window]
    assert rays.anchor_c2w[0, 0, 0].item() == float(TARGET)  # the first target frame's camera


def test_on_the_memory_frames_only_the_clients_slot_is_the_source_players():
    """The client's slot carries the source player's states and controls; every other player
    repeats its first-frame state with zero controls, and nobody is labelled visible."""
    window, _ = gather_window(_round_batch(), WITH_MEMORY)
    memory = slice(1, 17)
    other = 1 - CLIENT
    assert window["player_states"][0, CLIENT, memory, 0].tolist() == MEMORY_VIDEO_FRAMES
    assert window["player_states"][0, other, memory, 0].tolist() == [0.0] * 16
    assert window["player_control_substeps"][0, CLIENT, memory, 0, 0].tolist() == (
        MEMORY_VIDEO_FRAMES
    )
    assert not bool(window["player_control_substeps"][0, other, memory].any())
    assert not bool(window["player_control_substep_valid"][0, other, memory].any())
    for key in ("client_visibility", "client_visibility_valid"):
        assert not bool(window[key][0, :, memory].any())
        assert bool(window[key][0, :, 17:].all()) and bool(window[key][0, :, 0].all())
    assert window["player_states"][0, other, :, 0].tolist() == OWN_VIDEO_FRAMES[:1] + [0.0] * 16 + (
        OWN_VIDEO_FRAMES[1:]
    )


def test_the_continuous_columns_are_gathered_with_the_window():
    """The round's columns by latent frame, the memory frames' between them, under the key of a
    gathered window."""
    batch = _round_batch()
    round_columns = W.ROUND_CONTINUOUS_COLUMNS_KEY
    window, _ = gather_window({**batch, round_columns: batch[round_columns].double()}, WITH_MEMORY)
    columns = window[W.CONTINUOUS_COLUMNS_KEY]
    assert round_columns not in window
    assert columns.shape == (1, 21, PLAYERS, 7) and columns.dtype == torch.float32
    in_window = OWN_LATENTS[:1] + MEMORY_LATENTS + OWN_LATENTS[1:]
    assert all(columns[0, :, player, 3].tolist() == in_window for player in range(PLAYERS))
    memory_columns = "memory_frames_continuous_columns"
    without_memory_columns = {k: v for k, v in batch.items() if k != memory_columns}
    with pytest.raises(KeyError, match=memory_columns):
        gather_window(without_memory_columns, WITH_MEMORY)
    without_columns = {k: v for k, v in without_memory_columns.items() if k != round_columns}
    window, _ = gather_window(without_columns, WITH_MEMORY)
    assert W.CONTINUOUS_COLUMNS_KEY not in window


def test_gather_without_a_memory_entry():
    batch = {k: v for k, v in _round_batch().items() if not k.startswith("memory_frames_")}
    window, rays = gather_window(batch, WITHOUT_MEMORY)
    assert window["latents"].flatten().tolist() == OWN_LATENTS
    assert window["view_deltas"][0, :, 1].tolist() == OWN_VIDEO_FRAMES
    assert window["player_states"][0, :, :, 0].tolist() == [OWN_VIDEO_FRAMES] * PLAYERS
    assert window[W.CONTINUOUS_COLUMNS_KEY][0, :, 0, 3].tolist() == OWN_LATENTS
    assert rays.frame_c2w[0, :, 0, 0].tolist() == OWN_LATENTS
    assert rays.anchor_c2w[0, 0, 0].item() == float(TARGET)  # also without memory frames


def test_gather_passes_the_rounds_entries_through_and_drops_its_own_inputs():
    batch = _round_batch()
    window, _ = gather_window(batch, WITH_MEMORY)
    for key in ("client_slot", "player_team_ids", "media_id"):
        assert window[key] is batch[key]
    assert not any(key.startswith(("window_", "memory_frames_")) for key in window)


def test_gather_refuses_what_it_cannot_place():
    batch = _round_batch()
    with pytest.raises(KeyError, match="does not know batch key 'depth'"):
        gather_window({**batch, "depth": torch.zeros(1, FRAMES)}, WITH_MEMORY)
    with pytest.raises(KeyError, match="needs a tensor batch\\['view_deltas'\\]"):
        gather_window({k: v for k, v in batch.items() if k != "view_deltas"}, WITH_MEMORY)
    with pytest.raises(ValueError, match=r"buttons must start with \[1, 113\]"):
        gather_window({**batch, "buttons": batch["buttons"][:, :-4]}, WITH_MEMORY)
    one_player = {**batch, "client_visibility": batch["client_visibility"][:, :1]}
    with pytest.raises(ValueError, match=r"client_visibility must start with \[1, 2, 113\]"):
        gather_window(one_player, WITH_MEMORY)
    with pytest.raises(ValueError, match=r"client_slot \[2\] is outside the 2 players"):
        gather_window({**batch, "client_slot": torch.tensor([2])}, WITH_MEMORY)
    with pytest.raises(KeyError, match="memory_frames_c2w"):
        gather_window({k: v for k, v in batch.items() if k != "memory_frames_c2w"}, WITH_MEMORY)
    window, _ = gather_window(batch, WITH_MEMORY)
    with pytest.raises(ValueError, match="it is a window"):
        gather_window(window, WITH_MEMORY)
    risen = {**batch, "player_states": batch["player_states"].clone()}
    risen["player_states"][0, 0, :60, 5] = 0.0  # dead, then alive
    with pytest.raises(RuntimeError, match="resurrection"):
        gather_window(risen, WITH_MEMORY)
    bad_tans = {**batch, "window_tans": torch.zeros(1, FRAMES, 2)}
    with pytest.raises(ValueError, match="finite and positive"):
        gather_window(bad_tans, WITH_MEMORY)
    with pytest.raises(ValueError, match="ends past the round's 29 latent frames"):
        gather_window({**batch, "window_target_start": torch.tensor([26])}, WITH_MEMORY)


def test_gather_refuses_a_memory_input_of_another_shape():
    """A memory input has its entry's shape with the memory frames in place of its frames; a
    source player's states of another width would otherwise broadcast into the client's slot."""
    batch = _round_batch()
    narrow = {**batch, "memory_frames_states": batch["memory_frames_states"][..., :1]}
    with pytest.raises(ValueError, match=r"memory_frames_states must be \[1, 16, 6\]"):
        gather_window(narrow, WITH_MEMORY)
    short = {**batch, "memory_frames_latents": batch["memory_frames_latents"][:, :3]}
    with pytest.raises(ValueError, match=r"memory_frames_latents must be \[1, 4, 1, 1, 1\]"):
        gather_window(short, WITH_MEMORY)
    per_player = batch["memory_frames_weapon_ids"][:, None].expand(1, PLAYERS, 16)
    with pytest.raises(ValueError, match=r"memory_frames_weapon_ids must be \[1, 16\]"):
        gather_window({**batch, "memory_frames_weapon_ids": per_player}, WITH_MEMORY)
    without_memory = {k: v for k, v in narrow.items() if k != "memory_frames_c2w"}
    gather_window(without_memory, WITHOUT_MEMORY)  # the memory inputs are not read
