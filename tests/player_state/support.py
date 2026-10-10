"""Synthetic rounds for the player-state tests: a seeded round of ten players, a round of players
that stand where a test puts them, and the fitted physics prior."""

from pathlib import Path

import numpy as np
import torch

from worldcast.data.latents import video_frame_count
from worldcast.player_state.extrapolation import PhysicsPrior

REPO = Path(__file__).resolve().parents[2]
PRIOR = PhysicsPrior.load(REPO / "configs" / "state_model" / "physics_prior.json")
#: Ordered substeps per video frame, button channels, and the columns of a substep.
SUBSTEPS, BUTTONS = 4, 11
SUBSTEP_WIDTH = BUTTONS + 2


def _player_track(g: torch.Generator, frames: int, client: dict | None) -> dict:
    """One player's states over ``frames`` video frames: the client's own random walk, or a
    player placed around the client ``client`` (mostly in front of it, at 50-1500 u)."""
    if client is None:
        xy = torch.cumsum(torch.randn(frames, 2, generator=g) * 6.0, dim=0)
        xy = xy + torch.randn(1, 2, generator=g) * 100.0
        yaw = torch.rand(1, generator=g) * 360.0 - 180.0
        yaw = yaw + torch.cumsum(torch.randn(frames, generator=g) * 2.0, dim=0)
        pitch = torch.rand(1, generator=g) * 30.0 - 15.0
        pitch = pitch + torch.cumsum(torch.randn(frames, generator=g) * 0.5, dim=0)
        return dict(xy=xy, z=torch.zeros(frames), yaw=yaw, pitch=pitch)
    in_front = torch.rand(1, generator=g) < 0.8
    spread = torch.randn(1, generator=g) * (35.0 if in_front else 30.0)
    bearing = torch.deg2rad(client["yaw"] + spread + (0.0 if in_front else 180.0))
    distance = 50.0 + torch.rand(1, generator=g) * 1450.0
    distance = distance + torch.cumsum(torch.randn(frames, generator=g) * 3.0, dim=0)
    direction = torch.stack([torch.cos(bearing), torch.sin(bearing)], dim=-1)
    z = torch.randn(1, generator=g) * 30.0 + torch.cumsum(torch.randn(frames, generator=g), dim=0)
    yaw = torch.rand(1, generator=g) * 360.0 - 180.0
    yaw = yaw + torch.cumsum(torch.randn(frames, generator=g) * 3.0, dim=0)
    pitch = torch.randn(1, generator=g) * 10.0
    pitch = pitch + torch.cumsum(torch.randn(frames, generator=g) * 0.5, dim=0)
    return dict(xy=client["xy"] + direction * distance[:, None], z=z, yaw=yaw, pitch=pitch)


def _controls(g: torch.Generator, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One player's random ordered substeps ``[T, 4, 13]`` and their valid flags ``[T, 4]``."""
    substeps = torch.zeros(frames, SUBSTEPS, SUBSTEP_WIDTH)
    substeps[..., :BUTTONS] = (torch.rand(frames, SUBSTEPS, BUTTONS, generator=g) < 0.3).float()
    substeps[..., BUTTONS] = torch.randn(frames, SUBSTEPS, generator=g) * 0.6  # pitch turn / 5
    substeps[..., BUTTONS + 1] = torch.randn(frames, SUBSTEPS, generator=g) * 0.8  # yaw turn / 5
    return substeps, torch.rand(frames, SUBSTEPS, generator=g) < 0.9


def round_batch(
    seed: int,
    *,
    batch: int = 2,
    players: int = 10,
    latents: int = 9,
    client_slots: tuple[int, ...] = (3, 7),
    pitch_drift: bool = True,
) -> dict[str, torch.Tensor]:
    """A seeded batch of one client's round per sample, at video-frame rate, as the loader gives
    it.

    Some players die mid-window, one slot per sample is dead from the first frame and one has no
    recording (team 0, all zeros). With ``pitch_drift`` one player's pitch turns are biased so that
    the pitch clamp is reached.
    """
    g = torch.Generator().manual_seed(seed)
    frames = video_frame_count(latents)
    team_ids = torch.tensor(
        [[2] * (players // 2) + [3] * (players - players // 2)] * batch, dtype=torch.long
    )
    client_slot = torch.tensor(list(client_slots)[:batch], dtype=torch.long)
    states = torch.zeros(batch, players, frames, 6)
    substeps = torch.zeros(batch, players, frames, SUBSTEPS, SUBSTEP_WIDTH)
    valid = torch.zeros(batch, players, frames, SUBSTEPS, dtype=torch.bool)
    for b in range(batch):
        client = int(client_slot[b])
        absent, dead = (client + 5) % players, (client + 2) % players
        team_ids[b, absent] = 0
        client_track = _player_track(g, frames, None)
        for p in range(players):
            if p == absent:
                continue
            track = client_track if p == client else _player_track(g, frames, client_track)
            alive = torch.ones(frames)
            if p == dead:
                alive[:] = 0.0
            elif p != client and torch.rand(1, generator=g) < 0.5:
                alive[int(torch.rand(1, generator=g) * (frames - 2)) + 1 :] = 0.0
            states[b, p, :, 0:2], states[b, p, :, 2] = track["xy"], track["z"]
            states[b, p, :, 3], states[b, p, :, 4] = track["yaw"], track["pitch"]
            states[b, p, :, 5] = alive
            substeps[b, p], valid[b, p] = _controls(g, frames)
        drift = (client + 1) % players
        if pitch_drift and drift != absent:
            substeps[b, drift, :, :, BUTTONS] += 1.5  # about +30 degrees per frame
    label_valid = torch.rand(batch, players, frames, generator=g) < 0.85
    return {
        "player_states": states,
        "player_control_substeps": substeps,
        "player_control_substep_valid": valid,
        "player_team_ids": team_ids,
        "player_weapon_ids": (torch.rand(batch, players, frames, generator=g) * 52).long(),
        "client_slot": client_slot,
        "client_visibility": (torch.rand(batch, players, frames, generator=g) < 0.2) & label_valid,
        "client_visibility_valid": label_valid,
        "obs_flash_flag": (torch.rand(batch, latents, generator=g) < 0.2).long(),
        "obs_flash_valid": (torch.rand(batch, latents, generator=g) < 0.9).long(),
        "obs_scope_on": (torch.rand(batch, latents, generator=g) < 0.3).long(),
        "obs_scope_level": (torch.rand(batch, latents, generator=g) * 3).long().clamp(max=2),
        "obs_scope_valid": (torch.rand(batch, latents, generator=g) < 0.9).long(),
    }


def standing_round(
    positions: list[list[float]], *, latents: int = 9, client_slot: int = 0, yaw: float = 0.0
) -> dict[str, torch.Tensor]:
    """One round (batch size 1) of living players that stand at ``positions`` (feet, u) looking
    along ``yaw`` degrees, with no control held and every substep valid; no label is known."""
    frames, players = video_frame_count(latents), len(positions)
    states = torch.zeros(1, players, frames, 6)
    states[0, :, :, :3] = torch.tensor(positions)[:, None, :]
    states[..., 3], states[..., 5] = yaw, 1.0
    return {
        "player_states": states,
        "player_control_substeps": torch.zeros(1, players, frames, SUBSTEPS, SUBSTEP_WIDTH),
        "player_control_substep_valid": torch.ones(1, players, frames, SUBSTEPS, dtype=torch.bool),
        "player_team_ids": torch.full((1, players), 2),
        "player_weapon_ids": torch.zeros(1, players, frames, dtype=torch.long),
        "client_slot": torch.tensor([client_slot]),
        "client_visibility": torch.zeros(1, players, frames),
        "client_visibility_valid": torch.zeros(1, players, frames, dtype=torch.bool),
    }


def flat_depth(depth_u: float):
    """A depth function that sees a surface ``depth_u`` in front of every latent frame."""
    return lambda latents: np.full((int(latents.shape[0]), 4, 24, 42), np.log(depth_u), np.float32)
