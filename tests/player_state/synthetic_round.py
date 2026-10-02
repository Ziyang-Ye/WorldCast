"""A synthetic round batch (players, controls, visibility) and its compacted window."""

import torch

QUARTER = 4  # pixel rows per latent


# ----------------------------------------------------------------------------------- synthetic data
def _pixel_rows(latent: int) -> list:
    return [0] if latent == 0 else list(range(QUARTER * latent - 3, QUARTER * latent + 1))


def make_round_batch(
    seed: int,
    *,
    batch: int = 2,
    players: int = 10,
    latents: int = 9,
    observer_slots=(3, 7),
    pitch_drift: bool = True,
) -> dict:
    """A seeded single-POV loader batch (pixel rate) shaped like ``OpenCS2SinglePOVDataset`` output.

    Peers stand mostly in front of the observer at 50-1500 u (some behind, some off-frustum); some
    die mid-window, one seat per sample is dead from the first row and one has no media (team 0, all
    zeros). One player's pitch deltas are biased so the pitch clamp is exercised.
    """
    g = torch.Generator().manual_seed(seed)
    T = 1 + QUARTER * (latents - 1)
    Q, buttons = 4, 11
    A = buttons + 2

    def rand(*shape):
        return torch.rand(*shape, generator=g)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    team_ids = torch.tensor(
        [[2] * (players // 2) + [3] * (players - players // 2)] * batch, dtype=torch.long
    )
    observer_slot = torch.tensor(list(observer_slots)[:batch], dtype=torch.long)

    states = torch.zeros(batch, players, T, 6)
    substeps = torch.zeros(batch, players, T, Q, A)
    valid = torch.zeros(batch, players, T, Q, dtype=torch.bool)
    for b in range(batch):
        obs = int(observer_slot[b])
        absent = (obs + 5) % players  # a seat without media
        pre_dead = (obs + 2) % players  # dead from the first row
        team_ids[b, absent] = 0
        # Observer track.
        obs_xy = torch.cumsum(randn(T, 2) * 6.0, dim=0) + randn(1, 2) * 100.0
        obs_yaw = (rand(1) * 360.0 - 180.0) + torch.cumsum(randn(T) * 2.0, dim=0)
        obs_pitch = (rand(1) * 30.0 - 15.0) + torch.cumsum(randn(T) * 0.5, dim=0)
        for p in range(players):
            if p == absent:
                continue
            if p == obs:
                xy, z, yaw, pitch = obs_xy, torch.zeros(T), obs_yaw, obs_pitch
            else:
                bearing = obs_yaw + (randn(1) * 35.0 if rand(1) < 0.8 else 180.0 + randn(1) * 30.0)
                dist = 50.0 + rand(1) * 1450.0 + torch.cumsum(randn(T) * 3.0, dim=0)
                rad = torch.deg2rad(bearing)
                xy = obs_xy + torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1) * dist[:, None]
                z = randn(1) * 30.0 + torch.cumsum(randn(T), dim=0)
                yaw = rand(1) * 360.0 - 180.0 + torch.cumsum(randn(T) * 3.0, dim=0)
                pitch = randn(1) * 10.0 + torch.cumsum(randn(T) * 0.5, dim=0)
            alive = torch.ones(T)
            if p == pre_dead and p != obs:
                alive[:] = 0.0
            elif p != obs and rand(1) < 0.5:
                death_row = int(rand(1) * (T - 2)) + 1
                alive[death_row:] = 0.0
            states[b, p, :, 0:2] = xy
            states[b, p, :, 2] = z
            states[b, p, :, 3] = yaw
            states[b, p, :, 4] = pitch
            states[b, p, :, 5] = alive
            substeps[b, p, :, :, :buttons] = (rand(T, Q, buttons) < 0.3).float()
            substeps[b, p, :, :, buttons] = randn(T, Q) * 0.6  # pitch delta / 5
            substeps[b, p, :, :, buttons + 1] = randn(T, Q) * 0.8  # yaw delta / 5
            valid[b, p] = rand(T, Q) < 0.9
        if pitch_drift:
            drift = (obs + 1) % players
            if drift != absent:
                substeps[b, drift, :, :, buttons] += 1.5  # ~ +30 deg per row: saturates the clamp
    vis_valid = rand(batch, players, T) < 0.85
    visibility = (rand(batch, players, T) < 0.2) & vis_valid  # ~half the latents visible
    weapon_ids = (rand(batch, players, T) * 52).long().clamp(max=51)
    out = {
        "player_states": states,
        "player_action_substeps": substeps,
        "player_action_substep_valid": valid,
        "player_team_ids": team_ids,
        "player_weapon_ids": weapon_ids,
        "observer_slot": observer_slot,
        "observer_visibility": visibility,
        "observer_visibility_valid": vis_valid,
        "obs_flash_flag": (rand(batch, latents) < 0.2).long(),
        "obs_flash_valid": (rand(batch, latents) < 0.9).long(),
        "obs_scope_on": (rand(batch, latents) < 0.3).long(),
        "obs_scope_level": (rand(batch, latents) * 3).long().clamp(max=2),
        "obs_scope_valid": (rand(batch, latents) < 0.9).long(),
    }
    return out


def _slot_sources(round_batch: dict, slot_latents) -> list:
    """Per sample: the first seat (not the observer, with media) alive on every slot latent."""
    states = round_batch["player_states"]
    rows = [QUARTER * f for f in slot_latents]
    sources = []
    for b in range(states.shape[0]):
        obs = int(round_batch["observer_slot"][b])
        for p in range(states.shape[1]):
            if (
                p != obs
                and int(round_batch["player_team_ids"][b, p]) != 0
                and bool((states[b, p, rows, 5] > 0.5).all())
            ):
                sources.append(p)
                break
        else:
            raise ValueError("no alive slot source")
    return sources


def compact_batch(
    round_batch: dict,
    continuous_rows: torch.Tensor,
    *,
    target: int,
    recent: int = 12,
    slot_latents=None,
) -> dict:
    """Emulate the window gather (``worldcast.sampling.window``): sink | [slot] | recent | target,
    pixel and latent axes.

    Peer rows on slot latents repeat the sink row with zero actions and zero labels; the observer
    row there is the slot source's own recording, and its continuous columns are that source's
    engine angles, no corpse, own xyz (``worldplay_windowout.py:537-661``). Only used to build valid
    13-wide tables for the golden tests.
    """
    own = [0] + list(range(target - recent, target)) + list(range(target, target + 4))
    sources = None if slot_latents is None else _slot_sources(round_batch, slot_latents)
    pixel_keys = (
        "player_states",
        "player_action_substeps",
        "player_action_substep_valid",
        "player_weapon_ids",
        "observer_visibility",
        "observer_visibility_valid",
    )
    latent_keys = (
        "obs_flash_flag",
        "obs_flash_valid",
        "obs_scope_on",
        "obs_scope_level",
        "obs_scope_valid",
    )
    out = {k: v for k, v in round_batch.items() if k not in pixel_keys + latent_keys}
    batch, players = round_batch["player_states"].shape[:2]

    def gather_pixels(value, latents):
        rows = [r for f in latents for r in _pixel_rows(f)]
        return value.index_select(2, torch.as_tensor(rows, dtype=torch.long))

    pieces = {k: [gather_pixels(round_batch[k], own[:1])] for k in pixel_keys}
    lat_pieces = {k: [round_batch[k][:, own[:1]]] for k in latent_keys}
    cont_pieces = [continuous_rows[:, own[:1]]]
    if slot_latents is not None:
        slot_rows = [r for f in slot_latents for r in _pixel_rows(f)]
        n = len(slot_rows)
        for k in pixel_keys:
            ref = round_batch[k]
            if k == "player_states":
                piece = ref[:, :, 0:1].expand(batch, players, n, *ref.shape[3:]).clone()
            else:
                piece = torch.zeros((batch, players, n) + tuple(ref.shape[3:]), dtype=ref.dtype)
            if k not in ("observer_visibility", "observer_visibility_valid"):
                for b in range(batch):
                    obs = int(round_batch["observer_slot"][b])
                    piece[b, obs] = ref[b, sources[b]].index_select(0, torch.as_tensor(slot_rows))
            pieces[k].append(piece)
        for k in latent_keys:
            lat_pieces[k].append(round_batch[k][:, list(slot_latents)])
        slot_cols = (
            continuous_rows[:, 0:1].expand(batch, 4, players, continuous_rows.shape[-1]).clone()
        )
        for b in range(batch):
            obs = int(round_batch["observer_slot"][b])
            src = round_batch["player_states"][b, sources[b]].float()
            lat_rows = src[[QUARTER * f for f in slot_latents]]
            slot_cols[b, :, obs, 0] = lat_rows[:, 3]
            slot_cols[b, :, obs, 1] = lat_rows[:, 4]
            slot_cols[b, :, obs, 2] = 0.0
            slot_cols[b, :, obs, 3] = 0.0
            slot_cols[b, :, obs, 4:7] = lat_rows[:, :3]
        cont_pieces.append(slot_cols)
    for k in pixel_keys:
        pieces[k].append(gather_pixels(round_batch[k], own[1:]))
        out[k] = torch.cat(pieces[k], dim=2)
    for k in latent_keys:
        lat_pieces[k].append(round_batch[k][:, own[1:]])
        out[k] = torch.cat(lat_pieces[k], dim=1)
    cont_pieces.append(continuous_rows[:, own[1:]])
    out["peer_continuous_columns"] = torch.cat(cont_pieces, dim=1)
    return out
