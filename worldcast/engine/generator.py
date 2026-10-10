"""The generator as inference, training and evaluation load and call it: its dimensions from the
Wan2.2 snapshot, the fixed prompt's embedding, the Wan2.2 VAE and a window's inputs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldcast.data.camera import window_cameras
from worldcast.modeling.build import generator_config_from_snapshot
from worldcast.modeling.wan22.model import GeneratorConfig
from worldcast.modeling.wan22.vae import VAE_CHECKPOINT_NAME, Wan22VAE, load_wan22_vae
from worldcast.player_state import PlayerState, player_state_conditions, visible_latent_frames
from worldcast.sampling.conditions import control_conditions
from worldcast.utils.precision import generator_dtype

__all__ = [
    "WindowInputs",
    "backbone_config",
    "client_cameras",
    "load_prompt_embeds",
    "load_vae",
    "window_conditions",
    "window_inputs",
]


def backbone_config(wan22_root: str | Path | None) -> GeneratorConfig:
    """The generator's config with the backbone dimensions of the Wan2.2 snapshot's
    ``config.json`` under ``wan22_root``; without one, the 5B model's."""
    snapshot = Path(wan22_root or "") / "config.json"
    if wan22_root and snapshot.is_file():
        return generator_config_from_snapshot(snapshot)
    return GeneratorConfig()


def load_prompt_embeds(
    prompt_embedding: str | Path | None, wan22_root: str | Path | None, device: torch.device | str
) -> torch.Tensor:
    """The fixed prompt's umT5 embedding ``[1, 512, 4096]``: the file ``prompt_embedding``, else
    encoded with the umT5 of ``wan22_root``."""
    from worldcast.modeling.wan22.text_encoder import load_prompt_embedding, load_text_encoder

    if prompt_embedding:
        return load_prompt_embedding(prompt_embedding, device=device)
    if not wan22_root:
        raise ValueError("set the prompt embedding, or the Wan2.2 root to encode the prompt")
    return load_text_encoder(wan22_root, device=device).encode_prompt()


def load_vae(wan22_root: str | Path | None, device: torch.device | str) -> Wan22VAE:
    """The Wan2.2 VAE of the snapshot under ``wan22_root``, on ``device`` in the generator's dtype
    there (:func:`~worldcast.utils.precision.generator_dtype`)."""
    if not wan22_root:
        raise ValueError(f"set the Wan2.2 root: the VAE is its {VAE_CHECKPOINT_NAME}")
    return load_wan22_vae(
        Path(wan22_root) / VAE_CHECKPOINT_NAME, device=device, dtype=generator_dtype(device)
    )


def client_cameras(
    states: np.ndarray,
    signals: Mapping[str, Sequence[int]],
    weapon_ids: Sequence[int],
    latent_frames: int,
) -> dict[str, torch.Tensor]:
    """The client's recorded cameras over a window, as a batch holds them for the ray embedding.

    Args:
        states (np.ndarray): ``[T, 6]`` the client's player state per video frame.
        signals (Mapping[str, Sequence[int]]): its observer signals per latent frame.
        weapon_ids (Sequence[int]): its weapon id per video frame.
        latent_frames (int): latent frames ``F`` of the window.

    Returns:
        dict[str, Tensor]: ``window_c2w`` ``[1, F, 4, 4]`` float32 and ``window_tans``
        ``[1, F, 2]`` float32, ``(tan(hfov / 2), tan(vfov / 2))``.
    """
    c2w, tans = window_cameras(states, signals, weapon_ids, latent_frames)
    return {"window_c2w": c2w[None], "window_tans": torch.as_tensor(tans)[None]}


@dataclass
class WindowInputs:
    """A window as the generator reads it.

    Attributes:
        latents (Tensor): ``[B, F, 48, 24, 42]`` the window's clean latents, in the model dtype.
        conditions (dict[str, Any]): the generator's conditions.
        player_state (PlayerState | None): every player's state at the window's latent frames
            (``None`` without the player state field, stage 1).
        visible (Tensor | None): ``[B, F, P]`` bool, the visibility gate of the conditions.
    """

    latents: torch.Tensor
    conditions: dict[str, Any]
    player_state: PlayerState | None = None
    visible: torch.Tensor | None = None


def window_inputs(
    batch: Mapping[str, Any],
    *,
    prompt_embeds: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    observer_signals: bool = True,
    rays: Mapping[str, torch.Tensor] | None = None,
) -> WindowInputs:
    """The latents and the generator's conditions of a window or a whole round.

    The conditions: the prompt, the client's controls, the player-state conditions gated by the
    batch's visibility labels (the GT labels of Table 3, or the labels a closed-loop client
    predicted), the observer signals and the ray embedding's cameras.

    Args:
        batch (Mapping[str, Any]): a collated batch (``latents``, the controls, ``player_*``,
            ``client_slot``, ``client_visibility`` / ``client_visibility_valid``, ``obs_*``).
        prompt_embeds (Tensor): ``[1, L, 4096]`` the fixed prompt's embedding.
        device (torch.device | str): where the generator runs.
        dtype (torch.dtype): the generator's dtype.
        observer_signals (bool): the generator reads the observer signals (with scene state).
        rays (Mapping[str, Tensor] | None): the ray embedding's conditions (with scene state).
    """
    latents = batch["latents"]
    if latents.ndim != 5:
        raise ValueError("latents must be one client's [B, F, C, H, W]")
    latents = latents.to(device=device, dtype=dtype, non_blocking=True).contiguous()
    conditions, player_state, visible = _conditions(
        batch,
        prompt_embeds=prompt_embeds,
        device=device,
        dtype=dtype,
        observer_signals=observer_signals,
        rays=rays,
    )
    return WindowInputs(latents, conditions, player_state, visible)


def window_conditions(
    batch: Mapping[str, Any],
    *,
    prompt_embeds: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    rays: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """The generator's conditions of a window or a whole round with the observer signals and the
    ray embedding's conditions ``rays``: those of :func:`window_inputs`, without reading the
    batch's latents."""
    conditions, _, _ = _conditions(
        batch,
        prompt_embeds=prompt_embeds,
        device=device,
        dtype=dtype,
        observer_signals=True,
        rays=rays,
    )
    return conditions


def _conditions(
    batch: Mapping[str, Any],
    *,
    prompt_embeds: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    observer_signals: bool,
    rays: Mapping[str, torch.Tensor] | None,
) -> tuple[dict[str, Any], PlayerState, torch.Tensor]:
    player_state = PlayerState.from_batch(batch, device=device)
    visible = visible_latent_frames(batch, device=device)
    conditions = {
        "prompt_embeds": prompt_embeds.expand(len(player_state.xyz), -1, -1),
        **control_conditions(batch, device=device, dtype=dtype),
        **player_state_conditions(
            player_state, visible, observer_signals=batch if observer_signals else None
        ),
        **(rays or {}),
    }
    return conditions, player_state, visible
