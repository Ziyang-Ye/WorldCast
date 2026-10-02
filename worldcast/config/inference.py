"""Inference configuration: one dataclass tree whose defaults are the paper's deployed client.

Every default below is the value the Table-3 WorldCast run used: the deployed run's inference config merged over
``Wan22/configs/default_config.yaml`` of the paper's research code, plus the command line of its ``memory_deploy.py``
(docs/inference.md, "Configuration"). Each field's comment says what it is and which old key or flag it replaces.
Paths of artefacts have no default (no machine-specific path lives in code); they come from the YAML or the CLI.

Things that were side effects or environment variables in the research code are fields here:

* ``data.camera_encoding = "noclip"`` (was ``WC_CAMERA_ENCODING``, whose code default ``clip`` is wrong for these
  weights);
* ``sampler.model_input_dtype = "bfloat16"`` (was the FSDP root's input cast, timesteps included);
* ``sampler.tf32 = True`` (was set inside ``Trainer.__init__``);
* ``run.seed`` (was ``--seed``, used for the entry noise and the RNG of the ladder re-noise).

Load with :func:`load_config`; every config is validated when it is built (unknown keys, wrong types, the
context-noise band check of the old ``resolve_band``, the ladder check of ``configured_rungs`` and the window
arithmetic of ``resolve_window``), so an invalid config cannot exist. Nothing here imports torch.
"""

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

__all__ = [
    "SchedulerConfig",
    "SamplerConfig",
    "WindowConfig",
    "ModelConfig",
    "FieldConfig",
    "DataConfig",
    "MemoryConfig",
    "PlayerStateConfig",
    "PoolConfig",
    "PathsConfig",
    "RunConfig",
    "InferenceConfig",
    "check_context_noise",
    "check_ladder",
    "first_target_latent",
    "load_config",
    "config_from_dict",
    "config_to_dict",
    "with_overrides",
]

# ----------------------------------------------------------------------------------------------- constants
#: The 11 OpenCS2 buttons, in the order of the dataset and of the AdaLN control input (deploy yaml ``action_buttons``).
ACTION_BUTTONS = (
    "forward",
    "back",
    "move_left",
    "move_right",
    "jump",
    "duck",
    "speed",
    "attack",
    "attack2",
    "reload",
    "look_at_weapon",
)
#: The 9 control fractions written into the player state field (``peer_raster_action_signals``).
FIELD_ACTION_SIGNALS = (
    "forward",
    "back",
    "move_left",
    "move_right",
    "jump",
    "duck",
    "speed",
    "attack",
    "reload",
)
#: Extra inputs the generator accepts beside text and controls (``model_kwargs.condition_passthrough_keys``).
CONDITION_PASSTHROUGH_KEYS = (
    "peer_states",
    "peer_actions",
    "peer_observer_slot",
    "peer_team_ids",
    "peer_alive",
    "peer_visible",
    "peer_weapons",
    "state_wp_memory_c2w",
    "state_wp_anchor_c2w",
    "state_wp_memory_frames",
    "state_wp_frame_c2w",
    "state_wp_frame_tans",
    "obs_flash_flag",
    "obs_flash_valid",
    "obs_scope_on",
    "obs_scope_level",
    "obs_scope_valid",
)
CAMERA_ENCODINGS = (
    "noclip",
)  # only the paper's encoding is ported (``clip`` is the old code default)
MODEL_INPUT_DTYPES = (
    "bfloat16",
    "float32",
)  # float32 exists for CPU tests only; the paper runs bfloat16


# ----------------------------------------------------------------------------------------------- checks
def check_context_noise(context_noise: int, band: tuple[int, int]) -> int:
    """Validate the deployment context-noise timestep against the trained band; return it as an int.

    Port of ``wan_utils/worldplay_context_noise.resolve_band`` (WorldPlay-named declaration only; the legacy
    ``noise_augmentation_max_timestep`` spelling is not part of the release config). ``band = (min, max)`` are
    timesteps of the training draw (``memory_worldplay_context_noise_{min,max}_timestep``); the deployment value
    must lie in ``[max(min, 1), max]``. A band of ``(0, 0)`` (off) is refused: the paper student was trained inside
    a band, and a clean (t = 0) context write is not a release path.
    """
    lo, hi = (_nonneg_int(v, "context_noise_band") for v in band)
    if hi == 0:
        raise ValueError(
            "context_noise_band max is 0 (no trained band): the 4-step student was trained with "
            "context noise in [16, 32] and deploys at 16"
        )
    if lo >= hi:
        raise ValueError(f"context_noise_band min {lo} must lie below its max {hi}")
    value = _nonneg_int(context_noise, "context_noise")
    if not (max(lo, 1) <= value <= hi):
        raise ValueError(
            f"context_noise={value} must lie in [{max(lo, 1)}, {hi}]: deploy inside the trained"
            " band (or at its floor), never outside it"
        )
    return value


def check_ladder(rungs) -> tuple[int, ...]:
    """Validate the few-step ladder (port of ``wan_utils/fewstep_ladder.configured_rungs``).

    Rungs are integer timesteps in (0, 1000], strictly descending from noisy to clean, with no repeat.
    """
    out = tuple(int(v) for v in rungs)
    if not out:
        raise ValueError("the few-step ladder is empty")
    if any(v <= 0 or v > 1000 for v in out):
        raise ValueError(f"ladder rungs must lie in (0, 1000]: {list(out)}")
    if list(out) != sorted(out, reverse=True):
        raise ValueError(f"the ladder must descend from noisy to clean: {list(out)}")
    if len(set(out)) != len(out):
        raise ValueError(f"the ladder repeats a rung: {list(out)}")
    return out


def first_target_latent(min_target_latent: int, block: int = 4) -> int:
    """The first reconstituted block ``s``: the first ``1 + block * k >= min_target_latent`` (24 -> 25)."""
    s = 1
    while s < int(min_target_latent):
        s += int(block)
    return s


def _nonneg_int(value, name) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer timestep, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return int(value)


def _only(value, allowed, name):
    if value not in allowed:
        raise NotImplementedError(
            f"{name}={value!r}: only {allowed} (the paper path) is ported; the other values were"
            " ablations or experiments of the research code"
        )


# ----------------------------------------------------------------------------------------------- sections
@dataclass(frozen=True)
class SchedulerConfig:
    """The flow-matching table shared by the ladder, the context write and the flow -> x0 conversion."""

    num_train_timesteps: int = 1000  # num_train_timestep: entries of the timestep / sigma table
    shift: float = (
        5.0  # timestep_shift: sigma' = 5 s / (1 + 4 s); Wan2.2 requires 5.0 (wan_wrapper.py:288-292)
    )
    sigma_min: float = 0.0  # FlowMatchScheduler(sigma_min=0.0) built in wan_wrapper.py:488-491
    sigma_max: float = 1.0  # FlowMatchScheduler default sigma_max (scheduler.py:108)
    extra_one_step: bool = (
        True  # FlowMatchScheduler(extra_one_step=True): linspace over 1001 points, last dropped
    )

    def __post_init__(self):
        if self.num_train_timesteps != 1000:
            raise NotImplementedError(
                "the ladder warp indexes a 1000-entry table (fewstep_ladder.py:79-80)"
            )
        if self.shift != 5.0:
            raise ValueError("Wan2.2 requires timestep_shift = 5.0")
        if not self.extra_one_step or self.sigma_min != 0.0 or self.sigma_max != 1.0:
            raise NotImplementedError(
                "only the deployed table (sigma 1 -> 0, extra_one_step) is ported"
            )


@dataclass(frozen=True)
class SamplerConfig:
    """The 4-step ladder, the context noise and the numerics of every generator call."""

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ladder: tuple[int, ...] = (
        1000,
        750,
        500,
        250,
    )  # fewstep_denoising_step_list (= --expect-fewstep-ladder 1000,750,500,250)
    warp_ladder: bool = (
        True  # warp_denoising_step: rung r -> table[1000 - r] = 1000 / 937.5 / 833.3 / 625
    )
    context_noise: int = (
        16  # context_noise = memory_worldplay_infer_context_noise = --context-noise 16 (label; level snaps to t ~ 14.82)
    )
    context_noise_band: tuple[int, int] = (
        16,
        32,
    )  # memory_worldplay_context_noise_{min,max}_timestep: the trained band
    guidance_scale: float = (
        1.0  # guidance_scale: classifier-free guidance off (the student has no CFG branch)
    )
    model_input_dtype: (
        str
    ) = (  # FSDP root MixedPrecision(param_dtype=bf16): every float input, timesteps included, reaches the model in bf16
        "bfloat16"
    )
    tf32: bool = (
        True  # torch.backends.cuda.matmul / cudnn allow_tf32 (opencs2_bidirectional_diffusion.py:867-868)
    )
    framewise_condition_axes: dict[str, int] = field(
        default_factory=lambda: {"peer_alive": 1, "peer_visible": 1}
    )  # framewise_condition_axes: sliced per block on this axis

    def __post_init__(self):
        object.__setattr__(self, "ladder", check_ladder(self.ladder))
        if not self.warp_ladder:
            raise NotImplementedError(
                "warp_denoising_step: true is the distilled ladder; the unwarped one is not ported"
            )
        check_context_noise(self.context_noise, self.context_noise_band)
        if self.guidance_scale != 1.0:
            raise NotImplementedError(
                "the 4-step student has no CFG branch (guidance_scale must be 1.0)"
            )
        if self.model_input_dtype not in MODEL_INPUT_DTYPES:
            raise ValueError(f"model_input_dtype must be one of {MODEL_INPUT_DTYPES}")
        for key, axis in self.framewise_condition_axes.items():
            if isinstance(axis, bool) or not isinstance(axis, int) or axis < 1:
                raise ValueError(
                    f"framewise axis of {key!r} must be an integer >= 1 (axis 0 is the batch)"
                )


@dataclass(frozen=True)
class WindowConfig:
    """The compacted window sink | memory slot | recent | target and the plain prefix before it."""

    block: int = 4  # num_frame_per_block: latents per generated block
    recent: int = 12  # memory_worldplay_recent (--recent): own latents s-12 .. s-1 in the window
    min_target_latent: int = (
        24  # memory_worldplay_min_target_latent (--min-target-latent): first reconstituted block s = 25
    )
    memory_slot: int = 4  # one retrieved entry = one complete 4-latent block, window positions 1-4
    sink: (
        str
    ) = (  # --sink-mode episode_first: position 0 is the recorded first latent of the round, forever
        "episode_first"
    )
    independent_first_frame: bool = (
        True  # independent_first_frame: latent 0 is committed alone before the 4-latent blocks
    )
    max_prefix_latents: int = (
        41  # image_or_video_shape[1]: the plain prefix is one call of at most 41 latents
    )
    kv_cache_latents: int = (
        41  # --local-attn 41: KV cache capacity in latents (41 x 252 tokens); no eviction while <= 41 are cached
    )
    kv_sink_latents: int = (
        0  # --sink-size 0: attention-sink latents of the rolling cache (inert here)
    )

    def __post_init__(self):
        if self.block != 4:
            raise NotImplementedError("the student was trained on 4-latent blocks")
        if self.recent <= 0 or self.recent % self.block:
            raise ValueError(f"recent={self.recent} is not a positive multiple of {self.block}")
        if self.memory_slot != self.block:
            raise NotImplementedError(
                "the deployed memory slot is exactly one complete block (k = 1)"
            )
        _only(self.sink, ("episode_first",), "window.sink")
        if not self.independent_first_frame:
            raise NotImplementedError("the paper commits latent 0 alone (independent_first_frame)")
        first = self.first_target
        if first - self.recent < 1:
            raise ValueError(
                f"recent={self.recent} with min_target_latent={self.min_target_latent}: the first "
                f"block s={first} leaves no latent after the sink (s - recent < 1)"
            )
        if first > self.max_prefix_latents:
            raise ValueError(
                f"first reconstituted block s={first} exceeds"
                f" max_prefix_latents={self.max_prefix_latents}"
            )
        if self.kv_sink_latents != 0:
            raise NotImplementedError("the paper runs the KV cache with sink size 0")
        if self.kv_cache_latents < self.max_cached_latents:
            raise ValueError(
                f"kv_cache_latents={self.kv_cache_latents} < {self.max_cached_latents} latents the "
                "client caches at once (the plain prefix)"
            )

    @property
    def first_target(self) -> int:
        """First reconstituted block s (25 for the paper)."""
        return first_target_latent(self.min_target_latent, self.block)

    @property
    def plain_prefix_latents(self) -> int:
        """Latents 1 .. s-1 generated by the plain prefix (24 for the paper: six blocks)."""
        n = self.first_target - 1
        return n - n % self.block

    @property
    def window_latents(self) -> int:
        """Compacted window with an installed memory slot: 1 + 4 + 12 + 4 = 21."""
        return 1 + self.memory_slot + self.recent + self.block

    @property
    def window_latents_without_memory(self) -> int:
        """Compacted window when nothing is retrieved: 1 + 12 + 4 = 17."""
        return 1 + self.recent + self.block

    @property
    def max_cached_latents(self) -> int:
        """Most latents in the KV cache at once: the prefix's 1 + 24 = 25 (a window caches at most 21)."""
        return max(1 + self.plain_prefix_latents, self.window_latents)


@dataclass(frozen=True)
class FieldConfig:
    """The player state field F (paper eq. (2)): 23 channels on the 12 x 21 token grid, injected after DiT block 2."""

    inject_after_block: int = (
        1  # peer_raster_block: added after DiT block index 1 (the second of 30)
    )
    stem_hidden: int = (
        32  # peer_raster_hidden: Conv2d(23 -> 32) -> SiLU -> Conv2d(32 -> 3072), zero-init
    )
    stem_depth: int = 1  # peer_raster_stem_depth
    splat_kernel: str = (
        "ati"  # peer_raster_splat_kernel: Gaussian splat at the feet raised by the body radius
    )
    splat_topk: int = 2  # peer_raster_splat_topk: top-2 players per token by w * exp(-scaled depth)
    splat_sigma_floor: float = 0.5  # peer_raster_splat_sigma_floor: sigma = half a token
    splat_temperature: float = (
        220.0  # peer_raster_splat_temperature: inert, capped to 72 on 12 x 21 (= sigma 0.5 token)
    )
    visibility_confidence: bool = (
        True  # peer_raster_visibility_confidence: per-player confidence e_p
    )
    confidence_floor: float = 0.3  # peer_raster_confidence_floor: e_p = 0.3 + 0.7 * ema
    confidence_smooth: float = (
        0.5  # peer_raster_confidence_smooth: causal EMA alpha, reset each block
    )
    death_channels: bool = True  # peer_raster_death_channels: dying and corpse planes
    id_bands: bool = True  # peer_raster_id_bands: live and corpse identity bands
    weapon_channels: int = 4  # peer_raster_weapon_channels: nn.Embedding(52, 4)
    action_signals: tuple[str, ...] = (
        FIELD_ACTION_SIGNALS  # peer_raster_action_signals: 9 control fractions
    )
    duck_aware: bool = True  # peer_raster_duck_aware: duck-aware body height
    camera_delta_scale: float = 5.0  # peer_camera_delta_scale: substep camera delta = raw / 5.0
    action_substeps: int = 4  # peer_action_substeps: 4 ordered substeps per pixel frame
    continuous_camera: bool = (
        True  # memory_worldplay_continuous_camera: observer camera and death state integrated on the whole round
    )
    visibility_source: str = (
        "labels"  # GT visibility labels gate the field (Table 3); predicted visibility = closed loop
    )

    def __post_init__(self):
        _only(self.visibility_source, ("labels",), "field.visibility_source")
        if not self.continuous_camera:
            raise NotImplementedError(
                "the weights were trained with the continuous observer camera"
            )


@dataclass(frozen=True)
class ModelConfig:
    """The generator: Wan2.2-TI2V-5B causal DiT plus the WorldCast inputs."""

    name: str = "Wan2.2-TI2V-5B"  # model_kwargs.model_name: base snapshot (revision 921dbaf3...)
    latent_channels: int = 48  # image_or_video_shape[2]: Wan2.2 VAE latent channels
    latent_height: int = 24  # image_or_video_shape[3]: 384 px / 16
    latent_width: int = 42  # image_or_video_shape[4]: 672 px / 16
    patch_size: tuple[int, int, int] = (
        1,
        2,
        2,
    )  # DiT patch (t, h, w) of the snapshot config.json: 12 x 21 = 252 tokens per latent
    vae_time_compression: int = (
        4  # model_kwargs.vae_time_compression_ratio: pixel rows per latent (1 + 4(F-1) layout)
    )
    vae_spatial_compression: int = 16  # model_kwargs.vae_spatial_compression_ratio
    action_button_dim: int = 11  # model_kwargs.action_button_dim
    action_camera_dim: int = 2  # model_kwargs.action_camera_dim
    action_weapon_vocab_size: int = 52  # model_kwargs.action_weapon_vocab_size ("fine" weapon ids)
    action_weapon_embedding_dim: int = 32  # model_kwargs.action_weapon_embedding_dim
    action_history_frames: int = (
        20  # model_kwargs.action_history_frames: 20 pixel rows of controls per latent
    )
    action_hidden_dim: int = 1024  # model_kwargs.action_hidden_dim
    action_adaln_rank: int = 128  # model_kwargs.action_adaln_rank: per-block low-rank AdaLN adapter
    condition_passthrough_keys: tuple[str, ...] = (
        CONDITION_PASSTHROUGH_KEYS  # model_kwargs.condition_passthrough_keys
    )
    obs_signal_hidden: int = (
        128  # obs_signal.hidden: flash/scope branch (not described in the paper)
    )
    memory_form: str = (
        "input_tokens"  # memory_worldplay_form: retrieved latents are 4 latents of the window
    )
    ray_form: str = "plucker_dm"  # memory_worldplay_ray_form: r = (d, (o / 420) x d)
    ray_unit_u: float = (
        420.0  # memory_worldplay_ray_unit_u: length unit of the ray moment (world units)
    )
    rays_all_frames: bool = (
        True  # memory_worldplay_sym_rays: rays added to every latent of the window
    )
    camera_tans: bool = True  # memory_worldplay_camera_tans: per-frame field of view in the rays
    visibility_head_block: int = (
        20  # visibility_head_block: module kept only for the strict load, never run
    )
    visibility_head_hidden: int = 256  # visibility_head_hidden
    fixed_prompt: str = (
        "first-person Counter-Strike 2 gameplay"  # fixed_prompt (prompt_mode: fixed)
    )
    player_field: FieldConfig = field(
        default_factory=FieldConfig
    )  # the player state field F and its injector

    def __post_init__(self):
        _only(self.memory_form, ("input_tokens",), "model.memory_form")
        _only(self.ray_form, ("plucker_dm",), "model.ray_form")
        if self.latent_height % self.patch_size[1] or self.latent_width % self.patch_size[2]:
            raise ValueError("latent spatial dims are not divisible by the DiT patch size")

    @property
    def frame_seq_length(self) -> int:
        """Tokens per latent frame: (24 / 2) x (42 / 2) = 252."""
        return (self.latent_height // self.patch_size[1]) * (
            self.latent_width // self.patch_size[2]
        )


@dataclass(frozen=True)
class DataConfig:
    """How recordings become conditions (the data and state areas read these)."""

    camera_encoding: str = (
        "noclip"  # WC_CAMERA_ENCODING=noclip: mu-law camera deltas without the +-20 degree clip
    )
    action_buttons: tuple[str, ...] = ACTION_BUTTONS  # action_buttons: button order of the controls
    skip_frame: int = 2  # skip_frame: 32 fps source -> 16 fps pixel frames
    max_tick_gap_seconds: float = (
        0.0390625  # max_tick_gap_seconds: a frame is covered if a tick lies within this gap
    )
    weapon_mode: str = "fine"  # weapon_mode: 52-way weapon ids
    eye_height: float = 64.0  # memory_eye_height: camera height above the feet (world units)
    hfov_degrees: float = (
        106.26  # memory_hfov_degrees: tan(hfov / 2) = 1.3333, tan_v = tan_h * 9 / 16 = 0.75
    )
    pose_source: str = (
        "oracle"  # recorded poses for every player (Table 3); the state model is the closed loop
    )

    def __post_init__(self):
        _only(self.camera_encoding, CAMERA_ENCODINGS, "data.camera_encoding")
        _only(self.weapon_mode, ("fine",), "data.weapon_mode")
        _only(self.pose_source, ("oracle",), "data.pose_source")


@dataclass(frozen=True)
class MemoryConfig:
    """The scene state: views memory with picture depth, one entry read per block."""

    enabled: bool = True  # --arm memory_on (False = memory_off, the "scene state off" row)
    bound: int = 64  # --bound 64: B, the most own entries a client keeps
    k: int = 1  # --k 1 (K_DEPLOYED): entries read per block (one 4-latent block)
    start_latent: int = 25  # first read at latent 25 = the first reconstituted block of the window
    write_gate: float = 0.0  # --view-g 0: every block with at least one hit pixel is published
    holes: str = "reach"  # --view-holes reach: holes = pixels no recent point reaches
    read_rule: str = "fill"  # --view-read fill: the entry that fills the most holes
    own: str = "include"  # --view-own include: the reader's own entries are candidates
    write: str = (
        "own"  # --view-write own: the write gate and eviction read the writer's own entries
    )
    pool_scope: str = "all"  # --view-pool-scope all: every player of the round
    scene_depth: str = "picture"  # --scene-depth picture: depth head on the latents
    scene_memory: str = "views"  # --scene-memory views

    def __post_init__(self):
        if self.bound <= 0:
            raise ValueError("memory.bound must be positive")
        if self.k != 1:
            raise NotImplementedError("the views memory reads exactly one entry per block (k = 1)")
        if self.write_gate < 0:
            raise ValueError("memory.write_gate must be non-negative")
        for name, value, allowed in (
            ("holes", self.holes, ("reach",)),
            ("read_rule", self.read_rule, ("fill",)),
            ("own", self.own, ("include",)),
            ("write", self.write, ("own",)),
            ("pool_scope", self.pool_scope, ("all",)),
            ("scene_depth", self.scene_depth, ("picture",)),
            ("scene_memory", self.scene_memory, ("views",)),
        ):
            _only(value, allowed, f"memory.{name}")


@dataclass(frozen=True)
class PlayerStateConfig:
    """Where the player states come from (Sec. 3.4).

    ``recorded``: every player at its recorded position, GT visibility labels (Table 3, the default).
    ``predicted``: closed-loop deployment (Tables 2 and 4a): each client estimates its own position with the state
    model and publishes it, the other clients are extrapolated from their published positions and their controls,
    and visibility is predicted from the picture depth head. Needs ``paths.state_model`` and the tables under
    ``paths.state_model_cells``, ``paths.state_model_map_norm`` and ``paths.physics_prior``.
    """

    source: str = "recorded"  # recorded | predicted
    pose_radius_u: float = (
        40.0  # [--visibility-pose-radius] predicted visibility tests 8 points on this ring
    )

    def __post_init__(self):
        if self.source not in ("recorded", "predicted"):
            raise ValueError(
                f"player_state.source must be recorded or predicted, got {self.source!r}"
            )
        if not self.pose_radius_u >= 0:
            raise ValueError("player_state.pose_radius_u must be >= 0")


@dataclass(frozen=True)
class PoolConfig:
    """The shared pool through which the clients of one round exchange published blocks (lock-step)."""

    provenance: str = "live"  # --pool-provenance live: peers render at the same time
    address: str = "published"  # --live-address published: follow each writer's withdrawals
    wait_s: float = 1800.0  # --live-wait-s 1800: lock-step wait budget per block
    poll_s: float = 2.0  # --live-wait-poll-s 2.0: poll period (latency only)
    fail_on_timeout: bool = (
        True  # release default: a late or failed peer is an error (the old code degraded silently)
    )

    def __post_init__(self):
        _only(self.provenance, ("live",), "pool.provenance")
        _only(self.address, ("published",), "pool.address")
        if self.wait_s <= 0 or self.poll_s <= 0:
            raise ValueError(
                "pool.wait_s and pool.poll_s must be positive (0 makes the output timing dependent)"
            )


@dataclass(frozen=True)
class PathsConfig:
    """Artefacts. No defaults: every path comes from the YAML or the CLI."""

    checkpoint: str | None = None  # generator weights (old: <checkpoint_dir>/model.pt, step 600)
    checkpoint_key: str = (
        "generator_ema"  # --weight-source composite_ema -> model.pt[generator_ema]
    )
    wan22_root: str | None = (
        None  # $WAN22_MODEL_ROOT/Wan2.2-TI2V-5B: config.json, VAE, umT5, tokenizer
    )
    prompt_embedding: str | None = (
        None  # shipped umT5 embedding of the fixed prompt; unset = encode with umT5 from wan22_root
    )
    depth_head: str | None = None  # --picture-depth-head: ddp5-777/depthlat_ckpt.pt (44.2M params)
    depth_readout: str | None = (
        None  # --picture-depth-readout: readout_ddp5/readout/readout.pt (0.21M params)
    )
    round_index: str | None = None  # --index: wholeround_index.jsonl (one row per client)
    media_index: str | None = None  # covis_media_index: media_index.jsonl
    dataset_root: str | None = None  # covis_dataset_root: OpenCS2 tick parquet of every player
    latent_cache_root: str | None = (
        None  # latent_cache_root: <media>.npz, member win_000000 (latent 0 = the sink)
    )
    visibility_label_root: str | None = (
        None  # visibility_label_root: <media>.npz GT visibility labels
    )
    obs_signal_label_root: str | None = (
        None  # flashlabels/<media>.npz, scopelabels/<media>.npz (was a hard-coded path); a missing file is an error
    )
    live_pool_dir: str | None = None  # --live-pool-dir: one fresh directory per session
    out_dir: str | None = None  # --out: where latents.npy is written
    state_model: str | None = (
        None  # player_state.source=predicted: the state model checkpoint (not released)
    )
    state_model_cells: str | None = (
        None  # configs/state_model/cells_v1.json (the place head's cell table)
    )
    state_model_map_norm: str | None = None  # configs/state_model/map_norm_v1.json
    physics_prior: str | None = None  # configs/state_model/physics_prior_v1.json (extrapolation)

    def require(self, *names: str) -> None:
        """Raise if any of the named paths is unset (call before the run, not at import)."""
        missing = [n for n in names if getattr(self, n) in (None, "")]
        if missing:
            raise ValueError(
                f"paths not set: {missing} (set them in the config or on the command line)"
            )


@dataclass(frozen=True)
class WeightsConfig:
    """Where ``tools/download_weights.py`` fetches the release files. The client never reads this section."""

    hf_repo_id: str = "ZiyangYe/WorldCast"
    hf_revision: str | None = None  # pin a commit once released
    checkpoint_file: str = "worldcast_4step_bf16.safetensors"  # generator EMA, step 600, bf16
    depth_head_file: str = "depth_head.safetensors"  # DepthLat (44.2M params, fp32)
    depth_readout_file: str = "depth_readout.safetensors"  # Readout (0.21M params, fp32)
    prompt_embedding_file: str = "fixed_prompt_umt5xxl_bf16.safetensors"  # [1, 512, 4096] bf16
    wan22_repo_id: str = (
        "Wan-AI/Wan2.2-TI2V-5B"  # base snapshot: VAE (decode), umT5 + tokenizer, config.json
    )
    wan22_revision: str = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"


@dataclass(frozen=True)
class RunConfig:
    """One client's run."""

    seed: int = 20260917  # --seed: CPU entry noise generator and the RNG of the ladder re-noise
    latents: int = 441  # --latents 441 = 1 + 4 x 110 s (clipped to the recording's coverage)
    index_row: int | None = None  # --window: row of the round index (one client); d59 = 59
    max_blocks: int = 0  # --max-blocks: 0 = every block
    device: str = "cuda"  # one GPU per client
    latents_dtype: str = (
        "float32"  # latents.npy is always written in float32 (the old code chose fp16 when lossless)
    )

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("run.seed must be a non-negative integer")
        if self.latents < 5 or (self.latents - 1) % 4:
            raise ValueError(f"run.latents={self.latents} must be 1 + 4k with k >= 1")
        if self.max_blocks < 0:
            raise ValueError("run.max_blocks must be >= 0")
        _only(self.latents_dtype, ("float32",), "run.latents_dtype")


@dataclass(frozen=True)
class InferenceConfig:
    """The whole client configuration; the defaults reproduce the paper's Table-3 WorldCast row."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    player_state: PlayerStateConfig = field(default_factory=PlayerStateConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)

    def __post_init__(self):
        if self.memory.start_latent != self.window.first_target:
            raise ValueError(
                f"memory.start_latent={self.memory.start_latent} but the first reconstituted block "
                f"of the window is {self.window.first_target}"
            )
        if self.window.block != self.model.vae_time_compression:
            raise ValueError(
                "a block of 4 latents must match the VAE time compression of 4 pixel rows"
            )
        unreachable = set(self.sampler.framewise_condition_axes) - set(
            self.model.condition_passthrough_keys
        )
        if unreachable:
            raise ValueError(f"framewise conditions {sorted(unreachable)} are not generator inputs")
        if self.run.latents > 1 and self.run.latents < 1 + self.window.plain_prefix_latents:
            raise ValueError(
                f"run.latents={self.run.latents} is shorter than the plain prefix "
                f"(1 + {self.window.plain_prefix_latents})"
            )


# ----------------------------------------------------------------------------------------------- loading
def config_from_dict(data: Mapping[str, Any] | None) -> InferenceConfig:
    """Build (and validate) an :class:`InferenceConfig` from nested mappings; unknown keys are errors."""
    return _build(InferenceConfig, dict(data or {}), "")


def load_config(
    path: str | os.PathLike[str], overrides: Mapping[str, Any] | None = None
) -> InferenceConfig:
    """Read a YAML file (sections as in ``configs/infer/worldcast_4step.yaml``) and apply dotted overrides.

    ``overrides`` maps dotted keys to values, e.g. ``{"run.seed": 1, "paths.out_dir": "/tmp/x"}`` (the CLI).
    """
    import yaml

    with open(os.fspath(path), encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: the top level of the config must be a mapping")
    cfg = config_from_dict(data)
    return with_overrides(cfg, overrides) if overrides else cfg


def with_overrides(cfg: InferenceConfig, overrides: Mapping[str, Any]) -> InferenceConfig:
    """A new config with dotted keys replaced (re-validated)."""
    data = config_to_dict(cfg)
    for dotted, value in overrides.items():
        node = data
        parts = str(dotted).split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"unknown config section in override {dotted!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key in override {dotted!r}")
        node[parts[-1]] = value
    return config_from_dict(data)


def config_to_dict(cfg) -> dict[str, Any]:
    """Plain nested dict (tuples become lists), the inverse of :func:`config_from_dict`."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            value = config_to_dict(value)
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, dict):
            value = dict(value)
        out[f.name] = value
    return out


def _build(cls, data: Mapping[str, Any], where: str):
    if not isinstance(data, Mapping):
        raise ValueError(
            f"config section {where or '<root>'} must be a mapping, got {type(data).__name__}"
        )
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown config keys in {where or '<root>'}: {unknown}")
    kwargs = {}
    for name in names:
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], f"{where}.{name}" if where else name)
    return cls(**kwargs)


def _coerce(tp, value, where: str):
    origin, args = get_origin(tp), get_args(tp)
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    if origin in (Union, UnionType):  # Optional[X] and X | None
        inner = [a for a in args if a is not type(None)]
        if value is None and len(inner) < len(args):
            return None
        return _coerce(inner[0], value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, where) for v in value)
        if len(value) != len(args):
            raise ValueError(f"{where} must have {len(args)} entries, got {list(value)!r}")
        return tuple(_coerce(a, v, where) for a, v in zip(args, value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"{where} must be a mapping, got {value!r}")
        return {_coerce(args[0], k, where): _coerce(args[1], v, where) for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where} must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported config type {tp!r}")
