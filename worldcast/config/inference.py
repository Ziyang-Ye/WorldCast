"""The inference config: what a user sets for a client.

The defaults are the settings of the paper's Table-3 runs, except the length: Table 3 ran every
round for its recorded length (``tools/run_session.py --round-length``), and ``run.latent_frames``
defaults to 441.

What the paper fixes is a constant of the module that uses it: the window layout
(:mod:`worldcast.sampling.window`), the denoising steps and the context noise
(:mod:`worldcast.sampling.sampler`), the model
(:class:`worldcast.modeling.wan22.model.GeneratorConfig`, read from the Wan2.2 snapshot), the player
state field (:class:`worldcast.player_state.field.PlayerStateFieldConfig`) and the recordings'
sampling (:class:`worldcast.data.window.WindowSpec`). Artefact paths have no default:
``weights/paths.yaml`` (written by ``tools/download_weights.py``) and a copy of
``examples/data_paths.yaml`` set them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .loader import (
    Paths,
    build,
    check_at_least,
    check_choice,
    config_to_dict,
    load,
    require_set,
    set_dotted,
)

__all__ = [
    "MEMORY_BOUND",
    "PLAYER_STATE_SOURCES",
    "InferenceConfig",
    "ModelConfig",
    "PathsConfig",
    "PlayerStateConfig",
    "RunConfig",
    "SceneStateConfig",
    "WorldStateConfig",
    "load_config",
]

#: Where the player states come from (Sec. 3.4): ``gt``, the GT states of the recording (Table 3),
#: or ``predicted`` (the closed loop of Tables 2 and 4a).
PLAYER_STATE_SOURCES = ("gt", "predicted")
#: B, the most memory entries of its own a client keeps (Sec. 3.3).
MEMORY_BOUND = 64


@dataclass(frozen=True)
class PathsConfig:
    """Artefacts; every path comes from a YAML file or the command line.

    Attributes:
        checkpoint (str | None): the generator weights.
        wan22_root (str | None): the Wan2.2-TI2V-5B snapshot: ``config.json``, VAE, umT5, tokenizer.
        prompt_embedding (str | None): the umT5 embedding of the fixed prompt; unset: umT5 encodes
            it.
        depth_head (str | None): the depth head.
        depth_readout (str | None): its read-out.
        round_index (str | None): one row per client.
        media_index (str | None): every recording.
        dataset_root (str | None): the OpenCS2 tick tables of every player.
        latent_cache_root (str | None): ``<media>.npz``; its member ``win_<start_frame:06d>`` holds
            the client's first frame.
        visibility_label_root (str | None): the GT visibility labels, ``<media>.npz``.
        observer_signal_label_root (str | None): ``flashlabels/`` and ``scopelabels/<media>.npz``.
        world_state_dir (str | None): the session's shared world state, fresh per session.
        out_dir (str | None): where ``latents.npy`` and ``client.json`` are written.
        state_model (str | None): predicted player states: the state model weights.
        state_model_cells (str | None): ``configs/state_model/cells.json``.
        physics_prior (str | None): ``configs/state_model/physics_prior.json``.
    """

    checkpoint: str | None = None
    wan22_root: str | None = None
    prompt_embedding: str | None = None
    depth_head: str | None = None
    depth_readout: str | None = None
    round_index: str | None = None
    media_index: str | None = None
    dataset_root: str | None = None
    latent_cache_root: str | None = None
    visibility_label_root: str | None = None
    observer_signal_label_root: str | None = None
    world_state_dir: str | None = None
    out_dir: str | None = None
    state_model: str | None = None
    state_model_cells: str | None = None
    physics_prior: str | None = None

    def require(self, *names: str) -> None:
        """Raise ``ValueError`` if any of the named paths is unset."""
        require_set(self, "paths", names)


@dataclass(frozen=True)
class RunConfig:
    """One client's run.

    Attributes:
        seed (int): seeds the entry noise and the re-noise between denoising steps.
        latent_frames (int): the latent frames the client generates, ``1 + 4 k``: the first frame
            and ``k`` blocks of four, one per second (441: 110 s); clipped to the recording's
            coverage.
        index_row (int | None): the client's row of the round index, from 0.
        max_blocks (int): blocks to run after latent frame 24; 0: all.
        device (str): ``cuda`` (bf16) or ``cpu`` (float32).
    """

    seed: int = 20260917
    latent_frames: int = 441
    index_row: int | None = None
    max_blocks: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("run.seed must be a non-negative integer")
        # 4: the latent frames of a block (worldcast.data.latents.BLOCK, a module above this one)
        if self.latent_frames < 5 or (self.latent_frames - 1) % 4:
            raise ValueError(f"run.latent_frames={self.latent_frames} must be 1 + 4k with k >= 1")
        if self.index_row is not None:
            check_at_least("run.index_row", self.index_row, 0)
        check_at_least("run.max_blocks", self.max_blocks, 0)


@dataclass(frozen=True)
class ModelConfig:
    """The generator on the client's GPU.

    Attributes:
        attention (str): the attention kernel, a name of
            :data:`worldcast.modeling.wan22.attention.ATTENTION_KERNELS`: ``flash``,
            flash-attention 2, the kernel of the reference runs and the only one that gives their
            bits; ``fa3``, FlashAttention-3 when it is installed, else ``flash``; ``sdpa``,
            PyTorch's kernel, for GPUs without flash-attention (the CPU always uses it).
    """

    attention: str = "flash"


@dataclass(frozen=True)
class PlayerStateConfig:
    """Where the player states come from (Sec. 3.4).

    Attributes:
        source (str): one of :data:`PLAYER_STATE_SOURCES`. ``gt``: the GT states, every player at
            its recorded position, and the GT visibility labels (Table 3). ``predicted``: each
            client estimates its own position with the state model and publishes it, the other
            clients are extrapolated from their published positions and controls, and visibility is
            predicted from the depth head; needs ``paths.state_model``, ``paths.state_model_cells``
            and ``paths.physics_prior``.
    """

    source: str = "gt"

    def __post_init__(self) -> None:
        check_choice("player_state.source", self.source, PLAYER_STATE_SOURCES)


@dataclass(frozen=True)
class WorldStateConfig:
    """The shared world state through which the clients of a round advance in lockstep.

    Attributes:
        wait_s (float): the lockstep wait budget per block, seconds; a late client is an error.
        poll_s (float): the poll period, seconds (latency only).
    """

    wait_s: float = 1800.0
    poll_s: float = 2.0

    def __post_init__(self) -> None:
        if self.wait_s <= 0 or self.poll_s <= 0:
            raise ValueError("world_state.wait_s and world_state.poll_s must be positive")


@dataclass(frozen=True)
class SceneStateConfig:
    """The scene state (Sec. 3.3).

    Attributes:
        bound (int): B, the most memory entries of its own a client keeps.
    """

    bound: int = MEMORY_BOUND

    def __post_init__(self) -> None:
        check_at_least("scene_state.bound", self.bound, 1)


@dataclass(frozen=True)
class InferenceConfig:
    """The whole client configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    player_state: PlayerStateConfig = field(default_factory=PlayerStateConfig)
    world_state: WorldStateConfig = field(default_factory=WorldStateConfig)
    scene_state: SceneStateConfig = field(default_factory=SceneStateConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "InferenceConfig":
        """The config of nested mappings, validated; unknown keys are errors."""
        return build(cls, data or {})

    def with_overrides(self, overrides: Mapping[str, Any]) -> "InferenceConfig":
        """This config with dotted keys replaced (``{"run.seed": 1}``), validated."""
        return self.from_dict(set_dotted(config_to_dict(self), overrides))


def load_config(paths: Paths = (), overrides: Mapping[str, Any] | None = None) -> InferenceConfig:
    """The defaults, the YAML ``paths`` merged over them in order, then the dotted ``overrides``.

    Args:
        paths (Paths): YAML files with the sections of :class:`InferenceConfig`.
        overrides (Mapping[str, Any] | None): dotted keys, e.g. ``{"run.seed": 1}``.

    Returns:
        InferenceConfig: the validated config.
    """
    return load(paths, overrides, InferenceConfig.from_dict)
