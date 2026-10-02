"""Demo settings: YAML files merged in order, then dotted ``--set key=value`` overrides (docs/demo.md, "Config")."""

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "demo" / "demo.yaml"
SYNC_MODES = ("async", "lockstep")
MEDIA_ROUTES = ("direct", "proxy")
ENGINES = ("mock", "worldcast")
ACTION_MAPPINGS = ("latest", "realtime")


@dataclass
class CoordinatorConfig:
    host: str = "0.0.0.0"
    port: int = 8100
    #: ``direct``: browsers open the worker's own WebSocket; ``proxy``: through the coordinator (GPU hosts unreachable).
    media_route: str = "direct"
    max_rooms: int = 32
    #: Scene-state blocks kept per player for late joiners (the memory bound B of the paper).
    scene_blocks_per_player: int = 64
    #: Seconds an empty room stays open.
    empty_room_s: float = 120.0


@dataclass
class WorkerConfig:
    host: str = "0.0.0.0"
    port: int = 8101
    coordinator_url: str = "ws://127.0.0.1:8100"
    #: WebSocket base browsers use to reach this worker, e.g. ``ws://gpu-07.lan:8101``; None: the coordinator uses the
    #: address the worker connected from.
    advertise_url: str | None = None
    worker_id: str | None = None
    gpu_label: str | None = None
    jpeg_quality: int = 85
    #: Seconds a player may reconnect (page reload) before the seat is released.
    reconnect_grace_s: float = 8.0
    #: Lock-step only: longest wait for a peer's previous block before stepping without it.
    lockstep_timeout_s: float = 3.0


@dataclass
class MockConfig:
    """The mock engine's timing: one step = ``frames_per_step`` frames after ``step_ms`` of fake denoising."""

    frames_per_step: int = 4
    step_ms: float = 180.0
    decode_ms: float = 4.0
    jitter_ms: float = 15.0
    #: One fake scene-state block per this many frames (16 = one paper block of 4 latents).
    scene_block_frames: int = 16
    #: Payload of a fake scene block: 4 x 48 x 24 x 42 float32 latents = 774144 bytes.
    scene_block_bytes: int = 774144


@dataclass
class WorldCastEngineConfig:
    """The GPU engine (``worldcast.engine.realtime``): the inference YAMLs (the README's three), dotted overrides of them,
    the serving options (``RealtimeConfig`` fields) and where the player's own state comes from."""

    configs: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    realtime: dict[str, Any] = field(
        default_factory=lambda: {
            "generator": "fast",
            "cuda_graphs": True,
            "early_prefill": True,
            "decode_overlap": True,
            "encoder": "jpeg",
            "lockstep": False,
        }
    )
    #: ``state_model``: the closed loop (no recorded positions); ``gt``: the recorded track (paper Table 3).
    own_state: str = "state_model"


@dataclass
class EngineConfig:
    kind: str = "mock"
    mock: MockConfig = field(default_factory=MockConfig)
    worldcast: WorldCastEngineConfig = field(default_factory=WorldCastEngineConfig)


@dataclass
class PlayConfig:
    """What every browser is told: frame rate, jitter buffer and the input-to-frame mapping."""

    fps: float = 16.0
    #: Frames the browser buffers before it shows the first one (and after a stall).
    jitter_buffer_frames: int = 2
    #: How a block's inputs map onto its frames (demo/actions.py): ``latest`` (lowest latency) or ``realtime``.
    action_mapping: str = "latest"
    mouse_degrees_per_pixel: float = 0.055
    default_sync: str = "async"


@dataclass
class DemoConfig:
    #: ``library.json`` of the round starts (tools/build_demo_library.py); None: the built-in synthetic arena.
    library: str | None = None
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    play: PlayConfig = field(default_factory=PlayConfig)

    def validate(self) -> "DemoConfig":
        checks = (
            (self.coordinator.media_route, MEDIA_ROUTES, "coordinator.media_route"),
            (self.engine.kind, ENGINES, "engine.kind"),
            (self.play.action_mapping, ACTION_MAPPINGS, "play.action_mapping"),
            (self.play.default_sync, SYNC_MODES, "play.default_sync"),
        )
        for value, allowed, key in checks:
            if value not in allowed:
                raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
        if self.engine.mock.frames_per_step < 1:
            raise ValueError("engine.mock.frames_per_step must be at least 1")
        return self


def _build(cls, data: Mapping[str, Any]):
    """Instantiate dataclass ``cls`` from a nested mapping; unknown keys raise."""
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {unknown}")
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        kwargs[name] = (
            _build(type(default), value or {}) if dataclasses.is_dataclass(default) else value
        )
    return cls(**kwargs)


def merge_dicts(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["worker.port=8102"]`` -> nested ``{"worker": {"port": 8102}}`` (YAML-typed values)."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        node = out
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return out


def load_config(paths: Sequence[str] | None = None, overrides: Sequence[str] = ()) -> DemoConfig:
    """Merge the YAML files (default ``configs/demo/demo.yaml``), apply overrides, validate.

    A relative ``library`` path is resolved against the YAML file that set it.
    """
    data: dict[str, Any] = {}
    for path in paths or [str(DEFAULT_CONFIG)]:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if loaded.get("library"):
            loaded["library"] = str((Path(path).resolve().parent / loaded["library"]).resolve())
        data = merge_dicts(data, loaded)
    data = merge_dicts(data, parse_overrides(overrides))
    return _build(DemoConfig, data).validate()
