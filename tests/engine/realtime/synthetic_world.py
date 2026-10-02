"""A synthetic round and tiny random weights for the realtime CPU tests (no OpenCS2 data, no pyarrow).

Ten players walk and turn on one map; their tick tables are built in memory and served by a stand-in for
``read_player_ticks`` (:func:`patch_ticks`). Labels, observer signals and first latents are written in their
producers' formats. The weights are a tiny random generator (paper latent grid 48 x 24 x 42, 4 blocks of width 32),
a tiny depth head and a prompt embedding.
"""

import json
from pathlib import Path

import numpy as np
import torch

from worldcast.data.ticks import TickTable

MATCH, MAP, ROUND = 7, "de_nuke", 3
SEED = 20260917
TINY = dict(
    in_dim=48,
    out_dim=48,
    dim=32,
    ffn_dim=64,
    freq_dim=16,
    text_dim=4096,
    text_len=512,
    num_heads=2,
    num_layers=4,
)
SNAPSHOT = {
    "_class_name": "WanModel",
    "model_type": "ti2v",
    "patch_size": [1, 2, 2],
    "eps": 1e-06,
    **{k: v for k, v in TINY.items() if k != "text_dim"},
}
PROMPT = "first-person Counter-Strike 2 gameplay"
_BUTTONS = (
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
)


def media_id(slot: int) -> str:
    return f"{MATCH}-{MAP}-r{ROUND:02d}-p{slot:02d}"


def _ticks(slot: int, seconds: float, rng: np.random.Generator) -> TickTable:
    n = int(seconds * 64) + 1
    t = np.arange(n, dtype=np.float64) / 64.0
    team = 2 if slot < 5 else 3
    heading = rng.uniform(-180, 180)
    d_yaw = rng.normal(0.0, 0.15, n).astype(np.float32)
    d_pitch = rng.normal(0.0, 0.05, n).astype(np.float32)
    yaw = heading + np.cumsum(d_yaw.astype(np.float64))
    pitch = np.clip(np.cumsum(d_pitch.astype(np.float64)), -30, 30)
    start = rng.uniform(-300, 300, 2) + (0 if team == 2 else 600)
    rad = np.deg2rad(yaw)
    x = start[0] + np.cumsum(np.cos(rad)) * 3.0
    y = start[1] + np.cumsum(np.sin(rad)) * 3.0
    active = [
        [b for b in _BUTTONS if rng.random() < (0.6 if b == "forward" else 0.05)] for _ in range(n)
    ]
    return TickTable(
        media_id=media_id(slot),
        sha256="0" * 64,
        tick=1000 + np.arange(n, dtype=np.int64),
        t=t,
        x=x,
        y=y,
        z=np.zeros(n),
        yaw=yaw,
        pitch=pitch,
        is_alive=np.ones(n, dtype=bool),
        active=active,
        delta_pitch=d_pitch.astype(np.float64),
        delta_yaw=d_yaw.astype(np.float64),
        input_weapon=["ak47"] * n,
        team_num=np.full(n, team, dtype=np.int64),
    )


def _obs_labels(root: Path, mid: str, video_frames: int, rng: np.random.Generator) -> None:
    (root / "flashlabels").mkdir(parents=True, exist_ok=True)
    (root / "scopelabels").mkdir(parents=True, exist_ok=True)
    lum = rng.uniform(0.0, 0.9, video_frames).astype(np.float16)
    np.savez_compressed(
        root / "flashlabels" / f"{mid}.npz",
        lum=lum,
        stride=np.array(1, np.int16),
        hot_threshold=np.array(0.85, np.float32),
        n_hot=np.array(int((lum > 0.85).sum()), np.int32),
    )
    scoped = (rng.random(video_frames) < 0.1).astype(np.uint8)
    np.savez_compressed(
        root / "scopelabels" / f"{mid}.npz",
        corner_max=rng.uniform(0, 1, video_frames).astype(np.float16),
        center=rng.uniform(0, 1, video_frames).astype(np.float16),
        scoped_vis=scoped,
        attack2=np.zeros(video_frames, np.uint8),
        weapon_id=np.full(video_frames, 2, np.int16),
        level=scoped.astype(np.int8),
        ncov=np.array(video_frames - 2, np.int32),
        thresholds=np.array([0.04, 0.08], np.float32),
    )


def make_world(
    root: Path, *, clients=(0,), seconds: float = 8.5, seed: int = 0
) -> dict[str, object]:
    """Write the round under ``root``; returns the data paths and the in-memory tick tables (``tables``)."""
    rng = np.random.default_rng(seed)
    dirs = {k: root / k for k in ("dataset", "vislabels", "obslabels", "latents")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    tables, rows = {}, []
    video_frames = int(seconds * 32)
    for slot in range(10):
        mid = media_id(slot)
        tables[mid] = _ticks(slot, seconds, rng)
        rows.append(
            dict(
                media_id=mid,
                match_id=MATCH,
                map_name=MAP,
                round=ROUND,
                player_slot=slot,
                fps=32.0,
                video_frames=video_frames,
                ticks_path=f"rounds/{mid}.parquet",
                ticks_rows=len(tables[mid]),
                ticks_file_size=1,
                ticks_sha256="0" * 64,
            )
        )
        visible = rng.random((10, video_frames)) < 0.5
        np.savez(
            dirs["vislabels"] / f"{mid}.npz",
            _binary_visible=visible,
            _binary_eval_valid=np.ones_like(visible),
            _in_frustum=visible,
            _binary_offscreen=~visible,
        )
        _obs_labels(dirs["obslabels"], mid, video_frames, rng)
        np.savez(
            dirs["latents"] / f"{mid}.npz",
            win_000000=rng.standard_normal((1, 41, 48, 24, 42)).astype(np.float16),
        )
    media_index = root / "media_index.jsonl"
    media_index.write_text("".join(json.dumps(r) + "\n" for r in rows))
    group = [media_id(s) for s in clients]
    round_index = root / "round_index.jsonl"
    round_index.write_text(
        "".join(
            json.dumps(
                dict(
                    media_id=media_id(s),
                    start_frame=0,
                    match_id=MATCH,
                    round=ROUND,
                    map_name=MAP,
                    latent_key="win_000000",
                    player_slot=s,
                    group_media=group,
                    group_slots=list(clients),
                )
            )
            + "\n"
            for s in clients
        )
    )
    return dict(
        dataset_root=str(dirs["dataset"]),
        media_index=str(media_index),
        round_index=str(round_index),
        visibility_label_root=str(dirs["vislabels"]),
        obs_signal_label_root=str(dirs["obslabels"]),
        latent_cache_root=str(dirs["latents"]),
        tables=tables,
    )


def patch_ticks(monkeypatch, tables) -> None:
    """Serve the in-memory tick tables wherever the client and the engine read tick files."""

    def read(media, dataset_root):
        return tables[media.media_id]

    import worldcast.data.item
    import worldcast.engine.inference.client
    import worldcast.engine.realtime.engine

    for module in (
        worldcast.engine.inference.client,
        worldcast.data.item,
        worldcast.engine.realtime.engine,
    ):
        monkeypatch.setattr(module, "read_player_ticks", read)


def make_weights(out: Path) -> dict[str, str]:
    """Tiny random generator (release key names under ``model.``), depth head, read-out, prompt embedding."""
    from worldcast.config.inference import InferenceConfig
    from worldcast.modeling.build import generator_config_from_inference
    from worldcast.modeling.depth_head import DepthLat, Readout
    from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator
    from worldcast.modeling.wan22.text_encoder import PromptEmbedding

    out.mkdir(parents=True, exist_ok=True)
    cfg = config({}, {}, out, max_blocks=1)
    model = WorldCastGenerator(generator_config_from_inference(cfg, GeneratorConfig(**TINY)))
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    torch.save(
        {"generator_ema": {"model." + k: v for k, v in model.state_dict().items()}, "step": 0},
        out / "generator.pt",
    )
    torch.manual_seed(9)
    torch.save(DepthLat(cin=48, w=8, nb=1, ndb=1).state_dict(), out / "depth_head.pt")
    torch.save(Readout(w=8).state_dict(), out / "depth_readout.pt")
    embeds = torch.zeros(1, 512, TINY["text_dim"])
    embeds[:, :10] = torch.randn(1, 10, TINY["text_dim"], generator=g) * 0.5
    PromptEmbedding(prompt=PROMPT, embeds=embeds.to(torch.bfloat16), seq_len=10).save(
        str(out / "prompt.safetensors")
    )
    snapshot = out / "wan22"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "config.json").write_text(json.dumps(SNAPSHOT))
    assert isinstance(cfg, InferenceConfig)
    return dict(
        checkpoint=str(out / "generator.pt"),
        wan22_root=str(snapshot),
        depth_head=str(out / "depth_head.pt"),
        depth_readout=str(out / "depth_readout.pt"),
        prompt_embedding=str(out / "prompt.safetensors"),
    )


def config(
    world: dict[str, object],
    weights: dict[str, str],
    out: Path,
    *,
    max_blocks: int,
    row: int = 0,
    pool: str = "",
):
    """The release config of a CPU float32 client of the synthetic round."""
    from worldcast.config.inference import config_from_dict

    paths = {k: v for k, v in {**weights, **world}.items() if k != "tables"}
    if pool:
        paths["live_pool_dir"] = pool
    paths["out_dir"] = str(out)
    return config_from_dict(
        dict(
            model=dict(action_hidden_dim=64, action_adaln_rank=8, obs_signal_hidden=16),
            sampler=dict(model_input_dtype="float32"),
            pool=dict(wait_s=300.0, poll_s=0.01),
            paths=paths,
            run=dict(seed=SEED, latents=441, index_row=row, max_blocks=max_blocks, device="cpu"),
        )
    )


# ----------------------------------------------------------------------------------------------- closed loop
STATE_TABLES = Path(__file__).resolve().parents[3] / "configs" / "state_model"


def write_state_model(path: Path) -> dict[str, str]:
    """A small random state model and the release's tables; returns the ``paths.*`` overrides of the closed loop."""
    from worldcast.modeling.state_model import StateModel, StateTables

    torch.manual_seed(11)
    model = StateModel(
        StateTables.load(STATE_TABLES / "cells_v1.json", STATE_TABLES / "map_norm_v1.json"),
        dim=64,
        layers=2,
    )
    g = torch.Generator().manual_seed(12)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    torch.save(model.state_dict(), path)
    return {
        "paths.state_model": str(path),
        "paths.state_model_cells": str(STATE_TABLES / "cells_v1.json"),
        "paths.state_model_map_norm": str(STATE_TABLES / "map_norm_v1.json"),
        "paths.physics_prior": str(STATE_TABLES / "physics_prior_v1.json"),
    }


def patch_control_ticks(monkeypatch, tables) -> None:
    """Serve the state model's raw control stream from the in-memory tick tables (``read_control_ticks``)."""
    from worldcast.data.actions import IGNORED_ACTIVE_ACTIONS, OPENCS2_BUTTONS, ControlTicks

    index = {name: i for i, name in enumerate(OPENCS2_BUTTONS)}
    by_name = {f"{mid}.parquet": table for mid, table in tables.items()}

    def read(path):
        table = by_name[Path(path).name]
        buttons = np.zeros((len(table.active), len(OPENCS2_BUTTONS)), np.float32)
        for row, names in enumerate(table.active):
            for name in names or ():
                if name not in IGNORED_ACTIVE_ACTIONS:
                    buttons[row, index[name]] = 1.0
        return ControlTicks(
            t=table.t.astype(np.float64),
            buttons=buttons,
            delta_pitch=table.delta_pitch.astype(np.float32),
            delta_yaw=table.delta_yaw.astype(np.float32),
            unknown_rows=np.zeros(0, np.int64),
        )

    import worldcast.engine.inference.client
    import worldcast.engine.realtime.engine

    for module in (worldcast.engine.inference.client, worldcast.engine.realtime.engine):
        monkeypatch.setattr(module, "read_control_ticks", read)
