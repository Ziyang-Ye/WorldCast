"""A synthetic round and tiny random weights for the client's CPU tests (no data, no pyarrow),
blocks and world states for the tests of the shared world state, and an offline run with its
fingerprint table.

Ten players walk and turn on one map; their tick tables are built in memory and served by a stand-in
for ``read_player_ticks`` (:func:`patch_ticks`). Labels, observer signals and first latents are
written in their producers' formats. The weights are a tiny random generator (paper latent grid 48 x
24 x 42, 4 blocks of width 32), a tiny depth head and a prompt embedding.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.modeling.support import TINY as TINY_MODEL
from tests.modeling.support import randomize_
from worldcast.data.latents import BLOCK
from worldcast.data.recordings import TickTable
from worldcast.engine.inference.directory import DirectoryWorldState
from worldcast.engine.inference.world_state import PublishedBlock

MATCH, MAP, ROUND = 7, "de_nuke", 3
SEED = 20260917
#: Blocks of a test session after the first six: 33 latent frames in all.
MAX_BLOCKS = 2
#: The tests' tiny generator at the paper's latent channels and prompt embedding.
TINY = {**TINY_MODEL, "in_dim": 48, "out_dim": 48, "text_dim": 4096, "text_len": 512}
SNAPSHOT = {
    "_class_name": "WanModel",
    "model_type": "ti2v",
    "patch_size": [1, 2, 2],
    "eps": 1e-06,
    **{k: v for k, v in TINY.items() if k != "text_dim"},
}
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
    """Write the round under ``root``; returns the data paths and the tick tables (``tables``)."""
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
            _in_frustum=np.ones_like(visible),
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
        observer_signal_label_root=str(dirs["obslabels"]),
        latent_cache_root=str(dirs["latents"]),
        tables=tables,
    )


def patch_ticks(monkeypatch, tables) -> None:
    """Serve the in-memory tick tables wherever the client reads tick files."""

    def read(media, dataset_root, *, jump_recall=True):
        return tables[media.media_id]

    import worldcast.data.window
    import worldcast.engine.inference.loading

    for module in (worldcast.data.window, worldcast.engine.inference.loading):
        monkeypatch.setattr(module, "read_player_ticks", read)


def make_weights(out: Path) -> dict[str, str]:
    """A tiny random generator, depth head, read-out and prompt embedding; returns their paths."""
    from worldcast.modeling.depth_head import DepthHead, DepthReadout
    from worldcast.modeling.wan22.model import GeneratorConfig, WorldCastGenerator
    from worldcast.modeling.wan22.text_encoder import save_prompt_embedding

    out.mkdir(parents=True, exist_ok=True)
    model = randomize_(WorldCastGenerator(GeneratorConfig(**TINY)), 8)
    torch.save(
        {"generator_ema": {"model." + k: v for k, v in model.state_dict().items()}, "step": 0},
        out / "generator.pt",
    )
    torch.manual_seed(9)
    torch.save(DepthHead(width=8, blocks=1, half_blocks=1).state_dict(), out / "depth_head.pt")
    torch.save(DepthReadout(width=8).state_dict(), out / "depth_readout.pt")
    embeds = torch.zeros(1, 512, TINY["text_dim"])
    g = torch.Generator().manual_seed(10)
    embeds[:, :10] = torch.randn(1, 10, TINY["text_dim"], generator=g) * 0.5
    save_prompt_embedding(embeds.to(torch.bfloat16), out / "prompt.safetensors")
    snapshot = out / "wan22"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "config.json").write_text(json.dumps(SNAPSHOT))
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
    world_state: str = "",
):
    """The release config of a CPU float32 client of the synthetic round."""
    from worldcast.config.inference import InferenceConfig

    paths = {k: v for k, v in {**weights, **world}.items() if k != "tables"}
    if world_state:
        paths["world_state_dir"] = world_state
    paths["out_dir"] = str(out)
    return InferenceConfig.from_dict(
        dict(
            world_state=dict(wait_s=300.0, poll_s=0.01),
            paths=paths,
            run=dict(
                seed=SEED, latent_frames=441, index_row=row, max_blocks=max_blocks, device="cpu"
            ),
        )
    )


def offline_run(tmp: Path) -> dict:
    """Run one offline client of the synthetic round under ``tmp``.

    Returns:
        dict: its config ``cfg``, the ``world``, its ``latents``, its ``media_id`` and ``table``, a
        fingerprint table of two reference runs: the run itself and a longer one of the same client
        (other entry noise, no ``all_latents``).
    """
    from worldcast.engine.inference import reference
    from worldcast.engine.inference.client import run_client

    torch.set_num_threads(1)
    world = make_world(tmp / "world", clients=(0,))
    weights = make_weights(tmp / "weights")
    cfg = config(world, weights, tmp / "run", max_blocks=1, world_state=str(tmp / "world_state"))
    with pytest.MonkeyPatch.context() as patch:
        patch_ticks(patch, world["tables"])
        client = run_client(cfg)["media_id"]
    latents = np.load(tmp / "run" / "latents.npy")
    recorded = dict(
        entry_noise=reference.entry_noise_fingerprint(cfg.run.seed, len(latents)),
        first_frame=reference.first_frame_fingerprint(torch.from_numpy(latents[:1])),
        **reference.latent_fingerprints(latents),
    )
    longer = dict(recorded, all_latents=None)
    cases = dict(
        same=dict(latent_frames=len(latents), fingerprints={client: recorded}),
        longer=dict(latent_frames=len(latents) + 4, fingerprints={client: longer}),
    )
    table = tmp / "fingerprints.json"
    table.write_text(json.dumps(dict(cases=cases)))
    return dict(cfg=cfg, world=world, latents=latents, table=table, media_id=client, tmp=tmp)


# -------------------------------------------------------------------------- the shared world state
def directory_world_state(root: Path, client: str, poll_s: float = 2.0) -> DirectoryWorldState:
    """The world state on the directory ``root`` as ``client`` sees it."""
    return DirectoryWorldState(root, client=client, poll_s=poll_s)


def block_latents(seed: int) -> np.ndarray:
    """Random latents ``[4, 2, 3, 5]`` float32 of one block."""
    return np.random.default_rng(seed).standard_normal((BLOCK, 2, 3, 5)).astype(np.float32)


def block_key(owner: str, window_start: int, f0: int) -> PublishedBlock:
    """The key of a block, as a reader asks for it."""
    return PublishedBlock(owner, window_start, f0, 0, 0, "")


# -------------------------------------------------------------------------------------- closed loop
STATE_TABLES = Path(__file__).resolve().parents[3] / "configs" / "state_model"


def write_state_model(path: Path) -> dict[str, str]:
    """A small random state model; returns the closed loop's ``paths.*`` overrides."""
    from worldcast.modeling.state_model import StateModel, load_cell_table

    torch.manual_seed(11)
    model = StateModel(load_cell_table(STATE_TABLES / "cells.json"), dim=64, layers=2)
    torch.save(randomize_(model, 12, scale=0.05).state_dict(), path)
    return {
        "paths.state_model": str(path),
        "paths.state_model_cells": str(STATE_TABLES / "cells.json"),
        "paths.physics_prior": str(STATE_TABLES / "physics_prior.json"),
    }
