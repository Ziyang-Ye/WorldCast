#!/usr/bin/env python3
"""Replay a recorded client through the live path: the demo's inference in evaluation form.

The client of ``--index-row`` plays its recording's own controls as a live player would
(``player_state.source: live``, nobody else in its world, no lockstep) and writes
``<out_dir>/latents.npy`` ([N, 48, 24, 42] float32, N = ``run.latent_frames`` clipped to the
recording, as ``tools/run_client.py`` writes it) and ``<out_dir>/replay.json`` (the settings it ran
with). Without ``--demo`` it runs with the settings of the evaluation (``model.attention``,
``scene_state.bound`` and the rest of the config as given), so its latents are those of the
evaluation's engine on the controls of a live player; with ``--demo`` it runs with the inference
settings of the demo's worker (``configs/demo/worldcast.yaml``) and says which of the config's
settings they replace. ``run.latent_frames`` is the length of the session replayed; a live client
keeps a round of 29 latent frames (45 of the generator trained without scene state) and slides it.
Decode the latents with ``tools/decode.py``.

Example, the first client of an example case, in evaluation form and as the demo serves it::

    python tools/replay_live.py --config weights/paths.yaml \\
        --config examples/data/mirage_r16/config.yaml --index-row 0 --out-dir runs/replay/exact
    python tools/replay_live.py --config weights/paths.yaml \\
        --config examples/data/mirage_r16/config.yaml --index-row 0 --out-dir runs/replay/demo \\
        --demo configs/demo/worldcast.yaml

A ``--set`` names a setting of both runs; ``run.latent_frames`` is the session's alone.
"""

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from worldcast.config.inference import InferenceConfig, load_config
from worldcast.config.loader import (
    add_config_args,
    config_to_dict,
    differences,
    parse_overrides,
    usage_errors,
)
from worldcast.data import load_round_index_row
from worldcast.data.window import DataPaths
from worldcast.demo.config import load_config as load_demo_config
from worldcast.demo.worldcast_client import inference_config
from worldcast.engine.inference.client import Client
from worldcast.engine.inference.live_rollout import live_round_latents
from worldcast.engine.inference.loading import load_window
from worldcast.engine.inference.replay import replay_latents
from worldcast.engine.inference.serving import ServingOptions


def live_settings(
    cfg: InferenceConfig, overrides: dict, demo: str | None, configs: Sequence[str]
) -> tuple[InferenceConfig, ServingOptions]:
    """The inference config and serving options of the live client that replays ``cfg``'s
    client: the evaluation's settings made live, or the worker's of the demo config ``demo``
    (``overrides`` over either, but the session's length)."""
    kept = {key: value for key, value in overrides.items() if key != "run.latent_frames"}
    if demo is None:
        round_latents = live_round_latents(cfg.model.scene_state)
        live = cfg.with_overrides(
            {"player_state.source": "live", "run.latent_frames": round_latents}
        )
        return live.with_overrides(kept), ServingOptions(decoder="none", lockstep=False)
    config = load_demo_config([demo])
    worker = replace(config.client.worldcast, configs=list(configs))
    live = inference_config(worker, config.scene_state).with_overrides(kept)
    return live, replace(worker.serving_options(), decoder="none")


def main(argv: Sequence[str] | None = None) -> int:
    """Replay the client of ``--index-row``; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--index-row", type=int, help="the client's row of the round index")
    parser.add_argument("--out-dir", help="where latents.npy and replay.json are written")
    parser.add_argument("--device", help="torch device, e.g. cuda or cuda:1")
    parser.add_argument(
        "--demo",
        metavar="CONFIG",
        help="the demo's config: replay with its worker's inference settings, not the evaluation's",
    )
    args = parser.parse_args(argv)

    with usage_errors(parser):
        overrides = parse_overrides(args.overrides)
        for key, value in (
            ("run.index_row", args.index_row),
            ("paths.out_dir", args.out_dir),
            ("run.device", args.device),
        ):
            if value is not None:
                overrides[key] = value
        cfg = load_config(args.config, overrides)
        cfg.paths.require("round_index", "out_dir", *DataPaths.names())
        if cfg.run.index_row is None:
            raise ValueError("run.index_row is not set: pass --index-row")
        live, serving = live_settings(cfg, overrides, args.demo, args.config)
        if live.player_state.source != "live":
            raise ValueError(
                f"a replay plays the live path, not player_state.source {live.player_state.source}"
            )
        live.paths.require("state_model", "state_model_cells", "physics_prior")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    row = load_round_index_row(cfg.paths.round_index, cfg.run.index_row)
    window = load_window(cfg, row)
    latents = replay_latents(Client(live, serving), row, window)
    out = Path(cfg.paths.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "latents.npy", latents)
    changed = differences(cfg, live)
    summary = dict(
        media_id=row.media_id,
        index_row=cfg.run.index_row,
        seed=cfg.run.seed,
        latent_frames=len(latents),
        demo=args.demo,
        config=config_to_dict(live),
        serving=asdict(serving),
        differs_from_config={key: list(pair) for key, pair in changed.items()},
    )
    (out / "replay.json").write_text(json.dumps(summary, indent=1))
    print(
        f"{row.media_id}: {len(latents)} latents -> {out} (live"
        f"{', demo settings' if args.demo else ''}; settings that differ from the config:"
        f" {', '.join(changed) or 'none'}; total {time.time() - t0:.0f} s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
