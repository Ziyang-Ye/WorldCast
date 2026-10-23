#!/usr/bin/env python3
"""Make the client weight file of the four-step generator trained without scene state (stage
4_noscene; not the paper's client), for a client with ``model.scene_state: false``.

Reads ``--checkpoint``: the generator's weights under its own key names (a ``.safetensors`` file),
or a training checkpoint of the stage (its EMA entry). Keeps every weight of the generator the stage
builds (``worldcast.engine.stage.generator_config``: the backbone, the controls and the state
injector), cast to bf16 as the released generator is, and leaves out the weights of the modules it
does not build (the ray and the observer-signal embeddings, the visibility probe), which a
checkpoint may carry unused; a weight of anything else, a missing weight or another shape is an
error, and nothing is written. Writes, under ``--out-dir``:

* ``worldcast/`` + ``worldcast.modeling.build.NOSCENE_FILE``: the generator's weights, bf16, under
  its key names, with the metadata ``format``, ``stage`` and the checkpoint's ``step``;
* ``noscene.yaml``: ``paths.checkpoint`` (that file) and ``model.scene_state: false``, a config to
  give after ``weights/paths.yaml`` (``configs/demo/worldcast_noscene.yaml`` names it);

and prints, as JSON, what it kept, what it left out per module (tensors, parameters, the largest
absolute weight: the weights of a module the stage did not train are its initial ones) and the
stage and step the checkpoint names, if any (the keys alone do not tell this generator from the
released one, whose file holds the same modules). ``--wan22-root`` gives the backbone's dimensions
(default: the 5B model's)::

    python tools/make_noscene_generator.py --checkpoint <the stage's generator> --out-dir weights
    python tools/run_client.py --config weights/paths.yaml --config weights/noscene.yaml \
        --config examples/data/mirage_r16/config.yaml --index-row 0 \
        --world-state-dir runs/noscene/world_state --out-dir runs/noscene
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from worldcast.config.training import paper_config
from worldcast.engine.stage import generator_config
from worldcast.modeling.build import EMA_KEY, NOSCENE_FILE, client_weights, read_generator
from worldcast.utils.files import sha256_file
from worldcast.utils.weights import is_safetensors, read_state_dict

#: The training stage of the generator.
STAGE = "4_noscene"


def checkpoint_names(path: Path) -> dict[str, str]:
    """The ``stage`` and ``step`` a checkpoint names: a ``.safetensors`` file's metadata, a training
    checkpoint's entries; those it does not name are left out."""
    if is_safetensors(path):
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            named = handle.metadata() or {}
    else:
        named = read_state_dict(path)
    return {key: str(named[key]) for key in ("stage", "step") if named.get(key) is not None}


def main(argv: Sequence[str] | None = None) -> int:
    """Make the client weight file and its config; returns the exit status."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="the generator of stage 4_noscene (fp32 weights)"
    )
    parser.add_argument("--out-dir", required=True, help="receives worldcast/ and noscene.yaml")
    parser.add_argument("--wan22-root", help="the Wan2.2-TI2V-5B snapshot (its config.json)")
    args = parser.parse_args(argv)
    source = Path(args.checkpoint)
    if not source.is_file():
        parser.error(f"--checkpoint: no such file {source}")

    config = generator_config(paper_config(STAGE, {"model.wan22_root": args.wan22_root}))
    original = read_generator(source, EMA_KEY)
    try:
        state, dropped = client_weights(original, config)
    except (KeyError, ValueError) as error:
        print(f"{source}: not the generator of stage {STAGE}: {error}", file=sys.stderr)
        return 1
    left_out = {
        module: dict(
            tensors=len(keys),
            parameters=sum(original[key].numel() for key in keys),
            max_abs=max(float(original[key].abs().max()) for key in keys),
        )
        for module, keys in dropped.items()
    }
    del original
    out_dir = Path(args.out_dir)
    out = out_dir / "worldcast" / NOSCENE_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    named = checkpoint_names(source)
    metadata = {"format": "pt", "stage": STAGE}
    if "step" in named:
        metadata["step"] = named["step"]
    from safetensors.torch import save_file

    save_file(state, str(out), metadata=metadata)
    settings = {"paths": {"checkpoint": str(out.resolve())}, "model": {"scene_state": False}}
    (out_dir / "noscene.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))

    report = dict(
        checkpoint=str(source),
        checkpoint_stage=named.get("stage"),
        step=named.get("step"),
        out=str(out),
        sha256=sha256_file(out),
        tensors=len(state),
        parameters=sum(value.numel() for value in state.values()),
        dtype=str(torch.bfloat16),
        left_out=left_out,
    )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
