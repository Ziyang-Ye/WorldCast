#!/usr/bin/env python3
"""Check a release run against the paper run's fingerprints (cells d57-d92 of ``tests/reference``).

The table of fingerprints is read from ``tests/reference/test_reference_anchors.py`` (one copy). A fingerprint is the
first 32 hex digits of the sha256 of a float32 tensor's bytes. Bit-equality needs an NVIDIA H20, torch 2.9.1+cu128
and the paper's flash-attention build (see that test's docstring).

``prefix``: run one client of a cell up to the end of its plain prefix (latents 0-24: generator and sampler, no
scene state, no peers) and compare the entry noise, the initial latent and the plain prefix. One GPU, a few
minutes; the client code path is the real one (``worldcast.engine.inference.client.Client``), stopped right after
``Sampler.rollout_prefix``. Also writes ``prefix.npy`` and per-block fingerprints, for bisecting::

    python tools/verify_reference.py prefix --config configs/infer/worldcast_4step.yaml --config ref.yaml \\
        --cell d59 --out-dir runs/verify/d59-prefix

``check``: fingerprint the ``latents.npy`` of finished sessions (``tools/run_session.py``) and compare the plain
prefix and the final latents of every reference cell found under the given directories::

    python tools/verify_reference.py check runs/reference

Both modes print one line per fingerprint, write ``report.json`` (``--report``) and exit 1 on any difference.
"""

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # run from a checkout without installing the package
    sys.path.insert(0, str(REPO))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402

ANCHORS = REPO / "tests" / "reference" / "test_reference_anchors.py"


def reference_table(path: Path = ANCHORS) -> dict[str, Any]:
    """``SEED``, ``CELLS``, ``ROUNDS`` and ``PLAIN_PREFIX`` of the reference test, without importing it (no pytest)."""
    wanted = {"SEED", "CELLS", "ROUNDS", "PLAIN_PREFIX"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and {
            t.id
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(t, ast.Name)
        }
        & wanted
    ]
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
        {"dict": dict},
        namespace,
    )
    missing = wanted - set(namespace)
    if missing:
        raise RuntimeError(f"{path} does not define {sorted(missing)}")
    return namespace


def fingerprint(array) -> str:
    """``memory_deploy.tensor_fingerprint``: sha256 of the contiguous float32 bytes, first 32 hex digits."""
    import numpy as np

    a = (
        array.detach().float().cpu().numpy()
        if hasattr(array, "detach")
        else np.asarray(array, dtype=np.float32)
    )
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:32]


def environment() -> dict[str, Any]:
    """What decides the bits: torch / CUDA / GPU / attention kernel / TF32."""
    import platform

    import torch

    from worldcast.modeling.wan22.attention import flash_attention_backend

    env: dict[str, Any] = dict(
        python=platform.python_version(),
        machine=platform.machine(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        attention=flash_attention_backend(),
    )
    for name in ("flash_attn", "flash_attn_interface"):
        try:
            env[name] = getattr(__import__(name), "__version__", "present")
        except ImportError:
            env[name] = None
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    env["tf32"] = dict(
        matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32
    )
    return env


def _line(name: str, got: str, want: str | None) -> bool:
    ok = want is not None and got == want
    print(
        f"  {'MATCH' if ok else 'DIFF ' if want else 'NOREF'}  {name:22s} {got}"
        + ("" if ok else f"  (paper {want})"),
        flush=True,
    )
    return ok


class _PrefixDone(Exception):
    pass


def run_prefix(args) -> int:
    import numpy as np
    import torch

    from worldcast.data.index import load_round_index_row
    from worldcast.data.latents import load_first_latent
    from worldcast.engine.inference.client import Client
    from worldcast.modeling.wan22.attention import sdpa_attention

    table = reference_table()
    if args.cell not in table["CELLS"]:
        raise SystemExit(f"unknown cell {args.cell}; known: {sorted(table['CELLS'])}")
    row_index, media_id, latents, want = table["CELLS"][args.cell]
    out = Path(args.out_dir)
    pool = out / "pool"
    if pool.exists() and any(pool.iterdir()):
        raise SystemExit(f"{pool} is not empty: use a fresh --out-dir")
    overrides = parse_overrides(args.overrides)
    overrides.update(
        {
            "run.index_row": row_index,
            "run.latents": latents,
            "run.seed": table["SEED"],
            "paths.live_pool_dir": str(pool),
            "paths.out_dir": str(out / "client"),
        }
    )
    cfg = load_cli_config(args.config, overrides)
    row = load_round_index_row(cfg.paths.round_index, row_index)
    if row.media_id != media_id:
        raise SystemExit(
            f"row {row_index} of {cfg.paths.round_index} is {row.media_id}, not {media_id}: "
            "paths.round_index must be the paper's wholeround_index.jsonl"
        )

    report: dict[str, Any] = dict(
        mode="prefix",
        cell=args.cell,
        media_id=media_id,
        row=row_index,
        latents=latents,
        config=[str(c) for c in args.config or []],
        checkpoint=cfg.paths.checkpoint,
        prompt_embedding=cfg.paths.prompt_embedding,
    )
    print(f"{args.cell} ({media_id}, row {row_index}, N = {latents})", flush=True)

    g = torch.Generator(device="cpu").manual_seed(int(table["SEED"]))
    noise = torch.randn(
        (
            1,
            latents - 1,
            cfg.model.latent_channels,
            cfg.model.latent_height,
            cfg.model.latent_width,
        ),
        generator=g,
    )
    sink = load_first_latent(cfg.paths.latent_cache_root, media_id, row.start_frame)[None].to(
        torch.bfloat16
    )
    got = dict(noise=fingerprint(noise), initial=fingerprint(sink))
    del noise

    captured: dict[str, Any] = {}

    class PrefixOnly(Client):
        def _sampler(self, generator):
            sampler = super()._sampler(generator)
            inner = sampler.rollout_prefix

            def rollout_prefix(*a, **k):
                captured["environment"] = environment()  # as the client set it up (TF32 on)
                t0 = time.time()
                captured["prefix"] = inner(*a, **k)
                captured["seconds"] = time.time() - t0
                raise _PrefixDone

            sampler.rollout_prefix = rollout_prefix
            return sampler

    t0 = time.time()
    try:
        PrefixOnly(cfg, attention=sdpa_attention if args.attention == "sdpa" else None).run()
    except _PrefixDone:
        pass
    prefix = captured["prefix"][0].float().cpu().numpy()  # [25, C, H, W], latent 0 = the sink
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "prefix.npy", prefix)
    got["prefix"] = fingerprint(prefix)
    blocks = {f"{s}-{s + 3}": fingerprint(prefix[s : s + 4]) for s in range(1, prefix.shape[0], 4)}
    report["environment"] = captured["environment"]
    print(f"  environment: {json.dumps(captured['environment'])}", flush=True)
    print(
        f"  plain prefix: {captured['seconds']:.1f} s (client total {time.time() - t0:.0f} s)",
        flush=True,
    )
    ok = all([_line(k, got[k], want.get(k)) for k in ("noise", "initial", "prefix")])
    for name, fp in blocks.items():
        print(f"         prefix latents {name:7s} {fp}", flush=True)
    report.update(
        fingerprints=got,
        paper={k: want.get(k) for k in got},
        prefix_blocks=blocks,
        match=ok,
        prefix_seconds=round(captured["seconds"], 2),
    )
    Path(args.report or out / "report.json").write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


def run_check(args) -> int:
    import numpy as np

    table = reference_table()
    by_media = {cell: v for cell, v in table["CELLS"].items()}
    found: list[dict[str, Any]] = []
    for root in args.dirs:
        for cell, (_, media_id, latents, want) in sorted(by_media.items()):
            for path in sorted(Path(root).glob(f"**/{media_id}/latents.npy")):
                lat = np.load(path, mmap_mode="r")
                got = dict(prefix=fingerprint(lat[: table["PLAIN_PREFIX"]]), final=fingerprint(lat))
                print(f"{cell} {path} {lat.dtype} {tuple(lat.shape)}", flush=True)
                ok_shape = lat.dtype == np.float32 and tuple(lat.shape) == (latents, 48, 24, 42)
                if not ok_shape:
                    print(
                        f"  DIFF   shape/dtype (paper float32 {(latents, 48, 24, 42)})", flush=True
                    )
                ok = all([ok_shape] + [_line(k, got[k], want.get(k)) for k in ("prefix", "final")])
                found.append(
                    dict(
                        cell=cell,
                        path=str(path),
                        fingerprints=got,
                        paper={k: want.get(k) for k in got},
                        match=ok,
                    )
                )
    if not found:
        raise SystemExit(f"no reference cell's latents.npy under {args.dirs}")
    n_ok = sum(f["match"] for f in found)
    print(f"{n_ok}/{len(found)} cells match the paper bit for bit", flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(dict(mode="check", cells=found), indent=1))
    return 0 if n_ok == len(found) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser(
        "prefix", help="run one client's plain prefix and compare noise / initial / prefix"
    )
    add_config_args(p)
    p.add_argument("--cell", default="d59", help="reference cell (d57 ... d92; default d59)")
    p.add_argument(
        "--out-dir", required=True, help="fresh directory: pool, prefix.npy, report.json"
    )
    p.add_argument("--attention", choices=("flash", "sdpa"), default="flash")
    p.add_argument("--report", help="report path (default <out-dir>/report.json)")
    c = sub.add_parser("check", help="compare the latents.npy of finished sessions")
    c.add_argument("dirs", nargs="+", help="session output directories (searched recursively)")
    c.add_argument("--report", help="write a JSON report here")
    args = parser.parse_args(argv)
    return run_prefix(args) if args.mode == "prefix" else run_check(args)


if __name__ == "__main__":
    sys.exit(main())
