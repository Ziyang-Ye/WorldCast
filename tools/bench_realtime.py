#!/usr/bin/env python
"""Latency budget of one WorldCast client on a GPU: generation, decoding, encoding and a paced real-time run.

    python tools/bench_realtime.py --config configs/infer/worldcast_4step.yaml --config paths.yaml \
        --row 59 --blocks 24 --taehv weights/taew2_2.pth --out runs/bench

Sections (``--sections``, default all but ``session``):

* ``generation``: the same session (one client, recorded tracks, no peers) under each generator setting; per block
  the GPU time of the prefill, each ladder step and the field (CUDA events), the host time of the scene-state work;
  whether the latents equal the release generator's bit for bit.
* ``decode``: the paper decode (Wan2.2 VAE, chunks of 8) against the streamed decoders (Wan2.2 per latent frame,
  tiny ``taew2_2``): time per latent frame and per block, frames equal / PSNR against the paper decode.
* ``encode``: JPEG (nvJPEG, libjpeg) and H.264 (libx264 zerolatency) per frame.
* ``loop``: paced real-time runs of the engine (before / after), keypress-to-photon and frames per second.
* ``session``: ``--clients N`` engines in lock-step on N GPUs (one process each), lock-step waits.

Writes ``<out>/bench.json``.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldcast.config.loader import add_config_args, load_cli_config, parse_overrides  # noqa: E402
from worldcast.data.index import load_round_index_row  # noqa: E402
from worldcast.engine.realtime.config import RealtimeConfig  # noqa: E402
from worldcast.engine.realtime.engine import Engine, EngineModels  # noqa: E402
from worldcast.engine.realtime.latency import (  # noqa: E402
    LatencyModel,
    keypress_to_photon,
    stats,
    throughput,
)

SECTIONS = ("generation", "decode", "encode", "loop", "subblock")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def environment() -> dict:
    from worldcast.modeling.wan22.attention import flash_attention_backend

    return dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),
        attention=flash_attention_backend(),
        host=os.uname().nodename,
    )


# ----------------------------------------------------------------------------------------------- generation
GENERATION = {
    "release": dict(),
    "fast": dict(generator="fast"),
    "fast_graphs": dict(generator="fast", cuda_graphs=True),
    "fast_graphs_cudnn": dict(generator="fast", cuda_graphs=True, attention="cudnn"),
    "fast_graphs_compile": dict(generator="fast", cuda_graphs=True, compile=True),
    "fast_graphs_fa3": dict(generator="fast", cuda_graphs=True, attention="fa3"),
    "fast_graphs_compile_fa3": dict(
        generator="fast", cuda_graphs=True, compile=True, attention="fa3"
    ),
    "fast_graphs_fp8": dict(generator="fast", cuda_graphs=True),
}
#: variants whose generator weights are converted (a copy of the loaded generator)
WEIGHTS = {"fast_graphs_fp8": "fp8"}


def fp8_generator(generator):
    """A copy of the generator whose DiT linear layers run in FP8 (torchao: dynamic per-row activation scales,
    per-row weight scales); the modulation, the AdaLN adapters, the embeddings and the head stay bf16.
    """
    import copy

    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

    g = copy.deepcopy(generator)
    targets = {
        f"blocks.{i}.{name}"
        for i in range(len(g.blocks))
        for name in (
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            "cross_attn.q",
            "cross_attn.o",
            "ffn.0",
            "ffn.2",
        )
    }
    quantize_(
        g,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        filter_fn=lambda module, fqn: fqn in targets,
    )
    for name, module in g.named_modules():  # row-wise FP8 GEMMs take bf16 in (autocast's cast)
        if name in targets:
            module.register_forward_pre_hook(
                lambda m, args: (args[0].to(torch.bfloat16),) + tuple(args[1:])
            )
    return g


def run_session(cfg, models, rt, row, *, controls=None, pace=None, profile=True):
    """Run one session to its end; returns (engine records, frames, latents [N, ...] float32 CPU, seconds)."""
    engine = Engine(cfg, rt, models=models, profile=profile)
    engine.start(row=row)
    frames, t0 = [], time.monotonic()
    k = 0
    while not engine.finished:
        not_before = pace(engine, frames) if pace is not None else None
        frames.extend(engine.step(controls, not_before=not_before))
        k += 1
    seconds = time.monotonic() - t0
    latents = engine.latents[0].float().cpu()
    records = list(engine.records)
    gen = engine._generator
    engine.stop()
    stats_ = dict(getattr(gen, "stats", {}))
    del engine, gen
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return records, frames, latents, seconds, stats_


def block_rows(records, warmup: int = 3):
    """Per reconstituted block (after ``warmup`` of them): the timing fields, ms."""
    rows = []
    recon = [r for r in records if r.kind == "reconstituted"][warmup:]
    for r in recon:
        g, h = r.gpu_ms, r.host_ms
        row = dict(
            s=r.s,
            prefill=g.get("prefill", 0.0),
            ladder=g.get("ladder", 0.0),
            field=g.get("field", 0.0),
            ladder_wall=1e3 * (r.t_x0 - r.t_ladder),
        )
        for k in range(1, 5):
            row[f"step{k}"] = g.get(f"step{k}", 0.0)
        for k in range(1, 6):
            row[f"prefill_call{k}"] = g.get(f"prefill{k}", 0.0)
        row.update({k: v for k, v in h.items()})
        row["window"] = (r.read or {}).get("window")
        rows.append(row)
    return rows


def summarize_rows(rows):
    keys = sorted({k for r in rows for k in r if k not in ("s", "window")})
    return {k: stats([r[k] for r in rows if r.get(k) is not None]) for k in keys}


def section_generation(cfg, models, row, out, variants):
    res = {}
    ref = None
    for name in variants:
        rt = RealtimeConfig(decoder="none", lockstep=False, **GENERATION[name])
        log("generation", name, rt)
        try:
            use = models
            if WEIGHTS.get(name) == "fp8":
                use = dataclasses.replace(models, generator=fp8_generator(models.generator))
            records, _, latents, seconds, gstats = run_session(cfg, use, rt, row)
            del use
        except Exception as exc:  # noqa: BLE001 - a failed option is a result, not a crash
            traceback.print_exc()
            res[name] = dict(error=f"{type(exc).__name__}: {exc}")
            continue
        rows = block_rows(records)
        entry = dict(
            seconds=seconds,
            blocks=len(records),
            generator_stats=gstats,
            per_block=summarize_rows(rows),
            rows=rows,
        )
        if ref is None:
            ref = latents
            np.save(out / "latents_release.npy", latents.numpy())
        else:
            diff = (latents - ref).abs()
            per_block = [float(diff[s : s + 4].max()) for s in range(1, latents.shape[0], 4)]
            entry.update(
                bit_identical=bool(torch.equal(latents, ref)),
                max_abs_diff=float(diff.max()),
                first_diff_block=next((1 + 4 * i for i, d in enumerate(per_block) if d > 0), None),
            )
            if not entry["bit_identical"]:
                np.save(out / f"latents_{name}.npy", latents.numpy())
        res[name] = entry
        pb = entry["per_block"]
        log(
            name,
            json.dumps({k: v for k, v in entry.items() if k not in ("rows", "per_block")}),
            "prefill p50 {:.1f} ladder p50 {:.1f}".format(
                pb.get("prefill", {}).get("p50", 0.0), pb.get("ladder", {}).get("p50", 0.0)
            ),
        )
    if ref is not None and models.wan_vae is not None:
        inexact = [n for n, e in res.items() if e.get("bit_identical") is False]
        decoded = {"exact": paper_decode(models, ref)}
        for n in inexact:
            decoded[n] = paper_decode(models, torch.from_numpy(np.load(out / f"latents_{n}.npy")))
            res[n]["quality"] = compare_videos(decoded["exact"], decoded[n])
            log("quality", n, json.dumps(res[n]["quality"], default=float)[:600])
        res["exact_quality"] = frame_quality(decoded["exact"][97:], [])
        save_grids(decoded, out, "generation")
    return res


def paper_decode(models, latents: torch.Tensor) -> torch.Tensor:
    """uint8 ``[T, 3, H, W]`` frames of latents ``[N, 48, 24, 42]`` with the paper decode (Wan2.2, chunks of 8)."""
    from worldcast.engine.inference.decode import decode_frames

    lat = latents[None].to("cuda", torch.bfloat16)
    return torch.from_numpy(np.stack(list(decode_frames(models.wan_vae, lat, chunk=8)))).permute(
        0, 3, 1, 2
    )


def compare_videos(exact: torch.Tensor, other: torch.Tensor) -> dict:
    """PSNR of ``other`` against ``exact`` per second (the rollouts drift apart, so later seconds measure divergence,
    not quality) and the frame statistics of ``other`` after the plain prefix."""
    t = min(exact.shape[0], other.shape[0])
    by_second = [
        float(np.mean([psnr(exact[i], other[i]) for i in range(a, min(a + 16, t))]))
        for a in range(1, t, 16)
    ]
    return dict(psnr_db_by_second=by_second, frames=frame_quality(other[97:t], []))


def check_patch_linear(models) -> dict:
    from worldcast.engine.realtime.fast import patchify_linear

    g = models.generator
    x = torch.randn(1, 48, 4, 24, 42, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = g.patch_embedding(x.permute(0, 1, 2, 3, 4))
        b = patchify_linear(g.patch_embedding, x)
    return dict(
        bit_identical=bool(torch.equal(a, b)),
        max_abs_diff=float((a.float() - b.float()).abs().max()),
    )


# ----------------------------------------------------------------------------------------------- decode
def cuda_ms(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    e.synchronize()
    return out, float(s.elapsed_time(e))


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.float() - b.float()) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def section_decode(cfg, models, latents, out):
    from worldcast.engine.inference.decode import decode_frames
    from worldcast.engine.realtime.frames import TinyFrameDecoder, WanFrameDecoder

    res = {}
    lat = latents[None].to("cuda", torch.bfloat16)  # the bf16 output buffer, as latents.npy holds
    n = int(lat.shape[1])
    log("decode: paper decode (Wan2.2, chunk 8) of", n, "latents")
    t0 = time.monotonic()
    ref = torch.from_numpy(
        np.stack(list(decode_frames(models.wan_vae, lat, chunk=8)))
    )  # [T, H, W, 3] uint8
    torch.cuda.synchronize()
    res["wan_chunk8_total_s"] = time.monotonic() - t0
    ref = ref.permute(0, 3, 1, 2).contiguous()

    def streamed(decoder, chunk):
        decoder.reset()
        frames, per_call = [], []
        starts = [0] + list(range(1, n, chunk))
        for i, a in enumerate(starts):
            b = 1 if a == 0 else min(a + chunk, n)
            f, ms = cuda_ms(lambda: decoder.decode(lat[0, a:b]))
            frames.append(f.cpu())
            if a > 0:
                per_call.append(ms)
        return torch.cat(frames), per_call

    for name, dec in (
        ("wan", WanFrameDecoder(models.wan_vae)),
        ("taehv", TinyFrameDecoder(models.tiny)),
    ):
        if dec is None:
            continue
        for chunk in (1, 4):
            streamed(dec, chunk)  # warm-up (cuDNN autotune, allocator)
            frames, per_call = streamed(dec, chunk)
            key = f"{name}_chunk{chunk}"
            equal = (frames == ref).flatten(1).all(1)
            entry = dict(
                ms_per_call=stats(per_call[2:]),
                latents_per_call=chunk,
                frames=int(frames.shape[0]),
                frames_equal_paper=int(equal.sum()),
                psnr_db=stats([psnr(frames[i], ref[i]) for i in range(frames.shape[0])]),
            )
            res[key] = entry
            log("decode", key, json.dumps(entry))
            if name == "taehv" and chunk == 1:
                torch.save(dict(paper=ref[::40], taehv=frames[::40]), out / "decode_samples.pt")
    return res, ref


# ----------------------------------------------------------------------------------------------- encode
def section_encode(frames: torch.Tensor):
    from worldcast.engine.realtime.frames import H264Encoder, JpegEncoder

    res = {}
    sample = frames[100:164]
    for name, enc, batch in (
        ("jpeg_gpu_q90", JpegEncoder(90, "cuda"), 1),
        ("jpeg_gpu_q90_x4", JpegEncoder(90, "cuda"), 4),
        ("jpeg_cpu_q90", JpegEncoder(90, "cpu"), 1),
        ("jpeg_gpu_q75", JpegEncoder(75, "cuda"), 1),
        ("h264_x264_zerolatency", H264Encoder(672, 384), 1),
    ):
        src = sample.cuda() if "gpu" in name else sample
        times, sizes = [], []
        for i in range(0, src.shape[0], batch):
            t0 = time.perf_counter()
            data = enc.encode(src[i : i + batch])
            times.append(1e3 * (time.perf_counter() - t0) / batch)
            sizes.extend(len(d) for d in data)
        res[name] = dict(ms_per_frame=stats(times[4:]), bytes_per_frame=stats(sizes))
        log("encode", name, json.dumps(res[name]))
    return res


# ----------------------------------------------------------------------------------------------- loop
LOOP = {
    "before": dict(decoder="wan", encoder="jpeg"),
    "after_wan": dict(
        generator="fast",
        cuda_graphs=True,
        early_prefill=True,
        decode_overlap=True,
        decoder="wan",
        encoder="jpeg",
    ),
    "after_taehv": dict(
        generator="fast",
        cuda_graphs=True,
        early_prefill=True,
        decode_overlap=True,
        decoder="taehv",
        encoder="jpeg",
    ),
    "after_taehv_fast": dict(
        generator="fast",
        cuda_graphs=True,
        compile=True,
        attention="fa3",
        early_prefill=True,
        decode_overlap=True,
        decoder="taehv",
        encoder="jpeg",
    ),
    "after_taehv_closed": dict(
        generator="fast",
        cuda_graphs=True,
        early_prefill=True,
        decode_overlap=True,
        decoder="taehv",
        encoder="jpeg",
    ),
}
#: loop variants that run the closed loop (player_state.source = predicted) with a state model of the paper's size
CLOSED = ("after_taehv_closed",)
STATE_TABLES = Path(__file__).resolve().parents[1] / "configs" / "state_model"


def closed_loop_config(cfg, state_model: str):
    """``cfg`` under the closed loop with ``state_model``; a random state model of the paper's size is written there
    first if the file does not exist (timing only: its positions are meaningless)."""
    from worldcast.config.inference import with_overrides

    if not Path(state_model).exists():
        from worldcast.modeling.state_model import StateModel, StateTables

        torch.manual_seed(0)
        model = StateModel(
            StateTables.load(STATE_TABLES / "cells_v1.json", STATE_TABLES / "map_norm_v1.json")
        )
        with torch.no_grad():
            for prm in model.parameters():
                prm.mul_(0.1)
        torch.save(model.state_dict(), state_model)
    return with_overrides(
        cfg,
        {
            "player_state.source": "predicted",
            "paths.state_model": state_model,
            "paths.state_model_cells": str(STATE_TABLES / "cells_v1.json"),
            "paths.state_model_map_norm": str(STATE_TABLES / "map_norm_v1.json"),
            "paths.physics_prior": str(STATE_TABLES / "physics_prior_v1.json"),
        },
    )


class Pacer:
    """Start each ladder as late as the display allows: the block's first frame should be ready just when the
    display reaches it. ``lead`` = recent worst (first frame ready - controls taken) + margin."""

    def __init__(self, margin_s: float = 0.03, network_s: float = 0.005) -> None:
        self.margin, self.network = margin_s, network_s

    def __call__(self, engine, frames):
        recs = [r for r in engine.records if r.t_first_frame and r.kind == "reconstituted"]
        if len(recs) < 2 or not frames:
            return None
        lead = max(r.t_first_frame - r.t_sample for r in recs[-4:]) + self.margin
        from worldcast.engine.realtime.latency import display_times

        shown = display_times([f.t_ready for f in frames], network_s=self.network)
        next_slot = shown[-1] + 1.0 / 16
        return next_slot - self.network - lead


def measure(records, frames, *, warmup: int = 3) -> dict:
    """Keypress-to-photon, frame rate and the per-generation spans of a served session (after ``warmup``
    reconstituted generations)."""
    recs = [r for r in records if r.kind != "sink"]
    first = {}
    for i, f in enumerate(frames):
        first.setdefault(f.block, i)
    samples = [r.t_sample for r in recs]
    firsts = [first[r.s] for r in recs]
    ready = [f.t_ready for f in frames]
    k = next(i for i, r in enumerate(recs) if r.kind == "reconstituted") + warmup
    recon = recs[k:]
    return dict(
        keypress_to_photon=keypress_to_photon(
            samples[k - 2 :], firsts[k - 2 :], ready, model=LatencyModel(), skip_blocks=2
        ),
        throughput=throughput(firsts[k:], [r.t_first_frame for r in recon]),
        sample_to_first_frame_ms=stats([1e3 * (r.t_first_frame - r.t_sample) for r in recon]),
        sample_to_last_frame_ms=stats([1e3 * (r.t_last_frame - r.t_sample) for r in recon]),
        ladder_wall_ms=stats([1e3 * (r.t_x0 - r.t_ladder) for r in recon]),
        generation_period_ms=stats(np.diff([r.t_sample for r in recon]) * 1e3),
        lockstep_ms=stats([r.host_ms.get("lockstep_ms", 0.0) for r in recon]),
        prepare_host_ms=stats([sum(r.host_ms.values()) for r in recon]),
    )


def section_loop(cfg, models, row, taehv_path, variants, state_model=None):
    res = {}
    base = cfg
    for name in variants:
        cfg = closed_loop_config(base, state_model) if name in CLOSED else base
        kw = dict(LOOP[name])
        if kw.get("decoder") == "taehv":
            kw["taehv_path"] = taehv_path
        rt = RealtimeConfig(lockstep=False, **kw)
        live = name != "before"
        log("loop", name, rt)
        try:
            records, frames, latents, seconds, _ = run_session(
                cfg,
                models,
                rt,
                row,
                controls=(lambda: None) if live else None,
                pace=Pacer() if live else None,
                profile=False,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res[name] = dict(error=f"{type(exc).__name__}: {exc}")
            continue
        entry = dict(
            seconds=seconds,
            frames=len(frames),
            **measure(records, frames),
            latents_sha=__import__("hashlib").sha256(latents.numpy().tobytes()).hexdigest()[:16],
        )
        res[name] = entry
        log("loop", name, json.dumps(entry))
    return res


# ----------------------------------------------------------------------------------------------- sub-block
SUBBLOCK = {"block4": (4, 4), "commit2": (2, 4), "commit1": (1, 4), "target1": (1, 1)}


def frame_quality(frames: torch.Tensor, gen_starts) -> dict:
    """Sharpness (variance of the Laplacian of the luma) and frame-to-frame change (mean |frame - previous|) of
    uint8 ``[T, 3, H, W]`` frames that start at a latent frame (latent k gives frames 4k .. 4k + 3). The change is
    split by where it happens: into the first frame of a generation, into the first frame of another latent, or
    inside a latent. ``gen_starts``: frame indices where a generation's frames begin."""
    import torch.nn.functional as F

    luma = (0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]).float()[:, None]
    lap = F.conv2d(luma, torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3))
    sharp = lap.flatten(1).var(dim=1)
    step = (
        (frames[1:].float() - frames[:-1].float()).abs().mean(dim=(1, 2, 3))
    )  # change into frame i + 1
    t = int(frames.shape[0])
    gen = {int(i) for i in gen_starts if 0 < int(i) < t}
    latent = {i for i in range(4, t, 4)} - gen
    within = [i for i in range(1, t) if i not in gen and i not in latent]

    def at(idx):
        return stats([float(step[i - 1]) for i in sorted(idx)])

    return dict(
        sharpness=stats(sharp.tolist()),
        change_into_generation=at(gen),
        change_into_latent=at(latent),
        change_within_latent=at(within),
        sharpness_by_second=[float(sharp[i : i + 16].mean()) for i in range(0, t, 16)],
    )


def section_subblock(cfg, models, row, taehv_path, out, modes):
    from worldcast.engine.inference.decode import decode_frames

    res, decoded = {}, {}
    for name in modes:
        commit, target = SUBBLOCK[name]
        rt = RealtimeConfig(
            **dict(LOOP["after_taehv"], taehv_path=taehv_path),
            lockstep=False,
            commit_latents=commit,
            target_latents=target,
        )
        log("subblock", name, rt)
        try:
            records, frames, latents, seconds, _ = run_session(
                cfg, models, rt, row, controls=lambda: None, pace=Pacer(), profile=False
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res[name] = dict(error=f"{type(exc).__name__}: {exc}")
            continue
        committed = 1 + (len(frames) - 1) // 4
        lat = latents[:committed][None].to("cuda", torch.bfloat16)
        video = torch.from_numpy(
            np.stack(list(decode_frames(models.wan_vae, lat, chunk=8)))
        ).permute(0, 3, 1, 2)
        first = {}
        for i, f in enumerate(frames):
            first.setdefault(f.block, i)
        recon_starts = [first[r.s] for r in records if r.kind == "reconstituted"]
        entry = dict(
            commit=commit,
            target=target,
            seconds=seconds,
            latents=committed,
            **measure(records, frames),
            quality=frame_quality(video[97:], [i - 97 for i in recon_starts]),
        )
        res[name] = entry
        decoded[name] = video
        np.save(out / f"latents_{name}.npy", latents[:committed].numpy())
        save_strip(video, recon_starts, out / f"strip_{name}.jpg")
        log(
            "subblock",
            name,
            json.dumps({k: v for k, v in entry.items() if k != "quality"}),
            json.dumps({k: v for k, v in entry["quality"].items() if k != "sharpness_by_second"}),
        )
    save_grids(decoded, out, "subblock")
    return res


def save_strip(video: torch.Tensor, gen_starts, path: Path, around: int = 4) -> None:
    """Consecutive frames around three generation boundaries (the frame left of the bar ends a generation)."""
    from PIL import Image, ImageDraw

    starts = [i for i in gen_starts if 97 + 16 * 8 <= i < video.shape[0] - around][:3]
    if not starts:
        return
    h, w = video.shape[2] // 2, video.shape[3] // 2
    img = Image.new("RGB", (w * 2 * around + 4, h * len(starts)), "white")
    draw = ImageDraw.Draw(img)
    for r, i in enumerate(starts):
        for c, k in enumerate(range(i - around, i + around)):
            tile = Image.fromarray(video[k].permute(1, 2, 0).numpy()).resize((w, h))
            img.paste(tile, (c * w + (4 if k >= i else 0), r * h))
        draw.text(
            (4, r * h + 4),
            f"frames {i - around}-{i + around - 1}, generation starts at {i}",
            fill="yellow",
        )
    img.save(path, quality=90)


def save_grids(decoded: dict, out: Path, prefix: str) -> None:
    """Side-by-side frames: rows = modes, columns = seconds 8, 12, ..., and one strip of consecutive frames."""
    from PIL import Image, ImageDraw

    names = list(decoded)
    if not names:
        return
    t = min(v.shape[0] for v in decoded.values())
    cols = [i for i in range(16 * 8, t, 16 * 4)][:6]

    def grid(indices, path, scale=0.5):
        h, w = (int(d * scale) for d in decoded[names[0]].shape[2:])
        img = Image.new("RGB", (w * len(indices) + 90, h * len(names)), "white")
        draw = ImageDraw.Draw(img)
        for r, name in enumerate(names):
            draw.text((4, r * h + h // 2), name, fill="black")
            for c, i in enumerate(indices):
                tile = Image.fromarray(decoded[name][i].permute(1, 2, 0).numpy()).resize((w, h))
                img.paste(tile, (90 + c * w, r * h))
        img.save(path, quality=92)

    grid(cols, out / f"{prefix}_grid.jpg")
    mid = cols[len(cols) // 2] if cols else 0
    grid(list(range(mid, min(mid + 8, t))), out / f"{prefix}_strip.jpg")


# ----------------------------------------------------------------------------------------------- session
def _session_worker(job: dict) -> dict:
    """One client of a lock-step session, in its own process and GPU."""
    gpu = int(job["gpu"])
    torch.cuda.set_device(gpu)
    cfg = load_cli_config(job["configs"], {**job["overrides"], "run.device": f"cuda:{gpu}"})
    rt = RealtimeConfig(**job["rt"])
    models = EngineModels.load(cfg, rt, vae_path=job["vae"])
    from worldcast.engine.realtime.engine import RoundSpec

    spec = RoundSpec(**job["spec"])
    engine = Engine(cfg, rt, models=models, pool_dir=job["pool"])
    engine.start(spec, job["slot"])
    log("session client", job["slot"], "on GPU", gpu, "started")
    frames, t0 = [], time.monotonic()
    pacer = Pacer() if job["paced"] else None
    while not engine.finished:
        nb = pacer(engine, frames) if pacer is not None else None
        frames.extend(engine.step(lambda: None, not_before=nb))
    seconds = time.monotonic() - t0
    records = list(engine.records)
    engine.stop()
    recon = [r for r in records if r.kind == "reconstituted"][3:]
    first = {}
    for i, f in enumerate(frames):
        first.setdefault(f.block, i)
    ready = [f.t_ready for f in frames]
    recs = [r for r in records if r.kind != "sink"]
    k = next(i for i, r in enumerate(recs) if r.kind == "reconstituted") + 1
    return dict(
        slot=job["slot"],
        gpu=gpu,
        seconds=seconds,
        lockstep_ms=stats([r.host_ms.get("lockstep_ms", 0.0) for r in recon]),
        prepare_ms=stats([sum(r.host_ms.values()) for r in recon]),
        throughput=throughput([first[r.s] for r in recon], [r.t_first_frame for r in recon]),
        keypress_to_photon=keypress_to_photon(
            [r.t_sample for r in recs[k:]], [first[r.s] for r in recs[k:]], ready, skip_blocks=2
        ),
    )


def section_session(args, cfg, n_clients: int) -> dict:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from worldcast.data.index import read_round_index

    row = read_round_index(cfg.paths.round_index)[args.row]
    slots = list(row.group_slots)[:n_clients]
    spec = dict(
        match_id=row.match_id,
        map_name=row.map_name,
        round=row.round,
        start_frame=row.start_frame,
        clients=tuple(slots),
    )
    res = {}
    for name, rt_kw in (("lockstep", dict(lockstep=True)), ("latest", dict(lockstep=False))):
        pool = Path("/dev/shm") / f"wcrt-pool-{os.getpid()}-{name}"
        rt = dict(LOOP["after_taehv"], taehv_path=args.taehv, **rt_kw)
        jobs = [
            dict(
                gpu=i,
                slot=slot,
                spec=spec,
                pool=str(pool),
                rt=rt,
                paced=True,
                vae=args.vae,
                configs=args.config,
                overrides={**parse_overrides(args.overrides), "run.max_blocks": args.blocks},
            )
            for i, slot in enumerate(slots)
        ]
        log("session", name, "clients", slots)
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=mp.get_context("spawn")) as ex:
            results = [f.result() for f in [ex.submit(_session_worker, j) for j in jobs]]
        res[name] = results
        log("session", name, json.dumps(results, default=float))
    return res


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_args(parser)
    parser.add_argument("--row", type=int, default=59, help="row of the round index (default: d59)")
    parser.add_argument("--blocks", type=int, default=24, help="reconstituted blocks per session")
    parser.add_argument("--taehv", help="taew2_2.pth")
    parser.add_argument("--vae", help="Wan2.2_VAE.pth (default: paths.wan22_root/Wan2.2_VAE.pth)")
    parser.add_argument("--sections", default=",".join(SECTIONS))
    parser.add_argument("--generation", default=",".join(GENERATION))
    parser.add_argument("--loop", default=",".join(LOOP))
    parser.add_argument("--subblock", default=",".join(SUBBLOCK))
    parser.add_argument("--clients", type=int, default=3, help="session: clients (one GPU each)")
    parser.add_argument(
        "--state-model",
        help="loop, closed-loop variants: state model checkpoint (a random one of "
        "the paper's size is written here if missing)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_cli_config(
        args.config,
        {**parse_overrides(args.overrides), "run.max_blocks": args.blocks, "run.device": "cuda"},
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sections = [s for s in args.sections.split(",") if s]
    result = dict(env=environment(), row=args.row, blocks=args.blocks, sections=sections)
    log(json.dumps(result["env"]))

    def save():
        (out / "bench.json").write_text(json.dumps(result, indent=1, default=float))

    if sections == ["session"]:
        result["session"] = section_session(args, cfg, args.clients)
        save()
        log("done")
        return 0

    t0 = time.monotonic()
    models = EngineModels.load(cfg, RealtimeConfig(decoder="wan"), vae_path=args.vae)
    if args.taehv:
        models.load_decoder(RealtimeConfig(decoder="taehv", taehv_path=args.taehv), cfg)
    result["load_s"] = time.monotonic() - t0
    log("models loaded in %.0f s" % result["load_s"])
    row = load_round_index_row(cfg.paths.round_index, args.row)

    result["patch_linear"] = check_patch_linear(models)
    log("patch_linear", result["patch_linear"])
    if "generation" in sections:
        result["generation"] = section_generation(cfg, models, row, out, args.generation.split(","))
        save()
    frames = None
    if "decode" in sections:
        latents = torch.from_numpy(np.load(out / "latents_release.npy"))
        result["decode"], frames = section_decode(cfg, models, latents, out)
        save()
    if "encode" in sections and frames is not None:
        result["encode"] = section_encode(frames)
        save()
    if "loop" in sections:
        result["loop"] = section_loop(
            cfg, models, row, args.taehv, args.loop.split(","), args.state_model
        )
        save()
    if "subblock" in sections:
        result["subblock"] = section_subblock(
            cfg, models, row, args.taehv, out, args.subblock.split(",")
        )
        save()
    save()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
