# WorldCast — Code Style & Conventions

Project-level standards for `worldcast/`, `tools/`, `demo/` and `tests/`. They bind human contributors and code
agents. Reference style: the Wan2.2 and diffusers sources. The code must read like a well-kept research library
and reproduce the paper bit for bit (see the reproducibility contract below).

---

## Repo map

```
worldcast/
  config/         loader.py (the one YAML merge + --set), inference.py, training.py
  data/           round / media index, tick tables, controls, labels, first latents, memory-slot material
  modeling/       wan22/ (model.py: the block-causal DiT, attention.py, vae.py, text_encoder.py);
                  action, rays, state_injector, obs_signal (conditioning); depth_head, visibility_head,
                  state_model; build.py
  sampling/       schedulers.py (table, ladder, flow <-> x0, re-noise), rollouts.py (Sampler, KV cache),
                  window.py (sink | memory slot | recent | target)
  player_state/   field, projection, visibility, predicted_visibility, attributes, tables, extrapolate,
                  closed_loop                                                    (paper Sec. 3.2, 3.4)
  scene_state/    bank, geometry, state                                          (paper Sec. 3.3)
  engine/         inference/ (client, pool, decode), realtime/ (engine, fast, taehv, transport, ...),
                  training/, checkpoint/, optim/
  distributed/    process group, seeding, FSDP
  utils/          precision (TF32, bf16 casts), seed
tools/            entry points: run_client, run_session, decode, serve_demo, verify_reference, bench_realtime, ...
configs/          infer/, train/, demo/, state_model/
demo/             the web app: coordinator, worker, protocol, engines, static/
examples/         recorded rounds with GT player states (run.sh, grid.sh)
tests/            mirrors the package; tests/reference holds the paper run's GPU fingerprints
.claude/skills/   onboarding-worldcast, debug-worldcast, deploy-worldcast-live
```

## Vocabulary

Use the paper's terms in names, docstrings and docs, and no synonyms.

- **client**: one player's generator and state model; one process, one GPU.
- **player state**: a player's `[x, y, z, yaw, pitch, alive]` per latent frame (engine units, degrees), with the
  block's controls. **player state field**: every player's state projected into the client's camera, 23 channels
  on the 12 x 21 token grid, added after the second DiT block.
- **scene state**: the shared memory of generated blocks. **memory bank**: one client's copy of it (its own entries,
  at most B = 64, and its peers'). **memory entry**: one generated block with its four cameras and depth.
- **block**: 4 latent frames = 16 video frames = 1 s at 16 fps; the unit of generation, publishing and lock-step.
  **latent frame**: one Wan2.2 latent `[48, 24, 42]`.
- **window**: what a block attends to: sink | memory slot (4) | recent (12) | target (4), 21 latent frames, 17 when
  nothing is retrieved. The **plain prefix** (latents 1-24) runs before the first window.
- **ladder**: the 4-step schedule 1000 / 750 / 500 / 250, warped to 1000 / 937.5 / 833.3 / 625.
  **context noise**: the noise level of every context write into the KV cache (label 16).
- **state model**: reads the client's own generated latents and controls and estimates its position (Sec. 3.4).

## Reproducibility contract

The release reproduces the paper's latents bit for bit on the paper's stack (`docs/inference.md`, "Numerics that
the paper's numbers depend on"). The items below look like slack or bugs; they are what the weights were trained and
evaluated with. Keep each explicit in code with a neutral one-line *why* ("as trained"). Changing one changes every
latent after it.

- **bf16 generator inputs, timesteps included.** Parameters and every floating input are bf16: the model sees
  1000 / 936 / 832 / 624 for 1000 / 937.5 / 833.3 / 625 and converts flow to x0 at those timesteps; the re-noise
  between rungs uses the unrounded ones. Poses, fields of view, controls and peer states are rounded too; the ray
  pose math runs inside the generator's bf16 autocast.
- **TF32** on for fp32 CUDA matmuls and convolutions (the DiT's fp32 islands, the depth head), once per process,
  before any model runs.
- **Context noise**: labelled t = 16; the scheduler snaps it to table index 997, so the level is t ~ 14.82.
- **RNG draw order.** Entry noise: one CPU-generator draw for the requested `run.latents`, cut to N - 1 (the CPU
  normal kernel is platform dependent). Then `set_seed(seed)` right before the plain prefix; the device's global RNG
  gives the prefix's 25 draws and 3 ladder draws per later block; later context writes draw from CPU generators
  keyed by `(seed, block, role)`. Any other draw from the global RNG during a rollout (dropout, debug noise, a
  module's random init) shifts every later draw.
- **Data-flow dtypes**: prefix latents bf16-rounded; later blocks keep their fp32 x0 in the client's store (window,
  depth head, published payload); `latents.npy` is the bf16 output buffer written as float32.
- **Camera delta encoding** `noclip` (mu-law without the +-20 degree clip), passed down as `data.camera_encoding`,
  never read from the environment. `clip` agrees below 20 degrees per frame and is wrong for these weights.
- **Scene-state geometry** in numpy on the CPU: float32 eviction cache, float64 retrieval z-buffers. Retrieval ties
  break by `(t_last, seq, block)`, so the reader's ingestion order (own block, then peers by slot) is part of it.
- **Window**: 21 latent frames with the sink at position 0, no attention mask (context written range by range);
  RoPE and control-history positions are window indices, not round time.
- **One process per client**, never threads: numpy reprojections under a multithreaded BLAS are not
  bit-reproducible across threads of one process. One real-time engine per process.
- **Lock-step without timeouts** makes the output independent of timing; a timeout or a failed peer is fatal
  (`pool.fail_on_timeout`); every session starts on a fresh pool directory.
- **Attention kernel**: flash-attention as deployed. FA2 (the paper's) and FA3 give different bits; SDPA is a
  reference, not bit-equal. Record `flash_attention_backend()` next to every reference run.
- **Inputs checked by content**: tick tables by size and sha256 (the digest also keys the jump-button recall), never
  mtime; observer-signal labels come from `paths.obs_signal_label_root` and a missing file is an error.

## Testing

```bash
pip install -e ".[test]"
python -m pytest tests -q        # CPU; tests that need a GPU, the weights or the data skip and say why
black --check . && isort --check . && flake8
```

On the CPU the model runs in float32 (bf16 parameters need CUDA autocast), so the suite checks logic, not the
paper's bits. Those are checked on a GPU against the paper run's fingerprints (entry noise, initial latent, plain
prefix, final latents of the cells listed in `tests/reference`):

```bash
python tools/verify_reference.py prefix --config configs/infer/worldcast_4step.yaml --config ref.yaml \
    --cell <cell> --out-dir runs/verify/prefix       # one GPU, minutes: generator + sampler, no peers
python tools/verify_reference.py check runs/reference  # finished tools/run_session.py sessions
WORLDCAST_REFERENCE_CONFIG=ref.yaml python -m pytest tests/reference -q
```

`ref.yaml` points `paths` at the release weights and the paper's data. Bit equality holds on an NVIDIA H20 with
torch 2.9.1+cu128 and flash-attn 2.8.3 (FA2); the initial latent anywhere; the entry noise on x86-64 with torch
2.9.1. A change on the generation path keeps `tests/` green and the fingerprints equal. Compare `latents.npy`,
never the mp4.

---

## Type annotations

Python >= 3.10 built-ins (PEP 585) and unions (PEP 604): `list[int]`, `dict[str, Tensor]`, `X | None`, `A | B`.
Never `typing.List / Dict / Tuple / Set / Type / Optional / Union`; `Any`, `Callable`, `Iterator`, `Iterable`,
`Mapping`, `Sequence` are fine. No `from __future__ import annotations`: quote the rare forward reference instead.

## Docstrings — Google style

Public functions, classes and methods get a Google-style docstring with the type in each entry,
`name (type): ...`; tensors also state shape, dtype and units. Simple public functions take a one-liner; private
helpers, `@property` and dunders need none. A module starts with a one-line summary, optionally a short paragraph.

```python
def splat(uv: Tensor, radius: Tensor, eligible: Tensor, *, grid: tuple[int, int]) -> Tensor:
    """Gaussian splat weight of every player on the token grid (Sec. 3.2).

    Args:
        uv (Tensor): ``[B, F, P, 2]`` float32 token coordinates of each player's feet.
        radius (Tensor): ``[B, F, P]`` float32 projected radius, tokens.
        eligible (Tensor): ``[B, F, P]`` bool, alive, in front and visible.
        grid (tuple[int, int]): token grid ``(h, w)``, ``(12, 21)`` for the paper's latents.

    Returns:
        Tensor: ``[B, F, P, h, w]`` float32 weights, zero for ineligible players.
    """
```

## Imports and formatting

- isort + black (profile black, line length 100); flake8 clean (`.flake8`). Let black decide; don't hand-format.
  4-space indent, UTF-8, newline at EOF, no trailing whitespace.
- Order: stdlib, third-party, first-party (`worldcast`), local. Absolute imports for `worldcast.*`; relative imports
  inside a subpackage.
- Heavy or optional dependencies (flash_attn, pyarrow, imageio, trimesh, transformers, PIL) are imported inside the
  function that needs them, or behind a PEP 562 `__getattr__`, never at package top level.

## Naming and structure

- `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE` constants with a one-line `#:` meaning
  instead of magic numbers, `_leading_underscore` for non-public.
- Dataclasses for records and configs. Paper values are the config defaults; config keys and CLI flags are public.
- Every artefact path comes from the config or the CLI. One config loader: `worldcast/config/loader.py`.
- A categorical config value the paper did not use raises `NotImplementedError`; it never runs an unported branch.

## Comments

Default to no comment. Add one only for a non-obvious *why*, e.g. `# bf16 here is what the paper ran; fp32 changes
the latents`. Never restate the code; never leave commented-out code. No provenance: no old file names, `file:line`,
"port of", "the old ...", commit hashes or run names. Paper references ("Sec. 3.2", "Eq. (6)", "App. ...",
"Table 3") are good.

## What not to do

- No `typing.List` / `Dict` / `Tuple` / `Optional` / `Union`, no `from __future__ import annotations`.
- No wildcard imports; no heavy imports at package top level.
- No backwards-compat shims (`from_legacy` and friends), aliases, dead code, speculative options, defensive checks
  that cannot fire, or workarounds for earlier bugs of this codebase.
- No "fix" of an item of the reproducibility contract; no new draw from the global RNG on the rollout path.
- No second YAML merge or `--set` parser; no hard-coded paths.
- No synonyms for the paper's terms; no docstrings on trivial or private helpers just to fill space.
