# WorldCast — Code Style & Conventions

Project-level standards for `worldcast/`, `tools/` and `tests/`. They bind every contributor. Reference style: the Wan2.2 and diffusers sources. The code must read like a well-kept research library
and reproduce the reference runs bit for bit (see the reproducibility contract below).

---

## Repo map

```
worldcast/
  config/         loader.py (the one YAML merge + --set), inference.py, training.py (the stage table STAGES)
  data/           recordings (round / media index, tick tables), controls, game (the game's constants),
                  jump_recall, camera (the recordings' camera), labels (GT visibility, observer signals), latents
                  (shape, time grid), latent_cache, video, map_mesh, window (one client's window of a round);
                  for training: training (the training windows), stream (sampler, batch stream), memory_frames,
                  memory_selection, appearance, memory_mask (the mask m_k)
    preparation/  media_index, manifest, observer_labels, latent_cache, buckets, visibility (its helpers
                  _visibility_geometry, _visibility_labels, _visibility_scope), meshes (the training files from
                  OpenCS2 and the game's maps; tools/prepare_data.py)
  modeling/       wan22/ (model.py: the generator WorldCastGenerator; dit.py: the causal DiT block and the KV cache;
                  training.py: the training forward; attention.py, vae.py, text_encoder.py);
                  controls, ray_embedding, observer_signals, state_injector (conditioning); depth_head (DepthHead,
                  DepthReadout), state_model, visibility_probe; build.py (build and load the inference generator)
  sampling/       schedulers.py (table, denoising steps, flow -> x0, re-noise, the entry noise and its stream),
                  sampler.py (the Sampler), window.py (first frame | memory frames | recent context | target
                  frames; slide_window_entries: a live round's slide), conditions.py (a batch's controls),
                  unipc.py (20 steps)
  player_state/   states, projection, attributes, field, visibility, predicted_visibility   (paper Sec. 3.2);
                  extrapolation (within a block), closed_loop (the complementary filter)   (paper Sec. 3.4);
                  live play (not the paper's): live_position (the own position, guarded; the wall ahead and
                  the map's edge), live_peers (the other players: from their published rows; nobody else is in
                  the world), live (LiveLoop: a client's player states under live controls, in the closed
                  loop's place)
  scene_state/    bank, geometry, state                                          (paper Sec. 3.3)
  engine/         generator.py (the generator as inference, training and evaluation load and call it), stage.py (a
                  training stage's generator)
    inference/    client (Client, run_client), rollout (the block loop), controls, world_state (WorldState),
                  directory (DirectoryWorldState), messages (MessagePart, MessageWorldState), memory_blocks
                  (MemoryBlocks, LiveMemoryBlocks),
                  loading, serving (ServingOptions), fast (the generator call), taehv (tiny decoder), decode,
                  session (the clients of a round as processes), reference (comparison with the reference runs),
                  field_rows (FieldRows: the rows the player state field read, for the demo's page),
                  live_rollout (LiveRollout: the block loop under live controls on a round that slides,
                  player_state.source live), idle_context (IdleContext: the context of an idle
                  client), replay (RecordedControls: a recorded client played as a live player),
                  replicas (GeneratorReplica: the generator on more cards, for a live client's refreshes)
    training/     trainer (what every stage shares), build (a stage's run from its config), window (a micro-batch's
                  training window), losses, recipes/ (flow_matching: the step of stages 1-3; bidirectional: stages
                  1-2; teacher_forcing: stage 3; distillation: stage 4)
    evaluation/   protocols (unipc, four_step), metrics, samplers, evaluator, validation (in-training); the paper's
                  metrics (tools/evaluate_paper.py): boxes, detector, entity (Rendered, Unmatched), stats (the
                  bootstrap intervals), depth_agreement, matching (Mat. Pix., Inliers, RotErr), fvd, vbench
    checkpoint/   training (save, resume)
    optim/        builder (parameter groups, AdamW, clipping), ema
  demo/           WorldCast Live, the web demo: coordinator, worker, local (both in one process), launch (a worker
                  process per GPU), inputs, pacing, latency, wire, clients (ClientProtocol), worldcast_client, mock,
                  library, build_library, map_picker (the seats on a map's overview; the seat nearest to a click),
                  config, static/ (the page)
  distributed/    process_group (process group, RNG state), fsdp
  utils/          precision (TF32, bf16 casts), seed, weights (reading weight files), files (a file's sha256),
                  fingerprints, lazy (package exports)
  third_party/    gim_dkm/ (GIM's dense matcher of Mat. Pix., MIT, inference only: dkm, encoders, local_correlation,
                  model, padding)
  hub.py          downloads: the weights, the example cases, the weights of the paper's metrics, the
                  evaluation data
tools/            run_client, run_session, decode, download_weights, download_examples, download_eval_data (the
                  evaluation sets' data), check_examples, verify_reference, train, evaluate, evaluate_paper (the
                  paper's metrics), serve_demo (--role all: a worker per GPU), setup_demo (weights, checks, the serve
                  command), build_demo_library, bench_realtime, bench_live, make_prompt_embedding, make_overviews
                  (the lobby's map overviews), prepare_data (the training files), make_showcase (the GT boxes and the
                  map over a session's views), replay_live (a recorded client through the live path, the demo's
                  inference in evaluation form), make_noscene_generator (the client weight file of the generator
                  trained without scene state, model.scene_state false)
configs/          train/ (stage1, stage1_long, stage2, stage2s, stage3, stage4, *_noscene, ablations/); demo/
                  (mock_block16.yaml, worldcast.yaml, worldcast_wan_vae.yaml, worldcast_wan_vae_channels_last.yaml,
                  worldcast_noscene.yaml, worldcast_noscene_wan.yaml, worldcast_two_cards.yaml); state_model/ (the
                  closed loop's tables); data/ (the paper's held-out matches, excluded media, evaluation reserve,
                  validation index); eval/ (the evaluation sets: round_index, window_views, windows)
examples/         recorded rounds with GT player states: README.md, run.sh, grid.sh, manifest.json (the cases'
                  files and fingerprints), table3_fingerprints.json, the path templates (data_paths.yaml,
                  train_paths.yaml), data/<case>/ (the round and media indices)
docs/             installation.md, inference.md (client, numerics, data, weights), reproduce.md, demo.md (live demo,
                  latency), training.md, data.md (where the data comes from, how to build the training files),
                  development.md (this file)
tests/            mirrors the package (test_<module>.py; support.py and conftest.py per area); tests/tools: every
                  tool and the examples' scripts on the CPU; tests/docs: the docs against the code (links,
                  commands, settings, cited names, the weights tables, the library example); tests/reference: the
                  client against the paper's Table-3 fingerprints (GPU, the paper's data)
```

## Vocabulary

Use the paper's terms in names, docstrings and docs, and no synonyms.

- **client**: one player's generator and state model; one process, one GPU.
- **player state**: a player's `[x, y, z, yaw, pitch, alive]` per latent frame (engine units, degrees), with the
  block's controls. **player state field**: every other player's state projected into the client's camera, 23
  channels on the 12 x 21 token grid, added after the second DiT block.
- **scene state**: a **memory bank** shared by all clients, of which each client keeps a copy (its own entries, at
  most B = 64, and the other clients'). **memory entry**: one generated block with its four cameras and depth.
- **session**: one run of a round by its clients on one fresh shared world state (`tools/run_session.py`; a room of
  the live demo). `Client.start` ... `Client.stop` is one client's part in it.
- **block**: 4 latent frames = 16 video frames = 1 s at 16 fps; the unit of generation, publishing and lockstep.
  **latent frame**: one Wan2.2 latent `[48, 24, 42]`.
- **window**: two things, each named with what it is a window of. The **window of a block**
  (`worldcast/sampling/window.py`) is what the block attends to: first frame | memory frames (4) | recent context
  (12) | target frames (4), 21 latent frames, 17 when nothing is retrieved (the paper counts the 20 after the first
  frame, the round's recorded first frame). The first six blocks (latents 1-24) run before the first such window,
  at absolute positions and without scene state (the paper reads the scene state from latent 25 on). A **window of
  a recorded round** (`worldcast/data/window.py`, `WindowSpec`) is a span of the recording from a start frame: the
  ten-second training window (41 latent frames), or everything a client generates of a round. In live play a
  session has no end: its round is the first frame, the first six blocks and one more (29 latent frames), which
  slides by a block after each block from the seventh on, and the window of a block is the paper's.
- **denoising steps**: the four steps, at the timesteps 1000 / 937.5 / 833.3 / 625.
  **context noise**: the noise level of every context write into the KV cache (label 16).
- **state model**: reads the client's own generated latents and controls and estimates its position (Sec. 3.4):
  encoder, trunk, action encoder, motion head, place head; the complementary filter of Eq. (4) combines the heads.
  **depth head** and its **read-out**: depth per 16 x 16-pixel patch from latents (scene state, predicted
  visibility).
- **training**: stages 1-2 bidirectional, stage 3 teacher forcing, stage 4 distribution matching distillation with
  the roles generator, teacher and critic.
- Not in the paper, plain names: the **observer signals** (flash, scope; an input of the generator), the
  **visibility probe** (training only), **context noise**, and in live play an **idle** block (no key held, the view
  controls zero), a **stretch** (a run of idle blocks of one view) with its **clip** (its first three blocks; not the
  `clip` camera encoding) and **freeze** (`live.idle_context`: the store and the memory bank keep the clip's blocks in
  place of the later ones).

## Reproducibility contract

The release reproduces the latents of the reference runs bit for bit on their stack. What the latents depend on is
listed once, in `docs/inference.md` ("Numerics"): bf16 generator inputs, timesteps included; TF32; the context
noise; the RNG draw order; the dtypes of the data flow; the `noclip` camera encoding; the scene-state geometry in
numpy on the CPU; the predicted-visibility geometry in torch (`worldcast/player_state/predicted_visibility.py`), a
second arithmetic beside it (the two are not merged); inputs checked by content; the attention kernel. These are
easy to "tidy" by mistake, and they are what the weights were trained and evaluated with.

- Keep each explicit in code with a neutral one-line *why* ("as trained"). Changing one changes every latent after
  it; never re-baseline a fingerprint.
- No other draw from the global RNG during a rollout (dropout, debug noise, a module's random init): it shifts every
  later draw.
- **Window of a block**: 21 latent frames with the first frame at position 0, no attention mask (context written
  range by range); RoPE and control-history positions are window indices, not round time.
- **One process per client**, never threads: numpy reprojections under a multithreaded BLAS are not
  bit-reproducible across threads of one process.
- **Lockstep without timeouts** makes the output independent of timing; a timeout or a failed client is fatal;
  every session starts on a fresh world-state directory.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests -q        # CPU; tests that need a GPU, the weights or the data skip and say why
black --check worldcast tools tests && isort --check-only worldcast tools tests
flake8 worldcast tools tests
```

On the CPU the model runs in float32 (bf16 parameters need CUDA autocast), so the suite checks logic, not the
reference runs' bits. Those are checked on a GPU against the fingerprints the reference runs recorded: first frame,
latents 0-24 and all latents (`examples/manifest.json` for the examples), and for Table 3 also the entry noise
(`examples/table3_fingerprints.json`):

```bash
python tools/verify_reference.py client --config ref.yaml --index-row <row> \
    --out-dir runs/verify/client                     # one GPU, minutes: generator + sampler, no other client
python tools/verify_reference.py check runs/reference  # finished tools/run_session.py sessions
WORLDCAST_REFERENCE_CONFIG=ref.yaml python -m pytest tests/reference -q
```

`ref.yaml` points `paths` at the release weights and the paper's data. Bit equality holds on the reference runs'
stack, an NVIDIA H20 with torch 2.9.1+cu128 and flash-attn 2.8.3 (FA2); the first frame anywhere; the entry noise on
x86-64 with torch 2.9.1. A change on the generation path keeps `tests/` green and the fingerprints equal. Compare
`latents.npy`, never the mp4.

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
def input_to_photon(taken: np.ndarray, photon: np.ndarray) -> dict[str, dict[str, float]]:
    """The latency of inputs at evenly spread times.

    An input reaches the first block whose controls are taken after it and shows when that block's
    first frame is displayed.

    Args:
        taken (np.ndarray): ``[n]`` when each block's controls were taken, s.
        photon (np.ndarray): ``[n]`` when each block's first frame is shown, s.

    Returns:
        dict[str, dict[str, float]]: :func:`stats` of the ``total`` latency and of its
        ``lookahead``, the wait for the next block's controls.
    """
```

(`worldcast/demo/latency.py`, as it is in the code.)

## Imports and formatting

- isort + black (profile black, line length 100); flake8 clean (`.flake8`). Let black decide; don't hand-format.
  4-space indent, UTF-8, newline at EOF, no trailing whitespace.
- Order: stdlib, third-party, first-party (`worldcast`), local. Absolute imports for `worldcast.*`; relative imports
  inside a subpackage.
- Heavy or optional dependencies (flash_attn, pyarrow, imageio, trimesh, transformers, PIL; the metrics' ultralytics,
  kornia, decord, cv2, scipy) are imported inside the function that needs them, or behind a PEP 562 `__getattr__`,
  never at package top level. `worldcast/demo` needs
  its extra: its modules import aiohttp and Pillow at their top, and its coordinator and mock client load no torch.

## Naming and structure

- `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE` constants with a one-line `#:` meaning
  instead of magic numbers, `_leading_underscore` for non-public.
- Dataclasses for records and configs. What the paper fixes is a module constant, not a setting; the settings that
  remain default to the paper's values. Config keys and CLI flags are public.
- Every artefact path comes from the config or the CLI. One config loader: `worldcast/config/loader.py`.
- A config is checked when it is built: an unknown key, a value of the wrong type and a value a setting does not
  take raise `ValueError`, whether a file or `--set` names it, and a tool reports it as a usage error
  (`usage_errors`). A value the weights were not run with (`model.attention: sdpa`) is allowed and documented as not
  bit-equal.

## Comments

Default to no comment. Add one only for a non-obvious *why*, e.g. `# bf16 here is what the paper ran; fp32 changes
the latents`. Never restate the code; never leave commented-out code. No provenance: no old file names, `file:line`,
"port of", "the old ...", commit hashes or run names. Paper references ("Sec. 3.2", "Eq. (4)", "App. ...",
"Table 3") are good.

## What not to do

- No `typing.List` / `Dict` / `Tuple` / `Optional` / `Union`, no `from __future__ import annotations`.
- No wildcard imports; no heavy imports at package top level.
- No backwards-compat shims, aliases, dead code, speculative options, defensive checks that cannot fire, or
  workarounds for earlier bugs of this codebase.
- No "fix" of an item of the reproducibility contract; no new draw from the global RNG on the rollout path.
- No second YAML merge or `--set` parser; no hard-coded paths.
- No synonyms for the paper's terms; no docstrings on trivial or private helpers just to fill space.
