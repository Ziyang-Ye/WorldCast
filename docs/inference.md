# Inference

How a WorldCast client runs, what it computes in which order, its configuration, the numerical details that the
paper's numbers depend on, and where the code differs from the paper's text. The code is `worldcast/` (one module per
concern, listed below); `worldcast/engine/inference/client.py` only orchestrates.

## One client, many clients

Each player of a round runs one client on its own GPU. A client generates its player's view block by block (4 latents
= 16 video frames = 1 s at 16 fps) from:

- the player's own controls (recorded), and the recorded states of all ten players, projected into the **player
  state field** (23 channels on the 12 x 21 token grid, added after the second DiT block);
- the **scene state**: one retrieved block of 4 latents (the client's own or a peer's), placed in the window as
  memory frames;
- its own recent latents and the round's first latent (the sink).

Clients share nothing but a pool: each publishes every finished block and a per-block step record; each reads its
peers' blocks. In the paper's evaluation (and here) the pool is a shared directory and the clients run in
**lock-step**: before block `s` every client waits until every peer has published every block that ended before `s`.
With no timeout this makes the output independent of timing. One process per client: never threads (the scene
state's numpy reprojections are not bit-reproducible across threads of one process).

## The block loop

```
start ── config, TF32 on ── generator (bf16), depth head, prompt embedding ── client window (data)
  │
  ├─ entry noise  randn(1, latents-1, 48, 24, 42) from a CPU generator(seed); cut to N-1; bf16
  ├─ sink         latent 0 of the cached first window (bf16)
  ├─ seed         set_seed(seed)   (python, numpy, torch, cuda)
  ├─ plain prefix latents 1..24 = 6 blocks at absolute positions 1..24, KV cache grows to 25 latents,
  │               no memory; publish the 6 blocks
  │
  └─ for s = 25, 29, ..., N-4          (t = start_frame + 8 s, the block's first source frame)
       1  own write          scene.ingest_own        the own block that ended before t -> memory entry
       2  step record        pool.publish_step       own withdrawals (bound B = 64) and own resident entries
       3  lock-step wait     pool.wait_for_peers     every peer: blocks up to t - 8 and its step record for t
       4  admit              PeerBlocks.admit         peers' published blocks with orig_last < t -> entries
       5  follow             scene.follow_withdrawals apply peers' withdrawals; copy must equal their lists
       6  retrieve           scene.retrieve           at most one entry: the one that fills the most holes
       7  window             window.prepare_window    sink | slot 4 | recent 12 | target 4  (17 if none)
       8  KV prefill         Sampler.generate_block   context ranges re-noised to t=16, written in order
       9  4-step ladder      Sampler.generate_block   1000 / 937.5 / 833.3 / 625; field built at injection
      10  publish            pool.publish_block       the block's 4 clean latents (fp32)
  │
  └─ latents.npy (float32) ── DONE marker in the pool ── decode separately (tools/decode.py)
```

Window layout of a reconstituted block (positions are window indices, also the RoPE and control-history positions):

```
position   0      1  2  3  4      5 ... 16              17 18 19 20
           sink | memory slot  | recent own latents    | target block
           own0 | retrieved    | own s-12 .. s-1       | own s .. s+3      21 latents
           own0 |                own s-12 .. s-1       | own s .. s+3      17 latents (nothing retrieved)
```

The context is written to the KV cache range by range: `(0,1) (1,4) (5,4) (9,4) (13,4)`. There is no mask: each
range attends to the cache so far and itself; the target attends to all 21.

## Modules

| module | does |
|---|---|
| `worldcast/config/inference.py` | `InferenceConfig`: every setting, paper values as defaults, validated |
| `worldcast/utils/precision.py` | TF32, the bf16 parameter cast, the bf16 input cast that the paper's FSDP wrapper applied |
| `worldcast/modeling/` | `generator.py` (causal DiT, KV path), `action.py`, `rays.py`, `obs_signal.py`, `state_injector.py`, `attention.py` (flash-attn / SDPA), `loading.py`, `text.py` (umT5, prompt embedding), `vae.py`, `depth_head.py` |
| `worldcast/player_state/` | the player state field: `material.py`, `projection.py`, `visibility.py` (GT labels), `attributes.py`, `field.py`; the closed loop: `state_model.py`, `extrapolate.py`, `predicted_visibility.py`, `closed_loop.py` |
| `worldcast/scene_state/` | the scene state: `geometry.py`, `bank.py` (entries, bound, retrieval), `state.py` (per client) |
| `worldcast/engine/inference/pool.py` | `LocalDirPool` (the shared directory, lock-step), `PeerBlocks` (admission order) |
| `worldcast/sampling/window.py` | the compacted window and its conditions |
| `worldcast/sampling/rollouts.py` | scheduler, 4-step ladder, context re-noise, KV cache, plain prefix |
| `worldcast/data/` | round / media index, tick tables, controls, materials, labels, first latents, slot material |
| `worldcast/engine/inference/decode.py` | streaming VAE decode to mp4 |
| `worldcast/engine/inference/client.py` | the loop above |
| `worldcast/engine/realtime/` | serving one client live: a resident `Engine` that takes a block's controls and streams its frames (bit-identical by default; latency options and budget in [docs/latency.md](latency.md)) |

## Configuration

`configs/infer/worldcast_4step.yaml` holds every value, with the key or flag of the paper's research code that it
replaces in brackets; the defaults in `worldcast/config/inference.py` equal it. Only `paths` (and `run.index_row`) need setting:
the weight paths come from `weights/paths.yaml` (written by `tools/download_weights.py`), the data paths from a copy
of `examples/data_paths.yaml`. Off-paper values of the categorical switches raise `NotImplementedError` instead of
running an unported branch.

| section | key fields (paper value) |
|---|---|
| `model` | Wan2.2-TI2V-5B; latents 48 x 24 x 42, patch 1x2x2 (252 tokens / latent); controls 11 buttons + 2 camera + 52 weapons, history 20 rows; `player_field` (inject after block 1 = the second, stem 32, ATI splat top-2, sigma 0.5 token, confidence floor 0.3 / EMA 0.5); rays `plucker_dm`, unit 420 u; observer-signal hidden 128; fixed prompt |
| `sampler` | ladder `[1000, 750, 500, 250]` warped; `context_noise` 16 in band [16, 32]; `model_input_dtype` bfloat16; `tf32` true; framewise axes `peer_alive`, `peer_visible` |
| `window` | block 4, recent 12, first reconstituted block 25, memory slot 4, sink `episode_first`, KV capacity 41 latents |
| `data` | `camera_encoding: noclip`, buttons, 32 -> 16 fps, max tick gap 2.5/64 s, eye height 64 u, hfov 106.26 deg, `pose_source: oracle` |
| `memory` | bound B 64, k 1, write gate 0, holes `reach`, read `fill`, own entries `include`, scope `all`, picture depth, views memory |
| `player_state` | `source: recorded` (`predicted` = closed loop, below), `pose_radius_u: 40` |
| `pool` | wait 1800 s, poll 2 s, `fail_on_timeout: true` |
| `paths` | checkpoint, depth head / read-out, prompt embedding or `wan22_root`, data paths (`docs/data.md`), `live_pool_dir`, `out_dir`; closed loop: `state_model`, `state_model_cells`, `state_model_map_norm`, `physics_prior` |
| `run` | seed 20260917, latents 441, `index_row`, `max_blocks` 0, device |
| `weights` | download locations (only `tools/download_weights.py` reads them) |

Command line: `--config a.yaml --config b.yaml` merges files in order; `--set section.key=value` overrides one key.
Without `--config`, the scripts load `configs/infer/worldcast_4step.yaml`; with it, only the files given (the
defaults in `worldcast/config/inference.py` fill in the rest).

## Numerics that the paper's numbers depend on

These were side effects of the training stack in the paper's client; here each is explicit.

- **bf16 everywhere the paper's FSDP wrapper put it.** Parameters are bf16. Every floating input of a generator call is
  cast to bf16, *timesteps included*: the model sees 1000 / 936 / 832 / 624 for the ladder 1000 / 937.5 / 833.3 / 625,
  and converts its flow prediction to x0 with sigma at those bf16 timesteps, while the re-noise between rungs uses
  the unrounded ones. Poses, fields of view, controls and peer states are rounded to bf16 too. Never "fix" this.
- **TF32** on for fp32 CUDA matmuls and convolutions (the DiT's fp32 islands, the fp32 depth head).
- **Context noise.** Every context write is labelled t = 16, but the scheduler snaps 16 to its nearest table entry
  (index 997), so the actual level is sigma = 0.014822 (t ~ 14.82).
- **RNG order.** Entry noise: a CPU generator seeded with `run.seed`, drawn once for the requested length and cut
  to N - 1 (the CPU normal kernel differs across platforms: the paper's draw reproduces on x86-64 with torch
  2.9.1, not on arm64). Then `set_seed(seed)` right before the prefix; the global RNG of the GPU gives the prefix's
  25 draws (sink write, then per block 3 ladder + 1 cache write) and 3 ladder draws per reconstituted block. The
  context writes of reconstituted blocks draw from CPU generators keyed by `sha256(f"{seed}|{s}|{role}")`, roles
  `sink, slot0, recent0..2`. Anything else that draws from the GPU's global RNG during a rollout shifts every later
  draw.
- **dtypes of the data flow.** The prefix's latents are bf16-rounded; a reconstituted block keeps its unrounded
  fp32 x0 in the client's store, which feeds later windows (cast to bf16 there), the depth head and the published
  payload (fp32). `latents.npy` is the bf16 output buffer, written as float32.
- **Camera encoding** `noclip` (mu-law without the +-20 degree clip) is a config value passed down explicitly. The
  paper's research code read the environment variable `WC_CAMERA_ENCODING` and defaulted to `clip`, which is wrong for
  these weights.
- **Observer-signal labels** come from `paths.obs_signal_label_root`; a missing file is an error.
- **Retrieval geometry** stays in numpy on the CPU, with float32 for the eviction cache and float64 z-buffers for
  retrieval, as deployed (integer pixel counts can flip at boundaries otherwise).
- **Camera geometry in bf16.** The pose math of the ray code runs inside the generator's bf16 autocast, so on CUDA
  the ray codes are bf16-rounded. Do not move it out.
- **Tick files** are checked by size and sha256, never by mtime (a plain copy changes mtime, not content). The
  sha256 also keys the jump-button recall.
- **Attention backend.** FA2 and FA3 give different bits; record which one ran
  (`worldcast.modeling.wan22.attention.flash_attention_backend`) next to every reference run.

Bit-exactness: the release code equals the paper code bit for bit on CPU (the old-vs-new tests in `tests/golden/`,
including a three-client end-to-end session). Reproducing the paper's latents bit for bit additionally
needs the paper's stack: NVIDIA H20, torch 2.9.1+cu128 and flash-attn 2.8.3 (FA2; FA3 was not installed). On that
stack the release client reproduces all nine reference cells bit for bit: plain prefix and final latents of the three
rounds, every per-block memory read, and the decoded mp4 files byte for byte. `tests/reference` and
`tools/verify_reference.py` check the paper's fingerprints on such a machine.

## Closed loop

`--set player_state.source=predicted` runs closed-loop deployment (Sec. 3.4, App. "Closed-loop deployment"; Table 4a
with the released generator). Each client takes nothing from the recording beyond its first latent, its round-start
position and every player's controls:

- **Own position.** After every block the client reads its own position at each of the block's four latent frames
  off the frames it generated, with the state model (`state_model.py`: conv encoder and 16-layer causal trunk with the
  control encoder, motion head, place head over 1,860 cells), and fuses them with Eq. (6) at the constant weight 1/2
  from its round-start position. The model reads 10-s windows (41 latents, stride 40); a window still being
  generated is read with its future latents zeroed.
- **Exchange.** Before each block (and before its step record, in the reconstituted blocks) it publishes those
  positions to `<live_pool_dir>/<client>/state/state_<t>.json`; in the plain prefix the clients also wait for each
  other's records there. A client that publishes nothing holds its last position.
- **Extrapolation** (`extrapolate.py`). Over block s every client is drawn at its position at latent s-1 plus the
  physics prior's displacement over the block (movement keys at 64 Hz, per-key speeds, jump and gravity, no
  collisions); view angles integrate the player's own view controls. The client's rays, its retrieval queries and the
  keys of its memory entries use its own extrapolated cameras; peers' memory entries are keyed at the cameras their
  publishers drew them from. Players that run no client keep their recorded state.
- **Visibility** (`predicted_visibility.py`). Labels come from the picture depth head: test point feet + 40 u, 8 more
  on a 40 u ring, median of the 2 x 2 depth cells, 24 u margin; the block about to be drawn is tested against the
  last 12 drawn latents splatted into its cameras, and re-tested after the first rung on the depth of its x0.

Needs `paths.state_model` (a state model checkpoint, a plain state dict; none is released) and the tables shipped
under `configs/state_model/`:

    --set player_state.source=predicted --set paths.state_model=<state_model.pt> \
    --set paths.state_model_cells=configs/state_model/cells_v1.json \
    --set paths.state_model_map_norm=configs/state_model/map_norm_v1.json \
    --set paths.physics_prior=configs/state_model/physics_prior_v1.json

Reading the controls needs `pyarrow`. Table 2 (ten-second windows) ran the same loop on the four-step model trained
without scene state, whose weights are not released.

## Paper vs code

What the code (and therefore the paper's numbers) does where the paper's text says something else or less. The
release keeps the code's behaviour.

1. **Window size.** The paper bounds the input at 20 latents (4 memory, 12 recent, 4 target). The code adds a 21st:
   the sink, the round's recorded first latent, at position 0. Without a retrieved entry the window is 17.
2. **Attention of memory frames.** The paper says memory frames attend only to memory frames. There is no mask: the
   memory frames attend to the sink and to themselves (and the recent frames to everything before them).
3. **Context noise.** "Noised to 16" is the label; the applied level is t ~ 14.82 (above).
4. **Player states.** In Table 3 every player's position comes from the recording and visibility from GT labels. The
   closed loop (Tables 2 and 4a) is the `player_state.source: predicted` path below; its state model weights are not
   released.
5. **What a message carries.** The paper describes a message of four cameras and depths and a fetch of the one
   retrieved entry. In the code a reader fetches the latents of every admitted peer block (fp32, 774 kB per block)
   and recomputes their depth itself; depth is never sent.
6. **Every block a memory entry.** A block with no surface pixel is not stored; latent 0 is never published.
7. **Retrieval ties.** "The more recent on a tie" is `(t_last, seq, block)`: blocks of the same instant tie on
   `t_last`, so the reader's own ingestion order decides (own block first, then peers by slot).
8. **Headcount channel.** Capped at four *and divided by four*.
9. **Field composition.** Attribute channels are the top-2 weighted sums, but the live identity channel is
   winner-take-all, the corpse identity is argmax, and the dying / corpse planes are a max over players.
10. **Visibility gate.** Corpses skip it; for live players the only geometric test is depth > 1 (frustum and
    occlusion come from the GT labels).
11. **Kernel width.** The configured temperature 220 is capped to 72 on the 12 x 21 grid, which is exactly sigma =
    half a token.
12. **Observer signals.** The generator has an input the paper does not describe: flash and scope labels of the
    recorded video, embedded and added to every token of their frame.
13. **Fields of view.** Rays use each frame's field of view (scoped zoom included); retrieval uses one fixed field of
    view (tan 1.333 x 0.75) for every camera.
14. **Visibility re-test on x0** runs only with predicted states (closed loop); Table 3 uses labels.
15. **Eye height.** A constant 64 u (crouching only shrinks the projected body).

As the paper states: 30 DiT blocks with the field after block 2, 23 field channels, a 32-channel zero-initialised
stem, top-2 depth-discounted merge, confidence in [0.3, 1], B = 64 with older-on-tie eviction, z tolerance
max(24 u, 5%), holes from the 12 recent latents, own entries included, first read at latent 25, k = 1, depth head
44.2M / read-out 0.21M parameters, ladder 1000 / 750 / 500 / 250.

## Tests

`python -m pytest tests -q` (after `pip install -e ".[test]"`). The old-vs-new tests need the paper's research code,
which is not public, and are skipped without it. They find it through one environment variable, `WORLDCAST_OLD_TREE`
(the root of that code, with `Wan22/` and `shared/`), plus `WORLDCAST_DEPLOY_CONFIG` (its inference config) for the
few tests that read the paper's settings from there; without the `golden` extra installed they skip too.

- `tests/golden/<area>/`: each module against the paper code, CPU, exact equality.
- `tests/golden/player_state/`: the closed loop's modules against the research code it ran on
  (`WORLDCAST_CLOSED_LOOP_TREE`) and the state model's code (`WORLDCAST_STATE_MODEL_CODE`), with small random state
  model weights on both sides; the orchestration tests need neither.
- `tests/golden/engine/inference/`: three clients in lock-step, the paper's rollout code (`reconstituted_rollout` with
  its deploy driver, memory loop and live payload store) vs `Client` and `tools/run_session.py`; identical latents
  and pool directory. CPU, tiny random weights, float32 (the bf16 path needs CUDA autocast).
- `tests/reference/`: the paper run's fingerprints (GPU, real weights and data). `tools/verify_reference.py prefix`
  runs one client's plain prefix (one GPU, a few minutes) and `tools/verify_reference.py check` fingerprints
  finished sessions against the same table.
