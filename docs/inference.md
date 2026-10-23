# Inference

How a WorldCast client runs, its configuration, the numerics the reference latents depend on, the data a client reads
and the released weights. The code is `worldcast/`, one module per concern (the repo map is in
[development.md](development.md#repo-map), the paper's sections in the
[README](../README.md#from-the-paper-to-the-code)); `worldcast/engine/inference/client.py` (`Client`) orchestrates.

## One client, many clients

Each player of a round runs one client on its own GPU. A client generates its player's view block by block (4 latent
frames = 16 video frames = 1 s at 16 fps) from:

- the player's own controls, and the other players' states projected into the **player state field** (23
  channels on the 12 x 21 token grid, added after the second DiT block);
- the **scene state**: at most one retrieved block of 4 latent frames (the client's own or another client's), placed in
  the window as memory frames;
- its recent context (its own latest latent frames) and the round's recorded first latent frame (the first frame).

Clients share nothing but the **shared world state**: once per block each publishes a message (its finished block as a
memory entry and its withdrawals) and reads the other clients'. In the paper's evaluation (and here) the shared world
state is a directory and the clients run in **lockstep**: before block `s` every client waits until every other client
has published every block that ended before `s`. The output does not depend on timing. Without lockstep (live play,
[docs/demo.md](demo.md)) a client reads the messages that have arrived. Each client is one process
(`tools/run_session.py` starts them so), never a thread of another client's process.

## The block loop

```
start ── config, TF32 on ── generator (bf16), depth head, prompt embedding ── the round's recorded data
  │
  ├─ entry noise  randn(1, latents-1, 48, 24, 42) from a CPU generator(seed); cut to N-1; bf16
  ├─ first frame  latent 0 of the cached first window (bf16)
  ├─ seed         set_seed(seed)   (python, numpy, torch, cuda)
  ├─ latents 1-24 six blocks at absolute positions 1..24, KV cache grows to 25 latents, no scene state;
  │               publish them
  │
  └─ for s = 25, 29, ..., N-4          (t = start_frame + 8 s, the source frame of the block's first latent frame)
       1  own write          scene.publish_own        the own block that ended before t -> memory entry
       2  withdrawals        world_state.publish_step own withdrawals (bound B = 64) and own resident entries
       3  lockstep wait      world_state.wait_for_others every other client: blocks up to t - 8, its withdrawals for t
       4  admit              MemoryBlocks.admit       the others' published blocks that ended before t -> entries
       5  follow             scene.follow_withdrawals apply the others' withdrawals; the copies must equal their lists
       6  retrieve           scene.retrieve           at most one entry: the one covering the most missing pixels
       7  window             window.gather_window     first frame | memory 4 | recent 12 | target 4 (17 if none)
       8  KV prefill         Sampler.prefill_context  context ranges re-noised to t=16, written in order
       9  denoising steps    Sampler.denoise          1000 / 937.5 / 833.3 / 625; field built at injection
      10  publish            world_state.publish_block the block's 4 clean latents (fp32)
  │
  └─ latents.npy (float32) ── DONE marker in the world state ── decode separately (tools/decode.py)
```

Window of a block from latent 25 on (positions are window indices, also the RoPE and control-history positions):

```
position   0      1  2  3  4      5 ... 16              17 18 19 20
    first frame | memory frames | recent context       | target frames
           own0 | retrieved    | own s-12 .. s-1       | own s .. s+3      21 latents
           own0 |                own s-12 .. s-1       | own s .. s+3      17 latents (nothing retrieved)
```

The context is written to the KV cache range by range: `(0,1) (1,4) (5,4) (9,4) (13,4)`. There is no mask: each
range attends to the cache so far and itself; the target attends to all 21.

## As a library

The pieces work on their own: the generator, the player state field of any player states, a client of any recorded
round.

```python
import torch

from worldcast.config import load_config
from worldcast.data import load_round_index_row
from worldcast.engine.inference import Client, ClientModels, ServingOptions
from worldcast.player_state import player_state_field

cfg = load_config(["weights/paths.yaml", "examples/data/mirage_r16/config.yaml"])
serving = ServingOptions(decoder="none", lockstep=False)  # one client on its own; latents, no video
models = ClientModels.load(cfg, serving)  # the generator (bf16), the depth head, the prompt embedding

# the player state field of one latent frame: ten players in a row along +x, seen by player 0
states = torch.zeros(1, 1, 10, 6)  # [x, y, z, yaw, pitch, alive] of every player
states[..., 0], states[..., 5] = 200.0 * torch.arange(10), 1.0
field = player_state_field(  # states, control substeps, the client's slot, teams, alive, visible, weapons
    states, torch.zeros(1, 1, 10, 16, 14), torch.tensor([0]), torch.full((1, 10), 2),
    torch.ones(1, 1, 10), torch.ones(1, 1, 10), torch.zeros(1, 1, 10, dtype=torch.long),
    models.generator.state_injector.weapon_embedding.weight.float().cpu(),
)  # [1, 1, 23, 12, 21]

# a client of a recorded round, block by block (4 latent frames = 1 s)
client = Client(cfg, serving, models=models)
client.start(load_round_index_row(cfg.paths.round_index, 0))
while not client.finished:
    list(client.step())
latents = client.latents  # [1, N, 48, 24, 42]
client.stop()
```

## Configuration

The config holds only what a user sets. The defaults are the settings of the paper's Table-3 runs, except the length:
Table 3 ran every round for its recorded length (`tools/run_session.py --round-length`), and `run.latent_frames`
defaults to 441. Only `paths` (and `run.index_row`) need setting: the weight paths come from `weights/paths.yaml`
(written by `tools/download_weights.py`), the data paths from a copy of `examples/data_paths.yaml` or an example
case's `config.yaml` (written by `tools/download_examples.py`).

| section | fields (default) |
|---|---|
| `paths` | `checkpoint`, `depth_head`, `depth_readout`, `prompt_embedding` (or `wan22_root` to encode it), the data paths ([Data](#data)), `world_state_dir`, `out_dir`; closed loop and live player states: `state_model`, `state_model_cells`, `physics_prior` |
| `run` | `seed` 20260917, `latent_frames` 441 (requested; under live play the length of the round the client keeps, [the live round](#live-player-states): 29 in the demo), `index_row`, `max_blocks` 0 (blocks after latent 24; 0: all), `device` (`cuda`: bf16; `cpu`: float32) |
| `model` | `attention` `flash` (flash-attention 2, the reference runs' kernel; `fa3`: FlashAttention-3 when it is installed; `sdpa`: PyTorch's kernel, without flash-attention; neither is bit-equal), `scene_state` true (`false`: `paths.checkpoint` is the four-step generator trained without scene state, stage `4_noscene`, not the paper's client; [docs/demo.md](demo.md#the-generator-trained-without-scene-state)) |
| `player_state` | `source: gt` (the GT states; `predicted`: the [closed loop](#closed-loop); `live`: the live demo's [live player states](#live-player-states), not the paper's) |
| `world_state` | `wait_s` 1800 (a late or failed client is an error), `poll_s` 2 |
| `scene_state` | `bound` B = 64 |
| `live` | live play's own settings, which `gt` and `predicted` never read ([an idle client](#live-player-states)): `idle_context` `none` (`freeze`), `idle_with_peers` false, `wall_keys` `held` (`edge`, `walls`, `guard`: [a stopped block's movement keys are cleared](#live-player-states)), `dead_zone_u` 0 (above 0: [a tap and a slow walk move the position estimate as far as the view followed them](#live-player-states)), `hop` `step` (`spread`: [the camera of a block starts where the block before it ended](#live-player-states)), `refresh` false (true: the newest block stays open and is denoised again from newer controls until the next block commits it, [the demo's refresh](demo.md#latency)) |

What the paper fixes is a constant of the module that uses it: the block of 4 latent frames, the recent context of
12 and the first window at latent 25 (`BLOCK`, `RECENT`, `FIRST_TARGET`, `data/latents.py`), the window's layout
and the KV capacity of 41 latents (`sampling/window.py`), the denoising steps and the context noise
(`sampling/sampler.py`), the generator (`GeneratorConfig`), the player state field (`PlayerStateFieldConfig`), the
recordings' sampling (`WindowSpec`: 32 -> 16 fps, `noclip` camera deltas), the eye height 64 u and the field of view
106.26 deg (`data/game.py`).

Command line: `--config a.yaml --config b.yaml` merges the files over the defaults in order; `--set section.key=value`
overrides one key, in every tool that takes `--config`: `--set model.attention=sdpa` runs a client without
flash-attention. The tools spell what they share one way: `--index-row` is a client's row of the round index, from 0
(`run.index_row`); `--out-dir` is a directory a tool writes into, `--out` one output file; `--device` is the torch
device (`run.device`). A config a tool cannot run with (an unknown key, a wrong value, an unset path) ends it with the
usage message.

## Numerics

What the latents of the reference runs depend on; each is explicit in the code.

- **bf16 inputs.** Parameters are bf16. Every floating input of a generator call is
  cast to bf16, *timesteps included*: the model sees 1000 / 936 / 832 / 624 for the denoising steps 1000 / 937.5 /
  833.3 / 625 and converts its flow prediction to x0 at those bf16 timesteps, while the re-noise between steps uses the
  unrounded ones. Poses, fields of view, controls and player states are rounded to bf16 too.
- **TF32** on for fp32 CUDA matmuls and convolutions (the DiT's fp32 islands, the fp32 depth head).
- **Context noise.** Every context write is labeled t = 16; the scheduler takes the nearest table entry (index 997),
  so the level is sigma = 0.014822 (t ~ 14.82).
- **RNG order.** Entry noise: a CPU generator seeded with `run.seed`, drawn once for the requested length and cut to
  N - 1 (a live client draws the same noise block by block, `EntryNoiseStream`). Then `set_seed(seed)` right before
  latent 1; the GPU's global RNG gives the 25 draws of latents 0-24 (first-frame write, then per block 3 denoising + 1
  cache write) and 3 denoising draws per later block. The context writes of later blocks draw from CPU generators keyed
  by the seed, the block and the range written (`worldcast.sampling.schedulers.paired_context_noise`).
- **dtypes of the data flow.** Latents 1-24 are bf16-rounded; a later block keeps its unrounded fp32 x0 in the
  client's store, which feeds later windows (cast to bf16 there), the depth head and the published payload
  (fp32). `latents.npy` is the bf16 output buffer, written as float32.
- **Camera encoding** `noclip` (mu-law without the +-20 degree clip), a constant of the recordings' sampling
  (`WindowSpec`; per stage in `STAGES` for training); a recording's turn and a live player's go through one function,
  `encode_turn`.
- **Camera geometry in bf16.** The pose math of the ray embedding runs inside the generator's bf16 autocast, so on
  CUDA the Plücker coordinates are bf16-rounded.
- **Retrieval geometry** stays in numpy on the CPU: float32 for the reprojection cache (withdrawal), float64 z-buffers
  for retrieval.
- **Predicted-visibility geometry** (the closed loop) has its own arithmetic, in torch float64 on the CPU
  (`worldcast/player_state/predicted_visibility.py`), beside the scene state's numpy geometry
  (`worldcast/scene_state/geometry.py`). The depth of the generated frames is `exp` of the depth head's output in
  float32 (`_depth_frames`); the re-test of a block's x0 takes it in float64 (`block_depth_frames`). Live play, not
  the paper's, tests the labels of a block's controls cut with this arithmetic operation by operation (`_ztest` in
  `worldcast/player_state/live.py`).
- **Inputs checked by content.** Tick files by size and sha256, never mtime (a copy changes mtime, not content); the
  sha256 also keys the jump-button recall.
- **Attention kernel.** The reference runs use flash-attention 2 (`model.attention: flash`);
  `worldcast.modeling.wan22.attention.flash_attention_backend()` names the kernel in use.
- **VAE of the latent cache.** The cache encoder (`tools/prepare_data.py latent-cache`) runs the Wan2.2 VAE in bf16,
  weights and pixels, on CUDA, and stores float16 windows ([docs/data.md](data.md#latent-cache)).

The latents of the reference runs are reproduced bit for bit on their stack: NVIDIA H20, an x86-64 CPU (the entry
noise is drawn on it), torch 2.9.1+cu128 and flash-attn 2.8.3 (FA2); the paper's serving measurements (Fig. 6,
Table 7) ran on H200. The reference runs are the paper's Table-3 runs,
nine of whose clients recorded fingerprints (three rounds x three clients, `examples/table3_fingerprints.json`),
and the runs of the example cases (`examples/manifest.json`). A fingerprint is the first 32 hex digits of the sha256
of a float32 tensor: of the first frame (latent 0 in bf16), of latents 0-24 and of all latents; the Table-3 runs
also recorded that of the entry noise. `tools/verify_reference.py` compares a run with them:

```bash
python tools/verify_reference.py client --config weights/paths.yaml \
    --config examples/data/mirage_r16/config.yaml --index-row 0 \
    --out-dir runs/verify/client                                     # one GPU: latents 0-24
python tools/verify_reference.py check runs/examples/mirage_r16      # finished sessions
```

## Closed loop

`--set player_state.source=predicted` runs the closed loop (Sec. 3.4, App. A; Table 4a with the released
generator). Each client takes nothing from the recording beyond its first latent, its round-start position and every
player's controls:

- **Own position.** After every block the client reads its own position at each of the block's four latent frames
  off the frames it generated, with the state model (`modeling/state_model.py`: convolutional encoder, 16-layer
  causal trunk, action encoder, motion head, place head over 1,860 cells), and fuses them with the complementary
  filter of Eq. (4) at the constant weight 1/2 from its round-start position. The model reads 10-s windows (41
  latents, stride 40); a window still being generated is read with its future latents zeroed.
- **Exchange.** Before each block it publishes those positions to `<world_state_dir>/<client>/state/state_<t>.json`;
  during latents 1-24 the clients also wait for each other's records there. A client that publishes nothing holds its
  last position.
- **Extrapolation.** Over block s every client is drawn at its position at latent s-1 plus its displacement over
  the block extrapolated from the controls (`paths.physics_prior`: movement keys at 64 Hz, per-key speeds, jump and
  gravity, no collisions); view angles integrate the player's own view controls. The client's ray embedding, its
  retrieval queries and the keys of its memory entries use its own extrapolated cameras. Players that run no client
  keep their recorded state.
- **Visibility** (`player_state/visibility.py`). Labels come from the depth head: test point feet + 40 u, 8 more on a
  40 u ring, median of the 2 x 2 depth cells, 24 u margin; the block about to be generated is tested against the
  last 12 generated latent frames, back-projected into its cameras, and re-tested after the first denoising step on
  the depth of its x0.

A closed-loop session in lockstep is reproducible like one on GT states: a client keys another client's memory entry
when it admits it (item 4 of [the block loop](#the-block-loop)), at the positions that client generated the block
from, whichever of the two runs ahead.

`weights/paths.yaml` carries the released state model, its cell table and the extrapolation's speed table
(`configs/state_model/`), so the switch alone runs it.

## Live player states

`player_state.source: live` is the live demo's, not the paper's. Each block is generated from its player's live
controls (`LiveBlockControls`: the rows of `BlockControls` with each video frame's raw turn and session time), and every
camera of the client follows them. `LiveRollout` is the block loop with a `LiveLoop` (`player_state/live.py`) in the
closed loop's place:

- **Own camera.** Over block s the client stands at its own position at latent frame s-1 plus the physics prior's
  displacement of the block's keys (its integrator goes on from where the block before left it: the integral of the
  round's keys); its view angles integrate the live turns. The state model reads the own position off the generated
  video as in the closed loop, but through `GuardedPosition` (`player_state/live_position.py`), not Eq. (4). The
  position steps with what the camera moved, as far as the motion head says the generated view followed it: where the
  prior's step is at least 40 u (0.65 of the running step; the head reads a slower step less surely) the length of the
  head's displacement, in any direction, is a share of the camera's step. The step is whole where the share is at least
  0.4, none where it is under 0.15, in proportion between, by the larger share of the newest two latent frames (one is a
  glitch of the head). The place head corrects the position behind a gate and a rate limit (xy and height each inside
  its own gate), so the camera never jumps from a correction, and never while the client stands still. No step is longer
  than the prior's top speed (245 u/s), and none leaves the map's walkable extent: the columns of the place head's cells
  of the map and a margin of one column (`Walkable`).

  A place head rejected on most latent frames is not followed (the position steps with the camera) unless it disagrees
  persistently and is credible. It disagrees persistently when its robust position (the median of its newest 5 place
  terms) is outside the gate of the position in 14 of 16 latent frames, has not jumped by more than the gate, and moves
  with the player (at least a quarter of the camera's net way over the 16: a head stuck on a cell is not followed). It
  is credible when it agreed with the position, inside the gate, on 16 of the last 64 latent frames, or the prior's step
  led out of the map's extent on 4 of the last 16 (a position at the edge is wrong by construction). It is then followed
  at 16 u per latent frame (a block's camera is off by 64 u) until it is inside the gate. A head that has not agreed
  lately and holds a steady track away from the position is as often lost as right, and the position keeps its dead
  reckoning. At the map's edge a camera that keeps pushing outward runs back the distance a correction gained: the
  position leaves the edge once the keys stop pushing outward.

  Walls: a block that stays on the ground stops along its way where the depth head of latent frame s-1, read when the
  block is prepared, sees a wall ahead of the block's start (`wall_clearance`: a vertical surface across the hull's way,
  not a slope or stairs). Right before a wall the depth head no longer sees its lower part, so the plane a block stopped
  before stops the later blocks too while the client stands or pushes on (moving within 15 degrees of the way it met the
  wall), until the block's start or the followed place head is more than 64 u past it, or the motion head says the view
  moved on where a wall stopped the camera (then the next block is not stopped either). A block that leaves the ground
  is never stopped, and the depth head sees no wall beside or behind the client. Every block that moves stops at the
  edge of the walkable extent, as the position does. Where the motion head found the view did not follow the camera, the
  next block keeps a quarter of its way along the step, at least, while it holds the same movement keys and turns less
  than 15 degrees: the view follows a camera that moves, so a block that keeps moving shows whether it moved on, and a
  glitch of the head costs one slowed block, not the camera. A block cannot know that the view will stand still: where a
  push meets an obstacle the generator renders, that block's camera runs on and the position falls at most 64 u (two
  body widths) behind it within the block; the next block's camera starts from the position. With `live.wall_keys: edge`
  a block's movement keys are cleared from the row its camera is stopped at by the edge of the walkable extent, with
  `walls` also from the row a wall stops it at, in the controls the generator, the state model and the block's `live`
  part read; `held`, the default, leaves them. `guard` clears them only where a wall stops the camera, never at the
  edge of the extent (the camera stops there, the keys stay), for at most 3 blocks in a row: the block after them keeps
  its keys, as a probe, and so does every block once the keys change, are all released, or the turn takes the way off
  the stop, and once a block with cleared keys showed the view moving (the motion head) the clearing ends until the
  keys change, so that a player is never held at a stop the picture does not show.
  `live.dead_zone_u` above 0 (u per latent frame; 4 is the motion head's noise floor) reads the state model's motion
  head for a step shorter than the running one too, a tap or a slow walk: a displacement of the view below it is a
  view that stands, and the position estimate moves with the camera only as far as the view followed it, so that the
  next block does not start from a position the picture never reached. With `live.hop: spread` the camera of a block
  starts where the block before it ended and runs into the position estimate over the block (linearly, the last
  video frame is the estimate's), in the rays, the player state field and the rows the client publishes alike, so
  that what the view did not follow of the camera, and the place head's corrections, show as a glide and not as a
  step at the block's start that moves every other player's place in the picture. `step`, the default, and `0` leave
  the camera as it is. A turn that would take the pitch beyond 89 degrees is cut there, and its
  pitch view controls encoded again, so that every camera integral agrees.
  The player state field's own row, the rays, retrieval, the keys of the client's memory entries and its visibility
  labels all use this camera, as prepared and as cut.
- **The controls cut.** A block that reads the scene state is prepared before its controls with the last controls held:
  the newest row's buttons and weapon, and the newest block's turn row by row, so that a block that turns as the one
  before is prepared as it is cut, however its rows share the turn (the demo's `play.input_mapping`). At the cut, right
  before the denoising, the client takes in the parts that have arrived and writes the block's rows once: its own, and
  the other live players' at the block's session times. The prepared window's target frames are written again (rows,
  controls, observer signals, continuous columns, cameras, and visibility labels tested again against the same context);
  the anchor of the rays stays the camera the window's context was written with. The rows the block is denoised with
  leave as its `live` part; no position is published. The cut computes only what the block's controls change: the
  prior goes on from the integrator the block before left, the continuous columns of the block's four latent frames
  are written on their own, and the labels test the surface of the context read when the block was prepared (latents
  1-24: before the cut), reprojected on the client's device, once per set of the target frames' cameras (a cut whose
  rows are those its block was prepared with reprojects nothing); a block of latents 1-24 writes its own latent frames
  of the round's conditions; another player's newer part is extended only as far as the block's times need. The cut's
  arithmetic (the prior's substep scan, the view angles, the labels' z-test, the continuous columns) is torch's
  operation by operation, run in numpy but for its sums, cumulative sums, norms, einsums and trigonometry: a small torch
  call drops the GIL, and on a worker whose threads compete for it each costs tens of microseconds.
- **Other players.** A player who plays live is drawn at the rows its own camera used, paired by session time
  (`LivePeer`, `player_state/live_peers.py`): between received rows interpolated, past the newest extended by its held
  keys, then coasting and frozen, hidden after 3 s of silence; a newer part is blended in over the block however far it
  turns the player, so that it moves on the view instead of vanishing for a block, and only an offset beyond 1024 u
  (more than a player runs while it is drawn past its newest row) hides one block. A blended offset moves the player a
  quarter of it per latent frame on top of its own motion: one of 1024 u at about four times running speed. Only players
  a human plays are in the world: a slot becomes live with its first `live` part, whoever the client started with, and
  is drawn from then on; a seat nobody plays is nowhere, and the round's recorded players are never drawn, after the
  first frame as in it, for a seat whose first frame shows nobody. The demo's library offers only such seats and its
  client refuses another ([Library](demo.md#library)): the first frame, which every window attends, would hold people
  the field does not draw, and the first six blocks, which read the dataset's labels of latent frame 0, would list them
  there where a window does not. A player's memory entries are keyed at the cameras of their `live` parts. A player
  whose client is done is hidden from the next block, and one who takes its seat is drawn where it is from its first
  block, not blended from where the leaver stood. No slot is ever written dead, so nobody is drawn as a corpse and a
  seat can be taken at any time.
- **Observer signals.** Nothing flashes and the scope is off. While a weapon that zooms is held the scope is unknown
  (the learned "unknown") and the field of view stays unscoped: live play does not track the zoom, so a scoped view is
  not modelled. No block reads the recording after the first frame: every other player's rows stand at its first row,
  and the memory frames of a retrieved entry are made from its own `live` part (the turn in it is the turn the player
  made: the demo's `hybrid` input path feeds the generator a gain on the yaw, which is not in it).
- **An idle client.** `live.idle_context: freeze` keeps a client whose player holds still from feeding its own last
  blocks back into its context (`IdleContext`, `engine/inference/idle_context.py`; `none`, the default, leaves the block
  loop as it is, and `configs/demo/worldcast.yaml` sets `freeze`). A block is idle when no key is held on any of its
  frames and its view controls are all zero (the generator's encoding rounds a frame's turn under about 0.12 degrees to
  zero; [docs/demo.md](demo.md#gpu-integration) gives the turn that is for each input mapping), its cameras are within
  0.5 u and 0.5 degrees of the first idle block's of the run and it holds the same weapon. The first three blocks of
  such a run, a stretch, are the clip and are kept as generated; the blocks after them are generated and shown as
  always, but the store and the memory bank keep the clip's blocks in their place, in turn, so the context of every
  block is the clip however long the stretch lasts, and the memory entries other clients read are the clip's too. A
  block that is not idle ends the stretch: it is kept as generated and generated from the clip's context, and the next
  idle blocks make a new clip. The first six blocks are never idle.
  `live.idle_with_peers` says what other players do to it. `false`: a block is idle only with nobody else in the world
  over its recent context and target frames. `true` (the demo's setting): other players may be in the world as long as
  none is drawn, that is, labelled visible on a video frame of the block's recent context or target frames (the player
  state field of every such frame is then empty, as it is for a client alone); a block that draws a player is not idle.
- **The round.** A live session has no last block. Its round is the shortest the loader allows: the first frame, the
  first six blocks and one more (29 latent frames), and after each block from the seventh on the round slides: the
  frames after the first two (video frames after the first five) move up by a block, and the next block is generated
  at the same latent frame (25) of the same round. The window of a block (first frame, memory frames, recent
  context, target frames) reads only the first frame and the latent frames around the block, and the first frame is read
  with latent frame 1 (the depth head reads each latent frame with its neighbour), so a round that holds those gives every
  block the window a longer round would (`slide_window_entries`, `sampling/window.py`; a longer round, `run.latent_frames`,
  gives the same blocks). Everything the session counts is the session's, the round's latent frame plus the
  latent frames the round has slid by: the blocks' names, the source-frame times of the memory entries and of the causal
  cut, the keys of the context-write noise, the `live` and `scene` parts, and the page's frames. The entry noise is the
  paper's draw taken block by block (`EntryNoiseStream`: block `s` is `entry_noise(...)[:, s - 1 : s + 3]` of any length),
  so a seed gives a live client the noise of the evaluation's. A block's memory entry keeps its latents and
  its `live` part (`LiveMemoryBlocks`) for as long as the memory bank holds the entry, which can be longer than the
  round and than the message world state (`MessageWorldState(retain=...)`: the newest max(2 B, 6) blocks of each
  client in all its indexes; the first six entries are published together, after the sixth block). The state model
  reads its own latents through a window of its own (`LiveStateReader`: at most 41 latent frames and the live ticks of
  one window, a window of controls that equals the whole stream's bit for bit), and the continuous columns of a block
  are taken on from the block before (`continuous_block_state`: the yaw and pitch from latent frame s-1 with the turn
  of every row, the time since death of a player dead or absent from the first row counted from the session's first
  row), so no cost of a block depends on how many have been played. Every other player stands at its first row on
  every video frame until it is in the world (`hold_others`), so that the rows the slide vacates, which repeat the
  round's last, never differ from what a longer round holds. Nothing a block reads or recomputes grows with the blocks
  played. What no block reads is bounded too: a live client keeps the records of its newest 8 blocks
  (`Client.records`), the scene state keeps the key of another client's entry until that client withdraws it and a step
  record until the world state has none of it (`_uid_at`, `_empty` and `_followed`), and the list of known blocks
  holds a pointer a block (8 bytes: the scene state and the bank refer to a block by its place in it), the one residual,
  with the blocks skipped (`MemoryBlocks.skipped`) of a session with other players. A client keeps the rows and pitch
  the page reads of its newest 8 blocks only (`Client.field_rows`, `Client.camera_pitch`: a block older raises
  `ValueError`), each taken when its block was committed: whatever logs per block must log at the commit. A client
  that joins when another has played 2 B - 1 blocks or more uses none of that player's entries: it admits an entry
  once its own clock (its play time) has passed the entry's block, and by then the world state has retained the block
  out (`LiveMemoryBlocks.expired` counts them), while a joiner a few blocks behind admits and retrieves the earlier
  player's entries as two clients that start together do.

It needs the closed loop's paths, a world state over messages (the live demo's; `run_client` refuses `live`), each
block's live controls and a round on a map the state model knows. `Client.field_rows` says where each row comes from
(the client's own, a live player's, a recording's under `gt` and `predicted`, or hidden: every other player in live
play who is not in the world), which players were labelled visible at each block's
first denoising step, and the status of the own position's estimate (`GuardedPosition.status`): `reacquiring` while a
persistently disagreeing, credible place head is followed to where it is, `degraded` while the place head is not
followed (at least `DEGRADED_REJECTIONS` (12) of the last `DEGRADED_WINDOW` (16) latent frames' place estimates were
rejected by the gate, and it does not disagree persistently with a credential), else `guarded`.

## Data

A client renders one player of one recorded Counter-Strike 2 round of the
[OpenCS2 dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset) (CC BY 4.0). A *recording* is one
player's view of one round: its video and its tick table in OpenCS2, under the id `media_id` =
`<match>-<map>-r<round>-p<slot>` (the files below keep the dataset's word, "media", in their names and keys). A
client never reads the recorded video: it reads the recorded engine state and controls of all ten players, the
player's GT visibility labels, two observer-signal curves and the round's first latent. Every path comes from the
config; a missing file is an error.

| artifact | config key | file | used for |
|---|---|---|---|
| round index | `paths.round_index` (+ `run.index_row`, from 0) | JSONL, one row per client | which player, which start frame, the round's other clients |
| media index | `paths.media_index` | JSONL, one row per recording | `media_id` -> tick file, player slots of a round, source-frame count |
| tick tables | `paths.dataset_root` | `rounds/match_id=<m>/map_name=<map>/round=<rr>/player=<pp>/ticks.parquet` | states, controls, substeps, team of all ten players |
| first latents | `paths.latent_cache_root` | `<media_id>.npz`, member `win_<start:06d>` | latent 0 = the window's first frame |
| visibility labels | `paths.visibility_label_root` | `<media_id>.npz` | GT visibility of the other players (gates the player state field) |
| observer signals | `paths.observer_signal_label_root` | `flashlabels/<media_id>.npz`, `scopelabels/<media_id>.npz` | flash and scope inputs of the generator; scoped field of view |

`examples/data_paths.yaml` lists them; copy it to `data/paths.yaml`, edit the paths and pass it as an extra
`--config`. `tools/download_eval_data.py` writes such a config for the data of the paper's evaluation sets, e.g.
`data/evaluation/eval/rounds.yaml` for the rounds of Table 3 ([docs/reproduce.md](reproduce.md)).

**Round index** (one row per client; Table 3's, `configs/eval/round_index.jsonl`, has 96 rows: 32 rounds x 3
clients). Fields read:

```json
{"media_id": "2393226-de_mirage-r16-p00", "start_frame": 0, "match_id": 2393226, "round": 16,
 "map_name": "de_mirage", "latent_key": "win_000000", "player_slot": 0,
 "group_media": ["2393226-de_mirage-r16-p00", "2393226-de_mirage-r16-p01", "2393226-de_mirage-r16-p02"]}
```

`media_id` is the client's recording. `start_frame` is a source frame (32 fps); `latent_key` must equal
`win_<start_frame:06d>`. `group_media` lists the round's clients (this one included); a row whose `group_media` is
only itself runs alone. `tools/run_session.py --round-length` also reads `round_seconds` and requests
`1 + 4 * 10 * floor(min(round_seconds, 120) / 10)` latent frames (the cap is `--max-seconds`).

**Media index**: `media_id, match_id, map_name, round, player_slot, fps` (32), `video_frames` (source frames),
`ticks_path, ticks_rows, ticks_file_size`, `ticks_sha256` (optional), `capture_start_tick` (optional; equal within a
round), `video_path` (training). A player slot without a recording loads as an absent player (dead, silent, team 0).
`tools/prepare_data.py media-index` builds it from an OpenCS2 download and writes these thirteen fields
([docs/data.md](data.md#media-index)).

**Tick tables** (64 Hz, one per player, all starting at the round's capture start): `tick, t` (s), `x, y, z` (feet,
engine units, 1 u = 0.0254 m, +z up), `yaw, pitch` (degrees, pitch > 0 looks down), `is_alive`, `active` (held
buttons), `delta_pitch, delta_yaw` (degrees per tick), `input_weapon`, `team_num` (2 T, 3 CT). File size and, when the
index carries it, sha256 must match the media row.

**First latents**: the latent cache of training ([docs/data.md](data.md#latent-cache)): `<media_id>.npz`,
member `win_<start:06d>`, float16 `[1, 41, 48, 24, 42]`. Only latent 0 is read (widened to fp32, cast to bf16 by the
client); the example cases and the evaluation data ship it.

**Visibility labels**: `<media_id>.npz` with `_binary_visible`, `_binary_eval_valid`, `_in_frustum`,
`_binary_offscreen`, each `[10, T]` bool, one column per source frame (`T` is the media index's `video_frames`; engine
line-of-sight tests). Visible means confirmed visible; occluded and unknown both read as not visible. A latent counts
a player as visible if any of its video frames does. The file holds more arrays than these four
([docs/data.md](data.md#gt-visibility-labels)).

**Observer signals**: `flashlabels/<media_id>.npz` (`lum` float16 per source frame, `stride`, `hot_threshold` 0.85)
and `scopelabels/<media_id>.npz` (`scoped_vis`, `level` 0/1/2, ... per source frame), computed from the recorded
video; a client requires both files of a recording ([docs/data.md](data.md#flash-and-scope-labels)). Per latent: flash
= any of its video frames brighter than the threshold; scope on / level at its last video frame; `valid = 0` where a
curve cannot be sampled.

**A client's length** (`worldcast/data`): `N` = `run.latent_frames` (441 = 110 s), clipped to the player's tick
coverage and to `run.max_blocks`, rounded down to `1 + 4k`, at least 29 (latents 0-24 and one block). Under live
play `N` is the length of the round the client keeps, whose first 29 latent frames' 7 s are all the recording a seat
needs after its start frame; the session has no length ([the live round](#live-player-states)). Video frame
`k` is source frame `start + 2k` (16 fps); latent `f` covers video frames `4f-3 .. 4f`. States are sampled and held
from the last tick at or before each video frame (`alive` is 0 past the coverage); controls are the 11 buttons OR'd
per frame, the camera deltas summed and mu-law quantized (`noclip`) and the weapon id (52-way), with four ordered
substeps per frame for the field; cameras are camera-to-world from the recorded position (eye 64 u above the feet),
yaw and pitch.

The six example rounds ship in this format: their round and media indices in `examples/data/<case>/`, the rest
fetched by `tools/download_examples.py`. Those of the paper's evaluation sets, the 32 rounds of Table 3 among them,
are fetched by `tools/download_eval_data.py`, and the files of other rounds are built from OpenCS2 with
`tools/prepare_data.py` ([docs/data.md](data.md)).

## Weights

The files are in [ZiyangYe/WorldCast](https://huggingface.co/ZiyangYe/WorldCast). `tools/download_weights.py
--out-dir weights` fetches the files a client reads into `weights/worldcast/`, the Wan2.2 files into
`weights/Wan2.2-TI2V-5B/` and the tiny decoder to `weights/taew2_2.pth`, and writes `weights/paths.yaml`, which sets
the config keys below; `--with-training-checkpoints` then adds the training checkpoints to `weights/worldcast/`. It
also fetches the repository's `config.json`, a description of the release that no client reads: the Hub counts
downloads by it.

**What a client reads.**

| file | config key | what | format | size | sha256 |
|---|---|---|---|---|---|
| `worldcast_4step_bf16.safetensors` | `paths.checkpoint` | the four-step generator, EMA weights, stage 4 step 600 (Table 3) | 946 tensors, bf16, `WorldCastGenerator` names; metadata `{format: pt, stage: 4, step: 600}` | 10,196,422,752 B | `7cfd85b59de04968814a5d560cf6258c91ede0d3a850bd759fe5cc91c5c35b31` |
| `state_model.safetensors` | `paths.state_model` | the state model (Sec. 3.4; [Closed loop](#closed-loop)); 349,467,194 parameters; its cell table and the extrapolation's speed table are in `configs/state_model/` | 301 tensors, fp32, `StateModel` names | 1,399,660,936 B | `86fbf7e11418c568c2a33c8163774b52d1392dde9619e0dc5d9b277251547b23` |
| `depth_head.safetensors` | `paths.depth_head` | the depth head (width 384, 10 blocks, 5 half-resolution blocks; 44,199,600 parameters) | fp32, `DepthHead` names | 176,809,840 B | `ab599fcd68115142cf3b946e147f3cb89465d0b389371cc3501e1eb3101f6e76` |
| `depth_readout.safetensors` | `paths.depth_readout` | its read-out (width 128; 208,132 parameters) | fp32, `DepthReadout` names | 833,352 B | `6994480d726b7e958f519e835fa306b7045cfb19171ade4aebcb594a67ccd873` |
| `fixed_prompt_umt5xxl_bf16.safetensors` | `paths.prompt_embedding` | umT5-XXL embedding of "first-person Counter-Strike 2 gameplay" (`tools/make_prompt_embedding.py`) | `[1, 512, 4096]` bf16, rows 10-511 zero | 4,194,520 B | `a4157803a2c381835b219c079b7d811b7950c3fb2c47741d4552b7bba37cbd0a` |

**Training checkpoints** (`--with-training-checkpoints`).

| file | what | format | size | sha256 |
|---|---|---|---|---|
| `worldcast_stage3_ar_fp32.safetensors` | stage 3: block-causal with scene state, EMA, step 5000 (initialises stage 4) | 954 tensors, fp32, with the visibility probe | 20,396,189,916 B | `a8d39e8b88a0a9e8bd0d65ac804755778ece76839993667055e02eddf878e2c0` |
| `worldcast_stage2s_bidirectional_fp32.safetensors` | stage 2s: bidirectional with the field and scene state, EMA, 20,000 + 5,000 steps (initialises stage 3) | 954 tensors, fp32, with the visibility probe | 20,396,189,924 B | `83704993f1f6024ce9a4f65987d2dd69dc6172a39a038a3668b2c28dd26fa77b` |
| `worldcast_stage2_bidirectional_fp32.safetensors` | stage 2: bidirectional with the field, EMA, step 25,000: "WorldCast" of Table 1 (teacher and critic of stage 4; initialises stage 3 without scene state) | 930 tensors, fp32, without the scene-state modules | 20,353,235,472 B | `3af10355a31e19069dad8a810d38b31cd041c4d5af4460d4d4eb6966af09445e` |

**Third-party files**, fetched from their own repositories at a pinned revision.

| file | config key | what | size | sha256 |
|---|---|---|---|---|
| `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | `paths.wan22_root` | `config.json`, `Wan2.2_VAE.pth` (decode), the umT5 tokenizer; with `--with-t5` the umT5-XXL encoder | VAE 2,818,839,170 B | VAE `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |
| `taew2_2.pth` from `madebyollin/taehv` @ `011dfc2112197741c540e0bdd5b7b67bcc930771` | `client.worldcast.taehv_path` of the demo | the tiny Wan2.2 decoder of the live demo (MIT; 9.9M parameters; `--skip-taehv` skips it) | 22,884,021 B | `d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a` |

The four-step file holds the EMA weights of the step-600 generator cast to bf16, the parameter dtype of the paper's
client; the training checkpoints keep fp32, so training from them starts from the exact weights. Not released: the
models of stages 1 and 1_long.

**Generator names.** The four-step file has 946 tensors, 5,098,162,792 parameters, under the names of
`WorldCastGenerator.state_dict()`: a strict `load_state_dict` loads it. The training checkpoints add the visibility
probe of stages 2s-4 (8 tensors, 860,161 parameters): 954 tensors. The stage-2 file has neither the probe nor the
ray and observer-signal embeddings: 930 tensors, 5,088,284,952 parameters; stage 4 builds it as both the teacher and
the critic. The bidirectional and the block-causal stages are the same `WorldCastGenerator` (the attention mask is a
training input, not a weight). The checkpoints of `tools/train.py` (`.pt`, the same names under a `model.` root) of
the stages with scene state load as `paths.checkpoint` too ([docs/training.md](training.md#checkpoints)):
`worldcast.modeling.build.read_generator` strips the root, and the inference generator is built without the
visibility probe. A client runs the generator with the player state field and the scene state: `ClientModels.load`
refuses a checkpoint without them and names what it lacks (`tools/evaluate.py` scores such a model).

**Architecture.** Wan2.2-TI2V-5B made block-causal (30 blocks, dim 3072, 24 heads x 128, FFN 14336, patch 1x2x2, 48
latent channels, text 512 x 4096) plus the WorldCast inputs: the control embedding with per-block low-rank AdaLN
adapters, the observer-signal embedding, the ray embedding of Plücker coordinates (length scale 420 u) and the state
injector (Conv 23 -> 32 -> 3072: the player state field, added after DiT block 2).

**Loading** (`worldcast.modeling.build.load_generator`): the file is read by
`worldcast.utils.weights.read_state_dict` (a `.safetensors` file as it is; a torch file with `torch.load(...,
weights_only=True)`, memory-mapped when it is a zip archive, as torch writes them); `build_inference_generator`
then builds the model on the `meta` device, with the optional modules the file holds, assigns the weights, casts
them to bf16 and moves the model to its device. Both functions default to `device="cpu"`, where the model holds the
bf16-rounded weights in float32 and attends with `sdpa_attention`; a client passes `run.device` (`cuda`). The Wan2.2
backbone weights are never read: the checkpoint replaces every one.

**Checking a download.** `sha256sum weights/worldcast/*.safetensors` (macOS: `shasum -a 256`) against the tables.
