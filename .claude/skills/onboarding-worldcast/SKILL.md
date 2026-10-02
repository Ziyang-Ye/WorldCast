---
name: onboarding-worldcast
description: "Onboarding guide for newcomers to WorldCast, the distributed multiplayer world model. Two parts: Foundations (the client and its block loop, the window, player state and the player state field, scene state and retrieval, the lock-step pool, the state model and the closed loop, the training stages) and Pitfalls (non-obvious mistakes that break reproduction or quality). Use when someone starts reading or changing WorldCast code and needs the mental model, or wants to avoid known traps."
license: Apache-2.0
---

# onboarding-worldcast

Two parts:
- **Foundations** — the minimum needed to read the client, the sampler and the two shared states
- **Pitfalls** — non-obvious mistakes that break reproduction or quality

`docs/inference.md` is the reference for every number below, and its "Numerics that the paper's numbers depend on"
is the reproducibility contract.

---

## Part 1: Foundations

WorldCast gives each player a **client** on its own GPU: a video generator (Wan2.2-TI2V-5B made block-causal and
distilled to a 4-step ladder) and a **state model**. Clients do not generate jointly. They exchange two things:
**player states** (where everyone is) and **scene state** (the blocks they have generated). Adding a player adds a
client; there is no central model that grows with the number of players.

### The client and its block loop

A **block** is 4 latent frames = 16 video frames = 1 s at 16 fps. A client (`worldcast/engine/inference/client.py`)
rolls out one player's view of one recorded round:

```
start     TF32 on; generator (bf16), depth head, prompt embedding; the client window (recorded data)
          entry noise: one CPU draw for the whole rollout; sink = the round's recorded latent 0
prefix    latents 1-24: six blocks at absolute positions, KV cache grows to 25 latents, no memory
for s = 25, 29, ...                                     (t = the block's first source frame)
   1  own write        the own block that ended before t becomes a memory entry
   2  step record      publish own withdrawals and resident entries
   3  lock-step wait   every peer: its blocks ending before t, and its step record for t
   4  admit            peers' published blocks become memory entries
   5  follow           apply the peers' withdrawals (the copy is checked)
   6  retrieve         at most one entry: the one that fills the most holes
   7  window           sink | memory slot 4 | recent 12 | target 4    (17 latents if nothing retrieved)
   8  KV prefill       context written range by range at context noise 16
   9  4-step ladder    1000 / 937.5 / 833.3 / 625; player state field injected
  10  publish          the block's 4 clean latents (fp32)
end       latents.npy; decoding is a separate process (tools/decode.py, Wan2.2 VAE, streaming)
```

The real-time engine (`worldcast/engine/realtime/engine.py`) runs the same loop one `step()` per block, with the
controls taken right before step 9, and streams decoded frames.

### The window

```
position   0      1  2  3  4      5 ... 16              17 18 19 20
           sink | memory slot  | recent own latents    | target block
```

- Positions are window indices: they are the RoPE and control-history positions, not round time.
- No attention mask. The context is written to the KV cache range by range, `(0,1) (1,4) (5,4) (9,4) (13,4)`; each
  range attends to the cache so far and itself, the target to all 21.
- The paper's text says 20 latents and memory frames attending only to memory frames; the code that produced the
  paper's numbers has the sink and no mask (`docs/inference.md`, "Paper vs code"). The code wins.

### Player state and the player state field

- A player state is `[x, y, z, yaw, pitch, alive]` per latent frame plus the block's controls. In Table 3 every
  player's state is recorded; in the closed loop each client estimates its own and the clients exchange them.
- The **player state field** (`worldcast/player_state/field.py`) projects all ten players into the client's camera:
  23 channels on the 12 x 21 token grid (coverage, log depth, relative yaw, team, headcount, nine control
  fractions, a weapon embedding, dying and corpse planes, identity bands). Each player is a Gaussian splat of half
  a token; a token keeps its top two players by depth-discounted weight; a confidence in [0.3, 1] follows an EMA of
  visibility within the block.
- Only alive, visible players are written (GT visibility labels in Table 3, predicted from depth in the closed
  loop); corpses have their own planes. Memory frames carry no live players.
- Injection: the field is built at the injection point and added, through a zero-initialised stem
  (Conv 23 -> 32 -> 3072, `worldcast/modeling/state_injector.py`), after the second DiT block.
- The other inputs: controls (11 buttons, 2 camera, weapon; 20-row history) through per-block low-rank AdaLN
  adapters; Plücker rays of every frame's camera, relative to the first target latent, added before the first DiT
  block; observer signals (flash, scope of the recorded video) added to every token of their frame.

### Scene state: memory bank and retrieval

- A **memory entry** is one generated block with its four cameras and its depth. Depth is never sent: a reader
  fetches the fp32 latents of every admitted peer block and runs the depth head on them itself.
- A **memory bank** (`worldcast/scene_state/bank.py`) holds the client's own entries (at most B = 64; over the bound
  it withdraws the entry its other entries cover best, older on ties) and a copy of every peer's, which changes
  only by following that peer's published withdrawals.
- Retrieval (k = 1, from latent 25 on): the holes are the pixels of the next block's four cameras that no point of
  the 12 recent latents reaches; each candidate (own and peers', ended before t) scores the holes where it is
  front-most within max(24 u, 5 %); the best fills the memory slot. Score 0 means abstain: a 17-latent window.

### The lock-step pool

- The pool (`worldcast/engine/inference/pool.py`) is a shared directory in the paper's evaluation and messages in
  the real-time engine. Per block a client publishes its block, its step record and, in the closed loop, its
  position.
- Lock-step: before block s every client waits until every peer has published every block that ended before s.
  Each client publishes its step record before it waits and waits only for earlier blocks, so the protocol cannot
  deadlock; with no timeout the output does not depend on timing.
- A timeout (`pool.wait_s`) or a peer that marks itself failed stops the waiting client with an error.
- Live play runs without lock-step (the latest messages); the output then depends on timing.

### The state model and closed-loop deployment

- The state model (`worldcast/modeling/state_model.py`): a conv encoder and a 16-layer causal trunk with a control
  encoder, a motion head and a place head over 1,860 map cells. It reads 10-s windows of the client's own latents;
  a window still being generated is read with its future latents zeroed.
- Closed loop (`player_state.source: predicted`, `worldcast/player_state/closed_loop.py`): after each block the
  client reads its position at the block's four latent frames and fuses it with Eq. (6) (weight 1/2); it publishes
  the positions before the next block. Over block s every client is drawn at its position at latent s-1 plus the
  physics prior's displacement (`extrapolate.py`: keys at 64 Hz, jump and gravity, no collisions). Visibility is
  predicted from the depth head and re-tested once on the block's x0 after the first rung.
- Table 3 uses recorded states; the state model's weights are not released yet.

### Mental model of training

```
Wan2.2-TI2V-5B (bidirectional, pretrained)
        │  Stage 1 / 1b: bidirectional tuning on CS2 gameplay (5 s, then 10 s windows)
        ▼
bidirectional
        │  Stage 2: + player state field (zero-initialised stem)
        ▼
bidirectional + field ────────────────────────────────────────────┐  its EMA: teacher and critic of stage 4
        │  Stage 2s: + scene state (memory frames in the window;  │
        │            rays, observer signals, visibility head)     │
        ▼                                                         │
bidirectional + field + scene state                               │
        │  Stage 3: block-causal teacher forcing,                 │
        │           context noise in [16, 32)                     │
        ▼                                                         │
block-causal AR model (multi-step)                                │
        │  Stage 4: DMD with a self-forcing rollout  ◀────────────┘
        ▼
4-step block-causal student = the deployed client
```

Without scene state, stage 3 starts from stage 2 with a clean context and stage 4 follows (the Table-2 model). Each
stage's new modules start with zero-initialised outputs, so a stage starts as the model it was initialised from.
The training code is released later (`docs/training.md`).

---

## Part 2: Pitfalls

### Pitfall 1: "Fixing" a numeric that looks wrong

bf16 timesteps (the model sees 936 for 937.5), the context written at t ~ 14.82 under the label 16, ray pose math
in bf16, float32 eviction against float64 retrieval z-buffers: each is part of how the paper's numbers were
produced. A "fix" changes every latent after it. Leave them, and say why in a one-line comment.

### Pitfall 2: A stray draw from the global RNG

The ladder's re-noise draws from the device's global RNG after `set_seed`. A dropout left in train mode, a debug
`torch.randn` on the device, or a module randomly initialised after the seed shifts every later draw. Build modules
on `meta` and load weights; give new randomness an explicit generator.

### Pitfall 3: Several clients in one process

Clients as threads of one process are not bit-reproducible: the scene state's numpy reprojections differ under a
multithreaded BLAS. Run one process per client (`tools/run_session.py` does), and one real-time engine per process.

### Pitfall 4: Running part of a round, or reusing a pool

A client waits for every client in its row's `group_media`; started alone it never finishes. A pool directory
belongs to one session: republishing different bytes raises, and `tools/run_session.py` refuses a non-empty one.

### Pitfall 5: Treating window positions as time

Inside a window, positions 0-20 are window indices, and the control rows must be gathered in window order. The
plain prefix is different: absolute positions 0-24, no memory. Code that indexes controls or cameras by round time
inside a window is wrong even when it runs.

### Pitfall 6: The wrong camera encoding passes short tests

`noclip` and `clip` agree for turns under 20 degrees per frame, so slow or short test clips pass with the wrong one.
The released weights need `noclip` (`data.camera_encoding`). The training stages did not all use the same one
(`docs/training.md`).

### Pitfall 7: Judging reproduction on the wrong artefact or stack

Compare `latents.npy`, never the mp4. Bit equality needs the paper's stack (H20, torch 2.9.1+cu128, flash-attn 2.8.3
FA2); an installed FA3 is preferred by the kernel dispatch and changes the bits. CPU tests run the model in
float32. The entry noise depends on the CPU platform and on the requested `run.latents`, so whole-round runs use
`--round-length` as the paper did.

### Pitfall 8: Re-made inputs

Tick tables are checked by size and sha256, and the digest also keys the jump-button recall: a re-exported parquet
with other bytes is refused, and forcing it changes the controls. Use the shipped latent 0 of each client; a
re-encode is not bit-exact. A missing label file is an error, not "unknown".

### Pitfall 9: Live play still reads the recording

In the real-time engine, `own_state: gt` keeps the client's cameras and position on the recorded track; the keys
only drive the controls. The observer signals (flash, scope) always come from the recording. Seats nobody plays are
drawn from the recording unless the session drops them (`only_clients`).

### Pitfall 10: Latency options that change the bits

`generator: fast`, CUDA graphs, early prefill and overlapped decoding keep the latents exact. `attention` other than
`flash`, `compile`, and the sub-block modes do not, and an inexact rollout drifts away from the exact one over the
seconds: judge it by its own frames (`docs/latency.md`). Early prefill is exact with recorded states only.
