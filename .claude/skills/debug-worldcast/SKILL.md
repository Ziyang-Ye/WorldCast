---
name: debug-worldcast
description: "Diagnoses WorldCast inference, real-time engine and demo failures. Triggers on: latents differ from the reference run, lock-step session hangs or times out, a client dies and the others stall, players slide or pass through walls, scene content pops between blocks, other players missing or extra, colour or brightness drift, OOM, high keypress-to-photon latency, the demo shows no frames, flash-attn missing or SDPA differences, training loss NaN or hang. Use when someone describes a symptom and wants the root cause."
license: Apache-2.0
---

# debug-worldcast

A debugging companion for WorldCast clients, the real-time engine and the web demo. When you describe a symptom,
this skill narrows down the likely root cause and points to a fix, with a check to run before changing code. The
reproducibility contract (`docs/inference.md`, "Numerics that the paper's numbers depend on") lists what must not
change; many root causes below are one of its items broken.

## Diagnosis Protocol

For each symptom, the suggested response is structured as:

1. **Pattern match** — which known failure mode this seems to correspond to
2. **Likely root cause** — a one-sentence explanation
3. **How to verify** — what to check before making changes
4. **Suggested fix** — what to consider changing, and where

## Where to look first

- `client.json` next to each `latents.npy`: per block `window` (21, or 17 when nothing was retrieved), `entry`
  (the retrieved entry's client and first source frame), `score`, `holes`, `candidates`, `admitted`; the lock-step
  `wait` counters (`n_timeouts` must be 0); the `follow` counters.
- The pool directory: `<pool>/<media_id>/win_<start>/blk_<f0>.{json,npy}` (published blocks),
  `steps/step_<t>.json`, `state/state_<t>.json` (closed loop), `DONE.json` (`ok`, or `failed` with the error).
- `client.log` of every client of a `tools/run_session.py` session.
- Real-time engine: `Engine(..., profile=True)`, then `engine.records` (per block: when the controls were taken,
  the ladder started, the first frame was ready; GPU spans from CUDA events).
- Demo: the coordinator and worker logs, `GET /health` on a worker, the HUD's latency breakdown.

---

## Known Failure Patterns

### Latents differ from the reference run

**Symptom**: `tools/verify_reference.py` or `tests/reference` reports a mismatch, or two runs of one config differ.
Bisect by fingerprint: `initial` covers the data path (latent 0, cast to bf16) and matches anywhere; `noise` the CPU
entry-noise draw (x86-64, torch 2.9.1); `prefix` the generator and sampler without peers; `final` the whole round.

**Root cause A — not the paper's stack**: another GPU, torch, or attention kernel.
- Verify: the environment `verify_reference.py` writes to its report (`attention`, `flash_attn`,
  `flash_attn_interface`, torch, CUDA, GPU). An importable FA3 (`flash_attn_interface`) is preferred over FA2.
- Fix: NVIDIA H20, torch 2.9.1+cu128, flash-attn 2.8.3 (FA2, no FA3). Elsewhere compare decoded videos with a PSNR
  threshold instead of bits.

**Root cause B — `initial` differs**: wrong index row, wrong first-latent file or member (`win_<start:06d>`), or a
re-encoded latent instead of the shipped latent 0.

**Root cause C — `noise` differs**: another CPU platform (arm64 draws differently), seed, or requested length. The
noise is drawn for the requested `run.latents`, then cut to the generated length.
- Fix: whole-round runs use `tools/run_session.py --round-length`, as the reference did.

**Root cause D — `prefix` differs on the paper stack**: an item of the reproducibility contract is broken (bf16
input and timestep cast, TF32, an extra global-RNG draw, camera encoding, observer-signal labels, op order).
- Verify: `verify_reference.py prefix` also writes `prefix.npy` and fingerprints of the prefix blocks (latents
  1-4, 5-8, ...); between a good and a bad run (before and after a change) the first differing block localises it.
  Check `sampler.model_input_dtype: bfloat16`, `sampler.tf32: true`, `data.camera_encoding: noclip`, and the
  attention backend.
- Fix: restore the broken item; never re-baseline the fingerprints.

**Root cause E — only `final` differs**: the scene state or the peers. Not every client of the round ran, a
timeout under `pool.fail_on_timeout: false`, clients as threads, a reused pool directory, or a changed ingestion
order or geometry dtype.
- Verify: `wait.n_timeouts` is 0 in every `client.json`; the per-block `entry` / `score` / `holes` equal between runs.
- Fix: run the round with `tools/run_session.py` (fresh pool, one process per client, lock-step, fatal timeouts).

Sessions without lock-step (the real-time engine's default for play, the demo) depend on timing by design.

---

### Lock-step session hangs or times out

**Symptom**: clients sit in the lock-step wait; after `pool.wait_s` (1800 s by default) they stop with
`PeerTimeoutError: ... peer(s) [...] have not published up to orig_last ...`.

The protocol cannot deadlock: each client publishes its step record before it waits and waits only for earlier
blocks. A wait that does not end means a peer that is not running or not progressing.

**Root cause A — a client of the round never started**: a client waits for every peer in its row's `group_media`.
- Verify: every row of the group has a process, all on the same `--live-pool-dir`.
- Fix: `tools/run_session.py --group-of ROW` (one GPU per client, or two clients per GPU).

**Root cause B — a peer died without marking itself failed** (killed, OOM-killed, machine lost).
- Verify: the peer's newest `blk_*` and `step_*` stop advancing and it has no `DONE.json`; read its `client.log`.
- Fix: see the next pattern.

While debugging, lower `--set pool.wait_s=...`. Do not "fix" a hang with `pool.fail_on_timeout: false`: the client
then stops waiting for that peer and its output depends on timing.

---

### A client dies and the others stall

**Root cause A — the client raised**: it marks itself failed (`DONE.json`, `status: failed`, the error as note) and
re-raises; its peers stop at their next wait with `PeerFailedError`. Read the failed client's `client.log`.

**Root cause B — the process was killed**: no `DONE.json`, so its peers wait out `pool.wait_s`.

- Fix: a session cannot be resumed. Fix the cause (often memory, see OOM) and rerun the round on a fresh pool.
- Real-time engine: the same policy over messages. In the demo, room lock-step waits at most
  `worker.lockstep_timeout_s` and a failed engine ends only its own seat. Keep
  `engine.worldcast.realtime.lockstep: false` there: with engine lock-step a silent peer blocks for `pool.wait_s`.

---

### Players pass through walls or slide

**Root cause A — own state from the recording**: with `own_state: gt` (`player_state.source: recorded`) the
client's cameras (rays, retrieval, the field's observer) follow the recorded track; live keys only enter as
controls, so the view and the keys disagree.
- Fix: the closed loop (`own_state: state_model`) needs a state-model checkpoint, which is not released yet.

**Root cause B — closed-loop extrapolation**: between state-model readings each player is drawn at its last
position plus the physics prior's displacement over the block, which has no collisions; the state model corrects
once per block (Eq. (6), weight 1/2). With early prefill, retrieval and the ray anchor use the controls held at
prefill time.
- Verify: the published positions (`state/state_<t>.json`, or the position messages) against the map.

**Root cause C — stale peers**: a client that publishes nothing holds its last position; in async play a peer is
extrapolated from its latest state. The demo's mock engine extrapolates peers at constant velocity for the HUD
only: the minimap shows the messages' positions, not what the generator drew.

---

### Scene content pops or flickers between blocks

**Pattern A — the memory slot changed or emptied**: at most one entry is retrieved per block; another entry, or an
abstain (17-latent window), changes what the block is conditioned on.
- Verify: does the pop line up with a change of `window` / `entry` / `score` in `client.json`? Retrieval queries
  the client's own cameras for the block (recorded, or extrapolated in the closed loop) and the depth head's output.
- Fix: if the reads equal the reference run, it is the model; if not, see "Latents differ".

**Pattern B — a change at latent 25, about 6 s in**: the first reconstituted block. The window switches from the
plain prefix (absolute positions, no memory) to sink | memory slot | recent | target. Expected.

**Pattern C — a seam at every block in decoded video**: the decoder must stream across blocks (the Wan2.2 VAE
carries its causal conv cache across chunks; `taew2_2` uses a streaming decoder). Decoding each block as its own
sequence restarts the decoder.
- Verify: the frame count is `1 + 4 (N - 1)`; a seam is a jump in the frame-to-frame change into each block's first
  frame against the change inside a latent frame (the measure the `subblock` section of `tools/bench_realtime.py`
  reports, `docs/latency.md`).

**Pattern D — a one-latent target**: `target_latents=1` is outside what the student was trained on and can blur and
lose scene structure (`docs/latency.md`: tested, not recommended).

**Pattern E — no lock-step**: which peer blocks have arrived depends on timing, so retrieval differs between runs.

---

### Other players missing or extra in a view

**Root cause A — the visibility gate**: a player is written into the field only when alive and visible in that
latent frame: GT labels in Table 3 (occluded and unknown read as not visible), predicted from depth in the closed
loop. A player who becomes visible inside a block is written at reduced confidence until the next block (an EMA of
visibility that restarts at every block, mapped to [0.3, 1]).
- Verify: `_binary_visible` of that player in `<visibility_label_root>/<media_id>.npz` at that time.

**Root cause B — the player state**: `alive` is 0 (dead, or past the recording's tick coverage); a round slot
without media loads as an absent player; `only_clients=True` makes unplayed seats dead and unseen; a stale or
extrapolated state draws the player elsewhere.

**Root cause C — extra players in the GPU demo**: the engine fills every seat of the round from the recording
unless the session is started with `only_clients=True` (`docs/demo.md`, "GPU integration"); recorded players then
move through the world beside the live ones.
- Verify: whether the demo adapter's `start` passes `only_clients=True`, and whether the extra players follow
  the recording.

---

### Colour or brightness drift over long rollouts

**Root cause A — recorded observer signals**: the flash and scope inputs come from the recording's labels, also in
live play; a flash label brightens the frames, a scope label narrows the rays' field of view.
- Verify: `lum` against `hot_threshold` in `flashlabels/<media_id>.npz` around the jump.

**Root cause B — an inexact option**: `attention: cudnn | fa3` and `compile` (and the benchmark's FP8 variant)
change the last bits of each step, and the rollout drifts away from the exact one over the seconds
(`docs/latency.md`). Judge such a rollout by its own frames (sharpness, frame-to-frame change), not by PSNR to the
exact one.

**Root cause C — the decoder**: `taew2_2` is slightly softer than the Wan2.2 VAE (`docs/latency.md`). The Wan2.2
decode casts the latent mean and std to bf16 before inverting the std, as the paper decode did; inverting first
changes the scale of some channels.

**Root cause D — the sink**: every window attends to latent 0, the round's recorded first latent. Another latent
(another start frame, a re-encode) changes the look of the whole rollout. Verify the `initial` fingerprint.

---

### OOM

- Memory per client: `README.md` (Install); one GPU runs two clients, with the same latents.
- **Text encoder on the GPU**: without `paths.prompt_embedding` the client builds umT5-XXL on its device. Use the
  shipped prompt embedding.
- **Wan2.2 VAE in the engine**: `decoder: wan` loads it on the generator's device unless `decode_device` is set; the
  demo's default is the tiny decoder (`decoder: taehv`).
- **Engines stacked on one GPU**: a worker without its own `CUDA_VISIBLE_DEVICES`, or `run.device: cuda` without an
  index, runs on its creating thread's current device, usually GPU 0. Give every worker process one GPU.
- Training: the DMD stage holds three 5B models and umT5 (`docs/training.md`).

---

### Real-time engine slower than expected / high keypress-to-photon latency

Decompose first: the HUD breakdown (sampling, network, waiting for the next block, generating, encoding, jitter
buffer) or `engine.records`; then `tools/bench_realtime.py --sections generation,decode,encode,loop` against
`docs/latency.md`.

- **The paper settings**: `RealtimeConfig()` is the paper client (eager generator, Wan2.2 VAE, lock-step). Use
  `RealtimeConfig.low_latency(taehv_path=...)`: fast generator, CUDA graphs, early prefill, overlapped decoding,
  `taew2_2`, JPEG, no lock-step; the latents stay exact.
- **The Wan2.2 VAE**: on an H20 it alone takes longer than real time per block (`docs/latency.md`).
- **Lock-step**: every block waits for the slowest peer, rounded up to the poll period (`pool.poll_s` for a pool
  directory, `RealtimeConfig.poll_s` for messages). Play on the latest messages.
- **The block itself**: an input waits on average half a block for the next one to start, then the ladder
  (`docs/demo.md`, "Latency"). On an H20, keeping fewer latents per generation does not lower it, and the one-latent
  target that does costs quality (`docs/latency.md`, "Sub-block generation").
- **Pacing**: without `Engine.step(..., not_before=...)` (or the demo worker's pacing) the controls are taken early;
  `play.action_mapping: realtime` costs about one more block; `PACING_MARGIN_S` (demo worker) can shrink on a
  quiet machine.
- **Warm-up**: CUDA graphs are captured on the second occurrence of each call shape; time steady-state blocks.

---

### The demo connects but shows no frames

**Root cause A — the browser cannot reach the worker**: the lobby goes through the coordinator, but frames and
inputs use the worker's own WebSocket (`coordinator.media_route: direct`) at `worker.advertise_url`, or the address
the worker connected from. The page shows "Lost the connection to the GPU worker."; after 30 s the worker releases
the seat ("the player did not connect").
- Verify: the coordinator's log line "browsers reach it at ..."; open `http://<worker>:<port>/health` from the
  browser's machine.
- Fix: a reachable `worker.advertise_url`, or `--set coordinator.media_route=proxy` on the coordinator (one more
  hop). Behind HTTPS: a TLS proxy in front of the coordinator and `proxy`.

**Root cause B — the engine failed to start**: "the engine failed: ..." in the page and the worker log. Usual
causes: a round start without a cached first latent (`docs/demo.md`, "Library"), data paths missing from
`engine.worldcast.configs`, or `own_state: state_model` without a state-model checkpoint.

**Root cause C — no worker**: joining answers "every GPU is busy", or "the GPU worker did not answer" when a worker
does not accept the seat within 10 s.

---

### flash-attn missing / SDPA differences

- **Not installed**: `flash_attention_backend()` returns `none`. Install after torch:
  `pip install flash-attn --no-build-isolation`, or run with `--attention sdpa`.
- **SDPA runs but latents differ**: expected. SDPA is a reference kernel, never bit-equal to flash-attention; CPU
  engines use it automatically.
- **flash-attn installed, bits still differ**: FA3 is preferred when importable; the paper ran FA2. The engine's
  `attention: cudnn | fa3` options are inexact by design.
- Verify: `python -c "from worldcast.modeling.wan22.attention import flash_attention_backend as b; print(b())"`.

---

### Training loss NaN or hang

The training code ships later (`docs/training.md`). Known traps:

- Every rank must run the same collectives in the same order. The memory-frame draw (`memory.prob`) is made per
  optimizer step and is the same on every rank; keep any new random choice rank-symmetric.
- Stages 2s, 3 and 4 decode frames and ray-cast collision meshes inside the loader workers (trimesh with embree,
  OpenCV, scipy): slow storage or slow ray casting there starves every rank and can look like a hang.
- Resume refuses another world size or per-GPU batch; `data.camera_encoding` has no default and differs by stage.
- Stage 4: teacher and critic start from the same checkpoint, so the first DMD gradient is exactly 0. Expected.

---

## Quick Triage Questions

If the symptom description is unclear, ask in priority order:

1. Offline client or session (`tools/run_client.py`, `tools/run_session.py`), the real-time engine, or the demo?
2. On the paper's stack (H20, torch 2.9.1+cu128, flash-attn 2.8.3)? What does `flash_attention_backend()` say?
3. Recorded player states (`player_state.source: recorded`, `own_state: gt`) or the closed loop?
4. Lock-step or latest messages? Did every client of the round run, on one fresh pool directory?
5. Which fingerprint differs first, and what are the last lines of the failing client's `client.log`?
