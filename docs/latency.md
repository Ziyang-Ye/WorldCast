# Latency

How long a key press takes to reach the screen when one WorldCast client is served live, where that time goes, and
the settings of `worldcast.engine.realtime.Engine` that shorten it. Every number below was measured with
`tools/bench_realtime.py` (one run per row unless stated).

**Setup.** One NVIDIA H20 (96 GB; 148 TFLOPS dense bf16), torch 2.9.1 + CUDA 12.8, flash-attention 2.8.3, the
released 4-step student. One client of the paper's Table-3 index (Dust2, round 9, row 59), recorded player states
and controls, no peers (the lock-step section has three), the plain prefix and 24 blocks after it (30 s of video).
Per-block numbers are p50 / p95 over the blocks after the prefix and three warm-up blocks.

## Keypress to photon

A block is 4 latent frames, 16 video frames, 1 s at 16 fps, and its 16 frames are denoised together, so their
controls must all be known before its 4-step ladder starts. A key pressed at a random moment therefore:

1. **waits for the next block to take its controls** (input lookahead: up to one generation period);
2. waits for that block's **generation** (with early prefill only the ladder is left at this point);
3. waits for its first latent frame to be **decoded**;
4. is **encoded** and sent to the browser (network and display: assumed 5 ms one way on a LAN and 10 ms in the
   browser, not measured here);
5. in lock-step sessions, waits for the slowest **peer** before the block starts.

| H20, one client | before | after (exact) | after (fast, not exact) |
|---|---|---|---|
| keypress to photon, p50 / p95 | 3.38 / 4.72 s | 1.02 / 1.48 s | 0.97 / 1.42 s |
| of which input lookahead | 1.49 / 2.83 s | 0.51 / 0.97 s | 0.50 / 0.95 s |
| controls taken to first frame ready | 1.88 s | 0.49 s | 0.40 s |
| frames per second | 5.4 | 15.6 | 16.0 (real time, 0.81 s of generation a block) |
| latents | paper | bit-identical to the paper | differ (same quality, below) |

*Before*: the release client's pieces in a loop: take the controls, generate the block, decode it with the Wan2.2
VAE, encode, show. *After (exact)*: `RealtimeConfig.low_latency(taehv_path=...)`. *After (fast)*: the same plus
`compile=True, attention="fa3"`.

## Budget of one block

| part (H20, ms, p50 / p95) | before | after (exact) | on the latency path after? |
|---|---|---|---|
| scene state: own write (depth head) + retrieval | 9 + 36 / 44 | 9 + 36 / 46 | no (before the prefill) |
| window and conditions | 7 / 14 | 6 / 9 | no |
| KV prefill, 5 context writes | 767 / 814 | 492 / 495 | no (early prefill) |
| 4-step ladder | 702 / 712 (175 per step) | 475 / 481 (124, then 116 per step) | yes |
| of which the player state field | 46 | 32 (built once per block) | yes |
| VAE decode of one block | 1423 / 1431 (Wan2.2) | 8 / 8 (taew2_2) | first latent frame: 360 vs 3.3 |
| JPEG encode, 16 frames (q 90) | 3.7 (nvJPEG) | 27 (libjpeg; nvJPEG 3.7) | first frame: 1.7 |
| network + browser | 15 (assumed) | 15 (assumed) | yes |
| lock-step wait for the peers | 1.2 s average in the paper's runs (2 s poll) | 0 (latest messages); 25-89 ms mean in lock-step | yes, with lock-step |
| generation period (block to block) | 2.97 s | 1.02 s | |

With the exact settings the H20 generates a block in about 1.0 s, so the loop runs just below real time: the
generation period, not the display, sets when the next controls are taken. The fast settings bring the prefill and
the ladder to 421 + 392 ms (0.81 s a block), enough for 16 fps with time to spare; the engine then takes the
controls as late as the display allows (`Engine.step(..., not_before=...)`, or the demo worker's pacing).

## What changed

**Bit-identical** (same latents as the paper client, checked on the H20 for a whole session and on the CPU by
`tests/engine/realtime`):

- *Fast generator call* (`generator="fast"`): the KV cache's end indices on the host (no device sync per layer),
  the token grid as Python ints, flash-attention with device-resident sequence offsets, the text embedding and the
  30 blocks' cross-attention keys/values computed once per prompt, one rotary table per call, the patch embedding as
  one GEMM (bit-equal to the bf16 Conv3d it replaces on CUDA), and the ladder-invariant inputs (controls, rays,
  observer signals, player state field) built once per block instead of once per rung. 1451 -> 1038 ms a block.
- *CUDA graphs* (`cuda_graphs=True`): the 30 blocks, the field injection and the head of each recurring call shape
  (7 in this session) are captured on their second occurrence and replayed. 1038 -> 967 ms.
- *Early prefill* (`early_prefill=True`): the next block's scene-state work and context prefill run as soon as the
  previous block is done; `step()` then waits only for the ladder before the block's frames can be decoded.
  Controls given as a callable are taken right before the ladder (`Engine.step(LiveControls.sample)`).
- *Streaming, overlapped decode* (`decode_overlap=True`): each latent frame is decoded and encoded as soon as the
  block is done, on a second thread and CUDA stream, while the next block is generated. The Wan2.2 decoder streamed
  one latent frame at a time gives the paper's frames byte for byte (481 / 481 frames).
- *No lock-step* (`lockstep=False`): use the peers' messages that have arrived instead of waiting (the output then
  depends on timing, as in any live deployment).

**Not bit-identical** (opt-in):

| option | prefill + ladder (ms) | quality against the exact rollout |
|---|---|---|
| exact (`low_latency()`) | 492 + 475 | Laplacian sharpness 515 (mean of the frames after the prefix) |
| `decoder="taehv"` (taew2_2, 9.9M parameters) | decode 3.3 ms per latent frame vs 360 | PSNR 26.4 dB (p50) to the Wan2.2 frames; slightly softer fine texture |
| `compile=True, attention="fa3"` | 421 + 392 | sharpness 515, same frame-to-frame change; PSNR to the exact frames 37.4 dB in the first second, then about 20 dB as it drifts |
| FP8 linear layers (torchao, bench only) | 459 + 436 | sharpness 526; PSNR 27.3 dB already in the first second |

Alone, `attention="cudnn"`, `attention="fa3"` (FlashAttention-3) and `compile=True` give 466 + 457, 458 + 451 and
446 + 460 ms.

A different kernel changes the last bits of every step, and an autoregressive rollout then drifts away from the
exact one (PSNR to it falls over the seconds), so these options are judged by the quality of their own frames:
sharpness and frame-to-frame change against the exact rollout's, and side-by-side frames.

## Sub-block generation (tested, not recommended)

The input lookahead is the largest part left, and it is set by how often a block takes its controls. Two ways to
take them more often were tested on two rounds (Dust2 r09, 24 blocks; Mirage r16, 16 blocks) with the recorded
controls and cameras: keep only the first latent frames of each 4-latent generation and start the next generation
right after them (`commit_latents`), or denoise a one-latent target (`target_latents=1`). Quality is read off the
Wan2.2 decode of each rollout: Laplacian sharpness of the frames after the prefix, the frame-to-frame change into
the first frame of a generation against the change inside a latent frame (a seam shows as a jump), and side-by-side
frames at the same recorded cameras.

| mode | frames per second (H20) | keypress to photon p50 / p95 | sharpness, Dust2 / Mirage | change into a generation / inside a latent | frames |
|---|---|---|---|---|---|
| paper block: 4-latent target, keep 4 | 15.5 | 1.04 / 1.50 s | 515 / 272 | 13.3 / 15.6 | |
| 4-latent target, keep 2 (`commit_latents=2`) | 7.8 | 1.03 / 1.50 s | 476 / 278 | 15.0 / 15.1 | no visible loss; follows the paper block's scene |
| 4-latent target, keep 1 (`commit_latents=1`) | 3.9 | 1.03 / 1.49 s | 468 / 241 | 14.1 / 15.3 | no visible loss per frame; its scene departs earlier |
| 1-latent target (`target_latents=1`) | 5.7 | 0.56 / 0.88 s | 379 / 274 | 12.3 / 13.9 | Dust2: blurs and loses scene structure (a door seen at the same camera by every other mode is missing) |

Keeping the first latent frames of a 4-latent generation stays in the student's distribution: no seam at the
generation boundaries and no visible loss per frame. But it needs 2x or 4x the generation rate, and an H20 already
needs 1 s per generation, so the controls are still taken once a second, latency does not drop, and the frame rate
halves or quarters. The one-latent target is shorter to generate (175 ms ladder), which is why its latency drops,
but it is outside what the student was trained on and on one of the two rounds its frames blur and its scene drifts.
Both stay off. `commit_latents=2` becomes worth it only on hardware that generates a block in under 0.5 s.

## Lock-step

The paper's evaluation runs the clients of a round in lock-step: before each block a client waits until every peer
has published its previous block. Three engines of one round (Dust2 r09, three H20s of one node, exact low-latency
settings, the pool directory in shared memory, 2 ms poll):

| | lock-step wait per block, p50 / p95 / mean | frames per second | keypress to photon p50 / p95 |
|---|---|---|---|
| one client, no peers | 0 | 15.6 | 1.02 / 1.48 s |
| three clients, lock-step | 4-17 / 23-159 / 10-88 ms | 14.7 | 1.05 / 1.54 s |
| three clients, latest messages | 3 / 3-4 / 3 ms (reading the messages) | 14.9-15.1 | 1.03-1.05 / 1.51-1.53 s |

Each client waits for the slowest one, so lock-step costs about one frame per second and 60 ms at p95. With the
release's default 2 s poll (`pool.poll_s`) a wait rounds up to the next poll; the paper's runs waited 1.2 s a block
on average (shared file system, 2 s poll). For play, use the latest messages (`lockstep=False`): a client never
waits and reads whatever its peers have published; the output then depends on timing.

## Recommended settings for the demo

```python
from worldcast.engine.realtime import Engine, RealtimeConfig
rt = RealtimeConfig.low_latency(taehv_path="weights/taew2_2.pth")            # exact latents, H20: 15.6 fps
# rt = RealtimeConfig.low_latency(taehv_path=..., compile=True, attention="fa3")   # H20: 16 fps with headroom
engine = Engine(inference_config, rt)                    # own_state: "gt" now, "state_model" once its weights land
engine.start(round_spec, player_slot, only_clients=True) # seats nobody plays are left out of the world
for frame in engine.step(live.sample, not_before=...):   # controls taken right before the ladder
    send(frame.image)                                    # JPEG bytes
```

- `decoder="taehv"`: the Wan2.2 VAE needs 1.4 s per block on an H20, more than real time on its own.
- JPEG over the WebSocket, encoded on the CPU (`jpeg_device="cpu"`, the default): 1.7 ms a frame; 71 kB a frame at
  quality 90 (9 Mb/s), 41 kB at 75. nvJPEG takes 0.23 ms, but torchvision's encoder orders itself only against the
  stream current at its first use (on the H20, 72 of 193 frames came out wrong once the decode ran on another
  stream), so it is off. H.264 (libx264, zerolatency) is 30 kB a frame but 5.6 ms on the CPU.
- No lock-step for play; lock-step is for reproducing the paper's evaluation.
- Pace the blocks so the controls are taken as late as the display allows (the demo worker does).

## Reproduce

```bash
python tools/bench_realtime.py --config configs/infer/worldcast_4step.yaml --config paths.yaml \
    --row 59 --blocks 24 --taehv taew2_2.pth --out runs/latency \
    --sections generation,decode,encode,loop,subblock     # session: --sections session --clients 3
```
