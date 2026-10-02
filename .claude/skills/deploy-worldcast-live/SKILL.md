---
name: deploy-worldcast-live
description: "Step-by-step recipe for serving WorldCast Live, the networked multiplayer web demo: one coordinator, one GPU worker per player, the config, the ports, the tiny VAE decoder, own state from the recording or from the state model, the latency settings, how to check it works (mock engine first, then the real engine) and the common failures. Use when someone wants to run the demo on a laptop or on GPUs across machines."
license: Apache-2.0
---

# deploy-worldcast-live

> `docs/demo.md` is the reference for the architecture, protocol and config keys; `docs/latency.md` for the
> measured numbers. This skill is the order to do things in. Run every command from the repository root: the demo
> config's relative paths (inference configs, `taehv_path`) resolve against the working directory.

---

## What runs where

```
 browsers ── HTTP /api, WS /ws/lobby ──▶ coordinator (1, CPU, port 8100)
    ║                                      rooms, seats, library, room clock, worker registry; per room it relays
    ║                                      each player's state once per block and the engines' peer messages
    ║                                      (scene-state blocks, step records), keeping recent blocks for late joiners
    ║                                           ▲ WS /ws/worker (workers connect out to the coordinator)
    ╚══ WS /ws/play, /ws/watch ══════════▶ worker (1 per GPU, 1 player at a time, ports 8101, 8102, ...)
          direct, or through the              engine loaded once: worldcast/engine/realtime (or the mock)
          coordinator with media_route=proxy
```

A worker cuts the player's inputs into a block's controls right before the ladder, steps the engine and streams
each frame as soon as it is decoded. Clients share only player state and scene state, as in the paper.

---

## Step 1: Run the mock on one machine

No GPU, no weights: the mock engine plays pre-rendered frames (or a synthetic grid arena) and fakes the state model.

```bash
pip install -r requirements/demo.txt               # aiohttp, numpy, pillow, pyyaml
python tools/serve_demo.py --role local --workers 2
python -m pytest tests/demo -q                     # coordinator, three mock workers, three scripted players
```

Open `http://localhost:8100` in two browser windows, create a room, take a seat in each. To see what the paper's
16-frame block means for latency before a GPU is attached:

```bash
python tools/serve_demo.py --role local --config configs/demo/demo.yaml --config configs/demo/mock_block16.yaml
```

`--config` replaces the default demo config instead of adding to it: always pass `configs/demo/demo.yaml` first,
then your overrides. Without it the dataclass defaults apply (`own_state: state_model`, the Wan2.2 VAE decoder).

## Step 2: Build a library of round starts

The GPU engine needs real round starts whose start frame has a cached first latent (`docs/data.md`; usually frame 0).

```bash
python tools/build_demo_library.py --manifest library.yaml --out runs/demo-library   # format: the script's docstring
```

It needs, per round, the rendered video of every seat (previews, mock clips), the round's OpenCS2 tick tables
(spawn pose, team, weapons) and the map's nav mesh (the radar image). Every host reads the same `library.json`
(a shared file system or a copy).

## Step 3: Prepare a GPU worker host

- The package with its CUDA stack and flash-attention (`INSTALL.md`), the release weights
  (`tools/download_weights.py --out-dir weights`, which writes `weights/paths.yaml`) and the data paths
  (`data/paths.yaml`, `docs/data.md`).
- The tiny decoder `taew2_2.pth` from TAEHV (github.com/madebyollin/taehv, MIT) at `weights/taew2_2.pth`, the demo
  config's `engine.worldcast.realtime.taehv_path`.
- One GPU per worker process. Memory per client: `README.md`.

## Step 4: Check the engine on one GPU before the browser

```bash
python -m pytest tests/engine/realtime -q          # CPU, synthetic world: the engine and the demo adapter
python tools/bench_realtime.py --config configs/infer/worldcast_4step.yaml --config weights/paths.yaml \
    --config data/paths.yaml --row <index row> --blocks 24 --taehv weights/taew2_2.pth --out runs/bench \
    --sections generation,decode,loop
```

The `generation` section says whether each generator setting keeps the latents bit-identical; `loop` gives frames
per second and keypress-to-photon on recorded controls. Compare with `docs/latency.md` on the same GPU.

## Step 5: Start the coordinator

```bash
python tools/serve_demo.py --role coordinator --set library=/data/demo-library/library.json
```

Port 8100 (`coordinator.port`) must be reachable by browsers and by every worker. `GET /api/lobby` lists the rooms
and the workers (total, free).

## Step 6: Start one worker per GPU

```bash
CUDA_VISIBLE_DEVICES=0 python tools/serve_demo.py --role worker --set engine.kind=worldcast \
    --set library=/data/demo-library/library.json \
    --set 'engine.worldcast.configs=[configs/infer/worldcast_4step.yaml, weights/paths.yaml, data/paths.yaml]' \
    --set worker.coordinator_url=ws://lobby:8100 --set worker.port=8101 \
    --set worker.advertise_url=ws://gpu-1:8101 --set worker.gpu_label="H20 #0"
CUDA_VISIBLE_DEVICES=1 python tools/serve_demo.py --role worker ... --set worker.port=8102 \
    --set worker.advertise_url=ws://gpu-1:8102
```

- Each worker loads its engine once and serves one player at a time; its own `CUDA_VISIBLE_DEVICES` and port.
- `worker.advertise_url` is what browsers open (direct route). If browsers cannot reach the GPU hosts, run the
  coordinator with `--set coordinator.media_route=proxy` (one more hop); behind HTTPS, put a TLS proxy in front of
  the coordinator and use `proxy`.
- The worker logs "connected to ws://..." and the coordinator "worker ... connected (...), browsers reach it at ...".

## Step 7: Choose where each player's own state comes from

`engine.worldcast.own_state`:

| value | own position and cameras | needs |
|---|---|---|
| `gt` (the demo config's default) | the seat's recorded track (Table 3); the keys drive only the controls | nothing more |
| `state_model` | the closed loop: the state model reads the client's own frames, positions are exchanged once per block, visibility is predicted | `paths.state_model` (no checkpoint is released yet), `paths.state_model_cells`, `paths.state_model_map_norm`, `paths.physics_prior` (tables in `configs/state_model/`) |

Set the paths through `engine.worldcast.overrides` or an extra file in `engine.worldcast.configs`. With
`state_model`, early prefill is no longer exact: retrieval and the ray anchor use the controls held at prefill time.

## Step 8: Latency settings

The demo config's `engine.worldcast.realtime` block is `RealtimeConfig.low_latency(taehv_path=...)`:

| setting | value | effect |
|---|---|---|
| `generator`, `cuda_graphs` | `fast`, `true` | exact; fewer host syncs, graphs replayed per call shape |
| `early_prefill` | `true` | exact with recorded states; only the ladder waits for the controls |
| `decode_overlap`, `decoder` | `true`, `taehv` | frames decoded while the next block generates; the Wan2.2 VAE alone is slower than real time on an H20 |
| `encoder` | `jpeg` | nvJPEG in the engine; its quality is `engine.worldcast.realtime.jpeg_quality` (`worker.jpeg_quality` applies only to raw frames, e.g. the mock's) |
| `lockstep` | `false` | the room's sync mode decides, not the engine |

- Faster but not exact: `compile: true` with `attention: fa3` (needs FlashAttention-3). Quality notes:
  `docs/latency.md`.
- Leave `commit_latents` and `target_latents` at 4: the sub-block modes were tested and are not recommended.
- Room sync (`play.default_sync`, chosen per room): `async` for play. `lockstep` waits up to
  `worker.lockstep_timeout_s` for every peer's previous block, then steps without the late ones; it is not the
  deterministic lock-step of `tools/run_session.py`.
- `play.action_mapping: latest` is the lowest latency; `realtime` keeps tick timing inside a block at about one
  more block of latency. On a quiet machine `PACING_MARGIN_S` (`demo/worker.py`) can shrink.

## Step 9: Check it end to end

1. One worker and the coordinator on one host; join a room from a browser on another machine.
2. `GET http://<worker>:<port>/health` from the browser's machine: `engine`, `busy`, `room`, `seat`.
3. The HUD's latency number and breakdown (sampling, network, waiting for the block, generating, encoding,
   jitter buffer) against `docs/demo.md` and `docs/latency.md`.
4. A second player and a spectator (`/ws/watch`): each player's minimap shows the others moving.
5. Then add workers on other hosts. Bandwidth between workers grows with players: `docs/demo.md`, "Latency".

---

## Common failures

| symptom | cause | fix |
|---|---|---|
| join answers "every GPU is busy" | no free worker registered | start workers; check the coordinator's "worker ... connected" line |
| "could not start your client: the GPU worker did not answer" | the worker did not accept the seat within 10 s | read the worker log; restart the worker |
| page shows "Lost the connection to the GPU worker."; the worker later logs "the player did not connect" | the browser cannot open the worker's WebSocket | reachable `worker.advertise_url`, or `coordinator.media_route=proxy` |
| "the engine failed: ..." | session start failed: a round start without a cached first latent, data paths missing from `engine.worldcast.configs`, `state_model` without its checkpoint | the worker log's traceback; fix the library or the configs |
| sessions fail for missing state-model paths, or the worker loads the Wan2.2 VAE | a `--config` without `configs/demo/demo.yaml` before it: the dataclass defaults apply | pass the default demo config first |
| all workers on GPU 0, OOM | workers without their own `CUDA_VISIBLE_DEVICES` | one GPU per worker process |
| low frames per second | `decoder: wan`, lock-step rooms, CUDA graphs still warming up | `taehv`, `async`, look at steady-state blocks |
| worker log "lock-step: stepping block ... without seats [...]" | a peer slower than `worker.lockstep_timeout_s` | expected under load; play in `async` |
| recorded players move among the live ones | seats nobody plays are filled from the recording | start sessions with `only_clients=True` (`docs/demo.md`, "GPU integration") |
| the view drifts from the keys | `own_state: gt` keeps the cameras on the recorded track | expected; the closed loop needs a state-model checkpoint |
