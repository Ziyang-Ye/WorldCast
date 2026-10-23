# Training

How to run the paper's four training stages with this repository: the stages, the data, the commands, the
checkpoints, the compute of the paper runs, where the code differs from the paper's text, the evaluation and the
paper's metrics.

**Scope.** The release trains the generator (stages 1-4 and four of the trained ablations of Table 5) and ships the
checkpoints of stages 2, 2s and 3 that the later stages start from. The state model and the depth head are released
as weights ([docs/inference.md](inference.md#weights)), without their training code. Not in the release: the two
baselines (frame concatenation, Solaris) and the ablation "Trained without camera alignment". The paper's evaluation
sets ship in `configs/eval/` (256 windows and 32 rounds; [docs/data.md](data.md#the-papers-files)) and its metrics
(FVD, Rendered, Unmatched, the agreement measures, VBench) in `tools/evaluate_paper.py`
([The paper's metrics](#the-papers-metrics)); [Evaluation](#evaluation) scores PSNR, SSIM and LPIPS on the 64 windows
the training runs validated on.

| module | what it does |
|---|---|
| `worldcast/config/training.py` | the run's config: the stage table (`STAGES`), the settings a user sets, their paper values per stage |
| `worldcast/data/training.py` | the training windows: the bucket windows over the latent cache, the raw-video windows of stage 1 |
| `worldcast/data/stream.py` | the weighted resumable sampler and the per-rank batch stream |
| `worldcast/data/memory_selection.py`, `memory_mask.py`, `appearance.py`, `memory_frames.py` | the memory frames of a training window: which teammate block, the mask m_k and its appearance check, what a window reads of a block |
| `worldcast/engine/training/window.py` | the per-step memory draw and the training window of a micro-batch |
| `worldcast/engine/training/losses.py` | the weighted flow-matching loss, alpha_k, c_k, context noise, the distribution matching losses |
| `worldcast/engine/training/trainer.py` | what every stage shares: the step loop, metrics, validation, checkpoints and resume |
| `worldcast/engine/training/build.py` | a stage's run from its config: the generator, the data and the trainer of its recipe |
| `worldcast/engine/training/recipes/` | `flow_matching.py` (the step of stages 1-3), `bidirectional.py` (stages 1-2), `teacher_forcing.py` (stage 3), `distillation.py` (stage 4) |
| `worldcast/engine/optim/`, `worldcast/distributed/`, `worldcast/engine/checkpoint/training.py` | AdamW groups, clipping, the sharded EMA; process group, FSDP; checkpoints |

The entry point is `tools/train.py`; the configs are in `configs/train/`.

## The stages

| stage | config | recipe | starts from | steps | global batch |
|---|---|---|---|---|---|
| 1: bidirectional tuning, 5 s windows | `stage1.yaml` | bidirectional | Wan2.2-TI2V-5B | 34,000 | 256 |
| 1_long: stage 1 continued on 10 s windows | `stage1_long.yaml` | bidirectional | stage 1 @ 34,000 | 6,000 | 256 |
| 2: player state field | `stage2.yaml` | bidirectional | stage 1_long @ 40,000 (`model.pt`) | 25,000 | 64 |
| 2s: + scene state | `stage2s.yaml` | bidirectional | stage 2 @ 20,000 (EMA) | 5,000 | 64 |
| 3: block-causal, with scene state | `stage3.yaml` | teacher forcing | stage 2s @ 5,000 (EMA, released) | 5,000 | 64 |
| 3: without scene state | `stage3_noscene.yaml` | teacher forcing | stage 2 @ 25,000 (EMA, released) | 5,000 | 64 |
| 4: four-step distillation, with scene state (the released generator) | `stage4.yaml` | distillation | generator: stage 3 @ 5,000 (EMA, released); teacher and critic: stage 2 @ 25,000 (EMA, released) | 600 | 64 |
| 4: without scene state | `stage4_noscene.yaml` | distillation | generator: stage 3 without scene state @ 5,000 (EMA); teacher and critic as above | 600 | 64 |

```
Wan2.2-TI2V-5B -> stage 1 @34k -> 1_long @40k -> stage 2 --@20k EMA--> 2s @5k EMA -> 3 @5k EMA -> 4 @600 (released)
                                                          \--@25k EMA--> 3 no scene @5k EMA -> 4 no scene @600
                                    stage 2 @25k EMA = "WorldCast" of Table 1 = teacher and critic of both stage-4 runs
```

Steps count from stage 1 as in the paper (stage 1 continued on ten-second windows, `1_long`, from step 34,000 to
40,000); a release run counts its own steps from 0, so the last checkpoint of `1_long` is `checkpoint_model_006000`.

**Released checkpoints.** `tools/download_weights.py --out-dir weights --with-training-checkpoints` puts the EMA
weights of stage 2 @ 25,000, stage 2s @ 5,000 and stage 3 @ 5,000 into `weights/worldcast/` (fp32, the generator's
names; [docs/inference.md](inference.md#weights)). Stage 3, stage 3 without scene state and stage 4 start from
them, and so do the teacher and critic of both stage-4 runs; the models of stages 1 and 1_long are not released.

`checkpoint.init` reads a `.safetensors` file as it is, or a training checkpoint's `generator_ema` when it has one,
else its `generator` (stages 1 and 1_long keep no EMA). The modules a stage adds start fresh: the controls in stage 1, the
state injector in stage 2, the ray embedding, the observer-signal embedding and the visibility probe in stage 2s.
Their output layers start at zero, so a stage starts as the model it was initialised from; so do the state injector's
input columns for the dying, corpse and identity channels. A stage builds only the modules it trains: stages without
scene state, and the teacher and critic of stage 4, have no ray embedding, observer-signal embedding or visibility
probe.

The ablations of Table 5 are stage 2 with one setting changed (`configs/train/ablations/`): 6,000 steps from stage
1_long, 32 GPUs x 1 x 2 = 64, each parameter group clipped on its own. "Trained without foreground weight":
`no_foreground_weight` (alpha_k = 1); "Trained with late injection": `late_injection` (the field after the 23rd
block); "Trained with a coarse field": `coarse_field` (the field max-pooled by two); "Trained without visibility":
`no_visibility` (no visibility gate and no confidence).

What the paper fixes for a stage (its recipe, window length, camera encoding, which modules it has, the learning
rates per module, the loss weights, the memory frames) is in the code: the stage table `STAGES` and the constants of
the modules above. The config holds what a run sets: seed, steps, paths, batch, learning rate, clipping, FSDP sharding,
EMA, checkpointing. Its defaults are the paper run of the stage (`STAGE_SETTINGS` in `worldcast/config/training.py`;
`tools/train.py --print-config` prints them), so a stage file of `configs/train/` only names its stage, and the paths
come from `train_paths.yaml` and `--set`. An ablation file begins with `inherit: <file>` (a path relative to it) and
is that file with the keys below changed: the four of `configs/train/ablations/` inherit `base.yaml`, the run
settings they share. `model.dims` replaces dimensions of the generator by name, for a small
model in a smoke run (`--set 'model.dims={dim: 32, num_heads: 2, controls.hidden: 64}'`; a nested dimension as
`controls.hidden`); empty, the generator is the paper's. `checkpoint.keep`, `checkpoint.keep_shards` and the
in-training validation (`validation`) are left at their defaults (keep everything; off).

## Install and data

```bash
pip install -e ".[train]"          # after torch and flash-attn (README, Install)
pip install -e ".[train-decord]"   # optional: decode stage-1 video with decord (else OpenCV)
cp examples/train_paths.yaml train_paths.yaml   # then edit the paths
```

`train_paths.yaml` lists every path; a stage ignores the ones it does not read. Training reads the
[OpenCS2 dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset) (`data.dataset_root`: the recorded videos
and tick tables) and files derived from it, which `tools/prepare_data.py` builds: where the data comes from, its terms
and one command per file are in [docs/data.md](data.md).

| artefact | config key | used by | how to get it |
|---|---|---|---|
| media index | `data.media_index` | stages 2-4 | `tools/prepare_data.py media-index`: one row per player-round of OpenCS2 ([docs/inference.md](inference.md#data)) |
| raw-video manifest | `data.train_manifest` | stages 1, 1_long | `tools/prepare_data.py manifest`, in the format below |
| bucket index | `data.bucket_dir` | stages 2-4 | `tools/prepare_data.py buckets`: windows in the format below, five files with one map each (the paper's: `de_dust2`, `de_mirage`, `de_nuke`, `de_ancient` and an empty fifth) |
| evaluation reserve | `data.exclude_media_manifest` | stages 2-4 | the paper's: `configs/data/eval_reserve.txt`, 1,591 `de_dust2` recordings of training matches that training leaves out; or `tools/prepare_data.py buckets reserve` |
| latent cache | `data.latent_cache_root` | stages 2-4, evaluation | `tools/prepare_data.py latent-cache`: each window encoded with the Wan2.2 VAE as below (a re-encode is not bit-equal with the paper's cache) |
| GT visibility labels | `data.visibility_label_root` | stages 2-4 | `tools/prepare_data.py visibility`: engine line-of-sight tests of every player pair ([format](inference.md#data)) |
| flash and scope labels | `data.observer_signal_label_root` | stages 2-4 | `tools/prepare_data.py observer-labels`: computed from the recorded video ([format](inference.md#data)) |
| collision meshes | `data.collision_meshes` | stages 2s, 3, 4 | `tools/prepare_data.py meshes`: exported from your copy of the game's map files, which are Valve's and not redistributed ([docs/data.md](data.md#the-games-map-files)) |
| validation index | `validation.index` | evaluation | the paper's: `configs/data/validation_index.jsonl`, the 64 validation windows, checked by sha256; or `tools/prepare_data.py buckets validation` |
| umT5 embedding of the fixed prompt | `data.prompt_embedding` | all | released ([weights](inference.md#weights)) |
| Wan2.2-TI2V-5B snapshot | `model.wan22_root` | all | `tools/download_weights.py` fetches `config.json`, the VAE and the tokenizer; stage 1 also needs the backbone of [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) at the same revision, in that directory: the files `diffusion_pytorch_model*` (the `diffusion_pytorch_model-*.safetensors` shards and the `.safetensors.index.json` that lists them) |

Files you built yourself take two optional settings: `data.bucket_weights` (the sampling weight of each of the five
bucket files; the paper's unless set) and `validation.index_sha256` (the sha256 of the validation index, whose 64
windows are then selected from it; the paper's index unless set). `tools/prepare_data.py buckets` prints both for your
files.

The memory frames and m_k of stages 2s, 3 and 4 decode target and teammate frames from the videos and ray-cast the
collision meshes in the loader workers: trimesh with an embree backend, OpenCV and scipy (the `train` extra).

### Data formats

- **Bucket index**: five files, `train_q00-01.jsonl` ... `train_q07-10.jsonl`, each the windows of one map (the
  paper's: `train_q00-01.jsonl` de_dust2, `train_q01-03.jsonl` de_mirage, `train_q03-05.jsonl` de_nuke,
  `train_q05-07.jsonl` de_ancient, `train_q07-10.jsonl` empty; the names are slots, not ranges of a score, and
  `data.bucket_weights` weighs the files). One JSON object per 41-latent window of one player's view: `{"media_id": "2392812-de_dust2-r09-p09", "start_frame": 1288, "match_id": 2392812, "round": 9,
  "latent_key": "win_001288", "vis_available": true, "vis_pair_count": 3}`. `start_frame` is a 32 fps source frame (the
  window covers `start .. start + 320` at every second frame); `match_id` and `round` are checked against the media
  index; `latent_key` is optional and must be `win_<start:06d>`; `vis_available` and `vis_pair_count` (visible pairs of
  that player and another player over the window) weight the sampling. A `(media_id, start_frame)` pair occurs once
  across the files.
- **Evaluation reserve**: text, one media id per line; every id must occur in the buckets.
- **Latent cache**: `<media_id>.npz`, member `win_<start:06d>` per window, float16 `[1, 41, 48, 24, 42]`: Wan2.2 VAE
  latents (with its 48-channel normalisation) of the frames `start, start + 2, ..., start + 320`, resized bilinearly
  to 384 x 672 and mapped to `(x / 255 - 0.5) * 2`. A teammate's members are the candidates of its memory frames.
- **Collision meshes**: `{map_name: path}` to a glTF scene of the map's collision geometry, engine units, +z up; the
  paper's maps are `de_dust2`, `de_mirage`, `de_nuke`, `de_ancient`. `tools/prepare_data.py meshes` writes
  `<map>_world_collision_complete.glb`.
- **Raw-video manifest** (stages 1 and 1_long): JSONL, one record per media: `media_id, match_id, map_name, round,
  player_slot, video_path, ticks_path` (absolute, or relative to `data.dataset_root`), `video_frames,
  video_file_size, ticks_file_size, ticks_sha256, ticks_rows, fps, pixel_frames` (81 for stage 1, 161 for 1_long),
  `stride, max_tick_gap_seconds` (1.5 / 64), `start_frames` (sorted multiples of `stride`) and `n_windows`; the reader
  also takes `skip_frame` (`tools/prepare_data.py manifest` always writes it, 2) and `sample_weight` (per record) or
  `window_sample_weights` (per window), which it does not write.

## Commands

All stages run under torchrun; `torchrun ...` below stands for `torchrun --nnodes N --nproc_per_node 8 --node_rank
$RANK --master_addr $MASTER --master_port 29500` with the run's node count. `run.output_dir` receives `config.yaml`,
`metrics.jsonl` and the checkpoints. Any key can also be given with `--set` (the stage's settings follow `run.stage`
wherever it is set), and `--set model.attention=sdpa` runs without flash-attention (not bit-equal). A key the config
does not have is refused (`ValueError`), whether a file or `--set` names it.

From the released checkpoints (`examples/train_paths.yaml` sets the released stage 2 as stage 4's teacher and critic):

```bash
python tools/download_weights.py --out-dir weights --with-training-checkpoints

# stage 3 on 8 nodes (64 x 1), from the released stage 2s
torchrun ... tools/train.py --config configs/train/stage3.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage3 \
  --set checkpoint.init=weights/worldcast/worldcast_stage2s_bidirectional_fp32.safetensors

# stage 4 on 8 nodes (64 x 1), from the released stage 3
torchrun ... tools/train.py --config configs/train/stage4.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage4 --set checkpoint.init=weights/worldcast/worldcast_stage3_ar_fp32.safetensors

# without scene state, on 4 nodes (32 x 1 x 2): stage 3 from the released stage 2, then stage 4
torchrun ... tools/train.py --config configs/train/stage3_noscene.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage3_noscene \
  --set checkpoint.init=weights/worldcast/worldcast_stage2_bidirectional_fp32.safetensors
torchrun ... tools/train.py --config configs/train/stage4_noscene.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage4_noscene \
  --set checkpoint.init=runs/stage3_noscene/checkpoint_model_005000/model_ema.pt
```

From the start (each stage from the previous one's run):

```bash
# stage 1 on 8 nodes (64 x 4 x 1); it needs the Wan2.2 snapshot (backbone, VAE) and a raw-video manifest of 5 s windows
torchrun ... tools/train.py --config configs/train/stage1.yaml --config train_paths.yaml \
  --set data.train_manifest=data/stage1/train.jsonl --set run.output_dir=runs/stage1

# stage 1_long on 16 nodes (128 x 2 x 1): 10 s windows, from stage 1 at step 34000
torchrun ... tools/train.py --config configs/train/stage1_long.yaml --config train_paths.yaml \
  --set data.train_manifest=data/stage1_long/train.jsonl --set run.output_dir=runs/stage1_long \
  --set checkpoint.init=runs/stage1/checkpoint_model_034000/model.pt

# stage 2 on 8 nodes (64 x 1 x 1), from stage 1_long
torchrun ... tools/train.py --config configs/train/stage2.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage2 --set checkpoint.init=runs/stage1_long/checkpoint_model_006000/model.pt

# stage 2s on 4 nodes (32 x 1 x 2), from stage 2 at step 20000
torchrun ... tools/train.py --config configs/train/stage2s.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage2s --set checkpoint.init=runs/stage2/checkpoint_model_020000/model_ema.pt

# an ablation on 4 nodes (6,000 steps from stage 1_long)
torchrun ... tools/train.py --config configs/train/ablations/late_injection.yaml --config train_paths.yaml \
  --set run.output_dir=runs/late_injection --set checkpoint.init=runs/stage1_long/checkpoint_model_006000/model.pt
```

Stages 3 and 4 then take `runs/stage2s/checkpoint_model_005000/model_ema.pt` and
`runs/stage3/checkpoint_model_005000/model_ema.pt`, and stage 4's teacher and critic
`--set distillation.teacher=runs/stage2/checkpoint_model_025000/model_ema.pt` (and `distillation.critic`).

`--resume auto` (the default) continues from the newest complete checkpoint in `run.output_dir`; resume is exact
(weights, AdamW state, EMA, data position, RNG) and needs the checkpoint's world size and per-GPU batch.
`--print-config` prints the resolved config.

The data order depends on the world size (one weighted draw per epoch, sharded by rank; the loader is seeded `seed +
rank + 100003`), so reproducing a paper run's batches needs its topology
([Compute of the paper runs](#compute-of-the-paper-runs)). Whether a step trains on memory frames is drawn per
optimizer step, the same on every rank.

`metrics.jsonl` has one row per step (rank 0): the step, a UTC timestamp, the losses and gradient norms (means over
ranks), the learning rate of each parameter group, the data, forward-backward, optimizer and step times and on CUDA
the peak memory (maxima over ranks); stage 4 adds whether the step trained the generator and, per phase, the served
windows, the rollout's exit step and the score timesteps. Every `run.log_interval` steps the row is also printed (0:
never).

### Checkpoints

A checkpoint is written every `checkpoint.interval` steps and at the end of the run (0: at the end only).
`<output_dir>/checkpoint_model_<step:06d>/`: `model.pt` (`{"generator": ...}`; stage 4 also `"generator_ema"`),
`model_ema.pt` (`{"generator_ema": ...}`, stages 2-3 once the EMA exists), `critic.pt` (stage 4), one `rank_<r>.pt`
per GPU (its optimizer and EMA shards, RNG states and data position) and `checkpoint.ready.json` (written last). The
generator keys are the release's module names under a `model.` root, so the EMA weights of a stage with scene state
(2s, 3, 4) load into the inference generator, and the client takes the file as `paths.checkpoint`:

```python
from worldcast.modeling.build import load_generator
# stage 4: model.pt holds the EMA; stages 2s and 3: model_ema.pt
generator = load_generator("runs/stage4/checkpoint_model_000600/model.pt", device="cuda")
generator = load_generator("runs/stage3/checkpoint_model_005000/model_ema.pt", device="cuda")
```

A stage without scene state (2, 3 and 4 without scene state) has no ray embedding and no observer-signal embedding.
`load_generator` builds the generator with the modules the file holds, so its checkpoint loads the same way; a
client does not run it (`ClientModels.load` refuses it and names what it lacks), `tools/evaluate.py` scores it.

Sizes (fp32): a generator file (`model_ema.pt`, `critic.pt`, `model.pt`) is 19 GiB; stage 4's `model.pt` holds two
models, 38 GiB. The resume files hold the AdamW state (38 GiB per trained model) and the EMA shards (19 GiB) once per
FSDP replica: once in all with `fsdp.sharding: full`, once per node with `hybrid_full`. Per checkpoint:

| stage | GPUs, sharding | weights | resume files | checkpoints of the run |
|---|---|---|---|---|
| 1, 1_long | 64 / 128, hybrid (8 / 16 nodes) | 19 GiB | 304 / 608 GiB | 34 / 6 |
| 2 | 64, hybrid (8 nodes) | 38 GiB | 456 GiB | 50: 24 TiB |
| 2s, 3 | 32 / 64, full | 38 GiB | 57 GiB | 40 |
| 3 without scene state, ablations | 32, hybrid (4 nodes) | 38 GiB | 228 GiB | 10, 12 |
| 4 | 64, full | 57 GiB | 95 GiB | 6 |
| 4 without scene state | 32, hybrid (4 nodes) | 57 GiB | 380 GiB | 6 |

`checkpoint.keep: N` keeps the newest N complete checkpoints (0: all). `checkpoint.keep_shards: N` keeps the weights of
every checkpoint and the resume files of the newest N only; the older ones still load, but cannot be resumed.

## Compute of the paper runs

NVIDIA H20 GPUs (96 GB). Batch: GPUs x per-GPU batch x gradient accumulation; a node has 8 GPUs (`--nnodes` of the
`torchrun` line in [Commands](#commands)).

| run | `--nnodes` | GPUs x batch x accumulation | steps |
|---|---|---|---|
| stage 1 (5 s) | 8 | 64 x 4 x 1 | 34,000 |
| stage 1_long (10 s) | 16 | 128 x 2 x 1 | 6,000 |
| stage 2 | 8 | 64 x 1 x 1 | 25,000 |
| stage 2s | 4 | 32 x 1 x 2 | 5,000 |
| stage 3 | 8 | 64 x 1 x 1 | 5,000 |
| stage 3 without scene state | 4 | 32 x 1 x 2 | 5,000 |
| stage 4 | 8 | 64 x 1 x 1 | 600 |
| stage 4 without scene state | 4 | 32 x 1 x 2 | 600 |
| each ablation | 4 | 32 x 1 x 2 | 6,000 |

Stage 2 took 2,637 GPU-hours (the paper's App. C), stage 4 about 270 (4 h 15 min), stage 4 without scene state about
250 (82 s per generator step, 36 s per critic-only step).

Stage 4 holds three 5B models under FSDP: about 47 GB per GPU on generator steps and 40 GB on critic-only steps
(without scene state, 32 GPUs).

## The paper's text and the code

The release reproduces the code that trained the paper's models. Where that code differs from the paper's text, the
code wins:

1. **Loss.** The paper's loss is `sum_k alpha_k c_k ||v_theta - v||^2`. The code also multiplies by a per-frame
   timestep weight W(t) (Wan's Gaussian timestep weight, mean 1 over the table, 0 at t = 1000), and its noise and
   target are bf16 with t rounded to bf16 before the sigma lookup.
2. **alpha_k** uses an anisotropic Gaussian (sigma_u = max(r/2, 0.6), sigma_v = max(r, 0.6), centre raised by r above
   the feet) and is normalised by its mean over the GPU's micro-batch (one window), all frames.
3. **Memory frames** are drawn with p = 0.8 per optimizer step on every rank, not per sample; windows without memory
   frames (and, in 2s, without a teammate) train without them, so the effective share is below 0.8.
4. **m_k** also requires RGB texture, no player in front of the surface, a source within 420 units at half the
   target's resolution, and at least 20 unseen and 20 covered tokens; in stage 2s "the recent context" is the first
   frame alone.
5. **Stage-2s window**: 1 first frame + 4 memory + 32 recent + 4 target = 41 latents.
6. **Stage-3 context noise** is drawn in [16, 32) per 4-latent block of the whole context copy, the first frame and
   the memory frames included, on windows without memory frames as well. Without scene state the context is clean.
7. **Stage 4 updates the critic on every step** (600 of 600) and the generator on every fifth (120 of 600); the paper
   says "the critic otherwise".
8. **Stage-4 values not in the paper**: generator lr 2e-6 for every module, critic lr 4e-7, AdamW betas (0, 0.999)
   with the multi-tensor kernel, weight decay 0.01, a global clip of 1.0 per model, EMA 0.99 from step 200, context
   noise 16 (0 without scene state; with it the context of the teacher-forced pass that carries the gradient is
   noised to 16 including the first frame, which the KV cache holds clean), score timesteps uniform on [0, 1000)
   shifted by 5 and clamped to [20, 980], no CFG,
   deterministic algorithms. The generator is one FSDP unit; teacher and critic are wrapped by size. Teacher and
   critic start from the same checkpoint, so the first distribution matching gradient is exactly 0.
9. Stage 4 trains without memory frames (on windows drawn from those that have them) and passes no ray cameras.
   The ray embedding and the visibility probe stay in the optimizer with zero gradients; at lr 2e-6 and
   weight decay 0.01 AdamW leaves them unchanged (`1 - lr * wd` rounds to 1 in float32), so the released
   generator's ray embedding is stage 3's.
10. The ray embedding's MLP has its own learning rate (1e-4); the paper says it is "trained with the full backbone".
11. Not in the paper: the observer-signal embedding (flash, scope) in 2s, 3 and 4, and the detached visibility probe
    (masked BCE, weight 1) in 2s and 3.
12. **Camera encoding per stage**: `clip` in stages 1, 2, 3 and 4 without scene state and in the ablations; `noclip`
    in 2s, 3 and 4 with scene state (the stage-4 teacher was trained with `clip`). It is part of the stage table.
13. **Clipping differs by run** (`optim.clip`): stages 1, 1_long and 2 clipped the global gradient norm; stages 2s, 3,
    3 without scene state and the ablations clip each parameter group on its own. Stage 4 always clips each of its
    models globally.
14. **Stage-3 camera**: stage 3 integrates the player's own camera and the death state on the whole window and gathers
    them with the rows of a memory window; stage 2s re-integrates over the rows of the memory window.
15. The ablation "Trained without visibility" removes the label gate **and** the confidence from the field; its
    alpha_k still reads the labels.
16. **Observer-signal labels**: a missing scope label file (86 of the 118,390 training media) reads as "unknown".
17. **The field under teacher forcing**: stage 3 and the teacher-forced pass of stage 4 with scene state add the
    field to the noisy copy only; without scene state they add it to the clean context copy as well. At inference
    every forward adds the field, the cache writes included.
18. **Stages 1 and 1_long** run the WorldCast generator without its WorldCast modules. The paper's runs of these
    stages trained Wan2.2's own bidirectional model with the control embedding; the two compute the same function
    in another operation order (the timestep embedding per frame instead of per token), so a forward agrees to
    rounding (2.4e-6 on a flow of magnitude 7.5, fp32), not bit for bit.

## Evaluation

`tools/evaluate.py` scores a checkpoint on the windows the paper's training runs validated on: 64 ten-second
windows, 16 per map, from the held-out matches (`validation.index`; its sha256 and the selection's are checked).
They are not the paper's evaluation sets ([The paper's metrics](#the-papers-metrics)). Each window starts from its
recorded first latent and its own noise (a generator seeded per window); the rollout and the cached latents are
decoded with the Wan2.2 VAE and compared on every frame after the first: PSNR, SSIM (torchmetrics), LPIPS (AlexNet),
pixel and latent MSE, as means over the windows, per map and per stratum (`vis<n>`: the number of players visible in
the window), with every window's scores.

```bash
pip install -e ".[eval]"   # torchmetrics, lpips; LPIPS fetches torchvision's AlexNet once (alexnet-owt-7be5be79.pth
                           # into $TORCH_HOME/hub/checkpoints; copy it there on machines without internet)
python tools/download_eval_data.py --out-dir data/evaluation validation   # the windows' data, 0.3 GB
torchrun --nproc_per_node 4 tools/evaluate.py --config configs/train/stage2.yaml --config train_paths.yaml \
  --config data/evaluation/eval/validation.yaml \
  --checkpoint weights/worldcast/worldcast_stage2_bidirectional_fp32.safetensors --out runs/eval/stage2.json
```

`tools/download_eval_data.py` fetches the windows' cached latents, labels and tick tables, the paper's own files, from
the dataset [ZiyangYe/WorldCast-data](https://huggingface.co/datasets/ZiyangYe/WorldCast-data), so no video needs to
be downloaded or encoded; `validation.yaml` points `data` at them and `validation.index` at
`configs/data/validation_index.jsonl`, over `train_paths.yaml`.

`--checkpoint` takes a release `.safetensors` file or a training checkpoint (its `--weights` entry, `generator_ema`
by default); the stage config gives the model, the data paths and the sampler. `--windows N` scores the first N
windows only.

| `--sampler` | stages | how a window is sampled | reference (PSNR / SSIM / LPIPS) |
|---|---|---|---|
| `unipc` | 1_long to 3 (their default) | 20 UniPC steps (shift 5): the whole window at once for the bidirectional stages, block by block for stage 3 (KV cache, each block written clean) | the released stage 2 @ 25,000: 16.2587 / 0.4990 / 0.3971 |
| `four_step` | 4 (its default), 3 | the four denoising steps, block by block, the context written clean | stage 4 without scene state trained with this code (`stage4_noscene.yaml`, step 600): 16.0151 / 0.4811 / 0.4055 |

The stage-2 reference is the paper run's own validation row at that step: on 4 H20 (torch 2.9.1, flash-attn 2.8.3,
torchmetrics 1.9.0, lpips 0.1.4) `unipc` gives the paper runs' rows bit for bit. The four-step reference is what a
stage-4 run without scene state reaches with this code. `four_step` re-noises x0 between denoising steps from the
window's generator, so a score does not depend on how the windows are spread over GPUs.

In training, `validation.index` and `validation.interval` (0: off; the paper runs: 1000) score the EMA weights on
these windows with `unipc` and append an `event: validation` row to `metrics.jsonl` (stages 1_long to 3). Every RNG
state is restored afterwards, so the run continues as without it. The windows carry no memory frames.

### The paper's metrics

`tools/evaluate_paper.py` computes the metrics of the paper's tables beyond PSNR, SSIM and LPIPS: Rendered and Unmatched
(the entity criterion), FVD, depth agreement between clients, Mat. Pix., Inliers and RotErr, and the VBench bins, with
their intervals. Its steps (`--help` after each name):

| step | does | runs on | modules |
|---|---|---|---|
| `cut` | splits round-long videos into ten-second windows | CPU | |
| `detect` | the player detector on the recorded and the generated windows: a detection dump | GPU | `boxes`, `detector` |
| `entity`, `entity-ci` | Rendered and Unmatched per view and their medians; intervals and paired differences | CPU | `entity`, `stats` |
| `depth`, `depth-cells` | the rows of depth agreement of one arm; the cells of Tables 19, 10 and 4a (Depth) | GPU; CPU | `depth_agreement` |
| `match`, `match-cells` | Mat. Pix., Inliers and RotErr of every pair of one arm; the cells of Table 4a | GPU; CPU | `matching` |
| `fvd` | FVD of ten-second windows or of whole rounds | GPU | `fvd` |
| `vbench` | VBench's per-clip scores in the bins of Tables 4b, 13 and 14 | CPU | `vbench` |

The modules are in `worldcast/engine/evaluation/`; their docstrings hold the rules in full.

```bash
pip install -e ".[metrics]"            # scipy, decord, OpenCV
pip install -e ".[metrics-detector]"   # ultralytics (AGPL-3.0): the detect step
pip install -e ".[metrics-matching]"   # kornia (LoFTR), torchvision, einops, yacs (GIM): the match step
python tools/download_weights.py --out-dir weights --with-metrics   # paths.detector, gim_dkm, loftr, i3d
```

The weights are fetched from where each is published and checked by their sha256 (`worldcast.hub.METRIC_WEIGHTS`):
the detector from the WorldCast repository on the Hugging Face Hub (`detector/`, AGPL-3.0 as ultralytics), GIM's
matcher from its authors' Hugging Face Space at a pinned revision, LoFTR's outdoor weights from where kornia fetches
them, the I3D detector of FVD from a Hugging Face mirror at a pinned revision
([THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)).
VBench 0.1.5 pins numpy and transformers versions that conflict with these, so it runs in an environment of its own;
`vbench` reads its per-clip results.

**Inputs.** The evaluation sets are `configs/eval/window_views.jsonl` (the 512 views of the 256 ten-second windows of
Tables 1, 2, 5 and 9), `configs/eval/windows.jsonl` (their view pairs, Table 10) and `configs/eval/round_index.jsonl`
(the 32 whole rounds of Tables 3, 13, 16, 17 and 19). A step that reads videos of ten-second windows takes an
*evaluation directory*: `generation/<stem>.mp4` and `gt/<stem>.mp4` per view, `<stem>` being `<media_id>__start<NNNNNN>`
(the window's first source frame, 32 fps), `gt/` the recorded video (for the ten-second windows the VAE reconstruction
of the recording, re-encoded like the generated video). The whole rounds are rendered as in
[docs/reproduce.md](reproduce.md) (`runs/table3`); `cut` makes their windows. The data come from the data config that
`tools/download_eval_data.py` writes for each set, `data/evaluation/eval/windows.yaml` for the ten-second windows and
`data/evaluation/eval/rounds.yaml` for the rounds: the media index and the tick tables (`paths.media_index`,
`paths.dataset_root`), the GT visibility labels (`paths.visibility_label_root`,
[docs/data.md](data.md#gt-visibility-labels)). `fvd --rounds` and `match --recorded` read the recordings' video, and
`depth --rounds --recorded` the windows of their latent cache (`paths.latent_cache_root`) that start every 161 source
frames from 0, then from 80: they take a data config of your own that points at OpenCS2 (`data/paths.yaml` below, a
copy of `examples/data_paths.yaml`; [docs/inference.md](inference.md#data)). `depth` checks for those windows before it
runs and lists the missing ones in `<out>.windows.jsonl`, for `tools/prepare_data.py latent-cache --windows`.

```bash
python tools/download_eval_data.py --out-dir data/evaluation rounds windows   # the sets' data, 2.3 GB
python tools/evaluate_paper.py cut --videos runs/table3/*/*/video.mp4 --out-dir runs/table3_windows/generation
```

`cut` starts a window every 320 source frames (10 s) from the round's start, 161 video frames each, so neighbouring
windows share a frame; an incomplete last window is dropped.

**What the paper's numbers need.** Table 3's PSNR, SSIM and LPIPS against the recordings, and those columns of Tables 1,
2, 5, 9 and 17, have no tool in the release. The release does not write the videos of the ten-second windows
(`tools/evaluate.py` scores the bidirectional models without writing video) or the recorded windows of whole rounds; the
steps take them as evaluation directories. A session does not keep the player states that a client in closed loop gave
its generator: `entity --predicted-states` takes them in the format below (Table 2's predicted row), and `depth` does
not compute the closed-loop rows (Tables 4a and 19). The rounds of five clients of Table 4a are not in `configs/eval/`.
The inference-time ablations (Scene state off at inference, Only its own earlier blocks, Least-covering memory, Field
off and Visibility off at inference) are not client settings, and the one-hour rollouts, the damaged clips of Table 14
and the VBench run itself are not in the release (`vbench` aggregates per-clip scores you supply). Fresh rollouts use
another decoder build and mp4 writer than the paper's runs, so their scores match the paper's statistically, not digit
for digit. The paper's values were computed on NVIDIA H20 GPUs (x86-64) with torch 2.9.1+cu128, torchvision 0.24.1,
numpy 2.2.6, scipy 1.15.3, OpenCV 4.13.0.92, decord 0.6.0, ultralytics 8.4.110 and kornia 0.8.2, with torch's default
TF32 settings; the extras pin these packages but torch, torchvision and numpy. Where a value below is "checked", the
release reproduces it from the paper's stored intermediate files (detections, depth rows, per-pair results, features or
per-clip scores) as stated.

#### Rendered and Unmatched

The rule of App. "Evaluation protocols and long horizons in detail" (`worldcast/engine/evaluation/entity.py`), with
what its text leaves out:

- **The player's box** (`player_box`): the standing player (head 72 u above the origin, whatever the stance) seen from
  the eye 64 u above the observer, with HFOV 106.26 and VFOV 73.74 (rounded); `0.45 h` wide, centred on the projected
  head, `h` from head to feet; dropped when head or feet lie at most 1 u ahead, or when it lies entirely outside the
  frame grown by 15% on each side. A player's state at video frame `i` of a window from source frame `s` is the last
  tick at or before source frame `s + 2 i`, known up to 1.5 ticks after the player's last tick. The players of a round
  are those with a tick table in its directory (`player=<pp>/ticks.parquet`), also one without a media row.
- **Detection** (`detect`): the fine-tuned YOLO11x on every frame of the recorded and the generated video (at most 161),
  16 frames per call, decord's RGB frames handed over as BGR, letterboxed to 1280 px, confidence 0.05 (strictly above),
  ultralytics' defaults otherwise (NMS IoU 0.7, at most 300 detections, float32); a call whose NMS hits ultralytics'
  time limit is repeated. Detections are kept on the frames where the observer is known and alive.
- **Confirmed in the recorded video**: a detection overlaps the box clipped to the frame with IoU at least 0.30, or
  covers at least 0.55 of the smaller of the two; gaps of at most two frames are closed.
- **The visibility gate**: the visibility given to the generator, folded onto latent frames as the generator receives
  it: a frame is visible if any video frame of its latent frame is, counted from the rollout's first frame (the view's
  for ten-second windows; for whole rounds the sessions' first source frame, `entity --rollout-start`).
- **Rendered**: per frame, the players visible in the gate and confirmed are matched one to one to the generated
  detections within one body width (the clipped box's width), the most pairs first, then the least total distance;
  the box size and the recording's own hit are applied after the matching: a pair is scored when the clipped box
  covers at least 24^2 px and the recorded video's detections hit the player under the same rule (one to one among all
  confirmed players). On the partner alone (`entity --scope partner`: the window's other view player, from
  `configs/eval/window_views.jsonl`), the other players' pairs are dropped after the matching too.
- **Unmatched**: a generated detection with a short side of at least 24 px is unmatched when no expected centre lies
  within the larger of the two unclipped widths; expected are the recorded video's detections and the boxes of the
  players visible in the gate, each with a short side of at least 24 px. It is not one to one, does not ask for
  confirmation and counts every player on the partner alone too.
- **Summaries**: medians over the views with at least 8 scored pairs (Rendered) or 8 counted detections (Unmatched);
  Table 16 pools the scored pairs of the frames by the number of players in view (`entity --perframe`).

```bash
# one detection pass per evaluation directory and its arms, 8 shards on 8 GPUs
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python tools/evaluate_paper.py detect --config weights/paths.yaml \
    --config data/evaluation/eval/windows.yaml --eval-dir runs/windows/worldcast \
    --arm worldcast=runs/windows/worldcast --arm frame_concat=runs/windows/frame_concat \
    --shard $i/8 --out runs/entity/dets-$i.json.gz &
done; wait
python tools/evaluate_paper.py entity --config data/evaluation/eval/windows.yaml \
    --dets runs/entity/dets-*.json.gz --out runs/entity/table1   # whole rounds: rounds.yaml, --rollout-start 0
python tools/evaluate_paper.py entity-ci --scores runs/entity/table1.json \
    --difference worldcast,frame_concat --out runs/entity/table1_ci.json
```

`entity` writes `<out>.json` (the per-view Rendered and Unmatched of every arm, the medians and the views they take) and
`<out>_perview.tsv` (the counts of every view with a pair or a detection). Table 12's settings are `entity --radius 0.5`
or `2`, `--conf 0.25` or `0.5` (the stored detections at or above it, the recorded video's confirmation recomputed: what
the detector gives at that confidence but for exact ties and its cap of 300 detections) and `--min-side 48`, and
`entity-ci --statistic mean`. With states given to the generator (Table 2's predicted row), `--predicted-states DIR`
reads `<stem>.npz` per view: `states [P, R, 6]`, every player's state per video frame of the rollout, `labels [P, R]`,
whether the player is visible and valid there (both, as the GT gate reads the GT labels), and `start_frame`, the
rollout's first source frame; a view without one is left out and counted.
`entity-ci` resamples the windows, the two views of a window together (10,000 draws, seed 20260922), as Tables 11, 12
and 20 did.

| table | row | Rendered (views) | Unmatched (views) |
|---|---|---|---|
| 1 | WorldCast | 0.817 (391) | 0.039 (375) |
| 1 | Frame concatenation | 0.074 (391) | 0.771 (359) |
| 1 | Solaris | 0.056 (391) | 0.739 (323) |
| 2 | WorldCast (GT positions) | 0.897 (391) | 0.031 (385) |
| 2 | WorldCast (Predicted positions) | 0.830 (354) | 0.093 (371) |
| 5 | WorldCast (6,000 steps) | 0.763 | 0.076 |
| 5 | Trained without foreground weight / with late injection / with a coarse field | 0.691 / 0.549 / 0.708 | 0.060 / 0.203 / 0.083 |
| 5 | Visibility off at inference | 0.748 | 0.415 |
| App. "Training stages and scaling" | the partner alone: WorldCast / Frame concatenation / Solaris | 0.840 / 0.030 / 0.025 (358 each) | as Table 1 |

Checked: from the paper's detection dumps, every per-view count of these rows (the partner's alone too), of Table 9b
(frame concatenation and Solaris) and of Table 12 is reproduced exactly, and so are the medians and the intervals of Tables 11, 12 and 20 (Table
2's predicted row leaves out 2 of its 512 views, which have no player states); the detections' confirmation and the
recorded video's hits are reproduced too. The other cells of Tables 5 and 9 and the whole-round
cells (Tables 16 and 17) keep only their per-view counts, whose medians and pools are reproduced; their detections are
not stored. Table 15 is not computed by the release.

#### FVD

`worldcast/engine/evaluation/fvd.py`: the Kinetics-400 I3D (TorchScript) sees a clip whole, at 0-255 with its own
rescale and its own bilinear resize to 224 x 224 (no antialiasing, the aspect ratio not kept); a clip's feature is the
400 logits averaged over time. The distance is between the Gaussian fits in float64 (`np.cov`, N - 1; scipy's
`sqrtm`, retried with 1e-6 on both diagonals when not finite, an imaginary part up to 1e-3 dropped).

- **Ten-second windows** (Tables 1, 2, 5, 9): all 161 frames of a window, the recorded first frame included, against
  the window's `gt/` video; 512 clips (510 in Table 2). `fvd --eval-dir` refuses fewer than 64 clips; `--real-cache`
  keeps the recorded clips' features by the sha256 of their file.
- **Whole rounds** (Table 17): each complete ten-second segment `k` of a client's round, its generated frames
  `1 + 160 k .. 160 (k + 1)` (the first frame in none), against the recording's frames `2 i` over the same span (OpenCV,
  resized to 384 x 672 with `INTER_LINEAR`), the segments of all clients pooled. Table 17 (FVD 78.4 to 98.5) is
  reference-only: neither its features nor its clips are stored, so this rule is not checked against them.

```bash
python tools/evaluate_paper.py fvd --config weights/paths.yaml --eval-dir runs/windows/worldcast \
    --real-cache runs/fvd/real.npz --out runs/fvd/worldcast.json
python tools/evaluate_paper.py fvd --config weights/paths.yaml --config data/paths.yaml \
    --rounds configs/eval/round_index.jsonl --sessions worldcast=runs/table3 --out runs/fvd/rounds.json
```

| table | row | FVD | stored |
|---|---|---|---|
| 1 / 9a (1.6M) | WorldCast | 39.7 | 39.71825365112424 |
| 1 | Frame concatenation | 48.2 | 48.201015053362475 |
| 1 | Solaris | 49.0 | 48.96903236578515 |
| 2 | GT positions / Predicted positions (510 clips) | 47.5 / 53.0 | 47.54531143245859 / 52.95759711378901 |
| 5 / 9a (384k) | WorldCast (6,000 steps) | 45.7 | 45.67762674934986 |
| 5 | Trained without foreground weight / with late injection | 43.9 / 47.2 | 43.93288086619477 / 47.22784515327576 |
| 5 | Trained with a coarse field / Visibility off at inference | 41.9 / 60.8 | 41.94028523018097 / 60.76129566358054 |

Only these scalars are stored, not the features. No stored value: Table 5's "Trained without visibility" (45.8) and
"Field off at inference" (73.2), and Table 9a's 64k to 1.0M (53.0, 44.3, 43.3, 42.1, 43.3, 42.9).

#### Depth agreement

`worldcast/engine/evaluation/depth_agreement.py` (App. "Scene state in detail", Tables 19 and 10, the Depth row of
Table 4a): the depth head's depth of one view's latent frame (its last video frame) is back-projected with that view's
camera and reprojected into another view's; a reached pixel agrees when the two depths lie within the larger of 24 u
and 5%. What the text leaves out:

- The depth head's log depth is clamped to [0, log 4096] and rounded to float16
  (`DepthPredictor.log_depth(..., paper_numerics=True)`). A generated client's depth comes from latent frame 0 alone
  and then each block of four, as it was generated; in a round, the recorded video's from the depth head run once on
  each whole 41-latent window of the recording's latent cache that starts every 161 source frames from 0, then from
  80 (as far as one fits the recording), each latent frame taken from a window where it is the last frame of a latent
  frame, nearest that window's middle (on a tie, the window earlier in that order). The latent frames of a round past
  its recording's last such window have no recorded depth: their between-clients rows reach no pixel.
- Pixels at 2048 u or beyond are not reprojected; a point must land strictly inside the other frustum and more than
  1 u ahead, and the nearest point of a cell counts; the rays are float32 values and the tolerance is compared in
  float32, as the paper's rows were counted.
- A pair of views is scored at a latent frame when both players are alive (up to 0.5 s after a recording's last tick)
  and at least 50 of the first view's rays, placed at 128, 256, 512 or 1024 u, land in the second view, both on the
  recorded cameras (the eye 46 u above the origin while crouched), so every arm scores the same rows.
- Chance places the first view's depth from 10 s later (`k + 40`), or 10 s earlier near the end of a round, at its
  current camera; a frame with neither has no chance row. In ten-second windows it is the most distant instant.
- Table 19 keeps the between-clients and chance rows whose pair is co-visible in the recorded video (its row there has
  an agreeing pixel; `depth-cells --mask`), the recorded video included; the own-next-latent-frame rows are not masked.
  A unit (a round, a window) scores the ratio of the summed agreeing to reached pixels per segment, a cell is the mean
  over the units, and its interval resamples the units (10,000 draws), numbered in the order of their names as the
  paper's were.
- Table 10 differs from Table 19 in four ways: the depth head runs once on each whole window, each window's mp4 is
  encoded again with the Wan2.2 VAE (bf16), the recorded video's `gt/` one too (`depth --windows --recorded`), there is
  no mask, and chance takes the most distant instant.

```bash
python tools/evaluate_paper.py depth --config weights/paths.yaml \
    --config data/evaluation/eval/rounds.yaml --rounds configs/eval/round_index.jsonl \
    --session runs/table3 --out runs/depth/worldcast.npz
python tools/evaluate_paper.py depth --config weights/paths.yaml --config data/paths.yaml \
    --rounds configs/eval/round_index.jsonl --session runs/table3 --recorded --out runs/depth/recorded.npz
# scene_state_off.npz: the rows of an arm you render yourself
python tools/evaluate_paper.py depth-cells --rows worldcast=runs/depth/worldcast.npz \
    --rows scene_state_off=runs/depth/scene_state_off.npz --rows recorded=runs/depth/recorded.npz \
    --mask runs/depth/recorded.npz --difference worldcast,scene_state_off --out runs/depth/table19.json
python tools/evaluate_paper.py depth --config weights/paths.yaml \
    --config data/evaluation/eval/windows.yaml --windows configs/eval/windows.jsonl \
    --videos runs/windows/worldcast --out runs/depth/view_worldcast.npz
python tools/evaluate_paper.py depth --config weights/paths.yaml \
    --config data/evaluation/eval/windows.yaml --windows configs/eval/windows.jsonl \
    --videos runs/windows/worldcast --recorded --out runs/depth/view_recorded.npz
```

| table | row | first 30 s / first 5 s | after 30 s / last 5 s |
|---|---|---|---|
| 19 | WorldCast | 0.264 | 0.206 |
| 19 | Trained without scene state / Scene state off at inference | 0.259 / 0.229 | 0.135 / 0.133 |
| 19 | Only its own earlier blocks / Least-covering memory | 0.231 / 0.225 | 0.147 / 0.119 |
| 19 | Recorded video / Chance | 0.411 / 0.113 | 0.365 / 0.101 |
| 19 | WorldCast minus off: between clients / own next latent frame | +0.035 / +0.004 | +0.073 / +0.000 |
| 19, 4a | closed loop: WorldCast / Scene state off at inference | 0.247 / 0.243 | 0.120 / 0.092 |
| 10 | WorldCast / Frame concatenation / Solaris | 0.257 / 0.232 / 0.233 | 0.207 / 0.170 / 0.179 |
| 10 | Recorded video / Chance | 0.317 / 0.117 | 0.309 / 0.116 |

Checked: from the paper's rows, every cell of Tables 19 and 10 and the Depth row of Table 4a, with their intervals,
exactly. Rows recomputed from the depth arrays of one round equal the stored ones bit for bit with numpy's x86-64
float32 `exp`, also through `depth` itself (the recorded video's from the windows of its cache). The closed-loop
rows reproject each client's depth with the positions it published as its player state; `depth` reprojects with the
recorded cameras only, so those rows need the paper's camera tables.

#### Mat. Pix., Inliers and RotErr

`worldcast/engine/evaluation/matching.py` (Table 4a; App. "Scene state in detail", Agreement on the scene). The pairs
are the recorded video's between-clients depth rows (`depth --recorded`): ordered pairs of clients at a latent frame,
both alive and overlapping. Mat. Pix. is scored every 8 s (latent frames `k % 32 == 0`, so the first 30 s rest on 8,
16 and 24 s), Inliers and RotErr every second; a pair counts when it is co-visible in the recording in its own
direction.

- **Mat. Pix.**: GIM's dense matcher (DKM, the `gim_dkm_100h` weights; `worldcast/third_party/gim_dkm/`) on the two
  frames padded with black rows to 4:3, as GIM's demo runs it; the cells of DKM's 1152 x 1536 grid over the first frame
  whose certainty exceeds 0.05 (padding has none).
- **Inliers, RotErr**: LoFTR outdoor (kornia) on the grey frames, 8 consecutive pairs per call; its matches above
  confidence 0.5 give an essential matrix (OpenCV's MAGSAC++, 1 px, confidence 0.999, its default 1,000 iterations) at
  the intrinsics of HFOV 106.26 (fx = 252.0009, fy = 256.0010); Inliers is its inlier count (0 under 8 matches),
  RotErr the angle between the rotation recovered from it and the two recorded cameras' relative rotation.
- Per round and segment, Mat. Pix. and Inliers are means over the pairs and RotErr the median over those with an
  estimate; a cell is the mean over the rounds, and a difference between arms, per round, has the round bootstrap
  interval. Recordings are resized to 384 x 672 (bilinear, antialiased). With the screen mask (the lower right from
  half the width and 45% of the height, and the lower-left corner below 70% of the height and left of 30% of the
  width) a cell counts only if it and its match lie outside it, and LoFTR matches with an end inside it are dropped.

```bash
python tools/evaluate_paper.py match --config weights/paths.yaml \
    --config data/evaluation/eval/rounds.yaml --recorded-rows runs/depth/recorded.npz \
    --session runs/closed_loop_on --out runs/match/on.npz
python tools/evaluate_paper.py match --config weights/paths.yaml --config data/paths.yaml \
    --recorded-rows runs/depth/recorded.npz --recorded --out runs/match/recorded.npz
# off.npz: the pairs of an arm you render yourself
python tools/evaluate_paper.py match-cells --pairs on=runs/match/on.npz --pairs off=runs/match/off.npz \
    --pairs recorded=runs/match/recorded.npz --recorded-rows runs/depth/recorded.npz \
    --difference on,off --self-check recorded --out runs/match/table4a.json
```

| Table 4a | Off, first 30 s | On, first 30 s | Off, after 30 s | On, after 30 s |
|---|---|---|---|---|
| Mat. Pix. (K) | 72.0 | 92.1 | 31.1 | 44.3 |
| Inliers | 46.2 | 51.5 | 18.7 | 27.7 |
| RotErr (degrees) | 31.8 | 29.0 | 59.1 | 55.6 |

On minus off, first 30 s and after 30 s, with their intervals: Mat. Pix. +20.11K [5.08, 35.67] and +13.21K [4.51,
22.48], Inliers +5.23 [1.14, 9.34] and +8.96 [5.07, 13.15], RotErr -2.85 [-4.78, -0.90] and -3.43 [-6.65, -0.34]; with
the screen mask, after 30 s, +8.00K, +5.22 and -12.93. The recorded video's self-check, same instant against 10 s
apart: Mat. Pix. 163.1K against 52.3K, Inliers 128.3 against 36.4, RotErr 16.7 against 64.9. Checked: from the paper's
per-pair results (31 rounds and 155 clients: one round is left out and one is cut at 60 s), every cell and interval
exactly, and the pairs and keys are rebuilt in order.
The rounds of five clients are not in `configs/eval/`.

#### VBench

VBench 0.1.5 scores each clip in its own environment (its custom-input mode, local weights) on seven dimensions:
subject and background consistency, motion smoothness, temporal flickering, dynamic degree (a per-clip flag, so its
mean is the share of moving clips), imaging and aesthetic quality. `vbench` reads its per-clip results, one JSON file
per shard, `{"dims": [...], "clips": {name: {dimension: value}}}`, and takes the mean of each dimension over the clips
of a bin: the 20-minute bins of the one-hour rollouts (clips `<rollout>__s<SSSSS>.mp4`, Tables 4b and 13), the 10-s bins
of the first 60 s of the whole rounds (clips `<media_id>__start<NNNNNN>.mp4` as `cut` names them, Table 13), and the
percent change of the mean under each damage against the undamaged clips (`orig__<clip>` and `<damage>__<clip>`,
Table 14).

```bash
python tools/evaluate_paper.py vbench --hour runs/vbench/hour-*.json \
    --whole-rounds runs/vbench/rounds-*.json --sensitivity runs/vbench/damage-*.json --out runs/vbench/tables.json
```

Checked: from the paper's per-clip scores, all 105 cells of Tables 4b, 13 and 14. The paper generated the whole-round
clips of Table 13 with the four-step model trained without scene state, run at the `noclip` camera encoding although
it was trained with `clip`; `model.scene_state: false` runs it at `clip`, so its clips are not the paper's.
