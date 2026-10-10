# Training

How to run the paper's four training stages with this repository: the stages, the data, the commands, the
checkpoints, the run shapes and the evaluation.

The release trains the generator (stages 1-4 and four of the ablations of Table 5). [Evaluation](#evaluation) scores
PSNR, SSIM and LPIPS on the 64 validation windows.

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
| 3: block-causal, with scene state | `stage3.yaml` | teacher forcing | stage 2s @ 5,000 (EMA) | 5,000 | 64 |
| 3: without scene state | `stage3_noscene.yaml` | teacher forcing | stage 2 @ 25,000 (EMA) | 5,000 | 64 |
| 4: four-step distillation, with scene state (the released generator) | `stage4.yaml` | distillation | generator: stage 3 @ 5,000 (EMA); teacher and critic: stage 2 @ 25,000 (EMA) | 600 | 64 |
| 4: without scene state | `stage4_noscene.yaml` | distillation | generator: stage 3 without scene state @ 5,000 (EMA); teacher and critic as above | 600 | 64 |

```
Wan2.2-TI2V-5B -> stage 1 @34k -> 1_long @40k -> stage 2 --@20k EMA--> 2s @5k EMA -> 3 @5k EMA -> 4 @600 (released)
                                                          \--@25k EMA--> 3 no scene @5k EMA -> 4 no scene @600
                                    stage 2 @25k EMA = "WorldCast" of Table 1 = teacher and critic of both stage-4 runs
```

Steps count from stage 1 as in the paper (stage 1 continued on ten-second windows, `1_long`, from step 34,000 to
40,000); a release run counts its own steps from 0, so the last checkpoint of `1_long` is `checkpoint_model_006000`.

**Checkpoints.** Each stage starts from the checkpoint the stage before it wrote in your own run (`checkpoint.init`;
[Commands](#commands)): stage 2s from stage 2 @ 20,000, stage 3 from stage 2s @ 5,000, stage 3 without scene state from
stage 2 @ 25,000, stage 4 from stage 3 @ 5,000. The teacher and critic of both stage-4 runs are stage 2 @ 25,000
(`distillation.teacher`, `distillation.critic`). Stage 1 starts from the Wan2.2 backbone.

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
and tick tables) and files derived from it; [docs/data.md](data.md) lists the files and their formats.

| artefact | config key | used by | format |
|---|---|---|---|
| media index | `data.media_index` | stages 2-4 | one row per player-round of OpenCS2 ([docs/inference.md](inference.md#data)) |
| raw-video manifest | `data.train_manifest` | stages 1, 1_long | [docs/data.md](data.md#raw-video-manifest) |
| bucket index | `data.bucket_dir` | stages 2-4 | windows, five files with one map each ([docs/data.md](data.md#bucket-index-evaluation-reserve-and-validation-index)) |
| evaluation reserve | `data.exclude_media_manifest` | stages 2-4 | the media ids held out for evaluation |
| latent cache | `data.latent_cache_root` | stages 2-4, evaluation | each window encoded with the Wan2.2 VAE ([docs/data.md](data.md#latent-cache)) |
| GT visibility labels | `data.visibility_label_root` | stages 2-4 | engine line-of-sight tests of every player pair ([docs/data.md](data.md#gt-visibility-labels)) |
| flash and scope labels | `data.observer_signal_label_root` | stages 2-4 | computed from the recorded video ([docs/data.md](data.md#flash-and-scope-labels)) |
| collision meshes | `data.collision_meshes` | stages 2s, 3, 4 | one glTF binary per map ([docs/data.md](data.md#collision-meshes)) |
| validation index | `validation.index` | evaluation | the 64 validation windows, checked by sha256 |
| umT5 embedding of the fixed prompt | `data.prompt_embedding` | all | released ([weights](inference.md#weights)) |
| Wan2.2-TI2V-5B snapshot | `model.wan22_root` | all | `tools/download_weights.py` fetches `config.json`, the VAE and the tokenizer; stage 1 also needs the backbone of [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) at the same revision, in that directory: the files `diffusion_pytorch_model*` (the `diffusion_pytorch_model-*.safetensors` shards and the `.safetensors.index.json` that lists them) |

Files you built yourself take two optional settings: `data.bucket_weights` (the sampling weight of each of the five
bucket files; the paper's unless set) and `validation.index_sha256` (the sha256 of the validation index, whose 64
windows are then selected from it; the paper's index unless set).

The memory frames and m_k of stages 2s, 3 and 4 decode target and teammate frames from the videos and ray-cast the
collision meshes in the loader workers: trimesh with an embree backend, OpenCV and scipy (the `train` extra).

## Commands

All stages run under torchrun; `torchrun ...` below stands for `torchrun --nnodes N --nproc_per_node 8 --node_rank
$RANK --master_addr $MASTER --master_port 29500` with the run's node count. `run.output_dir` receives `config.yaml`,
`metrics.jsonl` and the checkpoints. Any key can also be given with `--set` (the stage's settings follow `run.stage`
wherever it is set), and `--set model.attention=sdpa` runs without flash-attention. A key the config
does not have is refused (`ValueError`), whether a file or `--set` names it.

Each stage from the previous one's run (`examples/train_paths.yaml` sets stage 4's teacher and critic to stage 2 @ 25,000):

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

# stage 3 on 8 nodes (64 x 1), from stage 2s
torchrun ... tools/train.py --config configs/train/stage3.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage3 --set checkpoint.init=runs/stage2s/checkpoint_model_005000/model_ema.pt

# stage 4 on 8 nodes (64 x 1), from stage 3; the teacher and critic are stage 2 at step 25000
torchrun ... tools/train.py --config configs/train/stage4.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage4 --set checkpoint.init=runs/stage3/checkpoint_model_005000/model_ema.pt \
  --set distillation.teacher=runs/stage2/checkpoint_model_025000/model_ema.pt \
  --set distillation.critic=runs/stage2/checkpoint_model_025000/model_ema.pt

# without scene state, on 4 nodes (32 x 1 x 2): stage 3 from stage 2 at step 25000, then stage 4
torchrun ... tools/train.py --config configs/train/stage3_noscene.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage3_noscene \
  --set checkpoint.init=runs/stage2/checkpoint_model_025000/model_ema.pt
torchrun ... tools/train.py --config configs/train/stage4_noscene.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage4_noscene \
  --set checkpoint.init=runs/stage3_noscene/checkpoint_model_005000/model_ema.pt

# an ablation on 4 nodes (6,000 steps from stage 1_long)
torchrun ... tools/train.py --config configs/train/ablations/late_injection.yaml --config train_paths.yaml \
  --set run.output_dir=runs/late_injection --set checkpoint.init=runs/stage1_long/checkpoint_model_006000/model.pt
```

`--resume auto` (the default) continues from the newest complete checkpoint in `run.output_dir`; resume is exact
(weights, AdamW state, EMA, data position, RNG) and needs the checkpoint's world size and per-GPU batch.
`--print-config` prints the resolved config.

The data order depends on the world size (one weighted draw per epoch, sharded by rank; the loader is seeded `seed +
rank + 100003`), so reproducing a paper run's batches needs its topology ([Compute of the paper runs](#compute-of-the-paper-runs)). Whether a step trains on
memory frames is drawn per optimizer step, the same on every rank.

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
FSDP replica: once in all with `fsdp.sharding: full`, once per node with `hybrid_full`.

`checkpoint.keep: N` keeps the newest N complete checkpoints (0: all). `checkpoint.keep_shards: N` keeps the weights of
every checkpoint and the resume files of the newest N only; the older ones still load, but cannot be resumed.

## Compute of the paper runs

Batch: GPUs x per-GPU batch x gradient accumulation; a node has 8 GPUs (`--nnodes` of the `torchrun` line in
[Commands](#commands)).

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

## Evaluation

`tools/evaluate.py` scores a checkpoint on the 64 validation windows: ten-second windows, 16 per map, from the
held-out matches (`validation.index`; its sha256 and the selection's are checked). Each window starts from its
recorded first latent and its own noise (a generator seeded per window); the rollout and the cached latents are
decoded with the Wan2.2 VAE and compared on every frame after the first: PSNR, SSIM (torchmetrics), LPIPS (AlexNet),
pixel and latent MSE, as means over the windows, per map and per stratum (`vis<n>`: the number of players visible in
the window), with every window's scores.

```bash
pip install -e ".[eval]"   # torchmetrics, lpips; LPIPS fetches torchvision's AlexNet once (alexnet-owt-7be5be79.pth
                           # into $TORCH_HOME/hub/checkpoints; copy it there on machines without internet)
torchrun --nproc_per_node 4 tools/evaluate.py --config configs/train/stage2.yaml --config train_paths.yaml \
  --checkpoint runs/stage2/checkpoint_model_025000/model_ema.pt --out runs/eval/stage2.json
```

`--checkpoint` takes a release `.safetensors` file or a training checkpoint (its `--weights` entry, `generator_ema`
by default); the stage config gives the model, the data paths and the sampler. `--windows N` scores the first N
windows only.

| `--sampler` | stages | how a window is sampled |
|---|---|---|
| `unipc` | 1_long to 3 (their default) | 20 UniPC steps (shift 5): the whole window at once for the bidirectional stages, block by block for stage 3 (KV cache, each block written clean) |
| `four_step` | 4 (its default), 3 | the four denoising steps, block by block, the context written clean |

`four_step` re-noises x0 between denoising steps from the window's generator, so a score does not depend on how the
windows are spread over GPUs.

In training, `validation.index` and `validation.interval` (0: off; the paper runs: 1000) score the EMA weights on
these windows with `unipc` and append an `event: validation` row to `metrics.jsonl` (stages 1_long to 3). Every RNG
state is restored afterwards, so the run continues as without it. The windows carry no memory frames.
