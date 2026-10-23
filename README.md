<p align="center">
  <img src="assets/banner.jpg" alt="WorldCast: Distributed Multiplayer World Models" width="100%">
</p>

<h1 align="center">WorldCast: Distributed Multiplayer World Models</h1>

<p align="center">
  <a href="https://ziyang-ye.github.io/">Ziyang Ye</a><sup>1</sup>&ensp;
  <a href="https://junchao-cs.github.io/">Junchao Huang</a><sup>1,2</sup>&ensp;
  Evelyn Zhang<sup>2</sup>&ensp;
  Zhihao Xie<sup>1</sup>&ensp;
  Ruicheng Zhang<sup>3</sup>&ensp;
  Boyao Han<sup>1</sup><br>
  Litao Ban<sup>4</sup>&ensp;
  Ziye Wang<sup>4</sup>&ensp;
  <a href="https://joyhuyy1412.github.io/">Xinting Hu</a><sup>5</sup>&ensp;
  <a href="https://shishaoshuai.com/">Shaoshuai Shi</a><sup>4</sup>&ensp;
  <a href="https://zhuotaotian.github.io/">Zhuotao Tian</a><sup>2</sup>&ensp;
  <a href="https://llijiang.github.io/">Li Jiang</a><sup>1,2&dagger;</sup>
</p>

<p align="center">
  <sup>1</sup>CUHK-Shenzhen&ensp;
  <sup>2</sup>SLAI&ensp;
  <sup>3</sup>Tsinghua SIGS&ensp;
  <sup>4</sup>Voyager Research, Didi Chuxing&ensp;
  <sup>5</sup>USTC<br>
  <sup>&dagger;</sup>Corresponding author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.12412"><img src="https://img.shields.io/badge/arXiv-2610.12412-b31b1b" alt="arXiv paper"></a>
  <a href="https://ziyang-ye.github.io/WorldCast-Page/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <a href="https://huggingface.co/ZiyangYe/WorldCast"><img src="https://img.shields.io/badge/Hugging_Face-Weights-yellow" alt="Hugging Face weights"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License: Apache-2.0"></a>
</p>

<p align="center"><b>TL;DR: WorldCast is a distributed multiplayer world model: each player runs a local client with its own video generator and state model, and on Counter-Strike 2 the camera-aligned player state field raises player rendering rates by over an order of magnitude over joint-generation methods.</b></p>

<p align="center"><img src="assets/teaser.jpg" alt="Three independently controlled clients render consistent views of one Counter-Strike 2 round" width="100%"></p>

## Overview

Multiplayer world models must generate independently controlled views that stay consistent in both the players and their shared environment. Most existing approaches couple the players through joint multi-view generation, whose cost grows with every added player. **WorldCast** is a distributed multiplayer world model: each player runs a local client with its own video generator and state model.

- **Player state.** A state model, trained on recorded player positions and map geometry, estimates each player's position from its generated video and controls. Clients exchange player states and project them into camera-aligned player state fields that guide where and how the other players are rendered.
- **Scene state.** A shared memory bank of generated blocks lets clients reuse each other's observations, which keeps the scene consistent across views.
- **Distributed by design.** Each client runs in real time on its own GPU and exchanges only player and scene states, so multiplayer generation has no centralized computational bottleneck.

On Counter-Strike 2, the camera-aligned player state field raises player rendering rates by over an order of magnitude over joint-generation methods, shared scene state improves visual consistency over whole rounds, and image quality stays stable over hour-long rollouts.

<p align="center"><img src="assets/overview.jpg" alt="Coupled joint generation versus WorldCast's distributed clients" width="100%"></p>
<p align="center"><em>(a) Coupled designs generate all players' views in one model. (b) Each WorldCast player runs its own client, and clients exchange only the shared world state.</em></p>

## Release

- [x] Inference: the clients of a round in lockstep on recorded player states, or in the closed loop ([docs/inference.md](docs/inference.md))
- [x] Weights: the four-step generator, the state model, the depth head, and the checkpoints of stages 2, 2s and 3 that training starts from ([docs/inference.md](docs/inference.md#weights))
- [x] State model: each client's own position from its generated video and controls ([docs/inference.md](docs/inference.md#closed-loop))
- [x] Training: stages 1-4 and four ablations of Table 5 ([docs/training.md](docs/training.md))
- [x] Data preparation: the training files from OpenCS2, `tools/prepare_data.py` ([docs/data.md](docs/data.md))
- [x] Evaluation: the paper's metrics (`tools/evaluate_paper.py`) with the player detector, and the evaluation data (`tools/download_eval_data.py`) ([docs/training.md](docs/training.md#the-papers-metrics))
- [x] Live demo: WorldCast Live, a networked multiplayer web demo ([docs/demo.md](docs/demo.md))

## Quick start

### Install

```bash
git clone https://github.com/Ziyang-Ye/WorldCast && cd WorldCast
conda env create -f environment.yml && conda activate worldcast    # Python 3.10, torch 2.9.1 (CUDA 12.8), the package and its extras
pip install flash-attn==2.8.3 --no-build-isolation                  # FlashAttention-2 (optional; for latents bit-exact with the paper)
```

Run every command from the repository root. For another CUDA build edit the index in `environment.yml`; to install into an existing environment, `pip install torch && pip install -e .`. Without flash-attention, add `--set model.attention=sdpa` to a tool (extras, attention kernels and GPU memory: [docs/installation.md](docs/installation.md)).

### Weights

```bash
python tools/download_weights.py --out-dir weights
```

Fetches the four-step generator, the state model, the depth head with its read-out and the prompt embedding from [ZiyangYe/WorldCast](https://huggingface.co/ZiyangYe/WorldCast), plus the Wan2.2 VAE and tokenizer and the live demo's tiny decoder, and writes `weights/paths.yaml` for the commands below. Files and checksums: [docs/inference.md](docs/inference.md#weights).

### Examples

```bash
python tools/download_examples.py --out-dir examples   # six recorded rounds
bash examples/run.sh mirage_r16                        # three clients, one GPU each
```

A case runs the clients of a recorded round in lockstep on their GT player states, decodes them and tiles the views into `runs/examples/mirage_r16/grid.mp4`. Each client's `latents.npy` and `video.mp4` are under `runs/examples/mirage_r16/<round>/<media_id>/`.

The [closed loop](docs/inference.md#closed-loop), where the clients take their positions from the state model, is the same case with a setting and a new output directory (`run.sh` starts only on one that does not exist):

```bash
OUT=runs/examples/mirage_r16_closed_loop bash examples/run.sh mirage_r16 --set player_state.source=predicted
```

The other five cases: [examples/README.md](examples/README.md); the 32 rounds of Table 3: [docs/reproduce.md](docs/reproduce.md).

### Live demo

```bash
pip install -e ".[demo]"
python tools/serve_demo.py --role local --workers 2   # a mock client, no GPU: open http://localhost:8100 in two windows
```

WorldCast Live is a networked multiplayer web demo: the lobby shows a top-down overview of each map, a click starts a room at the nearest round start, and each player's view comes from their own client on a GPU worker, played with the keyboard and mouse (the mouse turns the view once Play captures it), a gamepad or a phone's on-screen pad. On GPUs, `tools/setup_demo.py` fetches the weights, checks the library of round starts and prints the command that serves a worker per card ([Quick start](docs/demo.md#quick-start)). The controls and the demo's latency: [docs/demo.md](docs/demo.md).

## Training and evaluation

### Training

```bash
pip install -e ".[train]"
python tools/download_weights.py --out-dir weights --with-training-checkpoints
cp examples/train_paths.yaml train_paths.yaml   # then point it at the data
torchrun --nnodes 8 --nproc_per_node 8 --node_rank $RANK --master_addr $MASTER --master_port 29500 \
  tools/train.py --config configs/train/stage4.yaml --config train_paths.yaml \
  --set run.output_dir=runs/stage4 --set checkpoint.init=weights/worldcast/worldcast_stage3_ar_fp32.safetensors
```

This distils the released stage-3 model into the four-step generator, with the released stage-2 model (the WorldCast of Table 1) as teacher and critic. Each stage starts from the checkpoint of the stage before it (`checkpoint.init`). Training reads [OpenCS2](https://huggingface.co/datasets/blanchon/opencs2_dataset) and files derived from it, which `tools/prepare_data.py` builds ([docs/data.md](docs/data.md)). Every stage's command and its checkpoints: [docs/training.md](docs/training.md).

<details><summary>The stages and their configs</summary>

| stage | config | trains | starts from |
|---|---|---|---|
| 1 | `stage1.yaml` | Wan2.2-TI2V-5B, bidirectional, on gameplay: 5 s windows | Wan2.2-TI2V-5B |
| 1_long | `stage1_long.yaml` | stage 1 continued on 10 s windows | stage 1 |
| 2 | `stage2.yaml` | + the player state field | stage 1_long |
| 2s | `stage2s.yaml` | + the scene state (memory frames) | stage 2 |
| 3 | `stage3.yaml` | block-causal, teacher forcing | stage 2s (released) |
| 4 | `stage4.yaml` | four-step distribution matching distillation: the released generator | stage 3 (released); teacher and critic: stage 2 (released; `distillation.teacher`, `distillation.critic`) |

`stage3_noscene.yaml` and `stage4_noscene.yaml` train without scene state, `configs/train/ablations/` holds the ablations.
</details>

### Evaluation

```bash
pip install -e ".[eval]"
python tools/download_eval_data.py --out-dir data/evaluation validation   # the windows' data, 0.3 GB
torchrun --nproc_per_node 4 tools/evaluate.py --config configs/train/stage2.yaml --config train_paths.yaml \
  --config data/evaluation/eval/validation.yaml \
  --checkpoint weights/worldcast/worldcast_stage2_bidirectional_fp32.safetensors --out runs/eval/stage2.json
```

Writes PSNR, SSIM, LPIPS, pixel and latent MSE on the 64 ten-second windows the paper's training runs validated on (the paper's index, `configs/data/validation_index.jsonl`, and the windows' latents, labels and tick tables, fetched by `tools/download_eval_data.py`), overall, per map and per window. A stage-4 config scores the four-step generator. Samplers and reference values: [docs/training.md](docs/training.md#evaluation).

```bash
pip install -e ".[metrics,metrics-detector,metrics-matching]"
python tools/download_weights.py --out-dir weights --with-metrics
python tools/download_eval_data.py --out-dir data/evaluation windows rounds
```

`tools/evaluate_paper.py` computes the metrics of the paper's tables on its evaluation sets: Rendered and Unmatched, FVD, depth agreement between clients, Mat. Pix., Inliers and RotErr, and the VBench bins ([docs/training.md](docs/training.md#the-papers-metrics)). The player detector of Rendered and Unmatched, a fine-tuned YOLO11x, is in the weights repository as `detector/yolo11x_cs2_player.pt` and is AGPL-3.0 like ultralytics, unlike the other files there ([model card](https://huggingface.co/ZiyangYe/WorldCast#player-detector)).

## From the paper to the code

<details><summary>Where each part of the paper is in <code>worldcast/</code></summary>

| paper | code (`worldcast/`) |
|---|---|
| Sec. 3.1: the client; the generator $G_\theta$, Eq. (1) | `engine/inference/client.py` (`Client`); `modeling/wan22/model.py` (`WorldCastGenerator`), `sampling/` (the window, the four denoising steps) |
| Sec. 3.2: player state; the player state field $\mathbf{F}$; the state injector, Eq. (2) | `player_state/states.py` (`PlayerState`), `player_state/field.py` (`player_state_field`); `modeling/state_injector.py` (`StateInjector`) |
| Sec. 3.3: scene state, memory bank, retrieval; the ray embedding, Eq. (3) | `scene_state/` (`SceneState`, `MemoryBank`, `MemoryEntry`); `modeling/ray_embedding.py` (`RayEmbedding`) |
| Sec. 3.4: the state model; the complementary filter, Eq. (4); depth head; visibility | `modeling/state_model.py` (`StateModel`); `player_state/closed_loop.py` (`ComplementaryFilter`); `modeling/depth_head.py` (`DepthHead`); `player_state/visibility.py` |
| Sec. 3.5: the training stages; the loss, Eq. (5) | `config/training.py` (`STAGES`), `engine/training/recipes/`; `engine/training/losses.py` (`flow_matching_loss`) |
| Sec. 4 and the appendix: the metrics (Rendered and Unmatched, depth agreement, Mat. Pix., FVD, VBench) | `engine/evaluation/entity.py` (`EntitySummary`), `engine/evaluation/detector.py` (`detect_video`), `engine/evaluation/depth_agreement.py`, `engine/evaluation/matching.py`, `engine/evaluation/fvd.py` (`frechet_distance`), `engine/evaluation/vbench.py` |
| App. B: the deployment protocol (messages, lockstep) | `engine/inference/world_state.py` (`WorldState`), `engine/inference/directory.py` (`DirectoryWorldState`), `engine/inference/messages.py` (`MessagePart`, `MessageWorldState`) |

Each piece works on its own; [docs/inference.md](docs/inference.md#as-a-library) has a short example.
</details>

## Code structure

```
WorldCast/
├── worldcast/   the package: modeling/ (generator, state model, depth head), player_state/, scene_state/,
│                sampling/, engine/ (inference/, training/, evaluation/), data/ (preparation/), demo/, config/
├── tools/       one script per command: python tools/<name>.py --help
├── configs/     train/ (stage and ablation configs), demo/, state_model/ (cell and speed tables),
│                data/ and eval/ (the paper's held-out matches, validation index and evaluation sets)
├── examples/    the six recorded rounds, run.sh, the reference fingerprints
├── docs/        installation, inference, reproduce, demo, training, data, development
└── tests/       CPU tests mirroring the package: pip install -e ".[test]" && python -m pytest tests -q
```

The conventions for contributors (the repo map, the vocabulary, the reproducibility contract, the style) are in
[docs/development.md](docs/development.md).

## Citation

If you find WorldCast useful, please cite:

```bibtex
@article{ye2026worldcast,
  title   = {WorldCast: Distributed Multiplayer World Models},
  author  = {Ye, Ziyang and Huang, Junchao and Zhang, Evelyn and Xie, Zhihao and Zhang, Ruicheng and Han, Boyao and Ban, Litao and Wang, Ziye and Hu, Xinting and Shi, Shaoshuai and Tian, Zhuotao and Jiang, Li},
  journal = {arXiv preprint arXiv:2610.12412},
  year    = {2026}
}
```

## Acknowledgements

WorldCast builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) (Wan2.2-TI2V-5B) and is trained and evaluated on the [OpenCS2 dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset) (CC BY 4.0). The live demo's tiny decoder is [TAEHV](https://github.com/madebyollin/taehv) (MIT).

## License

Apache-2.0 ([LICENSE](LICENSE)); third-party code and weights: [NOTICE](NOTICE), [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
