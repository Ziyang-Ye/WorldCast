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
  <b>Paper</b> (coming soon)&ensp;|&ensp;<a href="https://ziyang-ye.github.io/WorldCast-Page/"><b>Project page</b></a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" alt="Three independently controlled clients render consistent views of one Counter-Strike 2 round" width="100%">
</p>

## Overview

Multiplayer world models must generate independently controlled views with consistent representations of both players and their shared environment. Most existing approaches coordinate multiple players through joint multi-view generation, whose cost grows with each additional player.

**WorldCast** is a distributed multiplayer world model in which each player runs a local client comprising a video generator and a state model:

- **Player state.** Using recorded player positions and map geometry during training, the state model estimates the player's position from generated video and control inputs. Clients exchange player states and project them into camera-aligned player state fields that guide where and how other players are rendered.
- **Scene state.** A shared memory bank of generated blocks lets clients reuse one another's generated observations, keeping the scene's appearance consistent across views.
- **Distributed by design.** Each client runs in real time on its own GPU and exchanges only player and scene states, so multiplayer generation has no centralized computational bottleneck.

Experiments on Counter-Strike 2 show that the camera-aligned player state field improves player rendering rates by over an order of magnitude over joint-generation methods, that shared scene state improves visual consistency over whole rounds, and that image quality remains stable over hour-long rollouts.

<p align="center">
  <img src="assets/overview.jpg" alt="Coupled joint generation versus WorldCast's distributed clients" width="100%">
</p>
<p align="center"><em>(a) Coupled designs generate all players' views in one model. (b) Each WorldCast player runs its own client, and clients exchange only the shared world state.</em></p>

## Release

- [x] Inference
- [ ] Weights
- [ ] State model
- [ ] Training

The inference code reproduces the paper's Table-3 WorldCast row: each player's client runs on its own GPU, with
recorded player states and GT visibility labels, and the clients of a round share their generated blocks through a
pool directory in lock-step. How it works, the configuration and the numerical details: [docs/inference.md](docs/inference.md).
Data: [docs/data.md](docs/data.md). Weights: [docs/checkpoints.md](docs/checkpoints.md).

### Install

Python 3.10 or newer, a CUDA build of PyTorch, and (for the paper's attention kernel) flash-attention. In a fresh
virtual environment:

```bash
git clone https://github.com/Ziyang-Ye/WorldCast && cd WorldCast
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip                  # editable installs need pip >= 21.3
pip install torch                          # the CUDA build for your driver: see pytorch.org
pip install -e .
pip install flash-attn --no-build-isolation   # builds against the installed torch
```

A client takes about 15 GiB of GPU memory (on an NVIDIA H20, the paper's GPU), so one GPU can run two clients
(`--gpus 0,1,2,3,4,5,6,7,0,1` runs ten clients on eight GPUs; the latents are the same as with one GPU each). Without
flash-attention, pass `--attention sdpa` to the scripts below (runs, but is not bit-equal to the paper's kernel). The
paper ran torch 2.9.1+cu128 and flash-attn 2.8.3 on Python 3.10; `pip install -e .` accepts any torch 2.x from 2.4 on.

Optional extras: `.[text]` to re-make the prompt embedding (`tools/make_prompt_embedding.py`), `.[test]` for the
tests.

### Download weights

```bash
python tools/download_weights.py --out-dir weights
```

This fetches the generator, the depth head and read-out and the prompt embedding from
[ZiyangYe/WorldCast](https://huggingface.co/ZiyangYe/WorldCast), plus the Wan2.2 VAE, tokenizer and `config.json`
from `Wan-AI/Wan2.2-TI2V-5B`, and the web demo's tiny decoder `taew2_2.pth` (`madebyollin/taehv`, MIT, pinned commit
and sha256). It writes `weights/paths.yaml` for the commands below.

### Prepare data

The clients read OpenCS2 tick tables, visibility and observer-signal labels, a round index and the first latent of
each client. [docs/data.md](docs/data.md) lists the files and their formats. The files of the paper's rounds are not
published yet (see Release above). Put their locations in `data/paths.yaml`:

```bash
mkdir -p data && cp examples/data_paths.yaml data/paths.yaml   # then edit the paths
```

### Run a three-client session

```bash
python tools/run_session.py --config configs/infer/worldcast_4step.yaml \
    --config weights/paths.yaml --config data/paths.yaml \
    --group-of 59 --gpus 0,1,2 --out-dir runs/dust2-r09
```

This starts the round's three clients (one process per GPU) on a fresh pool directory and waits for all of them.
Outputs: `runs/dust2-r09/<round>/<media_id>/latents.npy`. A client renders one row of the round index (rows count
from 0); the clients of a round are the rows of its `group_media`.

### Run the clients one by one

`run_session.py` starts one `tools/run_client.py` process per client. Start them yourself to spread a round over
machines that share a file system: one process per row of the round, each on its own GPU, all with the same
`--live-pool-dir` (a fresh directory):

```bash
for row in 57 58 59; do      # the three rows of the round of row 59
  CUDA_VISIBLE_DEVICES=$((row - 57)) python tools/run_client.py --config configs/infer/worldcast_4step.yaml \
      --config weights/paths.yaml --config data/paths.yaml \
      --index-row $row --live-pool-dir runs/dust2-r09-by-hand/pool --out-dir runs/dust2-r09-by-hand/row$row &
done; wait
```

Each client writes `latents.npy` and `client.json` to its `--out-dir`. Before every block a client waits for all the
other clients of its round (lock-step; after `pool.wait_s` = 1800 s it stops with an error), so a client started on
its own does not finish unless its row's `group_media` lists only itself.

### Decode

```bash
python tools/decode.py --config weights/paths.yaml --latents runs/dust2-r09/*/*/latents.npy
```

Each `latents.npy` becomes a 16 fps 672x384 `video.mp4` next to it (Wan2.2 VAE in bf16, streaming in chunks of 8
latents, as in the paper). Compare `latents.npy`, not the mp4, when checking reproducibility.

### Examples

Six recorded rounds, one or more per map, with everything a session needs (GT player states, labels, first latents):

```bash
python tools/download_examples.py      # 200 MB
bash examples/run.sh mirage_r16        # three clients, 30 s; see examples/README.md for all six
```

### Live demo

WorldCast Live is a networked multiplayer web demo: a coordinator serves the page and relays the shared world state,
and every player gets a GPU worker that runs a client from the browser's keyboard and mouse. To try it without a GPU:

```bash
pip install -r requirements/demo.txt
python tools/serve_demo.py --role local --workers 2      # then open http://localhost:8100 in two windows
```

Serving the real model on GPUs: [docs/demo.md](docs/demo.md).

### Reproduce the paper's whole-round setting

The paper's Table 3 ran 32 rounds x 3 clients (`wholeround_index.jsonl`, 96 rows), each round as long as recorded
(capped at 120 s), seed 20260917:

```bash
python tools/run_session.py --config configs/infer/worldcast_4step.yaml \
    --config weights/paths.yaml --config data/paths.yaml \
    --all-groups --round-length --gpus 0,1,2 --out-dir runs/wholeround
python tools/decode.py --config weights/paths.yaml --latents runs/wholeround/*/*/latents.npy
```

On an H20 with torch 2.9.1+cu128 and flash-attn 2.8.3 the latents match the paper run bit for bit (checked on nine of
its clients: latents, every scene-state read and the decoded mp4 bytes; `tools/verify_reference.py`). `tests/reference`
checks the recorded fingerprints of those nine clients:
`WORLDCAST_REFERENCE_CONFIG=<yaml with your paths> python -m pytest tests/reference -q`.

## Training

The training code (all four stages) will be released here.

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests -q
```

The tests run on the CPU. Tests that need a GPU, the weights or the paper's data skip and say why.

## Citation

If you find WorldCast useful, please cite:

```bibtex
@article{ye2026worldcast,
  title   = {WorldCast: Distributed Multiplayer World Models},
  author  = {Ye, Ziyang and Huang, Junchao and Zhang, Evelyn and Xie, Zhihao and Zhang, Ruicheng and Han, Boyao and Ban, Litao and Wang, Ziye and Hu, Xinting and Shi, Shaoshuai and Tian, Zhuotao and Jiang, Li},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

WorldCast builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) (Wan2.2-TI2V-5B) and is trained and evaluated on the [OpenCS2 dataset](https://huggingface.co/datasets/blanchon/opencs2_dataset) (CC BY 4.0).
