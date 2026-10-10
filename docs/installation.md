# Installation

```bash
git clone https://github.com/Ziyang-Ye/WorldCast && cd WorldCast
conda env create -f environment.yml && conda activate worldcast   # Python 3.10, torch 2.9.1 (CUDA 12.8), the package and its extras
pip install flash-attn==2.8.3 --no-build-isolation                 # FlashAttention-2 (optional, see below)
python tools/download_weights.py --out-dir weights
```

Run every command of the docs from the repository root. `environment.yml` pins the stack of the reference runs and installs the package with the extras below (`train`, `eval`, `showcase`, `test`); for another CUDA build change its `--extra-index-url` (pytorch.org). To install into an environment of your own: `pip install torch` (the CUDA build for your driver), `pip install -e .`, and the extras you need.

## What the extras are for

| extra | for |
|---|---|
| `.[showcase]` | `tools/make_showcase.py`: the drawing needs Pillow |
| `.[train]` | training: trimesh with embree, OpenCV, scipy ([docs/training.md](training.md)) |
| `.[train-decord]` | stage 1's video reader, decord, in place of OpenCV |
| `.[eval]` | `tools/evaluate.py`: SSIM and LPIPS as the paper's scorer computed them |
| `.[text]` | remaking the prompt embedding with umT5-XXL (`tools/make_prompt_embedding.py`) |
| `.[test]` | the tests (`python -m pytest tests -q`) |

Extras combine: `pip install -e ".[showcase,train,eval]"`.

## The attention kernel

`model.attention: flash` (flash-attention 2) is what the reference runs used; the latents of the paper's runs are
reproduced bit for bit with it on the reference stack: NVIDIA H20, torch 2.9.1+cu128, flash-attn 2.8.3
([docs/inference.md](inference.md), "Numerics"). `fa3` is FlashAttention-3 where it is installed. Without either, pass `--set model.attention=sdpa` to a tool: it runs, and its latents are not bit-equal.

## Memory

A client that writes latents (`tools/run_client.py`) takes about 15 GiB of GPU memory, so a GPU of 32 GB or more runs
two. The generator is 10.2 GB (bf16), the state model 1.4 GB and the depth head 0.18 GB
([docs/inference.md](inference.md), "Weights"); training's nodes are in [docs/training.md](training.md),
"Compute of the paper runs".

## Check it

```bash
python -m pytest tests -q                         # CPU; what needs a GPU, the weights or the data skips and says why
python tools/run_client.py --help
```

The examples are in the [README](../README.md#examples); the Table-3 rounds in [reproduce.md](reproduce.md).
