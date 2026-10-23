# Third-party licenses

WorldCast is licensed under the Apache License 2.0 (`LICENSE`). Parts of it are derived from the projects below.
Under each project, an item names a file of this repository and what in it is derived, as modified by the WorldCast
authors; a file or a name in parentheses is the project's own.

## Wan2.2

- Source: https://github.com/Wan-Video/Wan2.2 (audited at 1ea34ff48f87168174e12956e200b1d908b1c5ff)
- License: Apache License 2.0 (text identical to `LICENSE`)
- Copyright: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
- Used in (modified):
  - `worldcast/modeling/wan22/dit.py`: `sinusoidal_embedding_1d`, `rope_params`, `rope_table` and `apply_rope`
    (from `rope_apply`), `RMSNorm`, `LayerNorm`, `Attention`, `CausalDiTBlock`, `CausalHead` (from
    `wan/modules/model.py`)
  - `worldcast/modeling/wan22/model.py`: the backbone of `WorldCastGenerator`: its construction,
    `_init_backbone_weights`, `embed_timesteps`, `unpatchify`, `run_dit_blocks`, `forward` (from `WanModel` of
    `wan/modules/model.py`)
  - `worldcast/modeling/wan22/attention.py`: `flash_attention` (from `wan/modules/attention.py`)
  - `worldcast/modeling/wan22/vae.py`: the Wan2.2 VAE, `Wan22VAE` and its layers (from `wan/modules/vae2_2.py`)
  - `worldcast/modeling/wan22/text_encoder.py`: the umT5 encoder `UMT5Encoder` with its layers, the tokenizer
    `UMT5Tokenizer` and `TextEncoder` (from `wan/modules/t5.py` and `wan/modules/tokenizers.py`; `t5.py` states it is
    "Modified from transformers.models.t5.modeling_t5": Hugging Face Transformers,
    https://github.com/huggingface/transformers, Apache License 2.0, Copyright 2018 Mesh TensorFlow authors, T5
    Authors and HuggingFace Inc. team)
  - `worldcast/sampling/unipc.py`: `FlowUniPCSolver` (from `FlowUniPCMultistepScheduler` of
    `wan/utils/fm_solvers_unipc.py`, which adapts the `UniPCMultistepScheduler` of Hugging Face Diffusers,
    https://github.com/huggingface/diffusers, Apache License 2.0, to flow matching)

## DiffSynth-Studio

- Source: https://github.com/modelscope/DiffSynth-Studio (`diffsynth/schedulers/flow_match.py`, as of 2025-03/05)
- License: Apache License 2.0
- Copyright: Copyright 2023 Zhongjie Duan
- Used in (modified):
  - `worldcast/sampling/schedulers.py`: `FlowMatchScheduler`, the schedule and `add_noise`
  - `worldcast/engine/training/losses.py`: `training_weight_table`, `training_weight`

## Self Forcing

- Source: https://github.com/guandeh17/Self-Forcing (audited at 33593df3e81fa3ec10239271dd2c100facac6de1)
- License: Apache License 2.0
- Authors: Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman
- Used in (modified):
  - `worldcast/modeling/wan22/dit.py`: `KVCache`, the KV cache
  - `worldcast/modeling/wan22/model.py`: `allocate_kv_cache`
  - `worldcast/modeling/wan22/training.py`: `_masked_self_attention`, the self-attention under teacher forcing;
    `forward_train`
  - `worldcast/modeling/wan22/attention.py`: `MaskLayout` and `attention_intervals`, the teacher-forcing mask
    (re-implemented)
  - `worldcast/sampling/schedulers.py`: `shift_denoising_steps`; `noise_context`, the context noise
  - `worldcast/sampling/sampler.py`: `write_context`, the cache write under context noise
  - `worldcast/engine/training/losses.py`: `sample_timestep_index`, `training_weight`, `sample_flow_matching`
  - `worldcast/engine/optim/ema.py`: `ShardedEMA`
  - `worldcast/engine/training/recipes/flow_matching.py`: `encode_raw_frames`
  - `worldcast/engine/training/recipes/distillation.py`: `exit_step` and `rollout`, the self-forcing rollout with
    one exit step per window (the method; re-implemented for this release)
- Note: Self Forcing is built on CausVid. The CausVid-derived items that reached WorldCast through Self Forcing are
  listed under CausVid below.

## ATI

- Source: https://github.com/bytedance/ATI (`wan/modules/motion_patch.py`)
- License: Apache License 2.0
- Copyright: Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
- Used in (re-implemented):
  - `worldcast/player_state/field.py`: `splat`, the splat kernel; `compose`, the top-k merge

## Matrix-Game 3.0

- Source: https://github.com/SkyworkAI/Matrix-Game (`Matrix-Game-3/utils/cam_utils.py`)
- License: Apache License 2.0 (`Matrix-Game-3/LICENSE.txt`; the repository root is MIT, Copyright (c) 2025 SkyworkAI
  and contributors)
- Used in (modified):
  - `worldcast/modeling/ray_embedding.py`: `plucker_rays`, `se3_inverse`, `relative_c2w`

## OpenAI CLIP

- Source: https://github.com/openai/CLIP (`clip/simple_tokenizer.py`)
- License: MIT. Copyright (c) 2021 OpenAI.
- Used in:
  - `worldcast/modeling/wan22/text_encoder.py`: `_basic_clean`, `_whitespace_clean` (via Wan2.2)
- License text: below.

## CausVid

- Source: https://github.com/tianweiy/CausVid, commit fab2440fb0386c5c9d8c561869a924bd173e1986 (2025-03-18), whose
  `LICENSE` is MIT. The repository was relicensed to CC BY-NC-SA 4.0 at commit 161c1dbc66698a89f5b29efc03fcca7c639ad8bb.
- License of the snapshot used: MIT. Copyright (c) 2025-2026 Tianwei Yin.
- The code reached WorldCast through Self Forcing (Apache License 2.0). The functions in which non-trivial CausVid
  expression survived were rewritten for this release from the papers and the math. What remains are the short
  idioms, configuration and boilerplate below, each marked with an attribution comment at its site:
  - `worldcast/modeling/wan22/dit.py`: `rope_table`, the temporal offset of the RoPE table
  - `worldcast/modeling/wan22/model.py`: `run_dit_blocks`, handing each DiT block its cache entry and the call's
    token offsets
  - `worldcast/sampling/schedulers.py`: `add_noise`, the batched nearest-entry lookup
  - `worldcast/sampling/sampler.py`: `write_context` and `prefill_context`, rerunning a finished block to fill
    the KV cache
  - `worldcast/modeling/wan22/text_encoder.py`: `TextEncoder`, zeroing each prompt's padding rows
  - `worldcast/modeling/wan22/vae.py`: `latent_scale`, the reciprocal taken after the cast
  - `worldcast/modeling/wan22/attention.py`: `flex_block_mask`, padding to 128 and the eager call that builds the
    block mask
  - `worldcast/engine/training/losses.py`: `sample_timestep_index`, the per-block broadcast of the first index
  - `worldcast/distributed/fsdp.py`: `fsdp_wrap` and `full_state_dict`, the FSDP configuration and the
    full-state-dict idiom
  - `worldcast/utils/seed.py`: `set_seed`
- License text: below.

## TAEHV

- Source: https://github.com/madebyollin/taehv, commit 011dfc2112197741c540e0bdd5b7b67bcc930771
- License: MIT. Copyright (c) 2025 Ollin Boer Bohan.
- Used in (modified):
  - `worldcast/engine/inference/taehv.py`: `TinyDecoder`, the architecture of the decoder (`taew2_2`), under its
    checkpoint layout
- The weights `taew2_2.pth` (sha256 `d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a`) are not
  redistributed: `tools/download_weights.py` fetches them from that commit.
- License text: below.

## GIM

- Source: https://github.com/xuelunshen/gim, commit f09105a0555eef5c93db032d97b9e3d8a4cccb20
- License: MIT. Copyright (c) 2024 Xuelun Shen. GIM's dense matcher adapts DKM (https://github.com/Parskatt/DKM,
  Copyright (c) 2022 Johan Edstedt, whose `LICENSE` grants the MIT terms).
- Used in (modified; the inference code of the matcher of Mat. Pix., vendored with its `LICENSE` and a `README.md` that
  lists the changes):
  - `worldcast/third_party/gim_dkm/dkm.py`: `ConvRefiner`, `CosKernel`, `CAB`, `RRB`, `DFN`, `GP`, `Decoder`,
    `RegressionMatcher` (from `networks/dkm/models/dkm.py`)
  - `worldcast/third_party/gim_dkm/encoders.py`: `ResNet50` (from `networks/dkm/models/encoders.py`)
  - `worldcast/third_party/gim_dkm/local_correlation.py`: `local_correlation` (from
    `networks/dkm/utils/local_correlation.py`)
  - `worldcast/third_party/gim_dkm/model.py`: `DKMv3` (from `networks/dkm/models/model_zoo/DKMv3.py`)
  - `worldcast/third_party/gim_dkm/padding.py`: `get_padding_size` (from the `__init__.py` of GIM's `tools` package)
- The weights `gim_dkm_100h.ckpt` (sha256 `bbe056bcf1b401e8dbc0ea983e69ed168eb0a330dcee35f3d9cf53048bf69ab8`) are not
  redistributed: `tools/download_weights.py --with-metrics` fetches them from the authors' Hugging Face Space
  `xuelunshen/gim-online` at revision ab4a4eb1dbd5db062b6fc8d9912f3abff4f62cdf. No license is stated for them.
- License text: below.

## Fonts

The page of the live demo ships three fonts in `worldcast/demo/static/fonts/`, unmodified, each with its license
beside it:

- Inter (`inter-latin-opsz-normal.woff2`): Copyright 2016 The Inter Project Authors
  (https://github.com/rsms/inter). SIL Open Font License 1.1 (`OFL-Inter.txt`).
- Inter Tight (`inter-tight-latin-wght-normal.woff2`): Copyright 2022 The Inter Project Authors
  (https://github.com/rsms/inter-tight). SIL Open Font License 1.1 (`OFL-InterTight.txt`).
- JetBrains Mono (`jetbrains-mono-latin-wght-normal.woff2`): Copyright 2020 The JetBrains Mono Project Authors
  (https://github.com/JetBrains/JetBrainsMono). SIL Open Font License 1.1 (`OFL-JetBrainsMono.txt`).

## Model weights and data

- Wan2.2-TI2V-5B (https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B, revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e):
  Apache License 2.0. WorldCast weights are fine-tuned from it. The VAE and umT5-XXL (google/umt5-xxl, Apache 2.0) are
  used unmodified.
- OpenCS2 dataset (https://huggingface.co/datasets/blanchon/opencs2_dataset): CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/legalcode). Source demos are from HLTV; the dataset card notes that the
  original tournament terms apply to them. Attributing a shared file derived from OpenCS2:
  [docs/data.md](docs/data.md#opencs2).
- The weights of the paper's metrics (`tools/evaluate_paper.py`) are not part of this repository:
  `tools/download_weights.py --with-metrics` fetches each from where it is published and checks its sha256
  (`worldcast.hub.METRIC_WEIGHTS`).
  - The player detector `yolo11x_cs2_player.pt` of Rendered and Unmatched: YOLO11x of Ultralytics
    (https://github.com/ultralytics/ultralytics), fine-tuned by the WorldCast authors; AGPL-3.0, as Ultralytics' models
    and code. It is published in the WorldCast weights repository, https://huggingface.co/ZiyangYe/WorldCast
    (`detector/yolo11x_cs2_player.pt`), under AGPL-3.0 unlike the Apache-2.0 files there, and runs only with the
    `metrics-detector` extra.
  - GIM's matcher: under GIM above.
  - LoFTR's outdoor weights (https://github.com/zju3dv/LoFTR, Apache License 2.0), from the address kornia fetches them
    from.
  - The Kinetics-400 I3D of FVD: DeepMind's kinetics-i3d (https://github.com/google-deepmind/kinetics-i3d, Apache
    License 2.0) as TorchScript, the file of the StyleGAN-V evaluation, from the Hugging Face mirror
    `flateon/FVD-I3D-torchscript` (stated MIT) at revision 1c2d61711c52571ad617a1063e6fa691b212e184.
- ValveResourceFormat (Source 2 Viewer, https://github.com/ValveResourceFormat/ValveResourceFormat): MIT License,
  Copyright (c) 2015 ValveResourceFormat Contributors. `tools/prepare_data.py meshes` runs its command-line tool on the
  user's own copy of the game's map files; the tool is not distributed with WorldCast, and neither is a Valve game
  file or an export of one. The game's files are Valve's:
  [docs/data.md](docs/data.md#the-games-map-files).

## Dependencies (installed by pip, not distributed with WorldCast)

The packages of `pyproject.toml`:

- required: torch (BSD-3-Clause), numpy (BSD-3-Clause), pyyaml (MIT), safetensors (Apache-2.0), pyarrow
  (Apache-2.0), imageio (BSD-2-Clause), imageio-ffmpeg (BSD-2-Clause; it fetches an FFmpeg binary under FFmpeg's
  license), huggingface_hub (Apache-2.0);
- the paper's attention kernel, installed by hand: flash-attn (BSD-3-Clause);
- `text`: transformers (Apache-2.0), ftfy (Apache-2.0), regex (Apache-2.0), sentencepiece (Apache-2.0);
- `train`: trimesh (MIT), embreex (MIT; it wraps Embree, Apache-2.0), opencv-python-headless (Apache-2.0), scipy
  (BSD-3-Clause); `train-decord`: decord (Apache-2.0);
- `eval`: torchmetrics (Apache-2.0), lpips (BSD-2-Clause);
- `demo`: aiohttp (Apache-2.0), pillow (MIT-CMU), torchvision (BSD-3-Clause);
- `metrics`: scipy (BSD-3-Clause), decord (Apache-2.0), opencv-python-headless (Apache-2.0); `metrics-detector`:
  ultralytics (AGPL-3.0; an extra of its own, imported only by `load_detector` of
  `worldcast/engine/evaluation/detector.py`); `metrics-matching`: kornia (Apache-2.0), kornia_rs (Apache-2.0), einops
  (MIT), torchvision (BSD-3-Clause), yacs (Apache-2.0);
- VBench (https://github.com/Vchitect/VBench, Apache-2.0), which the paper's VBench scores come from, runs in an
  environment of its own and is not an extra;
- `test`: pytest (MIT); `lint`: black (MIT), isort (MIT), flake8 (MIT); `dev`: pre-commit (MIT).

## License texts

### Apache License 2.0

See `LICENSE`.

### SIL Open Font License 1.1 (the fonts)

See `worldcast/demo/static/fonts/OFL-Inter.txt`, `worldcast/demo/static/fonts/OFL-InterTight.txt` and
`worldcast/demo/static/fonts/OFL-JetBrainsMono.txt`.

### MIT License (OpenAI CLIP)

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT License (CausVid at fab2440f)

As published in that commit's `LICENSE`:

```
MIT License

Copyright (c) 2025-2026 Tianwei Yin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT License (GIM at f09105a)

As published in that commit's `LICENSE` (also `worldcast/third_party/gim_dkm/LICENSE`):

```
MIT License

Copyright (c) 2024 Xuelun Shen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT License (DKM)

The terms of DKM's `LICENSE` (https://github.com/Parskatt/DKM):

```
Copyright (c) 2022 Johan Edstedt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MIT License (TAEHV at 011dfc2)

As published in that commit's `LICENSE`:

```
MIT License

Copyright (c) 2025 Ollin Boer Bohan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
