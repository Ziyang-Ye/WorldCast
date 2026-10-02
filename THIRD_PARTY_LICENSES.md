# Third-party licenses

WorldCast is licensed under the Apache License 2.0 (`LICENSE`). Parts of it are derived from the projects below.
The files listed below contain code derived from these projects and modified by the WorldCast authors.

## Wan2.2

- Source: https://github.com/Wan-Video/Wan2.2 (audited at 1ea34ff48f87168174e12956e200b1d908b1c5ff)
- License: Apache License 2.0 (text identical to `LICENSE`)
- Copyright: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
- Used in (modified):
  - `worldcast/modeling/wan22/model.py`: `sinusoidal_embedding_1d`, `rope_params`, `rope_table` and `apply_rope`
    (from `rope_apply`), `RMSNorm`, `LayerNorm`, `Attention`, `CausalDiTBlock`, `CausalHead`,
    `WorldCastGenerator` backbone construction, `_init_backbone_weights`, `embed_timesteps`, `unpatchify`,
    `forward` (from `wan/modules/model.py`)
  - `worldcast/modeling/wan22/attention.py`: `flash_attention` (from `wan/modules/attention.py`)
  - `worldcast/modeling/wan22/vae.py`: the Wan2.2 VAE (from `wan/modules/vae2_2.py`)
  - `worldcast/modeling/wan22/text_encoder.py`: umT5 encoder and tokenizer wrapper (from `wan/modules/t5.py`, `tokenizers.py`).
    `t5.py` states it is "Modified from transformers.models.t5.modeling_t5" (Hugging Face Transformers,
    https://github.com/huggingface/transformers, Apache License 2.0, Copyright 2018 Mesh TensorFlow authors, T5 Authors
    and HuggingFace Inc. team).

## DiffSynth-Studio

- Source: https://github.com/modelscope/DiffSynth-Studio (`diffsynth/schedulers/flow_match.py`, as of 2025-03/05)
- License: Apache License 2.0
- Copyright: Copyright 2023 Zhongjie Duan
- Used in (modified): `worldcast/sampling/schedulers.py` `FlowMatchScheduler` (schedule, `add_noise`);
  `worldcast/engine/training/losses.py` `training_weight_table`, `training_weight`.

## Self Forcing

- Source: https://github.com/guandeh17/Self-Forcing (audited at 33593df3e81fa3ec10239271dd2c100facac6de1)
- License: Apache License 2.0
- Authors: Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman
- Used in (modified): the KV cache `KVCache`, teacher-forcing self-attention
  (`worldcast/modeling/wan22/model.py`); `warped_ladder`, the context-noise re-noise (`worldcast/sampling/schedulers.py`);
  the context-noise cache writes (`worldcast/sampling/rollouts.py`); `sample_timestep_index`, `training_weight`, `sample_flow_matching`
  (`worldcast/engine/training/losses.py`); sharded EMA (`worldcast/engine/optim/ema.py`); `encode_raw_frames`
  (`worldcast/data/training.py`). The exit-rung self-forcing rollout of `worldcast/engine/training/recipes/dmd.py`
  follows the Self Forcing method and was re-implemented for this release.
- Note: Self Forcing is built on CausVid. The CausVid-derived items that reached WorldCast through Self Forcing are
  listed under CausVid below.

## ATI

- Source: https://github.com/bytedance/ATI (`wan/modules/motion_patch.py`)
- License: Apache License 2.0
- Copyright: Copyright (c) 2024-2025 Bytedance Ltd. and/or its affiliates
- Used in (re-implemented): `worldcast/player_state/field.py` splat kernel (`splat_temperature`, `splat_weights`) and top-k
  merge (`compose_field`).

## Matrix-Game 3.0

- Source: https://github.com/SkyworkAI/Matrix-Game (`Matrix-Game-3/utils/cam_utils.py`)
- License: Apache License 2.0 (`Matrix-Game-3/LICENSE.txt`; the repository root is MIT, Copyright (c) 2025 SkyworkAI
  and contributors)
- Used in (modified): `worldcast/modeling/rays.py` `se3_inverse`, `relative_c2w`.

## OpenAI CLIP

- Source: https://github.com/openai/CLIP (`clip/simple_tokenizer.py`)
- License: MIT. Copyright (c) 2021 OpenAI.
- Used in: `worldcast/modeling/wan22/text_encoder.py` `_basic_clean`, `_whitespace_clean` (via Wan2.2).
- License text: below.

## CausVid

- Source: https://github.com/tianweiy/CausVid, commit fab2440fb0386c5c9d8c561869a924bd173e1986 (2025-03-18), whose
  `LICENSE` is MIT. The repository was relicensed to CC BY-NC-SA 4.0 at commit 161c1dbc66698a89f5b29efc03fcca7c639ad8bb.
- License of the snapshot used: MIT. Copyright (c) 2025-2026 Tianwei Yin.
- The code reached WorldCast through Self Forcing (Apache License 2.0). The functions in which non-trivial CausVid
  expression survived were rewritten for this release from the papers and the math. What remains are the short idioms, configuration and boilerplate below, each marked with an
  attribution comment at its site:
  - `worldcast/modeling/wan22/model.py`: `rope_table`, the temporal offset of the RoPE table;
    `WorldCastGenerator.forward`, handing each block its cache entry and the call's token offsets
  - `worldcast/sampling/schedulers.py`: `FlowMatchScheduler.add_noise` / `timestep_index`, the batched
    nearest-entry lookup
  - `worldcast/sampling/rollouts.py`: `Sampler.commit` / `rollout_prefix` / `generate_block`, rerunning a finished
    block to fill the KV cache
  - `worldcast/modeling/wan22/text_encoder.py`: `TextEncoder.forward`, zeroing each prompt's padding rows
  - `worldcast/modeling/wan22/vae.py`: `latent_scale`, the reciprocal taken after the cast
  - `worldcast/modeling/wan22/attention.py`: `flex_block_mask`, padding to 128 and the eager `create_block_mask` call
  - `worldcast/engine/training/losses.py`: `sample_timestep_index`, the per-block broadcast of the first index
  - `worldcast/distributed/fsdp.py`: `fsdp_wrap`, `full_state_dict` (FSDP configuration and idiom)
  - `worldcast/distributed/process_group.py`: `seed_everything`
  - `worldcast/utils/seed.py`: `set_seed`
- License text: below.

## TAEHV

- Source: https://github.com/madebyollin/taehv, commit 011dfc2112197741c540e0bdd5b7b67bcc930771
- License: MIT. Copyright (c) 2025 Ollin Boer Bohan.
- Used in (modified): `worldcast/engine/realtime/taehv.py`, the `taew2_2` decoder (architecture under its checkpoint
  layout). The weights `taew2_2.pth` (sha256 `d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a`)
  are not redistributed: `tools/download_weights.py` fetches them from that commit.
- License text: below.

## Model weights and data

- Wan2.2-TI2V-5B (https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B, revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e):
  Apache License 2.0. WorldCast weights are fine-tuned from it. The VAE and umT5-XXL (google/umt5-xxl, Apache 2.0) are
  used unmodified.
- OpenCS2 dataset (https://huggingface.co/datasets/blanchon/opencs2_dataset): CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/legalcode). Source demos are from HLTV; the dataset card notes that the
  original tournament terms apply to them.

## Runtime dependencies (installed by pip, not distributed with WorldCast)

torch (BSD-3-Clause), numpy (BSD-3-Clause), PyYAML (MIT), safetensors (Apache-2.0), pyarrow (Apache-2.0),
imageio (BSD-2-Clause), imageio-ffmpeg (BSD-2-Clause; it fetches an FFmpeg binary under FFmpeg's license),
huggingface_hub (Apache-2.0); optional: flash-attn (BSD-3-Clause), transformers (Apache-2.0), ftfy, regex,
sentencepiece, trimesh (MIT), embreex, opencv-python-headless (Apache-2.0), scipy (BSD-3-Clause), decord.

## License texts

### Apache License 2.0

See `LICENSE`.

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
