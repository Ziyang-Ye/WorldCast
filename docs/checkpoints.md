# Checkpoints

What a client loads, where it comes from, and how to check it. `tools/download_weights.py` fetches the files below
from [ZiyangYe/WorldCast](https://huggingface.co/ZiyangYe/WorldCast) (config `weights.hf_repo_id`).

## Files

| file (config key) | what | format | size | sha256 |
|---|---|---|---|---|
| `worldcast_4step_bf16.safetensors` (`paths.checkpoint`) | the 4-step student generator, EMA weights, stage 4 step 600 (the paper's Table-3 WorldCast row) | safetensors, 946 tensors, every tensor bf16, generator key names; metadata `{format: pt, stage: 4, step: 600}` | 10,196,422,552 B | `8737c86b94659ba08469778fc20a5e309d9b96b6e9b4bfb3cb816d538db21f17` |
| `depth_head.safetensors` (`paths.depth_head`) | picture depth head `DepthLat` (width 384, 10 blocks, 5 down blocks; 44,199,600 params) | safetensors, fp32, `DepthLat` key names; metadata `{format: pt, module: DepthLat}` | 176,809,840 B | `ab599fcd68115142cf3b946e147f3cb89465d0b389371cc3501e1eb3101f6e76` |
| `depth_readout.safetensors` (`paths.depth_readout`) | depth read-out `Readout` (width 128; 208,132 params) | safetensors, fp32, `Readout` key names; metadata `{format: pt, module: Readout}` | 833,352 B | `6994480d726b7e958f519e835fa306b7045cfb19171ade4aebcb594a67ccd873` |
| `fixed_prompt_umt5xxl_bf16.safetensors` (`paths.prompt_embedding`) | umT5-XXL embedding of "first-person Counter-Strike 2 gameplay" | `[1, 512, 4096]` bf16, rows 10-511 zero, metadata `{format, prompt, seq_len: 10}` | 4,194,520 B | `a4157803a2c381835b219c079b7d811b7950c3fb2c47741d4552b7bba37cbd0a` (H20, bf16; equal to the paper client's umT5 output) |
| `worldcast_stage3_ar_fp32.safetensors` (`--with-training-checkpoints`) | stage 3: the block-causal model with scene state, EMA weights, step 5000 (initialises stage 4) | safetensors, 954 tensors, fp32, generator key names with the visibility probe; metadata `{format: pt, stage: 3, step: 5000}` | 20,396,189,716 B | `e510c06a4040ee401ca1dde5f04005ad1052182f3a5b33ceca7bc1c1d1306dca` |
| `worldcast_stage2s_bidirectional_fp32.safetensors` (`--with-training-checkpoints`) | stage 2s: the bidirectional model with player state field and scene state, EMA weights, 20,000 stage-2 + 5,000 stage-2s steps (initialises stage 3) | safetensors, 954 tensors, fp32, generator key names with the visibility probe; metadata `{format: pt, stage: 2s, step: 5000}` | 20,396,189,716 B | `68883cf28a38b0cde657aa5864df453a419ed35797f09b6283468402765f9c94` |
| `taew2_2.pth` (`--with-taehv`; demo `realtime.taehv_path`) | TAEHV tiny decoder (MIT, [madebyollin/taehv](https://github.com/madebyollin/taehv) at `011dfc2`), fetched from upstream | torch state dict | 22,884,021 B | `d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a` |
| `Wan-AI/Wan2.2-TI2V-5B` @ `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` (`paths.wan22_root`) | `config.json` (backbone dimensions), `Wan2.2_VAE.pth` (decode only), `google/umt5-xxl/` tokenizer, optionally `models_t5_umt5-xxl-enc-bf16.pth` | as published | VAE 2,818,839,170 B; umT5 11,361,920,418 B | VAE `20eb7896...`, umT5 `7cace0da...`, `tokenizer.json` `6e197b4d...` |
| `taew2_2.pth` from `madebyollin/taehv` @ `011dfc2112197741c540e0bdd5b7b67bcc930771` (MIT; `RealtimeConfig.taehv_path`, the demo's `taehv_path`) | tiny Wan2.2 VAE decoder of the low-latency engine (`decoder="taehv"`; 9.9M params) | state dict, as published | 22,884,021 B | `d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a` |

The 4-step file holds the EMA weights of the deployed step-600 student, every tensor cast to bf16: the parameter
dtype the paper's client ran (its FSDP setup cast every parameter to bf16). Every tensor was checked equal to the
source cast to bf16, and the release client on this file reproduced the reference round d57-d59 bit for bit on an
H20 (latents and fingerprints, `tools/verify_reference.py`). The training checkpoints keep fp32, so training from
them starts from the exact weights.

## Generator: tensors and names

- The 4-step file has 946 tensors, 5,098,162,792 parameters, under the key names of
  `worldcast.modeling.wan22.model.WorldCastGenerator.state_dict()`: a strict `load_state_dict` loads it.
- The training checkpoints add the visibility probe of stages 2s-4 (`visibility_head.*`, 8 tensors, 860,161 params;
  the training model, `visibility_head` on): 954 tensors, 5,099,022,953 parameters. Stage 2s is bidirectional and
  stage 3 block-causal; both are the same `WorldCastGenerator` (the attention mask is a training input, not a weight).
- Torch checkpoints of the research format (`{"generator_ema": {...}}` with keys under `model.` or
  `model._fsdp_wrapped_module.`) also load: `worldcast.engine.checkpoint.formats.remap_generator_state` drops the visibility
  probe and the two scalars of the legacy sigmoid splat (`peer_raster.raster_write.raster_bias`, `raster_log_kappa`)
  and renames the modules (`KEY_RENAMES`): `peer_raster.raster_write.` -> `state_injector.`, `worldplay_memory.` ->
  `rays.`, `opencs2_action.` -> `action.`, `*.opencs2_action_adaln.` -> `*.action_adaln.`. Backbone names are
  Wan2.2's.
- Architecture: Wan2.2-TI2V-5B causal DiT (30 blocks, dim 3072, 24 heads x 128, FFN 14336, patch 1x2x2, 48 latent
  channels, text 512 x 4096) plus the WorldCast inputs: control conditioner and per-block low-rank AdaLN adapters,
  observer-signal branch (flash / scope), Plücker ray embedding (`ray_unit_u` 420), the player state field stem
  (Conv 23 -> 32 -> 3072, added after DiT block 2). See `docs/inference.md`.

Load path (`worldcast.modeling.build.load_generator`): a `.safetensors` file is read as it is (no remap); a torch
file with `torch.load(..., mmap=True, weights_only=True)` and the remap. Then build on the `meta` device, assign, cast
every parameter to bf16 (the `param_dtype` of the paper's FSDP setup; a no-op for the bf16 file), move to the GPU.
The Wan2.2 backbone safetensors are never read: the WorldCast checkpoint replaces every backbone weight.

## Checking a download

```bash
sha256sum weights/worldcast/*.safetensors     # macOS: shasum -a 256 weights/worldcast/*.safetensors
python - <<'EOF'
from safetensors import safe_open
from safetensors.torch import load_file
path = "weights/worldcast/worldcast_4step_bf16.safetensors"
with safe_open(path, framework="pt") as fh:
    print(fh.metadata())                       # {'format': 'pt', 'stage': '4', 'step': '600'}
state = load_file(path)
print(len(state), sum(v.numel() for v in state.values()), {str(v.dtype) for v in state.values()})   # 946 5098162792
EOF
```

## Not in this release

- The state model (closed-loop player positions, the paper's Table 2 / 4a): its code and checkpoint are not part of
  this release.
- The stage-1 and stage-2 models (the bidirectional model of Table 1).
