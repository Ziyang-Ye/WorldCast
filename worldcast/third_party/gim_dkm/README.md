# GIM DKM (vendored)

The dense matcher of GIM, inference only, as Mat. Pix. runs it
(`worldcast/engine/evaluation/matching.py`).

- Upstream: https://github.com/xuelunshen/gim at commit `f09105a0555eef5c93db032d97b9e3d8a4cccb20`
- License: MIT, Copyright (c) 2024 Xuelun Shen (`LICENSE`, unchanged)
- Weights: `gim_dkm_100h.ckpt`, not part of this repository; GIM distributes it.

## Files

| here | upstream |
|---|---|
| `dkm.py` | `networks/dkm/models/dkm.py`: `ConvRefiner`, `CosKernel`, `CAB`, `RRB`, `DFN`, `GP`, `Decoder`, `RegressionMatcher` |
| `encoders.py` | `networks/dkm/models/encoders.py`: `ResNet50` |
| `local_correlation.py` | `networks/dkm/utils/local_correlation.py` |
| `model.py` | `networks/dkm/models/model_zoo/DKMv3.py`: `DKMv3` |
| `padding.py` | `tools/__init__.py`: `get_padding_size` (as `demo.py` uses it) |

## Changes

The arithmetic of `RegressionMatcher.match` is unchanged. Otherwise:

- Imports are relative to this package; the code is formatted with black and isort; a
  placeholder docstring and a TODO comment are dropped.
- `torch.meshgrid` is called with `indexing="ij"`, its default.
- `ResNet50` builds torchvision's ResNet-50 with `weights=None` and no dilation (upstream:
  `pretrained=False`); the other encoders are left out.
- `DKMv3(h, w, symmetric=True)`: the `weights`, `sample_mode` and keyword arguments, which the
  function did not use, are left out.
- `RegressionMatcher` keeps the inference path: `match` for one pair (no `batched` mode), without
  `sample`, `to_pixel_coordinates`, the contrastive loss and the `og_transforms`; `Decoder` drops
  its unused `upsample_preds` method and arguments. With them go `utils/kde.py`,
  `utils/utils.py`, `utils/transforms.py` and the `DKMv3_outdoor` / `DKMv3_indoor` loaders.
- The dense matcher's own `Encoder` class (unused by `DKMv3`) is left out.
