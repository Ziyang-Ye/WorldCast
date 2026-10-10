"""A batch's controls as the generator reads them."""

import torch

from worldcast.modeling.controls import CONTROL_KEYS
from worldcast.sampling.conditions import control_conditions

VIDEO_FRAMES = 17


def _batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return {
        "buttons": torch.randint(0, 2, (1, VIDEO_FRAMES, 11), generator=g).float(),
        "view_deltas": torch.randn(1, VIDEO_FRAMES, 2, generator=g),
        "weapon": torch.randint(0, 52, (1, VIDEO_FRAMES), generator=g).float(),
        "frames": torch.zeros(1, 3, VIDEO_FRAMES, 4, 4),  # whatever else the batch holds
    }


def test_controls_in_the_generators_dtype():
    batch = _batch()
    controls = control_conditions(batch, device="cpu", dtype=torch.bfloat16)
    assert tuple(controls) == CONTROL_KEYS
    assert controls["buttons"].dtype == torch.bfloat16
    assert torch.equal(controls["view_deltas"], batch["view_deltas"].to(torch.bfloat16))
    assert controls["weapon"].dtype == torch.long
    assert torch.equal(controls["weapon"], batch["weapon"].long())
