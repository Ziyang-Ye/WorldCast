"""A batch's controls as the generator reads them."""

from collections.abc import Mapping
from typing import Any

import torch

from worldcast.modeling.controls import CONTROL_KEYS

__all__ = ["control_conditions"]


def control_conditions(
    batch: Mapping[str, Any], *, device: torch.device | str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """The generator's control conditions ``a_n`` (Eq. (1)) of a batch, on ``device``.

    Args:
        batch (Mapping[str, Any]): the entries
            :data:`~worldcast.modeling.controls.CONTROL_KEYS` over the ``T`` video frames of the
            window (the control history is never cut): ``buttons`` ``[B, T, 11]``, ``view_deltas``
            ``[B, T, 2]``, ``weapon`` ``[B, T]``; the control embedding checks the shapes.
        device (torch.device | str): where they go.
        dtype (torch.dtype): the generator's dtype, for the floating entries.

    Returns:
        dict[str, Tensor]: ``buttons`` and ``view_deltas`` in ``dtype``, ``weapon`` long.
    """
    buttons, view_deltas, weapon = (batch[key] for key in CONTROL_KEYS)
    controls = (
        buttons.to(device=device, dtype=dtype, non_blocking=True),
        view_deltas.to(device=device, dtype=dtype, non_blocking=True),
        weapon.long().to(device=device, non_blocking=True),
    )
    return dict(zip(CONTROL_KEYS, controls))
