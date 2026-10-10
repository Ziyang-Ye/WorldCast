"""The 20-step UniPC solver of the bidirectional and teacher-forced models (stages 1-3).

Multistep UniPC (Zhao et al., 2023) on a flow-matching model: order 2, the ``bh2`` variant, x0
updates, the predictor (UniP) after every call and the corrector (UniC) from the second call on,
order 1 at the first and last step, the final sigma 0. Each call's x0 is ``x_t - sigma_t flow``.

The arithmetic keeps the solver's tensor types as the trained models were scored: the sigmas are
0-dim float32 on the CPU, so the sample term is scaled in the sample's dtype (bf16 on the paper
path); the x0 terms and the sum are float32, and each update is rounded back to the sample's dtype
once.
"""

import numpy as np
import torch

from .schedulers import NUM_TRAIN_TIMESTEPS, TIMESTEP_SHIFT, shift_sigma

__all__ = ["FlowUniPCSolver"]

#: Solver order (the predictor uses the last two x0 estimates).
_SOLVER_ORDER = 2


class FlowUniPCSolver:
    """UniPC over ``num_steps`` timesteps of the shifted flow-matching schedule.

    ``timesteps`` ``[num_steps]`` int64 runs from 999 down; call :meth:`step` once per timestep, in
    order, with the model's flow at that timestep. A solver keeps the x0 estimates of its solve:
    build one per sample.

    Args:
        num_steps (int): solver steps (20 on the paper path).
        device (torch.device | str | None): device of ``timesteps``.
    """

    # Attribution: UniPC as in diffusers' UniPCMultistepScheduler (Apache-2.0), adapted to flow
    # matching as in Wan2.2's FlowUniPCMultistepScheduler (github.com/Wan-Video/Wan2.2, Apache-2.0,
    # (c) 2024-2025 The Alibaba Wan Team Authors).
    def __init__(self, num_steps: int, *, device: torch.device | str | None = None) -> None:
        # the largest sigma of the unshifted 1000-entry table, in the table's float32
        sigma_max = float(np.float32(1.0 - 1 / NUM_TRAIN_TIMESTEPS))
        sigmas = shift_sigma(np.linspace(sigma_max, 0.0, num_steps + 1)[:-1], TIMESTEP_SHIFT)
        self.timesteps = torch.from_numpy(sigmas * NUM_TRAIN_TIMESTEPS).to(
            device=device, dtype=torch.int64
        )
        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0]]).astype(np.float32))
        self.index = 0
        self._x0: list[torch.Tensor | None] = [None] * _SOLVER_ORDER
        self._last_sample: torch.Tensor | None = None
        self._order = 0

    def _alpha_sigma_lambda(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = self.sigmas[index]
        alpha = 1 - sigma
        return alpha, sigma, torch.log(alpha) - torch.log(sigma)

    @staticmethod
    def _corrector_weights(rks: list, hh: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """UniC's weights ``rho`` of order ``len(rks) + 1``: the solution of its ``bh2`` system."""
        rks = torch.tensor([*rks, 1.0], device=x.device)
        b_h = torch.expm1(hh)
        h_phi_k = b_h / hh - 1
        factorial = 1
        powers, b = [], []
        for i in range(1, len(rks) + 1):
            powers.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial / b_h)
            factorial *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial
        return torch.linalg.solve(torch.stack(powers), torch.tensor(b, device=x.device)).to(x.dtype)

    def _update(
        self,
        x: torch.Tensor,
        order: int,
        *,
        target: int,
        source: int,
        x0_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One UniP (``x0_t`` is None) or UniC update from ``x`` at ``source`` to ``target``."""
        alpha_t, sigma_t, lambda_t = self._alpha_sigma_lambda(target)
        _, sigma_s0, lambda_s0 = self._alpha_sigma_lambda(source)
        h = lambda_t - lambda_s0
        hh = -h
        b_h = torch.expm1(hh)
        m0 = self._x0[-1]
        rks, d1s = [], []
        for i in range(1, order):
            _, _, lambda_si = self._alpha_sigma_lambda(source - i)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((self._x0[-(i + 1)] - m0) / rk)
        if x0_t is None or order == 1:
            rhos = torch.tensor([0.5], dtype=x.dtype, device=x.device)
        else:
            rhos = self._corrector_weights(rks, hh, x)
        residual = 0
        if d1s:
            residual = torch.einsum("k,bkc...->bc...", rhos[: len(d1s)], torch.stack(d1s, dim=1))
        if x0_t is not None:
            residual = residual + rhos[-1] * (x0_t - m0)
        x_t = sigma_t / sigma_s0 * x - alpha_t * b_h * m0
        return (x_t - alpha_t * b_h * residual).to(x.dtype)

    def step(self, flow: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Advance ``sample`` one timestep given the model's ``flow`` at it.

        Args:
            flow (Tensor): the flow ``eps - x0`` at ``timesteps[index]``, float32.
            sample (Tensor): x_t, in the dtype the sampler keeps (bf16 on the paper path).

        Returns:
            Tensor: x at the next timestep, in ``sample.dtype``.
        """
        i = self.index
        if i >= len(self.timesteps):
            raise RuntimeError("the solver has made its last step; a new solve needs a new solver")
        x0 = sample - self.sigmas[i] * flow
        if i > 0:
            sample = self._update(self._last_sample, self._order, target=i, source=i - 1, x0_t=x0)
        self._x0 = [*self._x0[1:], x0]
        self._order = min(_SOLVER_ORDER, len(self.timesteps) - i, i + 1)
        self._last_sample = sample
        self.index += 1
        return self._update(sample, self._order, target=i + 1, source=i)
