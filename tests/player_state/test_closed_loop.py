"""The closed loop's orchestration (no research code needed): the position exchange between clients,
the agreement it produces between their tables, the rung-1 retest of the ladder and the config
switch."""

from pathlib import Path

import numpy as np
import pytest
import torch

from worldcast.config.inference import (
    ACTION_BUTTONS,
    InferenceConfig,
    config_from_dict,
    load_config,
)
from worldcast.engine.inference.client import Client
from worldcast.player_state import extrapolate as ex
from worldcast.player_state.closed_loop import ClosedLoop, StateExchange
from worldcast.sampling.schedulers import FlowMatchScheduler, ladder_denoise, warped_ladder

REPO = Path(__file__).resolve().parents[2]
PRIOR = ex.PhysicsPrior.load(REPO / "configs" / "state_model" / "physics_prior_v1.json")
N_LATENTS = 29
T_ROWS = 1 + 4 * (N_LATENTS - 1)
MEDIA = {
    0: "1-de_dust2-r2-p00",
    1: "1-de_dust2-r2-p01",
    2: "1-de_dust2-r2-p02",
    3: "1-de_dust2-r2-p03",
}


class FakeReader:
    """Stands in for the state model: a deterministic place estimate and motion per knot."""

    def __init__(self, seed: int) -> None:
        self.g = np.random.default_rng(seed)
        self.calls = []

    def read(self, latents, knots):
        self.calls.append((int(latents.shape[0]), list(knots)))
        k = np.asarray(knots, np.float64)[:, None]
        return 50.0 * k + self.g.normal(0, 5, (len(knots), 3)), self.g.normal(
            0, 1, (len(knots), 3)
        ).astype(np.float32)


def round_batch() -> dict:
    g = torch.Generator().manual_seed(0)
    states = torch.zeros(1, 4, T_ROWS, 6)
    states[..., :3] = torch.randn(1, 4, 1, 3, generator=g) * 400
    states[..., 5] = 1.0
    states[0, 3, :, 5] = 0.0
    sub = torch.cat(
        [
            (torch.rand(1, 4, T_ROWS, 4, 11, generator=g) < 0.3).float(),
            torch.randn(1, 4, T_ROWS, 4, 2, generator=g) * 0.2,
        ],
        -1,
    )
    return dict(
        player_states=states,
        player_action_substeps=sub,
        player_action_substep_valid=torch.ones(1, 4, T_ROWS, 4, dtype=torch.bool),
        observer_slot=torch.tensor([0]),
        observer_visibility=torch.zeros(1, 4, T_ROWS, dtype=torch.bool),
        observer_visibility_valid=torch.zeros(1, 4, T_ROWS, dtype=torch.bool),
    )


def client(slot: int, root: Path) -> ClosedLoop:
    batch = round_batch()
    batch["observer_slot"] = torch.tensor([slot])
    others = [MEDIA[p] for p in (0, 1, 2) if p != slot]
    return ClosedLoop.build(
        batch,
        reader=FakeReader(slot),
        me=MEDIA[slot],
        my_slot=slot,
        round_media=MEDIA,
        clients=others,
        n_latents=N_LATENTS,
        start_frame=0,
        motion_kwargs=dict(
            camera_delta_scale=5.0, channels=ex.prior_channels(ACTION_BUTTONS), prior=PRIOR
        ),
        eye_height=64.0,
        depth_fn=lambda x: np.full((int(x.shape[0]), 4, 24, 42), 7.0, np.float32),
        pose_radius=40.0,
        exchange=StateExchange(root, MEDIA[slot], poll_s=0.01, fatal=True),
    )


def test_exchange_round_trip(tmp_path):
    ex_a = StateExchange(tmp_path, "a", poll_s=0.01, fatal=True)
    xyz = np.random.default_rng(0).normal(0, 1e3, (4, 3))
    ex_a.publish(64, [4, 5, 6, 7], xyz)
    knots, got = ex_a.read("a", 64)
    assert knots == [4, 5, 6, 7] and np.array_equal(got, xyz)
    assert ex_a.read("b", 64) is None
    ex_a.wait(["a", "b"], 64, max_wait_s=1.0, is_done=lambda m: m == "b")
    with pytest.raises(TimeoutError):
        ex_a.wait(["b"], 64, max_wait_s=0.05, is_done=lambda m: False)


def test_clients_agree_on_each_other(tmp_path):
    """Every client draws every other client where that client placed itself (its own extrapolation
    of block s)."""
    clients = {p: client(p, tmp_path) for p in (0, 1, 2)}
    assert clients[0].table.track_of(MEDIA[3]) is None  # never alive: keeps the recording
    latents = torch.zeros(N_LATENTS, 48, 24, 42)
    for s in range(1, N_LATENTS - 3, 4):
        for c in clients.values():
            c.publish_own(s, 8 * s, latents)
        for c in clients.values():
            c.read_peers(s, 8 * s)
        for p, own in clients.items():
            mine = torch.from_numpy(own.own_cameras.block_rows(s)[:, :3]).float()
            for q, other in clients.items():
                assert torch.equal(other.table.states[0, p, 4 * np.arange(s, s + 4), :3], mine), (
                    s,
                    p,
                    q,
                )
    assert clients[1].reader.calls[1] == (N_LATENTS, [1, 2, 3, 4])


def test_denoise_retested():
    scheduler = FlowMatchScheduler(shift=5.0)
    ladder = warped_ladder((1000, 750, 500, 250), scheduler)

    class FakeSampler:
        def __init__(self):
            self.ladder, self.scheduler, self.rng, self.seen = ladder, scheduler, None, []

        def call(self, x, t, conditions, *, start, num_frames):
            self.seen.append(conditions["k"])
            return x, x * conditions["k"]

    noisy = torch.randn(1, 4, 2, 3, 3, generator=torch.Generator().manual_seed(0))
    sampler = FakeSampler()
    torch.manual_seed(1)
    got = Client._denoise_retested(
        sampler, noisy, {"k": 0.5}, start=13, retest=lambda x0: {"k": 0.25}
    )
    assert sampler.seen == [0.5, 0.25, 0.25, 0.25]
    torch.manual_seed(1)
    calls = iter([0.5, 0.25, 0.25, 0.25])
    want = ladder_denoise(lambda x, t: (x, x * next(calls)), noisy, ladder, scheduler)
    assert torch.equal(got, want)


def test_config_switch():
    assert InferenceConfig().player_state.source == "recorded"
    cfg = load_config(REPO / "configs" / "infer" / "worldcast_4step.yaml")
    assert cfg.player_state.source == "recorded" and cfg.player_state.pose_radius_u == 40.0
    assert (
        config_from_dict({"player_state": {"source": "predicted"}}).player_state.source
        == "predicted"
    )
    with pytest.raises(ValueError):
        config_from_dict({"player_state": {"source": "oracle"}})
