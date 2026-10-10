"""The training losses: the timesteps and the flow-matching sample, the training weight, the
foreground weight (its properties, its normalisation and the paper's values), the memory weight, the
angles of the foreground weight, the frame loss mask and the distribution matching losses."""

import pytest
import torch

from tests.engine.training.support import make_window_batch
from worldcast.engine.training import losses as L
from worldcast.player_state.states import integrate_camera_angles, pack_substeps
from worldcast.sampling.schedulers import FlowMatchScheduler

TOKEN_GRID = (12, 21)
LATENT_GRID = (24, 42)


# --------------------------------------------------------------------------------------- timesteps
def test_one_timestep_per_run_of_frames():
    torch.manual_seed(0)
    whole = L.sample_timestep_index(4, 41, frames_per_timestep=41, first_frame_alone=False, high=9)
    assert whole.shape == (4, 41) and whole.dtype == torch.long
    assert [int(row.unique().numel()) for row in whole] == [1, 1, 1, 1]
    blocks = L.sample_timestep_index(4, 41, frames_per_timestep=4, first_frame_alone=True, high=9)
    for row in blocks:
        runs = row[1:].reshape(10, 4)
        assert bool((runs == runs[:, :1]).all()) and int(row.unique().numel()) > 1
    band = L.sample_timestep_index(
        64, 9, frames_per_timestep=4, first_frame_alone=True, low=994, high=997
    )
    assert sorted(band.unique().tolist()) == [994, 995, 996]


def test_the_draw_of_a_block_timestep_has_the_windows_shape():
    """One ``randint`` of shape ``[B, F]``: the generator advances as for ``B F`` values."""
    torch.manual_seed(0)
    L.sample_timestep_index(2, 9, frames_per_timestep=4, first_frame_alone=True, high=1000)
    after = torch.rand(())
    torch.manual_seed(0)
    torch.randint(0, 1000, [2, 9])
    assert torch.equal(after, torch.rand(()))


def test_the_window_timestep_is_one_draw_per_sample():
    torch.manual_seed(0)
    index = L.sample_window_timestep_index(3, 21, high=1000)
    torch.manual_seed(0)
    assert torch.equal(index, torch.randint(0, 1000, [3, 1]).repeat(1, 21))


def test_the_context_band_is_three_table_steps():
    assert L.context_band_index_range(FlowMatchScheduler()) == (994, 997)


def test_the_training_weight_averages_to_one_and_vanishes_at_pure_noise():
    scheduler = FlowMatchScheduler()
    table = L.training_weight_table(scheduler)
    assert table.shape == (1000,) and float(table.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(table[0]) == 0.0 and float(scheduler.timesteps[0]) == 1000.0
    assert 450 < float(scheduler.timesteps[table.argmax()]) < 550
    # the weight of a timestep is that of the nearest table entry
    near = scheduler.timesteps[[3, 500]] + 0.01
    assert torch.equal(L.training_weight(near, scheduler), table[[3, 500]])
    assert L.training_weight(near.repeat(2, 1), scheduler).shape == (4,)


# ------------------------------------------------------------------------- the flow-matching draw
def test_pinned_frames_are_clean_at_timestep_zero():
    g = torch.Generator().manual_seed(1)
    noisy, clean = torch.randn(2, 9, 4, 6, 10, generator=g), torch.randn(
        2, 2, 4, 6, 10, generator=g
    )
    timestep, target = torch.full((2, 9), 500.0), torch.ones(2, 9, 4, 6, 10)
    pinned, pinned_t, pinned_target = L.pin_clean_frames(noisy, timestep, clean, target)
    assert torch.equal(pinned[:, :2], clean) and torch.equal(pinned[:, 2:], noisy[:, 2:])
    assert pinned_t.tolist() == [[0.0, 0.0] + [500.0] * 7] * 2
    assert float(pinned_target[:, :2].abs().sum()) == 0.0 and bool(
        (pinned_target[:, 2:] == 1).all()
    )
    assert float(timestep.min()) == 500.0 and bool((target == 1).all())  # the inputs are kept
    assert L.pin_clean_frames(noisy, timestep, clean)[2] is None


def test_a_flow_matching_sample():
    scheduler = FlowMatchScheduler()
    clean = torch.randn(2, 9, 4, 6, 8)
    torch.manual_seed(0)
    sample = L.sample_flow_matching(
        clean,
        scheduler,
        n_pin=1,
        frames_per_timestep=9,
        first_frame_alone=False,
        timestep_dtype=torch.float32,
    )
    assert sample.n_pin == 1 and sample.context_timestep is None
    assert torch.equal(sample.context, clean) and sample.context is not clean
    assert torch.equal(sample.noisy[:, :1], clean[:, :1]) and bool(
        (sample.timestep[:, 0] == 0).all()
    )
    assert float(sample.target[:, :1].abs().sum()) == 0.0
    assert torch.equal(sample.target[:, 1:], (sample.noise - clean)[:, 1:])
    # x_t = (1 - sigma) x0 + sigma eps at the window's one timestep
    sigma = (sample.timestep[:, 1] / 1000.0).view(2, 1, 1, 1, 1)
    expected = (1 - sigma) * clean[:, 1:] + sigma * sample.noise[:, 1:]
    assert torch.allclose(sample.noisy[:, 1:], expected, atol=1e-5)
    # the stage-1 draw: one [B, 1] index per window after the noise
    torch.manual_seed(0)
    stage1 = L.sample_flow_matching(
        clean,
        scheduler,
        n_pin=1,
        frames_per_timestep=9,
        first_frame_alone=False,
        window_timestep=True,
        timestep_dtype=torch.float32,
    )
    torch.manual_seed(0)
    torch.randn(clean.shape)
    index = torch.randint(0, 1000, [2, 1])
    assert torch.equal(stage1.timestep[:, 1], scheduler.timesteps[index[:, 0]])


def test_the_context_noise_is_drawn_apart_from_the_target_noise():
    scheduler = FlowMatchScheduler()
    clean = torch.randn(2, 21, 4, 6, 8).bfloat16()
    torch.manual_seed(0)
    s = L.sample_flow_matching(
        clean,
        scheduler,
        n_pin=1,
        frames_per_timestep=4,
        first_frame_alone=True,
        context_noise=True,
    )
    assert torch.equal(s.target[:, 1:], (s.noise - clean)[:, 1:])
    with_target_noise = scheduler.add_noise(
        clean.flatten(0, 1), s.noise.flatten(0, 1), s.context_timestep.flatten(0, 1)
    ).unflatten(0, (2, 21))
    assert not torch.equal(s.context, with_target_noise)
    assert not torch.equal(s.context[:, :1], clean[:, :1])  # the first frame is noised too
    assert bool(((s.context_timestep >= 16) & (s.context_timestep < 32)).all())


# ------------------------------------------------------------------------------ foreground weight
def _one_player(u=10.0, v=6.0, radius=1.2, visible=True):
    return torch.tensor([[[[u, v]]]]), torch.tensor([[[radius]]]), torch.tensor([[[visible]]])


def test_an_invisible_player_leaves_the_weight_at_one():
    w = L.foreground_weight(*_one_player(visible=False), grid=TOKEN_GRID)
    assert torch.equal(w, torch.ones_like(w))


def test_a_visible_player_is_weighted_at_its_own_tokens():
    w = L.foreground_weight(*_one_player(u=10.0, v=6.0, radius=1.2), grid=TOKEN_GRID)[0, 0, 0]
    peak = 1.0 + 2.0 * 1.5 / 1.2  # lambda = 3, beta = 1.5 / r
    assert float(w[int(6.0 - 1.2), 10]) == float(w.max()) and 2.5 < float(w.max()) <= peak
    assert w[0, 0] == pytest.approx(1.0, abs=1e-4)


def test_two_players_together_are_not_worth_double():
    uv = torch.tensor([[[[10.0, 6.0], [10.2, 6.0]]]])
    w = L.foreground_weight(
        uv, torch.tensor([[[1.2, 1.2]]]), torch.ones(1, 1, 2, dtype=bool), grid=TOKEN_GRID
    )
    assert float(w.max()) <= 1.0 + 2.0 * 1.5 / 1.2 + 1e-5


def test_foreground_weight_paper_values():
    """lambda = 3 and beta_p = min(3, max(1, 1.5 / r_p)): 3 at a near player's centre, 7 at a
    distant one's, and never more."""
    uv = torch.tensor([[[[10.5, 6.5 + 2.0], [30.5, 16.5 + 0.5]]]])  # centres on cell centres
    radius = torch.tensor([[[2.0, 0.5]]])
    visible = torch.ones(1, 1, 2, dtype=torch.bool)
    w = L.foreground_weight(uv, radius, visible, grid=LATENT_GRID)[0, 0, 0]
    assert float(w[6, 10]) == 3.0 and float(w[16, 30]) == 7.0
    assert float(w.max()) == 7.0 and float(w.min()) >= 1.0
    for r in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0, 12.0):
        w = L.foreground_weight(*_one_player(u=20.5, v=12.5 + r, radius=r), grid=LATENT_GRID)
        assert float(w.max()) == pytest.approx(1.0 + 2.0 * min(3.0, max(1.0, 1.5 / r)), rel=1e-6)


def _player_map(batch, frames, height, width, boost):
    base = torch.ones(batch, frames, 1, height, width)
    hot = torch.zeros_like(base)
    hot[:, :, :, height // 2, width // 2] = 1.0
    return base + (boost - 1.0) * hot


def _loss(pred, target, timestep, alpha, scheduler):
    weight = L.compose_weight(timestep, scheduler, n_pin=0, alpha_k=alpha)
    return L.flow_matching_loss(pred, target, weight)


def test_a_uniform_foreground_weight_leaves_the_loss_unchanged():
    scheduler = FlowMatchScheduler()
    torch.manual_seed(2)
    pred, target = torch.randn(2, 3, 4, 6, 8), torch.randn(2, 3, 4, 6, 8)
    for index in ((0, 10, 40), (300, 500, 700), (999, 998, 990)):
        t = scheduler.timesteps[torch.tensor([index, index])].bfloat16()
        uniform = _loss(pred, target, t, _player_map(2, 3, 6, 8, 1.0), scheduler)
        plain = L.flow_matching_loss(pred, target, L.compose_weight(t, scheduler, n_pin=0))
        assert torch.equal(uniform, plain)


def test_the_normalisation_keeps_the_loss_scale_and_the_emphasis():
    scheduler = FlowMatchScheduler()
    pred = torch.full((2, 3, 4, 6, 8), 3.0)
    target = torch.zeros_like(pred)
    t = scheduler.timesteps[torch.tensor([[100, 400, 600], [250, 500, 900]])].bfloat16()
    at_1 = _loss(pred, target, t, _player_map(2, 3, 6, 8, 1.0), scheduler)
    at_8 = _loss(pred, target, t, _player_map(2, 3, 6, 8, 8.0), scheduler)
    assert abs(float(at_8 / at_1) - 1.0) < 1e-4
    t = scheduler.timesteps[torch.tensor([[500]])].bfloat16()
    w = L.compose_weight(t, scheduler, n_pin=0, alpha_k=_player_map(1, 1, 6, 8, 8.0))
    assert abs(float(w[0, 0, 0, 3, 4] / w[0, 0, 0, 0, 0]) - 8.0) < 1e-4


def test_the_weight_of_pinned_frames_is_zero_and_the_memory_weight_multiplies():
    scheduler = FlowMatchScheduler()
    t = scheduler.timesteps[torch.tensor([[500, 500, 500]])]
    c_k = torch.full((1, 3, 1, 2, 2), 2.0)
    weight = L.compose_weight(t, scheduler, n_pin=1, c_k=c_k)
    table = L.training_weight_table(scheduler)
    assert weight.shape == (1, 3, 1, 2, 2) and float(weight[:, 0].abs().sum()) == 0.0
    assert torch.equal(weight[:, 1:], (2.0 * table[500]).expand(1, 2, 1, 2, 2))


def test_the_pitch_of_the_foreground_weight_is_clamped_as_a_sum():
    """A turn past the pitch limit is banked: the sum comes back down only once it re-enters the
    range, where the player state field's row-by-row clamp turns back at once."""
    batch = make_window_batch(0, batch=1, players=2, latents=3, pitch_saturation=False)
    batch["player_states"][..., 4] = 0.0
    substeps, valid = batch["player_control_substeps"], batch["player_control_substep_valid"]
    substeps[..., 11:] = 0.0
    valid[:] = True
    # player 0 turns up 40 degrees a row for three rows (to 120), then down 40 a row for two
    substeps[0, 0, 1:4, :, 11] = 2.0  # four substeps of 2 x 5 degrees
    substeps[0, 0, 4:6, :, 11] = -2.0
    _, pitch = L.foreground_angles(batch)
    assert pitch.shape == (1, 3, 2)  # at the latent frames: video frames 0, 4 and 8
    assert pitch[0, :, 0].tolist() == [0.0, 80.0, 40.0]
    assert pitch[0, :, 1].tolist() == [0.0, 0.0, 0.0]
    packed = pack_substeps(substeps, valid)
    _, by_row = integrate_camera_angles(batch["player_states"][:, :, 0], packed)
    assert by_row[0, 0, [0, 4, 8]].tolist() == [0.0, 49.0, 9.0]


def test_a_gathered_window_carries_its_angles():
    batch = make_window_batch(3, latents=5, continuous=True)
    yaw, pitch = L.foreground_angles(batch)
    columns = batch["player_continuous_columns"]
    assert torch.equal(yaw, columns[..., 0]) and torch.equal(pitch, columns[..., 1])


def test_memory_weight_is_the_papers_c_k():
    """c_k = 1/2 (1 + m_k / mean_f(m)): unit mean per target frame, 1 on a frame without marked
    positions and outside the target frames."""
    g = torch.Generator().manual_seed(7)
    m = torch.rand(3, 4, 12, 21, generator=g) < 0.2
    m[0, 0] = False  # an empty frame
    m[1, 2] = True  # a fully marked frame
    m[2, 3] = False
    m[2, 3, 5, 7] = True  # a single marked token
    weight = L.memory_weight(m, 41, grid=LATENT_GRID)
    assert bool((weight[:, :37] == 1.0).all())
    assert bool((weight[0, 37] == 1.0).all()) and bool((weight[1, 39] == 1.0).all())
    # one marked token of 252: c_k = (1 + 252) / 2 there, 1/2 elsewhere, on its 2 x 2 latent cells
    single = weight[2, 40, 0]
    assert torch.allclose(single[10:12, 14:16], torch.full((2, 2), 126.5))
    assert float(single.sum()) == pytest.approx(24 * 42, rel=1e-5) and float(single.min()) == 0.5
    assert all(
        abs(float(weight[b, 37 + f].mean()) - 1.0) < 1e-5 for b in range(3) for f in range(4)
    )


def test_frame_loss_mask_is_the_per_supervised_element_mean():
    weight = torch.ones(2, 21, 1, 1, 1)
    mask = torch.zeros(2, 21)
    mask[:, 17:] = 1.0
    w = L.apply_frame_loss_mask(weight, mask)
    assert float(w[:, :17].abs().sum()) == 0.0
    assert torch.allclose(w[:, 17:], torch.full((2, 4, 1, 1, 1), 21.0 / 4.0))
    assert abs(float(w.mean()) - 1.0) < 1e-6


def test_flow_matching_loss_values():
    x = torch.randn(2, 3, 4, 8, 8)
    assert L.flow_matching_loss(x, x).item() == 0.0
    assert L.flow_matching_loss(torch.ones(2, 3), torch.zeros(2, 3)).item() == pytest.approx(1.0)
    weight = torch.tensor([[2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    loss = L.flow_matching_loss(torch.ones(2, 3), torch.zeros(2, 3), weight)
    assert loss.item() == pytest.approx(6.0 / 6)


# ----------------------------------------------------------------------- distribution matching
def test_the_score_timestep_is_one_per_sample_within_the_clamp():
    torch.manual_seed(0)
    t = L.sample_score_timestep(4096, 3)
    assert t.shape == (4096, 3) and t.dtype == torch.float32 and bool((t == t[:, :1]).all())
    assert float(t.min()) == 20.0 and float(t.max()) == 980.0  # both ends of the clamp are hit
    # the draw is an integer s in [0, 1000), warped by the schedule's shift: 5 s / (1 + 4 s)
    torch.manual_seed(0)
    s = torch.randint(0, 1000, [4096, 1]) / 1000
    assert torch.allclose(t[:, :1], (1000 * 5 * s / (1 + 4 * s)).clamp(20, 980))


def test_the_gradient_is_normalised_per_sample_and_has_no_nan():
    critic, teacher = torch.full((2, 3, 4, 8, 8), 2.0), torch.ones(2, 3, 4, 8, 8)
    rollout = torch.zeros(2, 3, 4, 8, 8)
    rollout[1] = -1.0  # twice as far from the teacher: half the gradient
    gradient = L.distribution_matching_gradient(critic, teacher, rollout)
    assert bool((gradient[0] == 1.0).all()) and bool((gradient[1] == 0.5).all())
    ones = torch.ones(1, 1, 1, 1, 1)
    assert not torch.isnan(L.distribution_matching_gradient(ones, ones, ones)).any()


def test_the_distribution_matching_loss_hands_the_gradient_to_the_rollout():
    rollout = torch.ones(2, 3, 4, 8, 8, requires_grad=True)
    gradient = torch.full_like(rollout, 0.5)
    loss = L.distribution_matching_loss(rollout, gradient)
    assert loss.dtype == torch.float64 and loss.item() == 0.125  # 1/2 mean(g^2)
    loss.backward()
    assert torch.allclose(rollout.grad, gradient / rollout.numel())
    mask = L.distribution_matching_mask(rollout)
    assert mask.dtype == torch.bool and not bool(mask[:, 0].any()) and bool(mask[:, 1:].all())
    gradient[:, 0] = 100.0  # the first frame is masked out
    assert L.distribution_matching_loss(rollout, gradient, mask).item() == 0.125


def test_the_critic_loss_is_zero_for_the_exact_x0_and_skips_the_first_frame():
    scheduler = FlowMatchScheduler()
    g = torch.Generator().manual_seed(5)
    x, eps = torch.randn(2, 5, 4, 6, 6, generator=g), torch.randn(2, 5, 4, 6, 6, generator=g)
    timestep = scheduler.timesteps[torch.tensor([[300] * 5, [700] * 5])]
    noisy = scheduler.add_noise(x.flatten(0, 1), eps.flatten(0, 1), timestep.flatten(0, 1))
    noisy = noisy.unflatten(0, (2, 5))
    assert L.critic_loss(x, x, eps, noisy, timestep, scheduler).item() < 1e-8
    wrong = x.clone()
    wrong[:, 0] += 3.0  # the first frame is not scored
    assert L.critic_loss(x, wrong, eps, noisy, timestep, scheduler).item() < 1e-8
    wrong[:, 1:] += 0.5  # the velocity is off by 0.5 / sigma_t
    sigma = (timestep[:, 1] / 1000.0).double()
    expected = float(((0.5 / sigma) ** 2).mean())
    loss = L.critic_loss(x, wrong, eps, noisy, timestep, scheduler).item()
    assert loss == pytest.approx(expected, rel=1e-3)
