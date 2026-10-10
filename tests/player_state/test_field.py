"""The player state field: splatting, composition and the field of a window."""

import math

import pytest
import torch

from tests.player_state.support import round_batch
from worldcast.modeling.state_injector import (
    FIELD_CHANNEL_NAMES,
    FIELD_CHANNELS,
    log_compressed_depth,
)
from worldcast.player_state import (
    PlayerState,
    PlayerStateFieldConfig,
    compose,
    field_builder,
    player_state_conditions,
    player_state_field,
    splat,
    visible_latent_frames,
)
from worldcast.player_state.field import SPLAT_TEMPERATURE

GRID = dict(grid_h=12, grid_w=21)
CHANNEL = {name: index for index, name in enumerate(FIELD_CHANNEL_NAMES)}


def test_the_channels_are_the_papers_23_in_order():
    assert len(FIELD_CHANNEL_NAMES) == FIELD_CHANNELS == 23
    assert FIELD_CHANNEL_NAMES[:6] == (
        "coverage",
        "depth",
        "yaw_sin",
        "yaw_cos",
        "team",
        "headcount",
    )
    assert (CHANNEL["controls_forward"], CHANNEL["controls_reload"]) == (6, 14)
    assert (CHANNEL["weapon_0"], CHANNEL["weapon_3"]) == (15, 18)
    assert FIELD_CHANNEL_NAMES[19:] == ("dying", "corpse", "identity_live", "identity_corpse")


def test_splat_is_centred_on_the_body_with_half_a_token_sigma():
    assert SPLAT_TEMPERATURE == 72.0
    uv = torch.tensor([[[[10.5, 7.5]]]])  # feet one token below the centre of token (6, 10)
    radius = torch.ones(1, 1, 1)
    weight = splat(uv, radius, torch.ones(1, 1, 1, dtype=torch.bool), **GRID)
    assert weight.shape == (1, 1, 1, 12, 21)
    assert weight[0, 0, 0, 6, 10].item() == 1.0
    assert weight[0, 0, 0, 6, 11].item() == pytest.approx(math.exp(-2.0))  # one token: 2 sigma
    assert weight[0, 0, 0, 5, 10].item() == pytest.approx(math.exp(-2.0))
    half = splat(uv, radius, torch.ones(1, 1, 1), confidence=torch.full((1, 1, 1), 0.5), **GRID)
    assert torch.equal(half, weight * 0.5)
    assert not bool(splat(uv, radius, torch.zeros(1, 1, 1, dtype=torch.bool), **GRID).any())


def _compose(weights, *, depth, enemy):
    players = weights.shape[2]
    zeros = torch.zeros(1, 1, 12, 21)
    return compose(
        weights,
        depth,
        torch.zeros(1, 1, players),
        enemy,
        torch.zeros(1, 1, players, 9),
        torch.zeros(1, 1, players, 4),
        dying_plane=zeros,
        corpse_plane=zeros,
        corpse_ids=zeros,
    )


def test_compose_merges_the_two_nearest_players_of_a_token():
    weights = torch.zeros(1, 1, 3, 12, 21)
    weights[0, 0, :, 6, 10] = torch.tensor([0.5, 0.4, 0.3])
    depth = torch.tensor([[[100.0, 100.0, 100.0]]])
    enemy = torch.tensor([[[True, False, True]]])
    field = _compose(weights, depth=depth, enemy=enemy)
    assert field.shape == (1, 1, 23, 12, 21)
    at = field[0, 0, :, 6, 10]
    assert at[CHANNEL["coverage"]].item() == pytest.approx(0.9)  # the two largest weights
    # log1p(100) / log1p(4096) = 0.5548
    assert log_compressed_depth(depth)[0, 0, 0].item() == pytest.approx(0.5548, abs=1e-4)
    assert at[CHANNEL["depth"]].item() == pytest.approx(0.9 * 0.5548, abs=1e-4)
    assert at[CHANNEL["yaw_sin"]].item() == 0.0
    assert at[CHANNEL["yaw_cos"]].item() == pytest.approx(0.9)
    assert at[CHANNEL["team"]].item() == pytest.approx(0.5 - 0.4)
    assert at[CHANNEL["headcount"]].item() == pytest.approx((0.5 + 0.4 + 0.3) / 4.0)
    # the identity is the top-ranked player's alone: enemy slot 0 reads 0.50
    assert at[CHANNEL["identity_live"]].item() == pytest.approx(0.50 * 0.9)
    assert not bool(field[0, 0, :, 0, 0][[0, 1, 2, 3, 4]].any())


def test_compose_ranks_by_depth_discounted_weight():
    weights = torch.zeros(1, 1, 2, 12, 21)
    weights[0, 0, :, 6, 10] = torch.tensor([0.5, 0.45])
    enemy = torch.tensor([[[True, False]]])
    near_second = _compose(weights, depth=torch.tensor([[[4000.0, 10.0]]]), enemy=enemy)
    # slot 1 is much nearer: it outranks the larger weight of slot 0 and names the token
    assert near_second[0, 0, CHANNEL["identity_live"], 6, 10].item() == pytest.approx(0.22 * 0.95)


@pytest.fixture
def weapon_embedding():
    return torch.randn(52, 4, generator=torch.Generator().manual_seed(0))


def _two_players(player_xy, *, client_yaw=0.0, columns=6, frames=1) -> dict[str, torch.Tensor]:
    """The conditions of a client (slot 0) at the origin and one enemy (slot 1) at ``player_xy``
    that looks along +y, holds ``forward`` and, for half of each latent frame, ``reload``."""
    states = torch.zeros(1, frames, 2, columns)
    states[0, :, 1, :2] = torch.tensor(player_xy)
    states[0, :, 1, 3], states[..., 5] = 90.0, 1.0
    states[0, :, 0, 3] = client_yaw
    if columns == 13:  # a living player's continuous columns: its yaw, and its own position
        states[..., 6], states[..., 10:13] = states[..., 3], states[..., :3]
    controls = torch.zeros(1, frames, 2, 16, 14)
    controls[..., 13] = 1.0
    controls[0, :, 1, :, 0], controls[0, :, 1, :8, 9] = 1.0, 1.0
    return dict(
        player_state_table=states,
        player_controls=controls,
        client_slot=torch.tensor([0]),
        player_team_ids=torch.tensor([[2, 3]]),
        player_alive=torch.ones(1, frames, 2),
        player_visible=torch.ones(1, frames, 2),
        player_weapons=torch.tensor([[[0, 7]]]).expand(1, frames, 2),
    )


def test_one_visible_player_writes_its_attributes_where_its_body_is(weapon_embedding):
    field = player_state_field(**_two_players([500.0, 0.0]), weapon_embedding=weapon_embedding)
    assert field.shape == (1, 1, 23, 12, 21) and field.dtype == torch.float32
    # the feet are 64 u below the eye: row 7.02; the body's centre, 0.58 rows higher: token (6, 10)
    assert field[0, 0, CHANNEL["coverage"]].argmax().item() == 6 * 21 + 10
    at = field[0, 0, :, 6, 10]
    w = at[CHANNEL["coverage"]].item()
    assert w == pytest.approx(0.9946, abs=1e-3)  # exp(-72 d^2), 0.05 rows from the token centre
    expected = {
        "depth": w * math.log1p(500.0) / math.log1p(4096.0),
        "yaw_sin": w * 1.0,  # it looks 90 degrees to the client's left
        "yaw_cos": 0.0,
        "team": w * 1.0,  # an enemy
        "headcount": w / 4.0,
        "controls_forward": w * 1.0,
        "controls_reload": w * 0.5,
        "identity_live": w * 0.52,  # the enemy band, slot 1
        **{f"weapon_{k}": w * weapon_embedding[7, k].item() for k in range(4)},
    }
    for name in FIELD_CHANNEL_NAMES[1:]:
        assert at[CHANNEL[name]].item() == pytest.approx(expected.get(name, 0.0), abs=1e-5), name
    assert not bool(field[0, 0, :5, 0, 0].any())  # nobody in the corner


def test_a_13_column_table_carries_the_clients_angles_and_a_6_column_one_those_of_frame_0(
    weapon_embedding,
):
    def coverage(conditions):
        field = player_state_field(**conditions, weapon_embedding=weapon_embedding)
        return field[0, :, CHANNEL["coverage"], 6, 10].tolist()

    # the player stands along +y; the client's table yaw (column 3) says +x in every frame
    gathered = _two_players([0.0, 500.0], columns=13, frames=3)
    gathered["player_state_table"][0, :, 0, 6] = torch.tensor([0.0, 90.0, 90.0])  # continuous yaw
    seen = coverage(gathered)
    assert seen[0] == 0.0 and seen[1] > 0.99 and seen[2] > 0.99  # it follows columns 6, 7
    # six columns: frame 0's yaw plus the turns of the controls; columns 3, 4 of later frames are
    # not read
    contiguous = _two_players([0.0, 500.0], client_yaw=90.0, frames=3)
    contiguous["player_state_table"][0, 1:, 0, 3] = torch.tensor([0.0, -90.0])
    assert coverage(contiguous) == coverage(_two_players([0.0, 500.0], client_yaw=90.0, frames=3))
    assert min(coverage(contiguous)) > 0.3  # in view in every frame, at the confidence it has
    contiguous["player_controls"][0, 2, 0, 0, 12] = -18.0  # a turn of -90 degrees in frame 2
    assert coverage(contiguous)[2] == 0.0


def _conditions(batch):
    states = PlayerState.from_batch(batch)
    return player_state_conditions(states, visible_latent_frames(batch))


def test_a_blocks_field_is_its_slice_of_the_windows(weapon_embedding):
    c = _conditions(round_batch(0))
    whole = player_state_field(**c, weapon_embedding=weapon_embedding)
    assert whole.shape == (2, 9, 23, 12, 21) and whole.dtype == torch.float32
    assert bool(whole[:, :, CHANNEL["coverage"]].any()) and bool(whole[:, :, 20].any())
    for offset, frames in ((0, 1), (1, 4), (5, 4)):
        # the sampler hands the block's rows of the two framewise conditions
        block = dict(
            c,
            player_alive=c["player_alive"][:, offset : offset + frames],
            player_visible=c["player_visible"][:, offset : offset + frames],
        )
        got = player_state_field(
            **block, weapon_embedding=weapon_embedding, frame_offset=offset, num_frames=frames
        )
        assert torch.equal(got, whole[:, offset : offset + frames])
    built = field_builder()(c, weapon_embedding, 1, 4)
    assert torch.equal(built, whole[:, 1:5])


def test_the_visibility_gate_and_the_corpses(weapon_embedding):
    c = _conditions(round_batch(0))
    unseen = dict(c, player_visible=torch.zeros_like(c["player_visible"]))
    hidden = player_state_field(**unseen, weapon_embedding=weapon_embedding)
    live = [CHANNEL[name] for name in ("coverage", "depth", "team", "headcount", "identity_live")]
    assert not bool(hidden[:, :, live].any())
    assert bool(hidden[:, :, CHANNEL["corpse"]].any())  # corpses skip the gate
    # "Trained without visibility": every living player in front of the camera is written, at
    # confidence 1
    config = PlayerStateFieldConfig(confidence_floor=1.0)
    ungated = field_builder(config, visibility_gate=False)(unseen, weapon_embedding, 0, 9)
    all_visible = dict(c, player_visible=torch.ones_like(c["player_visible"]))
    assert torch.equal(
        ungated, player_state_field(**all_visible, weapon_embedding=weapon_embedding, config=config)
    )
    gated = player_state_field(**c, weapon_embedding=weapon_embedding)
    assert ungated[:, :, CHANNEL["coverage"]].sum() > gated[:, :, CHANNEL["coverage"]].sum()


def test_the_confidence_of_a_newly_visible_player_rises_over_its_block(weapon_embedding):
    conditions = _two_players([500.0, 0.0], frames=5)
    conditions["player_visible"][0, :2, 1] = 0.0  # unseen in the first frame and in frame 1
    field = player_state_field(**conditions, weapon_embedding=weapon_embedding)
    coverage = field[0, :, CHANNEL["coverage"], 6, 10]
    # block 1 = frames 1 .. 4: e_p = 0.3 + 0.7 ema with alpha 0.5, restarted at frame 1
    assert (coverage / coverage[4]).tolist() == pytest.approx(
        [0.0, 0.0, 0.65 / 0.9125, 0.825 / 0.9125, 1.0], abs=1e-5
    )


def test_the_field_refuses_inputs_that_do_not_fit(weapon_embedding):
    c = _conditions(round_batch(0))

    def field(**changes):
        return player_state_field(**{**c, "weapon_embedding": weapon_embedding, **changes})

    with pytest.raises(ValueError, match="6 columns wide"):
        field(player_state_table=c["player_state_table"][..., :5])
    with pytest.raises(ValueError, match="player_controls"):
        field(player_controls=c["player_controls"][:, :8])
    with pytest.raises(ValueError, match="weapon embedding"):
        field(weapon_embedding=weapon_embedding[:, :3])
    with pytest.raises(ValueError, match=r"latent frames \[7, 11\) lie outside the 9-frame window"):
        field(frame_offset=7, num_frames=4)
    with pytest.raises(ValueError, match="disagree"):
        field(player_alive=1.0 - c["player_alive"])
