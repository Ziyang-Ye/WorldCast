"""The context of a live client whose player is idle (``IdleContext``): the clip, what ``freeze``
keeps in the store and the memory bank's latents, and where a stretch ends, on a store of a few
numbers; the block loop's own tests are in ``test_live_rollout.py``."""

import numpy as np
import torch

from worldcast.data.latents import BLOCK, FIRST_TARGET
from worldcast.engine.inference.controls import LiveBlockControls
from worldcast.engine.inference.idle_context import (
    CLIP_BLOCKS,
    SAME_VIEW_DEGREES,
    SAME_VIEW_U,
    IdleContext,
    is_idle,
    same_view,
)

MEDIA = "m"
#: The round of a live client (29 latent frames): a window block is always the last one.
LATENTS = FIRST_TARGET + BLOCK


def camera(x: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """``[4, 4, 4]`` the cameras of a block: ``x`` u along x, turned ``yaw`` degrees about z."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c2w = np.eye(4)
    c2w[:2, :2] = [[c, -s], [s, c]]
    c2w[0, 3] = x
    return np.broadcast_to(c2w, (BLOCK, 4, 4)).copy()


def controls(*, buttons: bool = False, turn: float = 0.0) -> LiveBlockControls:
    held = np.zeros((16, 11), np.float32)
    held[:, 0] = float(buttons)
    deltas = np.zeros((16, 2), np.float32)
    deltas[:, 1] = turn
    return LiveBlockControls(
        buttons=held,
        view_deltas=deltas,
        weapon=np.zeros(16, np.int64),
        turn=np.zeros((16, 2), np.float32),
        times=np.arange(16, dtype=np.float64) / 16,
    )


class Play:
    """A live client's blocks as the rollout hands them to :meth:`IdleContext.keep`: block ``j``
    (every latent frame of it ``j + 1``) is written to the last block of the store, which then
    has the two blocks before it (the round slides after every block), and is kept or replaced
    before the memory bank's latents of the block are taken from the store."""

    def __init__(self, context: IdleContext) -> None:
        self.context = context
        self.store = torch.zeros(LATENTS, 2, 3, 4)
        self.known: dict[tuple[str, int], torch.Tensor] = {}
        self.j = 0

    def block(
        self,
        *,
        idle: bool = True,
        view: np.ndarray | None = None,
        weapons: tuple[int, ...] = (7,),
        f0: int | None = None,
    ):
        """One block; the store's newest block after it, ``[4, 2, 3, 4]``."""
        f0 = LATENTS - BLOCK if f0 is None else f0
        self.store[FIRST_TARGET - 2 * BLOCK : FIRST_TARGET - BLOCK] = self.store[
            FIRST_TARGET - BLOCK : FIRST_TARGET
        ]
        self.store[FIRST_TARGET - BLOCK : FIRST_TARGET] = self.store[FIRST_TARGET:LATENTS]
        self.store[f0 : f0 + BLOCK] = self.j + 1.0
        self.context.keep(self.store, f0, camera() if view is None else view, weapons, idle=idle)
        self.known[(MEDIA, f0 + BLOCK * self.j)] = self.store[f0 : f0 + BLOCK].clone()
        self.j += 1
        return self.store[f0 : f0 + BLOCK].clone()

    def recent(self) -> list[float]:
        """The three newest blocks of the store, which are the next block's recent context once the
        round slides, each by its first value."""
        return [
            float(self.store[f0, 0, 0, 0]) for f0 in range(FIRST_TARGET - 2 * BLOCK, LATENTS, BLOCK)
        ]


def test_a_block_is_idle_where_no_button_is_held_and_the_view_controls_are_zero():
    assert is_idle(controls())
    assert not is_idle(controls(buttons=True))
    assert not is_idle(controls(turn=0.5))


def test_two_views_are_one_within_half_a_u_and_half_a_degree():
    assert (SAME_VIEW_U, SAME_VIEW_DEGREES) == (0.5, 0.5)
    assert same_view(camera(), camera())
    assert same_view(camera(), camera(x=0.4, yaw=0.4))
    assert not same_view(camera(), camera(x=0.6))
    assert not same_view(camera(), camera(yaw=0.6))
    assert not same_view(camera(), camera(yaw=-0.6))
    assert not same_view(camera(), camera()[:1])  # not the same number of cameras
    assert same_view(
        camera(yaw=170.0), camera(yaw=170.2)
    )  # a rotation is compared, not its entries


def test_freeze_keeps_the_clip_and_gives_the_store_the_clip_in_turn_from_the_fourth_block_on():
    play = Play(IdleContext())
    kept = [float(play.block()[0, 0, 0, 0]) for _ in range(9)]
    # blocks 1-3 are the clip as generated, then the clip over and over, however many blocks go by
    assert kept == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    assert sorted(play.recent()) == [1.0, 2.0, 3.0]  # the next block's context is the clip
    assert (play.context.stretch, play.context.kept, len(play.context.clip)) == (9, 6, CLIP_BLOCKS)
    # the memory bank's latents are what the store kept
    session = LATENTS - BLOCK + BLOCK * 8
    assert torch.equal(play.known[(MEDIA, session)], play.store[LATENTS - BLOCK :])
    assert float(play.known[(MEDIA, session)][0, 0, 0, 0]) == 3.0


def test_a_stretch_ends_at_a_block_with_controls_and_the_next_one_has_a_new_clip():
    play = Play(IdleContext())
    for _ in range(5):
        play.block()
    assert play.block(idle=False)[0, 0, 0, 0] == 6.0 and play.context.clip == []
    assert play.context.stretch == 0
    # a new clip of the next three idle blocks (7, 8, 9), kept as generated; the clip is the new one
    assert [float(play.block()[0, 0, 0, 0]) for _ in range(5)] == [7.0, 8.0, 9.0, 7.0, 8.0]


def test_a_block_of_another_view_ends_the_stretch_and_starts_a_new_one():
    play = Play(IdleContext())
    for _ in range(5):
        play.block()
    turned = camera(yaw=2.0)
    assert (
        float(play.block(view=turned)[0, 0, 0, 0]) == 6.0
    )  # the first block of a stretch is the clip's
    assert play.context.stretch == 1
    assert [float(play.block(view=turned)[0, 0, 0, 0]) for _ in range(3)] == [7.0, 8.0, 6.0]
    # the camera of a block within the tolerance of the clip's first stays in its stretch
    nearly = camera(x=0.3, yaw=2.3)
    assert float(play.block(view=nearly)[0, 0, 0, 0]) == 7.0 and play.context.stretch == 5


def test_a_block_with_another_weapon_ends_the_stretch_as_one_of_another_view_does():
    play = Play(IdleContext())
    for _ in range(5):
        play.block()
    assert (
        float(play.block(weapons=(7, 9))[0, 0, 0, 0]) == 6.0
    )  # a switch inside the block: kept as generated
    assert play.context.stretch == 1
    assert [float(play.block(weapons=(7, 9))[0, 0, 0, 0]) for _ in range(3)] == [7.0, 8.0, 6.0]
    assert float(play.block(weapons=(9,))[0, 0, 0, 0]) == 10.0 and play.context.stretch == 1


def test_the_first_six_blocks_are_never_a_stretch():
    play = Play(IdleContext())
    for f0 in (1, 5, 9, 13, 17, 21):
        assert play.block(f0=f0)[0, 0, 0, 0] == play.j and play.context.stretch == 0
    assert play.context.kept == 0 and play.context.clip == []


def test_the_context_keeps_nothing_per_block_but_the_clip():
    play = Play(IdleContext())
    for _ in range(40):
        play.block()
    assert (play.context.stretch, play.context.kept) == (40, 37)
    assert len(play.context.clip) == CLIP_BLOCKS
    assert all(
        isinstance(c, torch.Tensor) and c.shape == (BLOCK, 2, 3, 4) for c in play.context.clip
    )


def test_the_clip_is_a_copy_of_the_store_not_a_view_of_it():
    play = Play(IdleContext())
    for _ in range(CLIP_BLOCKS):
        play.block()
    before = [c.clone() for c in play.context.clip]
    for _ in range(6):
        play.block()
    assert all(torch.equal(a, b) for a, b in zip(before, play.context.clip))
