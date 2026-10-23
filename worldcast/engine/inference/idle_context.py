"""The context of a live client whose player is idle (``live.idle_context``; not the paper's).

A client whose player is idle with nobody else drawn in its view hands the generator nothing that
changes from block to block except its own last blocks: the keys and the view controls are zero,
the player state field is empty, every camera of the window is the same camera, and the first frame
never changes. The recent context is three of the client's own blocks, so each block is generated
from the output of the blocks before it, and what the generator makes of a view it is shown again
and again compounds. :class:`IdleContext` cuts that loop: the first :data:`CLIP_BLOCKS` blocks of an
idle stretch are kept as generated, the **clip**; every block after them is generated and shown as
always, but the store, which the next windows are gathered from, and the latents of the memory
bank keep the clip's blocks in turn in its place, so that the context of the next block is always
the clip.

A **stretch** is a run of idle blocks of one view and one weapon. The caller says which blocks are
idle (:class:`~worldcast.engine.inference.live_rollout.LiveRollout`: the block's controls are
:func:`is_idle`, and no other player is drawn in its window). A block of the stretch has the
cameras of the stretch's first block (:func:`same_view`) and holds that block's weapon (a weapon
the clip does not show would flicker). The first six blocks, which read no scene state, are never
part of one. The stretch ends, and the clip is dropped, at the first block that is not idle or is of
another view or weapon; that block is kept as generated, and is generated from the clip's context.
Nothing is kept per block but the clip.
"""

import numpy as np
import torch

from worldcast.data.latents import BLOCK, FIRST_TARGET, RECENT

from .controls import BlockControls

__all__ = ["CLIP_BLOCKS", "SAME_VIEW_DEGREES", "SAME_VIEW_U", "IdleContext", "is_idle", "same_view"]

#: The blocks of the clip: as many as the recent context holds (of the generator trained without
#: scene state, as many as its window's recent context holds, :class:`IdleContext`).
CLIP_BLOCKS = RECENT // BLOCK
#: Two views are one when every camera is within this many u of its partner and turned by no more
#: than this many degrees: a standing camera is constant to the wire's 0.01 u.
SAME_VIEW_U, SAME_VIEW_DEGREES = 0.5, 0.5


def is_idle(controls: BlockControls) -> bool:
    """Whether a block's controls hold nothing: no button on any video frame and view controls (the
    generator's) that are all zero."""
    return not controls.buttons.any() and not controls.view_deltas.any()


def same_view(a: object, b: object) -> bool:
    """Whether two sets of cameras ``[n, 4, 4]`` (camera to world, array-likes) are one view
    (:data:`SAME_VIEW_U`, :data:`SAME_VIEW_DEGREES`)."""
    a = np.asarray(a, np.float64).reshape(-1, 4, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4, 4)
    if a.shape != b.shape:
        return False
    if np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1).max() > SAME_VIEW_U:
        return False
    cos = (np.einsum("nij,nij->n", a[:, :3, :3], b[:, :3, :3]) - 1.0) / 2.0  # trace(Ra^T Rb)
    return bool(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max() <= SAME_VIEW_DEGREES)


class IdleContext:
    """The clip of an idle stretch and what it does to the store (the module docstring).

    Args:
        clip_blocks (int): the blocks of the clip: :data:`CLIP_BLOCKS`, or as many as the recent
            context of the window holds.

    Attributes:
        stretch (int): the idle blocks of the stretch so far.
        clip (list[torch.Tensor]): the clip's blocks, ``[4, 48, 24, 42]`` float32 on the CPU.
        kept (int): the blocks the store has kept as the clip's so far.
    """

    def __init__(self, clip_blocks: int = CLIP_BLOCKS) -> None:
        self.clip_blocks = int(clip_blocks)
        self.stretch, self.kept = 0, 0
        self.clip: list[torch.Tensor] = []
        self._view: np.ndarray | None = None
        self._weapons: tuple[int, ...] = ()

    def reset(self) -> None:
        """The stretch is over: the clip is dropped."""
        self.stretch, self.clip, self._view, self._weapons = 0, [], None, ()

    def keep(
        self,
        store: torch.Tensor,
        f0: int,
        cameras: object,
        weapons: tuple[int, ...],
        *,
        idle: bool,
    ) -> None:
        """Before block ``f0`` of the round is published from ``store``: keep it as generated, or
        put the clip's block in its place.

        Args:
            store (torch.Tensor): ``[N, 48, 24, 42]`` float32 the client's latents; block ``f0`` is
                written.
            f0 (int): the block's first latent frame of the round.
            cameras (object): ``[4, 4, 4]`` the block's cameras.
            weapons (tuple[int, ...]): the weapon ids the block holds, sorted.
            idle (bool): whether the block is idle (the module docstring).
        """
        if f0 < FIRST_TARGET or not idle:
            self.reset()
            return
        view = np.asarray(cameras, np.float64).reshape(-1, 4, 4)
        if self._view is not None and not (
            same_view(self._view, view) and self._weapons == weapons
        ):
            self.reset()
        if self._view is None:
            self._view, self._weapons = view.copy(), weapons
        self.stretch += 1
        if self.stretch <= self.clip_blocks:
            self.clip.append(store[f0 : f0 + BLOCK].clone())
        else:
            store[f0 : f0 + BLOCK] = self.clip[(self.stretch - 1) % self.clip_blocks]
            self.kept += 1
