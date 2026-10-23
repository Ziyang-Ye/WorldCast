"""The rows the player state field read at one latent frame, which the demo's frame headers carry.

:meth:`worldcast.engine.inference.rollout.Rollout.field_rows` gives them. The module loads no torch,
so the source codes are read where torch is not loaded (the demo's coordinator and mock client).
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ["FieldRows"]


@dataclass
class FieldRows:
    """Every player's row as the player state field read it at one latent frame: what the live
    demo's frame headers carry.

    Attributes:
        states (np.ndarray): ``[P, 6]`` float64 ``x, y, z, yaw, pitch, alive`` (u, degrees, {0, 1})
            of every player.
        source (np.ndarray): ``[P]`` int8, where each row comes from: :attr:`OWN`, :attr:`LIVE`,
            :attr:`RECORDED` or :attr:`HIDDEN`; live play has no :attr:`RECORDED` rows.
        drawn (np.ndarray | None): ``[P]`` bool, the players the field draws (its visibility
            labels), or ``None`` where the rollout does not keep them.
        status (str): how the client's own row is known: ``player_state.source`` under ``gt`` and
            ``predicted``; a rollout with an estimator of its own reports the estimator's state.
    """

    #: The client's own row.
    OWN: ClassVar[int] = 0
    #: A player another client plays live, at the rows that client published.
    LIVE: ClassVar[int] = 1
    #: A player on its recording's controls: under ``gt`` and ``predicted``, never in live play.
    RECORDED: ClassVar[int] = 2
    #: A slot whose row the page does not show: a slot without media in the round
    #: (:class:`~worldcast.engine.inference.rollout.Rollout`), and in live play every player that
    #: is not in the world
    #: (:class:`~worldcast.engine.inference.live_rollout.LiveRollout`).
    HIDDEN: ClassVar[int] = 3

    states: np.ndarray
    source: np.ndarray
    drawn: np.ndarray | None
    status: str
