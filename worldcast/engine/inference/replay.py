"""A recorded client played as a live player: the demo's inference in evaluation form.

:class:`RecordedControls` hands a :class:`~worldcast.engine.inference.live_rollout.LiveRollout` the
recording's own controls block by block, as a player's live input would reach it: the rows the
evaluation reads (the buttons, the code of the turn, the weapon) and the turn they encode, which
the client's camera integrates, at the session times of the recording's video frames.
:func:`replay_latents` plays them through a live client and returns what the client generated.
The settings stay the caller's: a client served as ``tools/run_client.py`` serves it and a client
served as the demo's worker serves it differ in what ``docs/demo.md`` ("GPU integration") names.
"""

import numpy as np

from worldcast.data.latents import BLOCK, FPS, video_frames_of
from worldcast.data.recordings import RoundIndexRow
from worldcast.data.window import ClientWindow

from .client import Client
from .controls import LiveBlockControls

__all__ = ["RecordedControls", "replay_latents"]


class RecordedControls:
    """The recorded controls of a client's window as the blocks of a live session.

    Args:
        window (ClientWindow): the client's window of the recording, as long as the session to
            play (:func:`~worldcast.engine.inference.loading.load_window`).
    """

    def __init__(self, window: ClientWindow) -> None:
        self.frames, self.latent_frames = window.client_frames, window.spec.latent_frames

    def block(self, f0: int) -> LiveBlockControls:
        """The controls of the block at latent frame ``f0`` (1, 5, 9, ...).

        Args:
            f0 (int): the block's first latent frame.

        Returns:
            LiveBlockControls: its 16 video frames' buttons, mu-law view controls and weapon as the
            recording gives the evaluation, the turn they encode (degrees, pitch then yaw) and
            the session time of each frame, ``v / FPS`` s for video frame ``v``.

        Raises:
            ValueError: the recording's window ends before the block does.
        """
        if f0 < 1 or (f0 - 1) % BLOCK or f0 + BLOCK > self.latent_frames:
            raise ValueError(
                f"block {f0} is not a block of the recording's {self.latent_frames} latent frames"
            )
        rows = video_frames_of(f0, BLOCK)
        frames = self.frames
        return LiveBlockControls(
            buttons=frames.buttons[rows],
            view_deltas=frames.view_deltas[rows],
            weapon=frames.weapon_ids[rows],
            turn=frames.turn[rows],
            times=np.asarray(rows, np.float64) / FPS,
        )


def replay_latents(client: Client, row: RoundIndexRow, window: ClientWindow) -> np.ndarray:
    """Play a recorded client through a live ``client`` and return its latents.

    The client plays the player of ``row`` alone, as it does in a room where nobody else plays
    (``player_state.source: live``): its blocks are generated on the recording's controls, one
    after the other, as far as the recording's window goes.

    Args:
        client (Client): a live client (``player_state.source: live``) served with the decoder
            ``none``, so that its frames are its latent frames.
        row (RoundIndexRow): the client's row of the round index.
        window (ClientWindow): its window of the recording, as long as the session to play.

    Returns:
        np.ndarray: ``[N, 48, 24, 42]`` float32, the first frame and every block's latents rounded
        to the output's dtype, as ``latents.npy`` holds those of the paper's run.

    Raises:
        ValueError: the client is not live or decodes its frames.
    """
    if client.cfg.player_state.source != "live" or client.serving.decoder != "none":
        raise ValueError(
            "a replay plays a live client (player_state.source: live) served with the decoder none"
        )
    controls, latents = RecordedControls(window), []
    client.start(row)
    try:
        for f0 in range(1, window.spec.latent_frames - BLOCK + 1, BLOCK):
            latents += [frame.image for frame in client.step(lambda f0=f0: controls.block(f0))]
    finally:
        client.stop()
    return np.stack(latents)
