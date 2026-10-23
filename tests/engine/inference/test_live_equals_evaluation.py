"""The demo's inference is the evaluation's: a recorded client played as a live player
(:class:`~worldcast.engine.inference.replay.RecordedControls`) generates what the evaluation's
client generates of the same recording, wherever a live client's world allows it. On the CPU, with
the tiny models.

* A client that stands and turns, with nobody visible and nothing flashing (``solo_world``): the
  live rollout and the paper's rollout of the same recording take the same noise, build the same
  windows (the same batch, cameras, memory frames and retrieval) and the same inputs, and generate
  the same latents, bit for bit, also on a round that slides. What differs are the control
  substeps (the other players' valid flags, the own turn's split over a frame's four ticks) and the
  validity of the labels: the generator reads none of them here (``SOLO_INPUTS``, ``SOLO_WINDOW``).
* What differs where the recording is not such a client, and why: the own position is estimated
  (the camera follows the keys' physics, not the recorded path), the players of the recording are
  nowhere, no flash or scope is read and visibility is predicted; and a window block is prepared
  with the controls held before it, so that its retrieval reads the camera it would have if
  nothing changed, not the recorded camera of the block to come.
"""

import numpy as np
import pytest
import torch

import tests.engine.inference.support as sw
import worldcast.engine.inference.live_rollout as live_rollout_module
from tests.player_state.support import own_position_read
from worldcast.data.latents import BLOCK, FIRST_TARGET
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import Client
from worldcast.engine.inference.live_rollout import LIVE_ROUND_LATENTS, LiveRollout
from worldcast.engine.inference.loading import ClientModels, load_window
from worldcast.engine.inference.messages import MessageWorldState
from worldcast.engine.inference.replay import RecordedControls, replay_latents
from worldcast.engine.inference.serving import ServingOptions

#: The recording's blocks: latent frames 1-24, then three window blocks.
BLOCKS = list(range(1, 1 + BLOCK * 9, BLOCK))
#: What differs between the two on a client that stands and turns, and why. In the inputs of the
#: generator, the other players' control substeps: valid where the recording has ticks and not
#: where a live client holds a player at its first row, and the own player's split of a frame's turn
#: over its four ticks (the recording's ticks, a live client's even split); the player state field
#: reads the substeps of a visible player only. In the window, besides those, the validity of the
#: labels (the GT's, a live client's own).
SOLO_INPUTS = {"player_controls"}
SOLO_WINDOW = {"client_visibility_valid", "player_control_substep_valid", "player_control_substeps"}


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    torch.set_num_threads(1)
    tmp = tmp_path_factory.mktemp("equal")
    weights = sw.make_weights(tmp / "weights")
    live = {"player_state.source": "live", **sw.write_state_model(tmp / "state_model.pt")}
    worlds = {
        "steady": sw.solo_world(tmp / "steady", steady=True),
        "varying": sw.solo_world(tmp / "varying", steady=False),
        "walking": sw.make_world(tmp / "walking", clients=(0,), seconds=9.5),
    }
    return dict(tmp=tmp, weights=weights, live=live, worlds=worlds, models={})


def config(room, world: str, source: str, **run):
    """The config of the client of ``world``'s row 0 under ``source`` (``gt`` or ``live``), with
    ``run`` settings of the ``run`` section."""
    cfg = sw.config(room["worlds"][world], room["weights"], room["tmp"] / "out", max_blocks=3)
    live = room["live"] if source == "live" else {}
    return cfg.with_overrides({**live, **{f"run.{key}": value for key, value in run.items()}})


def played(room, monkeypatch, world: str, source: str, *, explicit=False, **run) -> dict:
    """A rollout under ``source`` on the recording of ``world``, played to the recording's end:
    on the recording's own controls (``gt``), or on the block's controls given by
    :class:`RecordedControls` (``live``; ``gt`` with ``explicit``). What it returns: the rollout
    (``rollout``), what its sampler was given and returned at each block (``calls``: ``noisy``,
    ``x0`` and the conditions of its first denoising step) and each window block as it was cut
    (``cuts``). A live rollout's state model reads its own position where the keys put it, on open
    ground."""
    cfg = config(room, world, source, **run)
    sw.patch_ticks(monkeypatch, room["worlds"][world]["tables"])
    sw.patch_extent(monkeypatch)
    if source not in room["models"]:
        room["models"][source] = ClientModels.load(cfg, ServingOptions(decoder="none"))
    row = load_round_index_row(cfg.paths.round_index, 0)
    recorded = RecordedControls(load_window(config(room, world, "gt"), row))
    rollout = sw.make_rollout(
        cfg, MessageWorldState(client=row.media_id, poll_s=0.01), models=room["models"][source]
    )
    if source == "live":
        sw.own_position_oracle(rollout)
    calls, cuts = [], []
    denoise, cut = rollout.sampler.denoise, rollout._at_cut

    def spy_denoise(noisy, conditions, **kwargs):
        x0 = denoise(noisy, conditions, **kwargs)
        kept = {k: v.clone() if torch.is_tensor(v) else v for k, v in conditions.items()}
        calls.append(dict(noisy=noisy.clone(), x0=x0.clone(), conditions=kept))
        return x0

    def spy_cut(f0, prepared, controls):
        cuts.append(cut(f0, prepared, controls))
        return cuts[-1]

    rollout.sampler.denoise, rollout._at_cut = spy_denoise, spy_cut
    for f0 in range(1, recorded.latent_frames - BLOCK + 1, BLOCK):
        given = (
            (lambda f0=f0: recorded.block(f0)) if explicit or source == "live" else (lambda: None)
        )
        sw.step(rollout, f0, given)
    return dict(rollout=rollout, calls=calls, cuts=cuts, recorded=recorded)


def trim(conditions: dict, f0: int, frames: int) -> dict:
    """What a call of one of the first six blocks reads of the round's conditions: the latent
    frames up to the block's and their video frames."""
    video = 4 * (frames - 1) + 1
    out = {}
    for key, value in conditions.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == frames:
            value = value[:, : f0 + BLOCK]
        elif torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == video:
            value = value[:, : 4 * (f0 + BLOCK - 1) + 1]
        out[key] = value
    return out


def differing(a: dict, b: dict) -> set[str]:
    """The keys of two dicts whose values are not the same bits (another shape included)."""
    found = set()
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if torch.is_tensor(x) and torch.is_tensor(y):
            same = x.shape == y.shape and torch.equal(x, y)
        else:
            same = bool(x == y)
        if not same:
            found.add(key)
    return found


def what_differs(evaluation: dict, live: dict) -> dict[str, set[str]]:
    """What is not the same bits in two played rollouts, over all their blocks: the sampler's
    ``inputs`` (the entry noise, the conditions and the x0 it returned), the cut ``window``
    (its gathered batch, its context's latents and its rays) and what retrieval ``read``
    (``window_frames``, ``entry``, ``coverage``, ``missing``, ``candidates``)."""
    found: dict[str, set[str]] = {"inputs": set(), "window": set(), "read": set()}
    for f0, a, b in zip(BLOCKS, evaluation["calls"], live["calls"]):
        conditions = []
        for call, played_ in ((a, evaluation), (b, live)):
            frames = played_["rollout"].latent_frames
            conditions.append(
                call["conditions"] if f0 >= FIRST_TARGET else trim(call["conditions"], f0, frames)
            )
        found["inputs"] |= differing(
            {"noisy": a["noisy"], "x0": a["x0"], **conditions[0]},
            {"noisy": b["noisy"], "x0": b["x0"], **conditions[1]},
        )
    for a, b in zip(evaluation["cuts"], live["cuts"]):
        # the target frames' slots of the window's latents are not read: the store's tail
        windows = [
            {**cut.batch, "latents": cut.batch["latents"][:, : cut.context]} for cut in (a, b)
        ]
        found["window"] |= differing(*windows) | differing(a.rays, b.rays)
        found["read"] |= differing(vars(a.read), vars(b.read))
    return found


def test_a_live_client_plays_a_recorded_client_that_stands_and_turns_to_the_bit(room, monkeypatch):
    """The paper's rollout of a client that stands and turns, and the live rollout on that
    recording's controls (each of its three window blocks retrieves a memory entry): the same
    noise, windows, retrievals and inputs but for the other players' substeps, the same latents,
    bit for bit."""
    evaluation = played(room, monkeypatch, "steady", "gt")
    live = played(room, monkeypatch, "steady", "live")
    assert type(live["rollout"]) is LiveRollout and live["rollout"].shift == 4
    assert [c.read.window_frames for c in live["cuts"]] == [21, 21, 21]
    assert what_differs(evaluation, live) == {
        "inputs": SOLO_INPUTS,
        "window": SOLO_WINDOW,
        "read": set(),
    }
    for a, b in zip(evaluation["calls"], live["calls"]):
        assert torch.equal(a["noisy"], b["noisy"]) and torch.equal(a["x0"], b["x0"])


def test_a_round_that_slides_plays_as_the_whole_round(room, monkeypatch):
    """The same client on a live round of 29 latent frames, which slides before its eighth and
    ninth blocks and once more when the tenth is prepared, and on the evaluation's whole round of
    37: every block's noise, window inputs and x0 are the same bits."""
    evaluation = played(room, monkeypatch, "steady", "gt")
    live = played(room, monkeypatch, "steady", "live", latent_frames=LIVE_ROUND_LATENTS)
    assert live["rollout"].latent_frames == 29 and live["rollout"].shift == 12
    assert len(live["calls"]) == len(evaluation["calls"]) == 9
    assert what_differs(evaluation, live) == {
        "inputs": SOLO_INPUTS,
        "window": SOLO_WINDOW,
        "read": set(),
    }
    for a, b in zip(evaluation["calls"], live["calls"]):
        assert torch.equal(a["noisy"], b["noisy"]) and torch.equal(a["x0"], b["x0"])


def test_the_recorded_controls_given_to_the_evaluation_are_the_recorded_ones(room, monkeypatch):
    """The paper's rollout on its own recorded controls, and on the block's controls given
    explicitly, as a live client takes them: the same bits."""
    recorded = played(room, monkeypatch, "steady", "gt")
    explicit = played(room, monkeypatch, "steady", "gt", explicit=True)
    assert what_differs(recorded, explicit) == {"inputs": set(), "window": set(), "read": set()}
    assert torch.equal(recorded["rollout"].output, explicit["rollout"].output)


def test_replaying_a_recording_gives_the_latents_of_the_live_rollout(room, monkeypatch):
    """:func:`replay_latents` plays a recording through a client: the first frame and each
    block's x0, the latents of the live rollout of the test above."""
    live = played(room, monkeypatch, "steady", "live")
    cfg = config(room, "steady", "live", latent_frames=LIVE_ROUND_LATENTS)
    row = load_round_index_row(cfg.paths.round_index, 0)
    recording = load_window(config(room, "steady", "gt"), row)
    client = Client(
        cfg, ServingOptions(decoder="none", lockstep=False), models=room["models"]["live"]
    )
    load_live_loop = live_rollout_module.load_live_loop

    def reading_the_keys_position(*args, **kwargs):
        loop = load_live_loop(*args, **kwargs)
        loop.reader.read = lambda latents, frames: own_position_read(loop.motion, frames)
        return loop

    monkeypatch.setattr(live_rollout_module, "load_live_loop", reading_the_keys_position)
    latents = replay_latents(client, row, recording)
    assert latents.shape == (37, 48, 24, 42) and latents.dtype == np.float32
    assert np.array_equal(latents[0], live["rollout"].output[0, 0].numpy())
    for k, call in enumerate(live["calls"]):
        assert np.array_equal(latents[1 + 4 * k : 5 + 4 * k], call["x0"][0].numpy())


def test_a_replay_plays_a_live_client_that_does_not_decode(room, monkeypatch):
    sw.patch_ticks(monkeypatch, room["worlds"]["steady"]["tables"])
    cfg = config(room, "steady", "live")
    row = load_round_index_row(cfg.paths.round_index, 0)
    recording = load_window(config(room, "steady", "gt"), row)
    serving = ServingOptions(decoder="none", lockstep=False)
    models = ClientModels.load(cfg, serving)
    decoding = Client(cfg, serving, models=models)
    decoding.serving = ServingOptions(decoder="wan", lockstep=False)
    offline = Client(config(room, "steady", "gt"), serving, models=models)
    for client in (decoding, offline):
        with pytest.raises(ValueError, match="plays a live client .* the decoder none"):
            replay_latents(client, row, recording)


def test_the_own_position_the_observer_signals_and_the_players_are_what_a_live_client_has_not(
    room, monkeypatch
):
    """On a recording that moves, with GT labels that show players and flashes: the live client's
    first six blocks differ from the evaluation's in the own camera (it follows the keys, the
    recorded path does not), the own row of the table, the observer signals (nothing flashes, the
    scope is off), the players' rows and controls (nobody but the player is in the world) and the
    labels (predicted: nobody is visible). The controls, the prompt, the first frame and the
    noise are the same."""
    evaluation = played(room, monkeypatch, "walking", "gt")
    live = played(room, monkeypatch, "walking", "live")
    frames = evaluation["rollout"].latent_frames
    found = set()
    for f0, a, b in zip(BLOCKS[:6], evaluation["calls"], live["calls"]):
        assert torch.equal(a["noisy"], b["noisy"])
        found |= differing(trim(a["conditions"], f0, frames), trim(b["conditions"], f0, frames))
    assert found == {
        "obs_flash_flag",
        "obs_scope_level",
        "obs_scope_on",
        "player_controls",
        "player_state_table",
        "player_visible",
        "ray_frame_c2w",
    }
    assert torch.equal(evaluation["rollout"].output[:, :1], live["rollout"].output[:, :1])
    assert not torch.equal(evaluation["calls"][0]["x0"], live["calls"][0]["x0"])


def test_a_window_is_prepared_with_the_controls_held_before_it(room, monkeypatch):
    """A client whose controls change from block to block: the first six blocks are the
    evaluation's to the bit (the cut writes their rows). A window block reads the scene state
    before its controls, with the turn of the block before, so that the camera its retrieval
    queries is not the recorded camera of the block to come: what retrieval reads differs from the
    first window block on."""
    evaluation = played(room, monkeypatch, "varying", "gt")
    live = played(room, monkeypatch, "varying", "live")
    for a, b in zip(evaluation["calls"][:6], live["calls"][:6]):
        assert torch.equal(a["noisy"], b["noisy"]) and torch.equal(a["x0"], b["x0"])
    assert what_differs(evaluation, live)["read"]
    assert not torch.equal(evaluation["calls"][6]["x0"], live["calls"][6]["x0"])
