"""The closed loop: the state model's windows and controls, the complementary filter of Eq. (4),
and the positions the clients publish once per block."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.data.support import tick_table
from tests.player_state.support import PRIOR, flat_depth, round_batch
from worldcast.data.memory_frames import MemoryBlock
from worldcast.player_state import ClosedLoop, ComplementaryFilter, StateReader
from worldcast.player_state.closed_loop import FUSION_WEIGHT, window_of

N_LATENTS = 29
MEDIA = {
    0: "1-de_dust2-r2-p00",
    1: "1-de_dust2-r2-p01",
    2: "1-de_dust2-r2-p02",
    3: "1-de_dust2-r2-p03",
}


# ----------------------------------------------------------------------------- Eq. (4)
def test_the_complementary_filter_is_eq_4():
    """``p_f = 1/2 (p_{f-1} + 64 Delta_f) + 1/2 (A_f - A_0 + p_0)``."""
    assert FUSION_WEIGHT == 0.5
    position = ComplementaryFilter(np.array([100.0, 0.0, 0.0]))
    # the place head's first estimate is 10 u off: its bias, removed from every later estimate
    assert position.step(0, [110.0, 0.0, 0.0], [9.0, 9.0, 9.0]).tolist() == [100.0, 0.0, 0.0]
    # one unit of displacement is 64 u: 1/2 (100 + 64) + 1/2 (180 - 110 + 100)
    assert position.step(1, [180.0, 0.0, 0.0], [1.0, 0.0, 0.0]).tolist() == [167.0, 0.0, 0.0]
    # 1/2 (167 + 0) + 1/2 (110 - 110 + 100), and half of -64 u along y
    assert position.step(2, [110.0, 0.0, 0.0], [0.0, -1.0, 0.0]).tolist() == [133.5, -32.0, 0.0]
    with pytest.raises(ValueError, match="latent frame 4 after 2"):
        position.step(4, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


# --------------------------------------------------------------------- the state model's windows
def test_a_latent_frame_is_read_from_the_window_that_ends_with_it():
    assert [window_of(f) for f in (0, 1, 40, 41, 80, 81)] == [
        (0, 0),
        (0, 1),
        (0, 40),
        (1, 1),
        (1, 40),
        (2, 1),
    ]
    with pytest.raises(ValueError):
        window_of(-1)


class StubStateModel:
    """Stands in for the state model: its place estimate is ``(latent value, position in the
    window, map)``, its displacement the turn summed over the latent frame's 16 substeps."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, latents, controls, map_id):
        self.calls.append((tuple(latents.shape), tuple(controls.shape), latents.dtype))
        frames = int(latents.shape[1])
        place = torch.stack(
            [
                latents[0].reshape(frames, -1)[:, 0],
                torch.arange(frames, dtype=torch.float32),
                torch.full((frames,), float(map_id[0])),
            ],
            dim=-1,
        )
        displacement = controls[0, :, :, 13:16].sum(1)  # pitch, yaw, valid
        return SimpleNamespace(place=place[None], displacement=displacement[None])


def test_the_state_reader_reads_each_latent_frame_from_its_window():
    model = StubStateModel()
    ticks = tick_table(seconds=21.0)  # holds forward and turns 30 / 64 degrees per tick
    reader = StateReader(
        model, ticks, map_name="de_mirage", source_fps=32.0, start_frame=0, device="cpu"
    )
    latents = (torch.arange(45, dtype=torch.float32) + 1.0 / 3.0).view(45, 1, 1, 1)
    place, displacement = reader.read(latents, [0])
    assert place.dtype == np.float64 and displacement.dtype == np.float32
    # latent frame 0 at position 0 of window 0; the latents are read in half precision
    assert place.tolist() == [[pytest.approx(0.33325195), 0.0, 2.0]]
    assert model.calls == [((1, 41, 1, 1, 1), (1, 41, 16, 16), torch.float32)]
    # latent frame 0 holds tick 0 alone: one valid substep, a turn of 30 / 64 / 5
    assert displacement.tolist() == [[0.0, pytest.approx(0.09375), 1.0]]

    place, displacement = reader.read(latents, [39, 40, 41, 42])
    # window 0 holds latent frames 0 .. 40, window 1 latent frames 40 .. 80
    assert place[:, 1].tolist() == [39.0, 40.0, 1.0, 2.0]
    assert place[:, 0].tolist() == pytest.approx([39.3333, 40.3333, 41.3333, 42.3333], abs=2e-2)
    assert displacement[:, 2].tolist() == [16.0] * 4  # sixteen valid substeps per latent frame
    assert displacement[:, 1].tolist() == pytest.approx([16 * 0.09375] * 4)
    with pytest.raises(ValueError, match="latent frame 45 requested with 45 generated"):
        reader.read(latents, [45])
    with pytest.raises(ValueError, match="the state model knows the maps .*, not 'de_cache'"):
        StateReader(model, ticks, map_name="de_cache", source_fps=32.0, start_frame=0, device="cpu")


def test_a_window_still_being_generated_is_read_with_its_future_zeroed():
    seen = []

    class Recording(StubStateModel):
        def __call__(self, latents, controls, map_id):
            seen.append(latents[0, :, 0, 0, 0].clone())
            return super().__call__(latents, controls, map_id)

    ticks = tick_table(seconds=11.0)
    reader = StateReader(
        Recording(), ticks, map_name="de_dust2", source_fps=32.0, start_frame=0, device="cpu"
    )
    reader.read(torch.ones(9, 1, 1, 1), [1, 2, 3, 4])
    assert seen[0].tolist() == [1.0] * 5 + [0.0] * 36  # latent frames 0 .. 4, then zeros


# ------------------------------------------------------------------------------- the closed loop
class FakeReader:
    """Stands in for the state model: a deterministic place estimate and motion per latent
    frame."""

    def __init__(self, seed: int) -> None:
        self.g = np.random.default_rng(seed)
        self.calls = []

    def read(self, latents, latent_frames):
        self.calls.append((int(latents.shape[0]), list(latent_frames)))
        k = np.asarray(latent_frames, np.float64)[:, None]
        n = len(latent_frames)
        place = 50.0 * k + self.g.normal(0, 5, (n, 3))
        return place, self.g.normal(0, 1, (n, 3)).astype(np.float32)


def _round() -> dict:
    """A round of four players; slot 3 is never alive."""
    batch = round_batch(0, batch=1, players=4, latents=N_LATENTS, client_slots=(0,))
    batch["player_states"][..., 5] = 1.0
    batch["player_states"][0, 3, :, 5] = 0.0
    batch["client_visibility"] = torch.zeros_like(batch["client_visibility"], dtype=torch.float32)
    batch["client_visibility_valid"] = torch.zeros_like(batch["client_visibility_valid"])
    return batch


class Positions:
    """One client's view of the positions the round's clients published (a ``PositionExchange``
    in memory)."""

    def __init__(self, client: str, published: dict) -> None:
        self.client, self.published = client, published

    def publish_position(self, t_target, latent_frames, xyz) -> None:
        self.published[self.client, int(t_target)] = (
            list(latent_frames),
            np.asarray(xyz, np.float64),
        )

    def read_position(self, media_id, t_target):
        return self.published.get((media_id, int(t_target)))


def client(slot: int, published: dict) -> ClosedLoop:
    batch = _round()
    batch["client_slot"] = torch.tensor([slot])
    return ClosedLoop.build(
        batch,
        reader=FakeReader(slot),
        client=MEDIA[slot],
        client_slot=slot,
        round_media=MEDIA,
        others=[MEDIA[p] for p in (0, 1, 2) if p != slot],
        latent_frames=N_LATENTS,
        start_frame=16,
        prior=PRIOR,
        depth_fn=flat_depth(1100.0),
        world_state=Positions(MEDIA[slot], published),
    )


def test_a_client_publishes_its_filtered_position_once_per_block():
    published: dict = {}
    own = client(0, published)
    start = _round()["player_states"][0, 0, 0, :3].double().numpy()
    latents = torch.zeros(N_LATENTS, 48, 24, 42)
    own.publish_position(1, latents)
    # block 1 publishes latent frame 0, the round start, under the block's first source frame
    frames, xyz = published[MEDIA[0], 16 + 8 * 1]
    assert frames == [0] and np.array_equal(xyz, start[None])
    own.publish_position(5, latents)
    frames, xyz = published[MEDIA[0], 16 + 8 * 5]
    assert frames == [1, 2, 3, 4]
    assert own.reader.calls == [(N_LATENTS, [0]), (N_LATENTS, [1, 2, 3, 4])]
    # the filter of Eq. (4) on the reader's estimates
    reader, check = FakeReader(0), ComplementaryFilter(start)
    place, displacement = reader.read(latents, [0])
    check.step(0, place[0], displacement[0])
    place, displacement = reader.read(latents, [1, 2, 3, 4])
    want = np.stack([check.step(f, place[i], displacement[i]) for i, f in enumerate(frames)])
    assert np.array_equal(xyz, want) and np.array_equal(own.cameras.track.xyz[1:], want)


def test_clients_agree_on_each_other():
    """Every client places every other client where that client placed itself (its own
    extrapolation of block s)."""
    published: dict = {}
    clients = {p: client(p, published) for p in (0, 1, 2)}
    assert clients[0].table.track_of(MEDIA[3]) is None  # never alive: keeps the recording
    assert clients[0].others == (MEDIA[1], MEDIA[2])
    latents = torch.zeros(N_LATENTS, 48, 24, 42)
    for s in range(1, N_LATENTS - 3, 4):
        for c in clients.values():
            c.publish_position(s, latents)
        for c in clients.values():
            c.read_positions(s)
        for p, own in clients.items():
            mine = torch.from_numpy(own.cameras.block_rows(s)[:, :3]).float()
            for q, other in clients.items():
                assert torch.equal(other.table.states[0, p, 4 * np.arange(s, s + 4), :3], mine), (
                    s,
                    p,
                    q,
                )
    assert clients[1].reader.calls[1] == (N_LATENTS, [1, 2, 3, 4])


def test_a_client_without_a_position_holds_its_last_one():
    published: dict = {}
    clients = {p: client(p, published) for p in (0, 1)}  # client 2 never publishes
    latents = torch.zeros(N_LATENTS, 48, 24, 42)
    for s in (1, 5, 9):
        for c in clients.values():
            c.publish_position(s, latents)
        for c in clients.values():
            c.read_positions(s)
    xyz = clients[0].table.states[0, 2, :, :3]
    # block 1 (video frames 1-16) is extrapolated from the round start; later frames hold its last
    assert not torch.equal(xyz[16], xyz[0])
    assert torch.equal(xyz[17:], xyz[16:17].expand_as(xyz[17:]))


def test_another_clients_block_is_keyed_where_that_client_generated_it():
    published: dict = {}
    clients = {p: client(p, published) for p in (0, 2)}
    latents = torch.zeros(N_LATENTS, 48, 24, 42)
    for c in clients.values():
        c.publish_position(1, latents)
    for c in clients.values():
        c.read_positions(1)
    recorded = _round()["player_states"]
    # client 2's block of latent frames 1 .. 4, in a window that starts with the round
    block = MemoryBlock.at(MEDIA[2], 2, 16, 1, torch.zeros(4, 4, 4))
    generated = clients[0].as_generated(block, recorded)
    table = clients[0].table.states
    assert torch.equal(generated.c2w[:, :2, 3], table[0, 2, [4, 8, 12, 16], :2])  # the eyes' x, y
    assert not torch.equal(table[0, 2, 16, :3], recorded[0, 2, 16, :3])  # not the recorded place
    inputs = clients[0].inputs_as_generated({"states": recorded[0, 2, 1:17].clone()}, block)
    assert torch.equal(inputs["states"][:, :3], table[0, 2, 1:17, :3])
    assert torch.equal(inputs["states"][:, 3:], recorded[0, 2, 1:17, 3:])
    with pytest.raises(ValueError, match="off the round's latent grid"):
        clients[0].as_generated(MemoryBlock.at(MEDIA[2], 2, 17, 1, None), recorded)
