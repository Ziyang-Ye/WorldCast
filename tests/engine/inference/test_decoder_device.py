"""The decoder on a device of its own (``ServingOptions.decoder_device``), on the CPU: where its
weights go, the frames it gives (the same as the client's own device's, byte for byte), and that the
generation thread never waits for it.

A second card is a second CPU here: the device is a name, the copy of a latent frame to it and the
decode there run through the same code."""

import contextlib
import queue
import threading
import types

import numpy as np
import torch

import tests.engine.inference.support as sw
from tests.engine.inference.test_client import RoundedOutput
from worldcast.config.inference import InferenceConfig
from worldcast.data.recordings import load_round_index_row
from worldcast.engine.inference.client import Client, _RunningRollout
from worldcast.engine.inference.loading import ClientModels
from worldcast.engine.inference.serving import ServingOptions


class Placed(torch.nn.Module):
    """A decoder that says where it was moved."""

    def __init__(self) -> None:
        super().__init__()
        self.moved: list[torch.device] = []
        self.channels_last: list[bool] = []

    def to(self, device, *args, **kwargs):
        self.moved.append(device)
        return self

    def set_decoder_channels_last(self, on: bool) -> None:
        self.channels_last.append(on)


def models_with(**decoders) -> ClientModels:
    return ClientModels(generator=None, prompt_embeds=None, depth=None, **decoders)


def test_a_decoder_goes_to_the_decoder_device_or_stays_on_the_clients():
    cfg = InferenceConfig().with_overrides({"run.device": "cpu"})
    models = models_with(wan_vae=Placed(), tiny=Placed())
    models.load_missing(cfg, ServingOptions(decoder="wan"))
    models.load_missing(cfg, ServingOptions(decoder="wan", decoder_device="cuda:1"))
    # only the decoder of the serving options moves: the other one is not in use
    assert models.wan_vae.moved == [torch.device("cpu"), torch.device("cuda:1")]
    assert models.tiny.moved == []
    models.load_missing(cfg, ServingOptions(decoder="taehv", taehv_path="x", decoder_device="cpu"))
    models.load_missing(cfg, ServingOptions(decoder="none"))
    assert models.tiny.moved == [torch.device("cpu")] and len(models.wan_vae.moved) == 2


def test_the_channels_last_setting_reaches_the_vae_and_only_the_vae():
    cfg = InferenceConfig().with_overrides({"run.device": "cpu"})
    models = models_with(wan_vae=Placed(), tiny=Placed())
    models.load_missing(cfg, ServingOptions(decoder="wan"))
    models.load_missing(cfg, ServingOptions(decoder="wan", decoder_channels_last=True))
    models.load_missing(cfg, ServingOptions(decoder="wan", decoder_device="cuda:1"))
    models.load_missing(cfg, ServingOptions(decoder="taehv", taehv_path="x"))
    # each setting puts the layout it names: a decoder shared by settings keeps no other's
    assert models.wan_vae.channels_last == [False, True, False]
    assert not hasattr(models.tiny, "channels_last") or models.tiny.channels_last == []


def test_the_decoder_on_its_own_device_gives_the_frames_of_the_same_device(
    synthetic, tiny_vae, monkeypatch
):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "live", max_blocks=2
    )
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    models.wan_vae = tiny_vae
    frames = {}
    for device in (None, "cpu"):
        serving = ServingOptions(decoder="wan", lockstep=False, decoder_device=device)
        client = Client(cfg, serving, models=models)
        client.start(load_round_index_row(cfg.paths.round_index, 0))
        frames[device] = [frame.image for frame in client.step()]
        client.stop()
    assert len(frames[None]) == 1 + 16  # the first frame and the first block
    assert frames[None][0].shape == (384, 672, 3) and frames[None][0].dtype == np.uint8
    assert all(np.array_equal(a, b) for a, b in zip(frames[None], frames["cpu"]))


def test_the_channels_last_decoder_gives_nearly_the_frames_of_the_default_and_off_gives_them_back(
    synthetic, tiny_vae, monkeypatch
):
    sw.patch_ticks(monkeypatch, synthetic["world"]["tables"])
    cfg = sw.config(
        synthetic["world"], synthetic["weights"], synthetic["tmp"] / "live", max_blocks=2
    )
    models = ClientModels.load(cfg, ServingOptions(decoder="none"))
    models.wan_vae = tiny_vae
    frames = []
    for channels_last in (False, True, False):
        serving = ServingOptions(decoder="wan", lockstep=False, decoder_channels_last=channels_last)
        client = Client(cfg, serving, models=models)
        client.start(load_round_index_row(cfg.paths.round_index, 0))
        frames.append([frame.image for frame in client.step()])
        client.stop()
    assert all(np.array_equal(a, b) for a, b in zip(frames[0], frames[2]))
    assert len(frames[1]) == len(frames[0]) and frames[1][1].dtype == np.uint8
    worst = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(*frames[:2]))
    assert worst <= 2  # the rounding of another kernel, not another picture


class Event:
    """A CUDA event that may be recorded; any wait for it by the host raises."""

    calls: list[str] = []

    def record(self) -> None:
        self.calls.append("record")

    def synchronize(self) -> None:
        raise AssertionError("the host waited for the generator's stream")

    query = wait = synchronize


class Stream:
    """The decoder's stream: it waits for events, on the device, and nothing else."""

    def __init__(self) -> None:
        self.waited: list[Event] = []

    def wait_event(self, event: Event) -> None:
        self.waited.append(event)


class OtherDevice:
    """A decoder on another device: it takes a latent frame and gives its pixels."""

    def __init__(self) -> None:
        self.latents: list[torch.Tensor] = []
        self.threads: set[str] = set()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.latents.append(latents.clone())
        self.threads.add(threading.current_thread().name)
        return torch.full((1, 3, 2, 2), 0.5)


def test_the_generation_thread_only_records_an_event_for_the_decoder(monkeypatch):
    """The decode of a block waits for its latents on the decoder's stream, on the device: the
    thread that generates the block records one event and no more, whatever the decoder's device
    does (an event that is waited for by the host raises)."""
    Event.calls = []
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    rollout, stream, decoder = RoundedOutput(open_ended=True), Stream(), OtherDevice()
    running = object.__new__(_RunningRollout)
    running.client = types.SimpleNamespace(stream=stream, records=[])
    running.rollout, running.decoder = rollout, decoder
    running.delivered, running.next_frame = queue.Queue(), 0
    running.jobs, running._lock = [], threading.Lock()

    generating = threading.Thread(target=running._block, args=(1, None), name="generation")
    generating.start()
    generating.join()
    assert Event.calls == ["record"] and stream.waited == []
    (delivery,) = running.delivered.queue
    assert isinstance(delivery.ready, Event)

    frames = list(running._drain(1))
    assert [frame.index for frame in frames] == [0, 1, 2, 3]  # the block's four latent frames
    assert stream.waited == [delivery.ready] and Event.calls == ["record"]
    # the latent frames reached the decoder one at a time, on the thread that took the frames
    assert [latents.shape for latents in decoder.latents] == [(1, 2, 3, 4)] * 4
    assert decoder.threads == {threading.current_thread().name}
    assert torch.equal(torch.cat(decoder.latents), delivery.latents[0])
