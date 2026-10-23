"""How a client is served."""

import pytest

from worldcast.engine.inference.serving import ServingOptions


def test_options_are_checked():
    with pytest.raises(ValueError, match="decoder"):
        ServingOptions(decoder="vae")
    with pytest.raises(ValueError, match="taehv_path"):
        ServingOptions(decoder="taehv")


def test_the_decoder_runs_on_the_clients_device_unless_asked_otherwise():
    assert ServingOptions().decoder_device is None
    assert ServingOptions.low_latency("taew2_2.pth").decoder_device is None
    for device in ("cpu", "cuda", "cuda:1", "cuda:12"):
        assert ServingOptions(decoder="wan", decoder_device=device).decoder_device == device
    for refused in ("gpu1", "cuda:", "cuda:x", "cuda:1 ", "xpu:0", ""):
        with pytest.raises(ValueError, match="decoder_device must be cpu, cuda or cuda:<index>"):
            ServingOptions(decoder="wan", decoder_device=refused)
    with pytest.raises(ValueError, match="decoder='none' has none"):
        ServingOptions(decoder="none", decoder_device="cuda:1")


def test_the_channels_last_decoder_is_off_unless_asked_and_is_the_vaes():
    assert ServingOptions().decoder_channels_last is False
    assert ServingOptions.low_latency("taew2_2.pth").decoder_channels_last is False
    on = ServingOptions.low_latency(None, decoder="wan", decoder_channels_last=True)
    assert on.decoder_channels_last and on.decoder == "wan"
    for decoder in ("taehv", "none"):
        with pytest.raises(ValueError, match="decoder_channels_last is the Wan2.2 VAE's"):
            ServingOptions(decoder=decoder, taehv_path="x", decoder_channels_last=True)


def test_the_original_vae_on_a_second_card_needs_no_tiny_decoder_weights():
    wan = ServingOptions.low_latency(None, decoder="wan", decoder_device="cuda:1")
    assert (wan.decoder, wan.decoder_device, wan.compile, wan.lockstep) == (
        "wan",
        "cuda:1",
        True,
        False,
    )
    with pytest.raises(ValueError, match="taehv_path"):
        ServingOptions.low_latency(None)
    with pytest.raises(TypeError):
        ServingOptions.low_latency("taew2_2.pth", speed=2)


def test_low_latency_is_the_web_demos_setting():
    paper, fast = ServingOptions(), ServingOptions.low_latency("taew2_2.pth")
    changed = {name for name in vars(fast) if getattr(fast, name) != getattr(paper, name)}
    assert changed == {"compile", "decoder", "taehv_path", "lockstep"}
    assert (fast.decoder, fast.taehv_path, fast.lockstep) == ("taehv", "taew2_2.pth", False)
    assert not ServingOptions.low_latency("taew2_2.pth", compile=False).compile
