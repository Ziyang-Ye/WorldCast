"""How a client is served."""

import pytest

from worldcast.engine.inference.serving import DECODERS, ServingOptions


def test_options_are_checked():
    assert DECODERS == ("wan", "none")
    with pytest.raises(ValueError, match="decoder"):
        ServingOptions(decoder="vae")
