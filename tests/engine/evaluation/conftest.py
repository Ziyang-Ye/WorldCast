"""Fixtures of the evaluation tests: synthetic rows of a window index."""

import pytest

#: Maps of the synthetic index rows.
MAPS = ("de_dust2", "de_mirage", "de_nuke", "de_ancient")


@pytest.fixture
def index_rows() -> list[dict]:
    """24 synthetic window-index rows over the four maps."""
    return [
        {
            "media_id": f"{2392000 + i}-{MAPS[i % 4]}-r{i:02d}-p0{i % 10}",
            "start_frame": 80 * i,
            "map_name": MAPS[i % 4],
            "match_id": 2392000 + i,
            "round": i,
        }
        for i in range(24)
    ]
