"""The round index."""

import pytest

from worldcast.data.index import RoundIndexRow


def test_round_index_rejects_bad_latent_key():
    with pytest.raises(ValueError):
        RoundIndexRow.from_row(
            {
                "media_id": "m",
                "start_frame": 80,
                "match_id": 1,
                "round": 1,
                "map_name": "de_x",
                "latent_key": "win_000000",
            }
        )
