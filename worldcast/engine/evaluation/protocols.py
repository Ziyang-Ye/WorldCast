"""The evaluation protocols and their sealed window selection.

Both protocols score the 64 ten-second windows the training runs validated on, 16 per map, drawn
from the held-out matches (``validation.index``): GT-paired PSNR, SSIM, LPIPS, pixel and latent MSE
of every frame after the first. They differ in the sampler:

* ``unipc``: 20 UniPC steps, the whole window at once for a bidirectional model (stages 1_long, 2
  and 2s), block by block for the teacher-forced one (stage 3); the in-training validation;
* ``four_step``: the four denoising steps of the four-step generator (stage 4; stage 3 too),
  block by block.

The windows' order and each window's noise follow from the protocol seed alone, so a score does not
depend on how the windows are spread over GPUs.
"""

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldcast.sampling.sampler import DENOISING_STEPS
from worldcast.utils.files import sha256_file

__all__ = [
    "FOUR_STEP",
    "PROTOCOLS",
    "UNIPC",
    "WINDOW_SET",
    "EvalWindow",
    "Protocol",
    "check_selection",
    "load_windows",
    "read_index",
    "select_windows",
    "selection_sha256",
    "window_noise_seed",
]

#: The window set of both protocols, as a result row names it.
WINDOW_SET = "worldcast-maps4-eval64-v1"


@dataclass(frozen=True)
class Protocol:
    """One evaluation protocol.

    Attributes:
        sampler (str): ``unipc`` or ``four_step``.
        denoising_steps (int): generator calls per block (per window for a bidirectional model).
        tf32 (bool): TF32 for fp32 CUDA matmuls while scoring.
        window_set (str): the name of the windows.
        index_sha256 (str): sha256 of the window index file.
        selection_sha256 (str): sha256 of the selected ``[[media_id, start_frame], ...]``.
        count (int): windows scored.
        seed (int): orders the windows and seeds each window's noise.
    """

    sampler: str
    denoising_steps: int
    tf32: bool
    window_set: str = WINDOW_SET
    index_sha256: str = "a15580a409d02b580e7220073e1ff5f037724c49e749b86905e335fe80030c49"
    selection_sha256: str = "44feaaa89dba69b17d5d303de6543dd24e85f0cd72118f812d1295e236f7b9b5"
    count: int = 64
    seed: int = 20260829

    def fields(self) -> dict[str, Any]:
        """The protocol fields of a result row."""
        return {
            "window_set": self.window_set,
            "sampler": self.sampler,
            "denoising_steps": self.denoising_steps,
            "index_sha256": self.index_sha256,
            "selection_sha256": self.selection_sha256,
            "seed": self.seed,
        }


UNIPC = Protocol(sampler="unipc", denoising_steps=20, tf32=True)
FOUR_STEP = Protocol(sampler="four_step", denoising_steps=len(DENOISING_STEPS), tf32=False)
#: The protocols by their sampler.
PROTOCOLS = {protocol.sampler: protocol for protocol in (UNIPC, FOUR_STEP)}


@dataclass(frozen=True)
class EvalWindow:
    """One selected window.

    Attributes:
        index (int): position in the selection (0-63).
        media_id (str): the client's recording.
        start_frame (int): first source frame.
        map_name (str): the map.
        stratum (str): ``vis<n>`` of the index row's ``num_visible_players`` (else its
            ``quality``), ``vis?`` for a row with neither.
        noise_seed (int): seed of the window's noise generator.
        record (dict): the index row.
    """

    index: int
    media_id: str
    start_frame: int
    map_name: str
    stratum: str
    noise_seed: int
    record: dict


def read_index(path: str | Path, protocol: Protocol) -> list[dict]:
    """The rows of the window index, checked against the protocol's sha256."""
    path = Path(path)
    observed = sha256_file(path)
    if observed != protocol.index_sha256:
        raise ValueError(
            f"{path}: sha256 {observed} is not the index of {protocol.window_set}"
            f" ({protocol.index_sha256})"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_windows(
    path: str | Path, protocol: Protocol, index_sha256: str | None = None
) -> tuple[Protocol, list[EvalWindow]]:
    """The protocol's windows of the index at ``path`` and the protocol they are scored under.

    The paper's index has the sha256 and the selection the protocol pins. An index you rebuilt
    names its own sha256 (``validation.index_sha256``): the selection that index determines is then
    taken as drawn, and the returned protocol holds both digests, for the result row.

    Args:
        path (str | Path): the window index.
        protocol (Protocol): the protocol.
        index_sha256 (str | None): the sha256 of a rebuilt index; ``None``: the protocol's.

    Returns:
        tuple[Protocol, list[EvalWindow]]: the protocol and its selected windows.
    """
    if index_sha256 in (None, protocol.index_sha256):
        windows = select_windows(read_index(path, protocol), protocol)
        check_selection(windows, protocol)
        return protocol, windows
    protocol = dataclasses.replace(protocol, index_sha256=index_sha256)
    windows = select_windows(read_index(path, protocol), protocol)
    return dataclasses.replace(protocol, selection_sha256=selection_sha256(windows)), windows


def window_noise_seed(seed: int, media_id: str, start_frame: int) -> int:
    """The seed of a window's noise generator: the first 8 bytes of
    ``sha256(f"{seed}\\0{media_id}@{start_frame}")``, modulo ``2^63 - 1``."""
    digest = hashlib.sha256(f"{int(seed)}\0{media_id}@{int(start_frame)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def selection_sha256(windows: Sequence[EvalWindow]) -> str:
    """sha256 of the compact JSON ``[[media_id, start_frame], ...]`` of a selection."""
    payload = [[w.media_id, w.start_frame] for w in windows]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def select_windows(rows: Sequence[Mapping[str, Any]], protocol: Protocol) -> list[EvalWindow]:
    """The protocol's windows: the rows ordered by ``sha256(f"{seed}\\0{media_id}\\0{start}")``
    (one per ``(media_id, start_frame)``), the first ``count``.

    Args:
        rows (Sequence[Mapping]): the index rows (:func:`read_index`).
        protocol (Protocol): the protocol.
    """
    unique: dict[tuple[str, int], dict] = {}
    for row in rows:
        unique.setdefault((str(row["media_id"]), int(row["start_frame"])), dict(row))
    if len(unique) < protocol.count:
        raise ValueError(
            f"the index holds {len(unique)} windows, the protocol needs {protocol.count}"
        )

    def rank(identity: tuple[str, int]) -> str:
        media_id, start = identity
        return hashlib.sha256(f"{protocol.seed}\0{media_id}\0{start}".encode()).hexdigest()

    ordered = sorted(unique, key=lambda identity: (rank(identity), *identity))[: protocol.count]
    windows = []
    for i, (media_id, start) in enumerate(ordered):
        record = unique[(media_id, start)]
        stratum = record.get("num_visible_players", record.get("quality"))
        windows.append(
            EvalWindow(
                index=i,
                media_id=media_id,
                start_frame=start,
                map_name=str(record["map_name"]),
                stratum="vis?" if stratum is None else f"vis{stratum}",
                noise_seed=window_noise_seed(protocol.seed, media_id, start),
                record=record,
            )
        )
    return windows


def check_selection(windows: Sequence[EvalWindow], protocol: Protocol) -> None:
    """Raise unless ``windows`` are the protocol's sealed selection."""
    observed = selection_sha256(windows)
    if observed != protocol.selection_sha256:
        raise ValueError(
            f"the selected windows are not those of {protocol.window_set} (selection sha256"
            f" {observed}, expected {protocol.selection_sha256})"
        )
