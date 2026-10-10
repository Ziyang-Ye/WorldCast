"""A session: the clients of a round run together, each in its own process and on a GPU.

Never threads: the scene state's numpy reprojections are not bit-reproducible across threads of one
process (docs/inference.md, "One client, many clients"). A client that fails marks itself failed in
the shared world state, and the other clients of its round stop with an error.
"""

import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from worldcast.data.latents import BLOCK
from worldcast.data.recordings import RoundIndexRow

__all__ = [
    "CLIENT_ENV",
    "CLIENT_MEMORY_GIB",
    "MAX_ROUND_SECONDS",
    "ROUND_STEP_SECONDS",
    "check_gpu_memory",
    "gpu_memory",
    "place_clients",
    "round_latents",
    "round_name",
    "rounds_of",
    "run_round",
    "visible_gpus",
]

#: The environment of every client process, as the reference runs set it: a fixed hash seed, four
#: BLAS threads per client (the scene state's reprojections are numpy on the CPU) and unbuffered
#: output for ``client.log``. A variable the caller sets wins: the thread count changes the speed,
#: not the latents (they equal the reference runs' with 1 thread as with 4), and it is the caller's
#: to fit to the machine's cores.
CLIENT_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTHONUNBUFFERED": "1",
}
#: The GPU memory a client takes, GiB (14.9 GiB in use on an H20, for 81 to 241 latent frames).
CLIENT_MEMORY_GIB = 15
#: Table 3 ran each round for its recorded length, rounded down to a multiple of
#: ``ROUND_STEP_SECONDS`` and at most ``MAX_ROUND_SECONDS``.
ROUND_STEP_SECONDS, MAX_ROUND_SECONDS = 10, 120


def rounds_of(rows: Sequence[RoundIndexRow]) -> list[list[int]]:
    """The index rows of each round (the rows of its clients, sharing ``start_frame``), in order.

    Raises:
        ValueError: the rows of a round do not cover its clients.
    """
    rounds: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        key = (tuple(sorted(row.clients or (row.media_id,))), row.start_frame)
        rounds.setdefault(key, []).append(i)
    for (clients, _), index in rounds.items():
        if sorted(rows[i].media_id for i in index) != list(clients):
            raise ValueError(f"index rows {index} do not cover their round {list(clients)}")
    return list(rounds.values())


def round_latents(round_seconds: int, max_seconds: int = MAX_ROUND_SECONDS) -> int:
    """``run.latent_frames`` of a round of ``round_seconds`` as the paper ran it: one block per
    second."""
    seconds = min(int(round_seconds), int(max_seconds))
    return 1 + BLOCK * (seconds // ROUND_STEP_SECONDS * ROUND_STEP_SECONDS)


def round_name(row: RoundIndexRow) -> str:
    """``<match_id>-<map_name>-r<round>-s<start_frame>``: a round's output directory."""
    return f"{row.match_id}-{row.map_name}-r{row.round:02d}-s{row.start_frame:06d}"


def visible_gpus() -> list[str]:
    """The GPUs this process sees, as a client's ``CUDA_VISIBLE_DEVICES`` names them: the entries
    of the caller's ``CUDA_VISIBLE_DEVICES`` where it is set, else ``0, 1, ...``."""
    import torch

    count = torch.cuda.device_count()
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if listed is None:
        return [str(i) for i in range(count)]
    return [gpu.strip() for gpu in listed.split(",")][:count]


def place_clients(clients: int, gpus: Sequence[str]) -> list[str]:
    """The GPU of each of ``clients`` clients: ``gpus`` in turn, so that a GPU runs two clients (or
    more) where there are fewer GPUs than clients (:func:`check_gpu_memory`)."""
    return [gpus[i % len(gpus)] for i in range(clients)]


def gpu_memory() -> dict[str, float]:
    """The memory of each GPU of the machine, GiB, by the ids ``CUDA_VISIBLE_DEVICES`` can name it
    with, its index and its UUID, as ``nvidia-smi`` lists them (no CUDA context in this process);
    empty without ``nvidia-smi``. Its indices are in PCI bus order, as CUDA's are on a machine of
    one GPU model (or with ``CUDA_DEVICE_ORDER=PCI_BUS_ID``)."""
    query = ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"]
    try:
        listed = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    memory = {}
    for line in listed.splitlines():
        index, uuid, mib = (field.strip() for field in line.split(","))
        if mib.isdigit():  # not "[N/A]"
            memory[index] = memory[uuid] = int(mib) / 1024
    return memory


def check_gpu_memory(gpus: Sequence[str], memory: Mapping[str, float]) -> None:
    """Raise ``ValueError`` naming each GPU whose clients need more memory than it has, at
    :data:`CLIENT_MEMORY_GIB` a client: they would fail after loading their weights.

    Args:
        gpus (Sequence[str]): the GPU of each client.
        memory (Mapping[str, float]): GiB by GPU (:func:`gpu_memory`); a GPU it does not list is
            not checked.
    """
    short = [
        f"GPU {gpu} has {memory[gpu]:.0f} GiB for {clients} client{'s' if clients > 1 else ''}"
        for gpu, clients in Counter(gpus).items()
        if gpu in memory and clients * CLIENT_MEMORY_GIB > memory[gpu]
    ]
    if short:
        raise ValueError(
            f"{'; '.join(short)}: a client takes about {CLIENT_MEMORY_GIB} GiB, place them on more"
            " GPUs (--gpus, CUDA_VISIBLE_DEVICES)"
        )


def run_round(
    command: Sequence[str],
    index: Sequence[int],
    rows: Sequence[RoundIndexRow],
    *,
    gpus: Sequence[str],
    out_dir: Path,
) -> list[str]:
    """Run the clients of the index rows ``index`` together and wait for them.

    Args:
        command (Sequence[str]): the command of one client (``tools/run_client.py`` with its
            ``--config`` and ``--set`` arguments); each client's ``--index-row``,
            ``--world-state-dir`` and ``--out-dir`` are appended.
        index (Sequence[int]): the clients' rows of the round index.
        rows (Sequence[RoundIndexRow]): the round index.
        gpus (Sequence[str]): one GPU id per client.
        out_dir (Path): receives ``world_state/`` and, per client,
            ``<media_id>/{latents.npy,client.json,client.log}``.

    Returns:
        list[str]: the media ids of the clients that failed.
    """
    if len(gpus) < len(index):
        raise ValueError(
            f"{len(index)} clients run at once and need a GPU id each, got {len(gpus)}:"
            f" {','.join(gpus)}"
        )
    world_state = Path(out_dir) / "world_state"
    if world_state.exists() and any(world_state.iterdir()):
        raise ValueError(f"{world_state} is not empty: every session needs a fresh one")
    world_state.mkdir(parents=True, exist_ok=True)
    clients = []
    for i, gpu in zip(index, gpus):
        client_dir = Path(out_dir) / rows[i].media_id
        client_dir.mkdir(parents=True, exist_ok=True)
        log = open(client_dir / "client.log", "w")
        process = subprocess.Popen(
            [*command, "--index-row", str(i), "--world-state-dir", str(world_state)]
            + ["--out-dir", str(client_dir)],
            env={**CLIENT_ENV, **os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        clients.append((rows[i].media_id, process, log))
    failed = []
    for media_id, process, log in clients:
        if process.wait() != 0:
            failed.append(media_id)
        log.close()
    return failed
