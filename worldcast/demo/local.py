"""The whole demo in one process: a coordinator and its workers on this machine.

For a laptop, with the mock client. The WorldCast client runs one per process: start its workers
with ``--role worker``, one per GPU.
"""

import asyncio
import logging
import socket
from collections.abc import Sequence
from dataclasses import replace

from aiohttp import web

from .clients import ClientProtocol, make_client
from .config import DemoConfig
from .coordinator import Coordinator
from .library import Library
from .worker import Worker

__all__ = ["LOOPBACK", "LocalDemo", "listen", "local_config", "run"]

log = logging.getLogger(__name__)
#: The address the coordinator and the workers of one machine reach each other at.
LOOPBACK = "127.0.0.1"


def listen(host: str, port: int) -> socket.socket:
    """A socket bound to ``host:port`` (port 0: any free port)."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    return sock


class LocalDemo:
    """A coordinator and one worker per client, served from this process.

    The coordinator listens on ``coordinator.host`` and ``coordinator.port``, worker ``i`` on the
    loopback port ``worker.port + i`` (a port 0: any free port). Use it as an async context manager.

    Args:
        config (DemoConfig): the demo's settings.
        library (Library): the round starts.
        clients (Sequence[ClientProtocol]): one client per worker.
    """

    def __init__(
        self, config: DemoConfig, library: Library, clients: Sequence[ClientProtocol]
    ) -> None:
        self.cfg, self.library, self.clients = config, library, list(clients)
        self.coordinator: Coordinator | None = None
        self.workers: list[Worker] = []
        self.url = ""
        self._runners: list[web.AppRunner] = []

    async def _serve(self, app: web.Application, sock: socket.socket) -> None:
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        self._runners.append(runner)
        await web.SockSite(runner, sock).start()

    async def __aenter__(self) -> "LocalDemo":
        cfg = self.cfg
        sock = listen(cfg.coordinator.host, cfg.coordinator.port)
        port = sock.getsockname()[1]
        cfg = replace(cfg, coordinator=replace(cfg.coordinator, port=port))
        self.coordinator = Coordinator(cfg, self.library)
        self.url = f"http://{LOOPBACK}:{port}"
        await self._serve(self.coordinator.app(), sock)
        for i, client in enumerate(self.clients):
            sock = listen(LOOPBACK, cfg.worker.port + i if cfg.worker.port else 0)
            worker = replace(
                cfg.worker,
                host=LOOPBACK,
                port=sock.getsockname()[1],
                coordinator_url=f"ws://{LOOPBACK}:{port}",
                advertise_url=None,
                worker_id=f"local-{i + 1}",
            )
            self.workers.append(Worker(replace(cfg, worker=worker), client, self.library))
            await self._serve(self.workers[-1].app(), sock)
        return self

    async def __aexit__(self, *error: object) -> None:
        for runner in reversed(self._runners):
            await runner.cleanup()


def local_config(config: DemoConfig, workers: int) -> DemoConfig:
    """``config`` as the local role serves it with ``workers`` workers.

    Browsers reach the workers through the coordinator's port (``coordinator.frame_route: proxy``),
    so another device on the network can join too. Each worker's own ``worker`` section (loopback,
    port ``worker.port + i``) is set by :class:`LocalDemo`.

    Raises:
        ValueError: more than one WorldCast client, which needs a process of its own.
    """
    if config.client.kind == "worldcast" and workers > 1:
        raise ValueError(
            "--role local serves every worker from this process, and a WorldCast client needs its"
            " own: use --workers 1, or start one --role worker per GPU"
        )
    return replace(config, coordinator=replace(config.coordinator, frame_route="proxy"))


def run(config: DemoConfig, workers: int) -> None:
    """Serve a coordinator and ``workers`` workers until interrupted.

    Args:
        config (DemoConfig): the settings, as :func:`local_config` returns them.
        workers (int): the workers to start, one client each.
    """
    library = Library.load(config.library)
    clients = [make_client(config, library) for _ in range(workers)]

    async def serve() -> None:
        async with LocalDemo(config, library, clients) as demo:
            log.info(
                "WorldCast Live: %s (%d workers, %s client)", demo.url, workers, config.client.kind
            )
            await asyncio.Event().wait()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
