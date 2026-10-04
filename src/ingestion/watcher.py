from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.core.logger import logger


class _Handler(FileSystemEventHandler):
    def __init__(self, notify: Callable[[Path], None]) -> None:
        self._notify = notify

    def _handle(self, event: FileSystemEvent, path: str | bytes) -> None:
        if not event.is_directory:
            self._notify(Path(path.decode() if isinstance(path, bytes) else path))

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event, event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event, event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle(event, event.dest_path)


class DataDirectoryWatcher:
    """Submits files that appear in (or change under) one directory.

    Events are debounced - a file is submitted only after it has been quiet for ``debounce``
    seconds, so a large copy in progress is never read half-written - and files that are not
    ingestible or are temporary (leading dot) are ignored. Submitting is idempotent downstream:
    an unchanged file is skipped by the ingestion ledger.
    """

    def __init__(
        self,
        directory: Path,
        extensions: frozenset[str],
        submit: Callable[[Path], Awaitable[None]],
        debounce: float,
    ) -> None:
        self._directory = directory
        self._extensions = extensions
        self._submit = submit
        self._debounce = debounce
        self._observer = Observer()
        self._pending: dict[Path, asyncio.TimerHandle] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._observer.schedule(_Handler(self._from_thread), str(self._directory), recursive=False)
        self._observer.start()
        logger.info("watching %s", self._directory)

    def _from_thread(self, path: Path) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._touch, path)

    def _touch(self, path: Path) -> None:
        if path.name.startswith(".") or path.suffix.lower() not in self._extensions:
            return
        assert self._loop is not None
        previous = self._pending.pop(path, None)
        if previous is not None:
            previous.cancel()
        self._pending[path] = self._loop.call_later(self._debounce, self._fire, path)

    def _fire(self, path: Path) -> None:
        self._pending.pop(path, None)
        if not path.is_file():
            return
        task = asyncio.ensure_future(self._submit(path))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._log_failure)

    @staticmethod
    def _log_failure(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.error("watcher failed to submit a file", exc_info=exc)

    async def stop(self) -> None:
        for handle in self._pending.values():
            handle.cancel()
        self._pending.clear()
        if self._observer.is_alive():
            self._observer.stop()
            await asyncio.to_thread(self._observer.join, 5)
        for task in list(self._tasks):
            task.cancel()
