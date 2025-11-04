from __future__ import annotations

import threading
from pathlib import Path
from queue import Queue
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.core.config import settings
from src.core.logger import logger


class _ReindexEventHandler(FileSystemEventHandler):
    def __init__(self, enqueue: Callable[[Path], None]) -> None:
        super().__init__()
        self.enqueue = enqueue

    def on_created(self, event):  # noqa: N802
        if event.is_directory:
            return
        self.enqueue(Path(event.src_path))

    def on_modified(self, event):  # noqa: N802
        if event.is_directory:
            return
        self.enqueue(Path(event.src_path))

    def on_moved(self, event):  # noqa: N802
        if event.is_directory:
            return
        self.enqueue(Path(event.dest_path))


class DataDirectoryWatcher:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.observer = Observer()
        self.queue: "Queue[Path]" = Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.ingestion_callback: Callable[[Path], None] | None = None

    def _worker(self) -> None:
        while True:
            path = self.queue.get()
            if path is None:  # sentinel for shutdown
                break
            if self.ingestion_callback:
                try:
                    self.ingestion_callback(path)
                except Exception as exc:  # pragma: no cover
                    logger.exception("Failed to reindex %s: %s", path, exc)
            self.queue.task_done()

    def start(self, ingestion_callback: Callable[[Path], None]) -> None:
        self.ingestion_callback = ingestion_callback
        handler = _ReindexEventHandler(self.queue.put)
        self.observer.schedule(handler, str(self.data_dir), recursive=False)
        self.observer.start()
        if not self.thread.is_alive():
            self.thread.start()
        logger.info("Started DataDirectoryWatcher on %s", self.data_dir)

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join(timeout=5)
        self.queue.put(None)
        self.thread.join(timeout=5)
        logger.info("Stopped DataDirectoryWatcher")


__all__ = ["DataDirectoryWatcher"]
