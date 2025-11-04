from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

from fastapi import UploadFile

from src.core.config import settings
from src.core.logger import logger

INDEX_STATE_FILENAME = ".index_state.json"


def compute_checksum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return SHA256 checksum for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


class IndexStateManager:
    """Persist checksum and metadata for indexed files to detect changes."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.state_path = self.data_dir / INDEX_STATE_FILENAME
        self.state: Dict[str, str] = {}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Index state file appears corrupted; starting fresh.")
                self.state = {}
        else:
            self.state = {}

    def _dump(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def needs_reindex(self, path: Path) -> bool:
        checksum = compute_checksum(path)
        recorded = self.state.get(str(path))
        return recorded != checksum

    def mark_indexed(self, path: Path) -> None:
        checksum = compute_checksum(path)
        self.state[str(path)] = checksum
        self._dump()

    def remove(self, path: Path) -> None:
        if str(path) in self.state:
            self.state.pop(str(path))
            self._dump()

    def iter_tracked_paths(self) -> Iterable[Path]:
        return (Path(p) for p in self.state)


class DataFileManager:
    """Store files and track their indexing status."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state = IndexStateManager(self.data_dir)

    def store_upload(self, upload: UploadFile) -> Path:
        destination = self.data_dir / upload.filename
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        logger.info("Stored upload at %s", destination)
        return destination

    def store_local_file(self, source: Path) -> Path:
        destination = self.data_dir / source.name
        shutil.copy2(source, destination)
        logger.info("Copied %s into data directory as %s", source, destination)
        return destination

    def file_needs_reindex(self, path: Path) -> bool:
        return self.state.needs_reindex(path)

    def mark_indexed(self, path: Path) -> None:
        self.state.mark_indexed(path)

    def purge_missing(self) -> None:
        for tracked in list(self.state.iter_tracked_paths()):
            if not tracked.exists():
                self.state.remove(tracked)


__all__ = [
    "DataFileManager",
    "IndexStateManager",
    "compute_checksum",
]
