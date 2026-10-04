from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from src.core.errors import InvalidRequestError, PayloadTooLargeError
from src.core.logger import logger
from src.core.specs import CollectionSpec


def compute_checksum(path: Path, chunk_size: int = 1 << 20) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


class DataStore:
    """Where uploaded files live: ``DATA_DIR/<collection subdir>/<file name>``.

    Names are reduced to a plain file name (no directories, no hidden files), uploads are
    size-capped while streaming, and files appear atomically (write to a temp file, then
    rename), so a half-written upload is never visible to a reader or a directory watcher.
    """

    def __init__(self, data_dir: Path, max_bytes: int) -> None:
        self.data_dir = data_dir
        self.max_bytes = max_bytes
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def directory(self, collection: CollectionSpec) -> Path:
        directory = (self.data_dir / collection.subdir).resolve()
        if directory != self.data_dir.resolve() and self.data_dir.resolve() not in directory.parents:
            raise InvalidRequestError(f"collection '{collection.name}' resolves outside the data directory")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def safe_name(filename: str | None) -> str:
        raw = (filename or "").replace("\\", "/")
        name = Path(raw).name
        if (
            not raw.strip()
            or raw.endswith("/")  # names a directory, not a file
            or not name.strip()
            or name.startswith(".")
            or "\x00" in name
            or len(name) > 255
        ):
            raise InvalidRequestError(f"invalid file name: {filename!r}")
        return name

    def save(self, collection: CollectionSpec, filename: str | None, stream: BinaryIO) -> Path:
        """Blocking; call from a worker thread."""
        name = self.safe_name(filename)
        directory = self.directory(collection)
        descriptor, temp_name = tempfile.mkstemp(dir=directory, prefix=".upload-")
        written = 0
        try:
            with os.fdopen(descriptor, "wb") as out:
                while chunk := stream.read(1 << 20):
                    written += len(chunk)
                    if written > self.max_bytes:
                        raise PayloadTooLargeError(
                            f"upload exceeds the {self.max_bytes // (1024 * 1024)} MB limit"
                        )
                    out.write(chunk)
            destination = directory / name
            os.replace(temp_name, destination)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        logger.info("stored upload %s (%d bytes)", destination, written)
        return destination

    def copy_in(self, collection: CollectionSpec, source: Path) -> Path:
        destination = self.directory(collection) / self.safe_name(source.name)
        shutil.copy2(source, destination)
        return destination

    def within_data_dir(self, path: Path) -> bool:
        root = self.data_dir.resolve()
        resolved = path.resolve()
        return resolved == root or root in resolved.parents
