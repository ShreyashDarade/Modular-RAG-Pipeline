from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

from src.ports.parsing import OcrEngine, OcrResult


class LazyOcr:
    """Builds the (heavy, model-loading) OCR engine on first use - API replicas that never
    ingest never pay for it - and bounds how many images are processed at once."""

    def __init__(self, factory: Callable[[], OcrEngine], concurrency: int) -> None:
        self._factory = factory
        self._engine: OcrEngine | None = None
        self._build_lock = threading.Lock()
        self._slots = asyncio.Semaphore(concurrency)

    def _get(self) -> OcrEngine:
        with self._build_lock:
            if self._engine is None:
                self._engine = self._factory()
            return self._engine

    async def read(self, image: bytes, language_hint: str | None) -> OcrResult:
        async with self._slots:
            return await asyncio.to_thread(lambda: self._get().read(image, language_hint))

    async def close(self) -> None:
        if self._engine is not None:
            await asyncio.to_thread(self._engine.close)
