"""Use cases. :class:`RagService` is the one place a use case is implemented; HTTP routes and the embedded
SDK are thin adapters over it (see ``docs/framework.md``, section 4)."""

from src.application.service import IngestOutcome, RagService

__all__ = ["IngestOutcome", "RagService"]
