from __future__ import annotations

import asyncio
from pathlib import Path

from src.ingestion.watcher import DataDirectoryWatcher

EXTENSIONS = frozenset({".pdf", ".txt"})


async def run_watcher(directory: Path, debounce: float = 0.4):
    submitted: list[Path] = []

    async def submit(path: Path) -> None:
        submitted.append(path)

    watcher = DataDirectoryWatcher(directory, EXTENSIONS, submit, debounce)
    await watcher.start()
    return watcher, submitted


async def settle(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def test_a_new_file_is_submitted_once_after_the_debounce(tmp_path):
    watcher, submitted = await run_watcher(tmp_path)
    (tmp_path / "doc.txt").write_text("hello")
    await settle(0.2)
    assert submitted == [], "not before the quiet period"
    await settle(0.8)
    assert submitted == [tmp_path / "doc.txt"]
    await watcher.stop()


async def test_a_file_being_written_in_pieces_is_submitted_once_when_it_goes_quiet(tmp_path):
    watcher, submitted = await run_watcher(tmp_path, debounce=0.5)
    with (tmp_path / "big.pdf").open("wb") as handle:
        for _ in range(8):
            handle.write(b"x" * 1000)
            handle.flush()
            await settle(0.15)  # each write restarts the quiet period
    await settle(1.0)
    assert submitted == [tmp_path / "big.pdf"], "one submission, after the last write"
    await watcher.stop()


async def test_hidden_temporary_and_unsupported_files_are_ignored(tmp_path):
    watcher, submitted = await run_watcher(tmp_path)
    (tmp_path / ".upload-abc123").write_text("partial")
    (tmp_path / ".hidden.txt").write_text("secret")
    (tmp_path / "program.exe").write_text("MZ")
    (tmp_path / "sub").mkdir()
    await settle(1.0)
    assert submitted == []
    await watcher.stop()


async def test_an_atomic_rename_into_place_is_picked_up_once(tmp_path):
    """The upload path writes a hidden temp file and renames it: only the final name counts."""
    watcher, submitted = await run_watcher(tmp_path)
    temp = tmp_path / ".upload-xyz"
    temp.write_text("content")
    temp.replace(tmp_path / "final.txt")
    await settle(1.0)
    assert submitted == [tmp_path / "final.txt"]
    await watcher.stop()


async def test_files_deleted_before_the_quiet_period_ends_are_not_submitted(tmp_path):
    watcher, submitted = await run_watcher(tmp_path)
    (tmp_path / "gone.txt").write_text("x")
    await settle(0.1)
    (tmp_path / "gone.txt").unlink()
    await settle(0.9)
    assert submitted == []
    await watcher.stop()
