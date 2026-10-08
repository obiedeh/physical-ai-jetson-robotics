from __future__ import annotations

from pathlib import Path

import pytest

from synria_lerobot.recording_transaction import RecordingRecoveryError, RecordingTransaction


def _transaction(tmp_path: Path) -> RecordingTransaction:
    storage = RecordingTransaction(tmp_path / "dataset")
    storage.root.mkdir()
    (storage.root / "meta").mkdir()
    (storage.root / "meta" / "info.json").write_bytes(b"original metadata")
    (storage.root / "physical_quality_records.jsonl").write_bytes(b"original records\n")
    return storage


def test_rollback_restores_metadata_and_append_lengths_without_touching_old_media(
    tmp_path: Path,
) -> None:
    storage = _transaction(tmp_path)
    media = storage.root / "videos" / "observation.images.wrist" / "chunk-000"
    media.mkdir(parents=True)
    old_media = media / "old.mp4"
    old_media.write_bytes(b"original media")
    storage.begin()
    (storage.root / "meta" / "info.json").write_bytes(b"changed metadata")
    with (storage.root / "physical_quality_records.jsonl").open("ab") as output:
        output.write(b"failed append\n")
    new_media = media / "file-001.mp4"
    new_media.write_bytes(b"new media")
    storage.rollback()
    assert (storage.root / "meta" / "info.json").read_bytes() == b"original metadata"
    assert (storage.root / "physical_quality_records.jsonl").read_bytes() == b"original records\n"
    assert old_media.read_bytes() == b"original media"
    assert not new_media.exists()
    assert not storage.journal.exists()
    storage.close()


def test_exclusive_lock_blocks_second_writer_and_releases_normally(tmp_path: Path) -> None:
    storage = _transaction(tmp_path)
    with pytest.raises(RecordingRecoveryError, match="exclusive recording lock"):
        RecordingTransaction(storage.root)
    storage.close()
    replacement = RecordingTransaction(storage.root)
    replacement.close()


@pytest.mark.parametrize("root", [Path("/"), Path.home(), Path.cwd()])
def test_broad_roots_are_refused_without_writes(root: Path) -> None:
    with pytest.raises(ValueError, match="broad directory"):
        RecordingTransaction(root)


@pytest.mark.parametrize("relative", ["../outside", "/outside", "C:/outside", "nested\\..\\x"])
def test_unsafe_journal_paths_are_refused(tmp_path: Path, relative: str) -> None:
    storage = _transaction(tmp_path)
    with pytest.raises(RecordingRecoveryError, match="unsafe"):
        storage.safe_path(relative)
    storage.close()


def test_symlink_and_changed_journal_are_refused(tmp_path: Path) -> None:
    storage = _transaction(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    linked = storage.root / "linked"
    try:
        linked.symlink_to(outside)
    except OSError:
        storage.close()
        pytest.skip("file symlinks are unavailable on this test host")
    with pytest.raises(RecordingRecoveryError, match="symlinks|escapes"):
        storage.begin()
    linked.unlink()
    storage.begin()
    storage.journal.write_text('{"files":["../outside.txt"]}', encoding="utf-8")
    with pytest.raises(RecordingRecoveryError, match="journal changed"):
        storage.rollback()
    assert outside.read_text() == "keep"
    assert storage.journal.exists()
    with pytest.raises(RecordingRecoveryError, match="retained"):
        storage.close()


@pytest.mark.parametrize("fault", ["unexpected_file", "missing_existing_file"])
def test_rollback_refuses_unrelated_changes_before_deleting_any_file(
    tmp_path: Path, fault: str
) -> None:
    storage = _transaction(tmp_path)
    storage.begin()
    new_data = storage.root / "data" / "chunk-000" / "file-001.parquet"
    new_data.parent.mkdir(parents=True)
    new_data.write_bytes(b"new data")
    if fault == "unexpected_file":
        (storage.root / "operator-note.txt").write_text("keep", encoding="utf-8")
    else:
        (storage.root / "meta" / "info.json").unlink()
    with pytest.raises(RecordingRecoveryError, match="recovery blocked"):
        storage.rollback()
    assert new_data.read_bytes() == b"new data"
    assert storage.journal.exists()


def test_unfinished_journal_blocks_a_later_session(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / ".recording_transaction.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RecordingRecoveryError, match="operator recovery"):
        RecordingTransaction(root)
