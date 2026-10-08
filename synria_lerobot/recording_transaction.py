"""Exclusive, bounded rollback for one closed-boundary recording append."""

from __future__ import annotations

import base64
import json
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

MUTABLE_METADATA = ("meta/info.json", "meta/stats.json", "meta/tasks.parquet")
APPEND_SIDECARS = (
    "physical_episode_metadata.jsonl",
    "physical_quality_records.jsonl",
    "physical_capture_provenance.jsonl",
)
JOURNAL_NAME = ".recording_transaction.json"


class RecordingRecoveryError(RuntimeError):
    """The preserved journal needs operator review before any further append."""


class RecordingTransaction:
    """Track metadata and new paths without copying existing dataset media."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise ValueError("dataset root must not be a symlink")
        self.root = root.resolve()
        if self.root in {
            Path(self.root.anchor), Path.home().resolve(), Path.cwd().resolve(),
            Path(__file__).resolve().parents[1],
        }:
            raise ValueError("refusing a broad directory as a recording dataset root")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        # A sibling lock can be acquired before upstream creates a new root.
        self._lock = self.root.parent / f".{self.root.name}.recording.lock"
        self._token = uuid.uuid4().hex
        self._closed = False
        self._snapshot: dict[str, Any] | None = None
        self._journal_text: str | None = None
        if self.journal.exists() or self.journal.is_symlink():
            raise RecordingRecoveryError("unfinished recording journal requires operator recovery")
        try:
            with self._lock.open("x", encoding="utf-8") as output:
                output.write(self._token)
        except FileExistsError as error:
            raise RecordingRecoveryError(
                "dataset already has an exclusive recording lock"
            ) from error

    @property
    def journal(self) -> Path:
        return self.root / JOURNAL_NAME

    @property
    def pending(self) -> bool:
        return self._snapshot is not None or self.journal.exists() or self.journal.is_symlink()

    def safe_path(self, relative: str) -> Path:
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts or any(
            character in relative for character in ("\\", ":")
        ):
            raise RecordingRecoveryError("unsafe recording journal path")
        path = self.root.joinpath(*pure.parts)
        if path == self.root or self.root not in path.resolve().parents:
            raise RecordingRecoveryError("recording path escapes dataset root")
        for ancestor in (path, *path.parents):
            if ancestor == self.root:
                break
            if ancestor.is_symlink():
                raise RecordingRecoveryError("recording paths must not contain symlinks")
        return path

    def files(self) -> set[str]:
        files = set()
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root).as_posix()
            self.safe_path(relative)
            if path.is_file():
                files.add(relative)
        return files

    def begin(self) -> None:
        if self.pending:
            raise RecordingRecoveryError("unfinished recording transaction blocks another save")
        files = self.files()
        self._snapshot = {
            "files": sorted(files),
            "metadata": {
                name: base64.b64encode(self.safe_path(name).read_bytes()).decode("ascii")
                if name in files else None
                for name in MUTABLE_METADATA
            },
            "sidecar_lengths": {
                name: self.safe_path(name).stat().st_size if name in files else 0
                for name in APPEND_SIDECARS
            },
        }
        self._journal_text = json.dumps(self._snapshot, sort_keys=True) + "\n"
        with self.journal.open("x", encoding="utf-8") as output:
            output.write(self._journal_text)

    def _require_owned_journal(self) -> dict[str, Any]:
        self.safe_path(JOURNAL_NAME)
        if self._snapshot is None or self.journal.read_text(encoding="utf-8") != self._journal_text:
            raise RecordingRecoveryError("recording journal changed; recovery is blocked")
        return self._snapshot

    def commit(self) -> None:
        self._require_owned_journal()
        self.journal.unlink()
        self._snapshot = None
        self._journal_text = None

    def rollback(self) -> None:
        snapshot = self._require_owned_journal()
        baseline = set(snapshot["files"])
        for name in baseline:
            self.safe_path(name)
        current = self.files()
        if baseline - current:
            raise RecordingRecoveryError(
                "a preexisting recording file is missing; recovery blocked"
            )
        new_files = current - baseline - {JOURNAL_NAME}
        for name in new_files:
            owned = name in (*MUTABLE_METADATA, *APPEND_SIDECARS) or any(
                re.fullmatch(pattern, name) for pattern in (
                    r"(?:data|meta/episodes)/chunk-\d+/file-\d+\.parquet",
                    r"videos/observation\.images\.(?:wrist|front)/chunk-\d+/file-\d+\.mp4",
                    r"images/observation\.images\.(?:wrist|front)/episode-\d+/frame-\d+\.png",
                    r"final_stills/episode_\d+_\d+\.jpg",
                )
            )
            if not owned:
                raise RecordingRecoveryError("unexpected new file retained; recovery blocked")
        for name in sorted(new_files):
            self.safe_path(name).unlink()
        for name in MUTABLE_METADATA:
            encoded = snapshot["metadata"][name]
            if encoded is not None:
                self.safe_path(name).write_bytes(base64.b64decode(encoded, validate=True))
        for name in APPEND_SIDECARS:
            if name in baseline:
                path = self.safe_path(name)
                length = snapshot["sidecar_lengths"][name]
                if not path.is_file() or path.stat().st_size < length:
                    raise RecordingRecoveryError("existing recording sidecar was shortened")
                with path.open("r+b") as output:
                    output.truncate(length)
        self.commit()

    def close(self) -> None:
        if self._closed:
            return
        if self.pending:
            raise RecordingRecoveryError(
                "unfinished recording journal and lock retained for recovery"
            )
        if self._lock.is_symlink() or self._lock.read_text(encoding="utf-8") != self._token:
            raise RecordingRecoveryError("exclusive recording lock changed; refusing to remove it")
        self._lock.unlink()
        self._closed = True
