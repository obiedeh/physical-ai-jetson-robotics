"""Maintain a rebuildable local session catalog beside recorder-owned datasets."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synria_lerobot.exclusions import append_exclusion, read_exclusions
from synria_lerobot.recording_transaction import JOURNAL_NAME


def utc_now() -> str:
    """Use timezone-aware wall time for audit facts, never capture scheduling."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    """Encode strict JSON so catalog facts remain portable and inspectable."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    """Read unique committed sidecar rows; partial or duplicate evidence is refused."""
    if path.is_symlink():
        raise ValueError("dataset sidecars must not be symlinks")
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        index = value.get("episode_index")
        if type(index) is not int or index < 0 or index in result:
            raise ValueError("invalid or duplicate sidecar episode index")
        result[index] = value
    return result


class Catalog:
    """Catalog disk facts and append audit events without taking dataset ownership."""

    def __init__(
        self, workspace: Path, *, repository: Path | None = None, backup: bool = True
    ) -> None:
        """Open one thread-safe WAL database outside git and migrate its schema forward."""
        self.workspace = workspace.resolve()
        self.repository = (repository or Path(__file__).resolve().parents[2]).resolve()
        if (
            workspace.is_symlink()
            or self.workspace.is_relative_to(self.repository)
            or any(
                (parent / ".git").exists() for parent in (self.workspace, *self.workspace.parents)
            )
        ):
            raise ValueError("console workspace must be outside every git tree and not a symlink")
        if self.workspace in {Path(self.workspace.anchor), Path.home().resolve()}:
            raise ValueError("console workspace must be a dedicated directory")
        self.workspace.mkdir(parents=True, exist_ok=True)
        if any(
            (self.workspace / name).is_symlink()
            for name in (
                "console.sqlite3",
                "console.sqlite3-wal",
                "console.sqlite3-shm",
                "backups",
                "sessions",
                "sessions-demo",
                "trash",
            )
        ):
            raise ValueError("console-owned workspace paths must not be symlinks")
        self._lock = threading.RLock()
        self._busy: set[str] = set()
        self._recovery_blocks: dict[str, str] = {}
        self.connection = sqlite3.connect(
            self.workspace / "console.sqlite3", check_same_thread=False
        )
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA journal_mode=WAL")
            self._migrate()
            if backup:
                self.backup()
        except BaseException:
            self.connection.close()
            raise

    def _migrate(self) -> None:
        """Create version one without guessing how to downgrade a newer database."""
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise ValueError("console catalog schema is newer than this application")
        if version == 1:
            return
        self.connection.executescript("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, slug TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL, source_kind TEXT NOT NULL, task_id TEXT NOT NULL,
                task_definition_sha256 TEXT NOT NULL, recording_purpose TEXT NOT NULL,
                dataset_path TEXT NOT NULL, repo_id TEXT NOT NULL, settings_json TEXT NOT NULL,
                operator TEXT NOT NULL, scene TEXT NOT NULL, follower_serial TEXT NOT NULL,
                leader_serial TEXT NOT NULL, power_state_start TEXT NOT NULL,
                power_state_end TEXT NOT NULL, target_episodes INTEGER NOT NULL,
                notes TEXT NOT NULL,
                git_sha TEXT NOT NULL, host TEXT NOT NULL, created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL, closed_utc TEXT, trashed_utc TEXT, trash_reason TEXT,
                tombstone_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE capture_runs (
                id INTEGER PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id),
                started_utc TEXT NOT NULL, ended_utc TEXT, git_sha TEXT, host TEXT,
                measured_state_rate REAL, callback_count INTEGER, maximum_gap REAL,
                preflight_json TEXT NOT NULL, exit_reason TEXT
            );
            CREATE TABLE episodes (
                session TEXT NOT NULL REFERENCES sessions(id), episode_index INTEGER NOT NULL,
                capture_run INTEGER REFERENCES capture_runs(id), operator_label TEXT NOT NULL,
                started_utc TEXT, duration_s REAL NOT NULL, frame_count INTEGER NOT NULL,
                requested_fps REAL NOT NULL, achieved_sample_rate_hz REAL, final_still_path TEXT,
                gate_passed INTEGER, failed_gates_json TEXT NOT NULL, excluded INTEGER NOT NULL,
                exclusion_reason_code TEXT, exclusion_note TEXT, excluded_utc TEXT,
                note TEXT NOT NULL DEFAULT '', PRIMARY KEY(session,episode_index)
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY, utc TEXT NOT NULL, session TEXT REFERENCES sessions(id),
                episode_index INTEGER, kind TEXT NOT NULL, detail_json TEXT NOT NULL
            );
            CREATE TRIGGER events_no_update BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
            CREATE TRIGGER events_no_delete BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
            PRAGMA user_version=1;
        """)
        self.connection.commit()

    def close(self) -> None:
        """Flush catalog work without modifying recorder-owned files or locks."""
        with self._lock:
            self.connection.close()

    def backup(self) -> Path:
        """Create a consistent SQLite backup and retain the newest five copies."""
        with self._lock:
            root = self.workspace / "backups"
            root.mkdir(exist_ok=True)
            path = (
                root
                / f"console-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{uuid.uuid4().hex}.sqlite3"
            )
            target = sqlite3.connect(path)
            try:
                self.connection.backup(target)
            finally:
                target.close()
            for old in sorted(root.glob("console-*.sqlite3"))[:-5]:
                old.unlink()
            return path

    def _decode(self, row: sqlite3.Row) -> dict[str, Any]:
        """Expose structured facts without leaking SQLite-specific row objects."""
        value = dict(row)
        for key in tuple(value):
            if key.endswith("_json"):
                value[key.removesuffix("_json")] = json.loads(value[key])
        return value

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Look up an opaque catalog identity; callers never supply dataset paths."""
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown session")
            value = self._decode(row)
            value["counts"] = self._counts(session_id)
            value["disk_size_bytes"] = self._size(self._directory(value))
            value["evidence_warning"] = self._evidence_warning(value)
            journal = Path(value["dataset_path"]) / JOURNAL_NAME
            recovery_error = self._recovery_blocks.get(session_id)
            if journal.exists() or journal.is_symlink():
                recovery_error = "unfinished recording journal requires operator recovery"
            value["recovery_blocked"] = recovery_error is not None
            value["recovery_error"] = recovery_error
            return value

    def _directory(self, session: dict[str, Any]) -> Path:
        """Validate catalog paths against known workspace locations before filesystem access."""
        path = Path(session["dataset_path"]).parent
        allowed = self.workspace / (
            "trash"
            if session["status"] in {"trashed", "purged"}
            else "sessions-demo"
            if session["source_kind"] == "synthetic_demo"
            else "sessions"
        )
        if (
            path.parent != allowed
            or allowed.is_symlink()
            or not allowed.resolve().is_relative_to(self.workspace)
            or path.is_symlink()
            or path.resolve().parent != allowed.resolve()
        ):
            raise ValueError("session path is not confined to its workspace collection")
        if session["status"] not in {"trashed", "purged"} and path.name != session["slug"]:
            raise ValueError("session slug differs from its directory")
        if session["status"] in {"trashed", "purged"} and not re.fullmatch(
            re.escape(session["slug"]) + r"-\d{8}T\d{12}", path.name
        ):
            raise ValueError("trash directory must match its owned session slug and timestamp")
        if Path(session["dataset_path"]).name != "dataset":
            raise ValueError("session dataset path must use the recorder-owned dataset directory")
        if (path / "dataset").is_symlink():
            raise ValueError("session dataset must not be a symlink")
        return path

    def _size(self, root: Path) -> int:
        """Count regular bytes without following links outside the session."""
        return sum(
            path.stat().st_size
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )

    def _counts(self, session_id: str) -> dict[str, int]:
        """Compute counters from committed episode rows, retaining failures and exclusions."""
        rows = self.connection.execute(
            "SELECT operator_label,excluded FROM episodes WHERE session=?", (session_id,)
        ).fetchall()
        return {
            "saved": len(rows),
            "success": sum(row[0] == "success" for row in rows),
            "failure": sum(row[0] == "failure" for row in rows),
            "excluded": sum(bool(row[1]) for row in rows),
            "discarded": self.connection.execute(
                "SELECT count(*) FROM events WHERE session=? AND kind='episode_discarded'",
                (session_id,),
            ).fetchone()[0],
        }

    def _evidence_warning(self, session: dict[str, Any]) -> str | None:
        """Warn about retained repository evidence without editing its history."""
        if (self.repository / "reports/ludo_flagship/data" / session["slug"]).exists():
            return (
                "Deleting this session leaves committed evidence pointing at removed data; "
                "record a correction."
            )
        return None

    def sessions(self, search: str = "", include_trashed: bool = False) -> list[dict[str, Any]]:
        """List named sessions and live catalog counts, with trash opt-in."""
        with self._lock:
            ids = self.connection.execute(
                "SELECT id,name,status FROM sessions ORDER BY updated_utc DESC"
            ).fetchall()
            return [
                self.get_session(row[0])
                for row in ids
                if search.casefold() in row[1].casefold()
                and (include_trashed or row[2] not in {"trashed", "purged"})
            ]

    def known_follower_serials(self) -> list[str]:
        """Return previously recorded physical follower serials for local form reuse."""
        with self._lock:
            rows = self.connection.execute(
                "SELECT DISTINCT follower_serial FROM sessions "
                "WHERE source_kind='physical' AND status!='purged' "
                "AND trim(follower_serial)!='' ORDER BY follower_serial COLLATE NOCASE"
            ).fetchall()
            return [str(row[0]) for row in rows]

    def _mirror(self, session: dict[str, Any]) -> None:
        """Atomically mirror session facts beside the dataset for catalog reconstruction."""
        directory = self._directory(session)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / ".session.json.tmp"
        if temporary.is_symlink() or (directory / "session.json").is_symlink():
            raise ValueError("session mirror must not be a symlink")
        temporary.write_text(
            _json(
                {
                    k: v
                    for k, v in session.items()
                    if k
                    not in {
                        "counts",
                        "disk_size_bytes",
                        "evidence_warning",
                        "recovery_blocked",
                        "recovery_error",
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(directory / "session.json")

    def create_session(self, values: dict[str, Any]) -> dict[str, Any]:
        """Create an identity and mirror without prematurely creating the dataset root."""
        with self._lock:
            name = values.get("name", "").strip()
            if not name or len(name) > 160:
                raise ValueError("session name must contain 1 to 160 characters")
            source = values.get("source_kind")
            if source not in {"physical", "synthetic_demo"}:
                raise ValueError("explicit physical or synthetic_demo source kind required")
            identity = uuid.uuid4().hex
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "session"
            slug += "-" + identity[:8]
            directory = (
                self.workspace
                / ("sessions-demo" if source == "synthetic_demo" else "sessions")
                / slug
            )
            now = utc_now()
            row: dict[str, Any] = {
                key: str(values.get(key, ""))
                for key in (
                    "task_id",
                    "task_definition_sha256",
                    "recording_purpose",
                    "repo_id",
                    "operator",
                    "scene",
                    "follower_serial",
                    "leader_serial",
                    "power_state_start",
                    "power_state_end",
                    "notes",
                    "git_sha",
                    "host",
                )
            }
            target = values.get("target_episodes", 0)
            if type(target) is not int or target < 0:
                raise ValueError("target episodes must be a non-negative integer")
            row.update(
                id=identity,
                name=name,
                slug=slug,
                status="active",
                source_kind=source,
                dataset_path=str(directory / "dataset"),
                target_episodes=target,
                settings_json=_json(values.get("settings", values.get("settings_json", {}))),
                created_utc=now,
                updated_utc=now,
                closed_utc=None,
                trashed_utc=None,
                trash_reason=None,
                tombstone_json="{}",
            )
            with self.connection:
                self.connection.execute(
                    f"INSERT INTO sessions ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                    tuple(row.values()),
                )
                self._event(identity, "session_created", {"source_kind": source})
            self._mirror(row)
            return self.get_session(identity)

    def _event(
        self,
        session_id: str | None,
        kind: str,
        detail: dict[str, Any],
        episode_index: int | None = None,
    ) -> None:
        """Insert one immutable event within the caller's catalog transaction."""
        self.connection.execute(
            "INSERT INTO events(utc,session,episode_index,kind,detail_json) VALUES (?,?,?,?,?)",
            (utc_now(), session_id, episode_index, kind, _json(detail)),
        )

    def record_event(
        self,
        session_id: str | None,
        kind: str,
        detail: dict[str, Any],
        episode_index: int | None = None,
    ) -> None:
        """Persist operator and controller transitions in the append-only audit log."""
        with self._lock, self.connection:
            self._event(session_id, kind, detail, episode_index)

    def events(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return chronological audit rows, optionally scoped to one session."""
        with self._lock:
            query = "SELECT * FROM events" + (" WHERE session=?" if session_id is not None else "")
            return [
                self._decode(row)
                for row in self.connection.execute(
                    query + " ORDER BY id", (session_id,) if session_id is not None else ()
                )
            ]

    def update_session(self, session_id: str, **changes: Any) -> dict[str, Any]:
        """Update operator metadata without permitting contract or saved-label edits."""
        allowed = {"name", "status", "notes", "power_state_end", "target_episodes"}
        if not changes or set(changes) - allowed:
            raise ValueError(
                "only session name, status, notes, end power and target can be updated"
            )
        with self._lock:
            current = self.get_session(session_id)
            if current["recovery_blocked"]:
                raise RuntimeError("session awaits recovery; metadata writes refused")
            if current["status"] in {"trashed", "purged"}:
                raise ValueError("restore a trashed session before editing")
            status = changes.get("status", current["status"])
            if status not in {"active", "closed"}:
                raise ValueError("use audited trash or restore for lifecycle changes")
            if status == "closed":
                self._require_idle(current)
            if "name" in changes and (
                not str(changes["name"]).strip() or len(changes["name"]) > 160
            ):
                raise ValueError("nonempty session name required")
            if "target_episodes" in changes and (
                type(changes["target_episodes"]) is not int or changes["target_episodes"] < 0
            ):
                raise ValueError("target episodes must be a non-negative integer")
            changes["updated_utc"] = utc_now()
            if "status" in changes:
                changes["closed_utc"] = utc_now() if status == "closed" else None
            with self.connection:
                self.connection.execute(
                    f"UPDATE sessions SET {','.join(key + '=?' for key in changes)} WHERE id=?",
                    (*changes.values(), session_id),
                )
                kind = (
                    "session_renamed"
                    if "name" in changes
                    else (
                        "session_closed"
                        if status == "closed"
                        else "session_opened"
                        if "status" in changes
                        else "session_updated"
                    )
                )
                self._event(session_id, kind, changes)
            result = self.get_session(session_id)
            self._mirror(result)
            return self.get_session(session_id)

    def set_busy(self, session_id: str, busy: bool) -> None:
        """Protect open sources and pending in-memory captures from lifecycle mutation."""
        with self._lock:
            if busy:
                self._busy.add(session_id)
            else:
                self._busy.discard(session_id)

    def _require_idle(self, session: dict[str, Any]) -> None:
        """Refuse busy, locked or recovery-blocked sessions without touching their guards."""
        root = self._directory(session) / "dataset"
        guards = (root.parent / ".dataset.recording.lock", root / JOURNAL_NAME)
        if (
            session.get("recovery_blocked")
            or session["id"] in self._busy
            or any(p.exists() or p.is_symlink() for p in guards)
        ):
            raise RuntimeError(
                "session is recording, locked or awaiting recovery; writes and deletion refused"
            )

    def _require_plain_tree(self, directory: Path) -> None:
        """Reject redirected session entries before catalog-driven filesystem changes."""
        if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob("*")):
            raise ValueError("session paths must not contain symlinks; inspect before mutation")

    def start_capture_run(self, session_id: str, **facts: Any) -> int:
        """Record one source-open attempt with its rate and preflight evidence."""
        with self._lock, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO capture_runs(session,started_utc,git_sha,host,measured_state_rate,"
                "callback_count,maximum_gap,preflight_json) VALUES (?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    utc_now(),
                    facts.get("git_sha"),
                    facts.get("host"),
                    facts.get("measured_state_rate"),
                    facts.get("callback_count"),
                    facts.get("maximum_gap"),
                    _json(facts.get("preflight", {})),
                ),
            )
            self._event(session_id, "capture_run_started", facts)
            return int(cursor.lastrowid or 0)

    def end_capture_run(self, run_id: int, exit_reason: str) -> None:
        """Close source-open evidence even after failures or interrupted pending capture."""
        with self._lock, self.connection:
            row = self.connection.execute(
                "SELECT session FROM capture_runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown capture run")
            self.connection.execute(
                "UPDATE capture_runs SET ended_utc=?,exit_reason=? WHERE id=?",
                (utc_now(), exit_reason, run_id),
            )
            self._event(row[0], "capture_run_ended", {"run_id": run_id, "exit_reason": exit_reason})

    def episodes(self, session_id: str) -> list[dict[str, Any]]:
        """List catalog episode facts without treating them as an independent data source."""
        with self._lock:
            return [
                self._decode(row)
                for row in self.connection.execute(
                    "SELECT * FROM episodes WHERE session=? ORDER BY episode_index", (session_id,)
                )
            ]

    def _reconcile_episodes(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        """Repair episode facts after a dataset commit wins a crash race with the catalog."""
        root = self._directory(session) / "dataset"
        if (root / JOURNAL_NAME).exists() or (root / JOURNAL_NAME).is_symlink():
            raise RuntimeError("recording recovery blocks reindex; preserve the journal and lock")
        metadata = _rows(root / "physical_episode_metadata.jsonl")
        quality = _rows(root / "physical_quality_records.jsonl")
        capture = _rows(root / "physical_capture_provenance.jsonl")
        if set(metadata) != set(quality) or set(metadata) != set(capture):
            raise ValueError("committed episode sidecars disagree; recovery review required")
        exclusions = read_exclusions(root)
        if set(exclusions.entries) - set(metadata):
            raise ValueError("exclusion log references an episode absent from disk")
        existing = {row["episode_index"]: row for row in self.episodes(session["id"])}
        differences = []
        for index, meta in metadata.items():
            record = quality[index]
            if meta["operator_label"] != record["operator_label"]:
                raise ValueError("saved operator label differs across sidecars")
            prior = existing.get(index, {})
            started_event = self.connection.execute(
                "SELECT utc FROM events WHERE session=? AND episode_index=? "
                "AND kind='episode_started' ORDER BY id DESC LIMIT 1",
                (session["id"], index),
            ).fetchone()
            from synria_lerobot.quality_gates import (
                EpisodeQualityRecord,
                GateConfig,
                evaluate_episode,
                load_limits,
            )

            quality_record = EpisodeQualityRecord.from_dict(record)
            gate = evaluate_episode(
                quality_record,
                load_limits(Path(__file__).resolve().parents[2] / "config/synria_limits.yaml"),
                GateConfig(quality_record.min_episode_s, quality_record.max_episode_s),
            )
            exclusion = exclusions.entries.get(index, {})
            values = {
                "session": session["id"],
                "episode_index": index,
                "capture_run": prior.get("capture_run"),
                "operator_label": meta["operator_label"],
                "started_utc": started_event[0] if started_event else prior.get("started_utc"),
                "duration_s": record["duration_s"],
                "frame_count": len(record["frames"]),
                "requested_fps": record["fps"],
                "achieved_sample_rate_hz": record.get("achieved_sample_rate_hz"),
                "final_still_path": meta["final_still"],
                "gate_passed": int(gate.passed),
                "failed_gates_json": _json(gate.failed_gates),
                "excluded": int(exclusion.get("action") == "exclude"),
                "exclusion_reason_code": exclusion.get("reason_code"),
                "exclusion_note": exclusion.get("note"),
                "excluded_utc": exclusion.get("utc"),
                "note": prior.get("note", ""),
            }
            if any(prior.get(key) != value for key, value in values.items()):
                self.connection.execute(
                    f"INSERT OR REPLACE INTO episodes ({','.join(values)}) "
                    f"VALUES ({','.join('?' for _ in values)})",
                    tuple(values.values()),
                )
                detail = {
                    "episode_index": index,
                    "correction": "inserted" if not prior else "disk facts restored",
                }
                differences.append(detail)
                self._event(session["id"], "reconcile_correction", detail, index)
        for index in set(existing) - set(metadata):
            self.connection.execute(
                "DELETE FROM episodes WHERE session=? AND episode_index=?", (session["id"], index)
            )
            detail = {"episode_index": index, "correction": "removed catalog row absent from disk"}
            differences.append(detail)
            self._event(session["id"], "reconcile_correction", detail, index)
        return differences

    def reconcile(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Rebuild missing sessions from mirrors and let committed disk episode facts win."""
        with self._lock, self.connection:
            differences = []
            if session_id is None:
                for collection in ("sessions", "sessions-demo", "trash"):
                    for mirror in sorted((self.workspace / collection).glob("*/session.json")):
                        if mirror.is_symlink() or mirror.parent.is_symlink():
                            raise ValueError("session mirrors must not be symlinks")
                        row = json.loads(mirror.read_text(encoding="utf-8"))
                        row["dataset_path"] = str(mirror.parent / "dataset")
                        if collection == "trash":
                            row["status"] = "trashed"
                        else:
                            expected = (
                                "synthetic_demo" if collection == "sessions-demo" else "physical"
                            )
                            if row["source_kind"] != expected or row["slug"] != mirror.parent.name:
                                raise ValueError(
                                    "session mirror collection or slug is inconsistent"
                                )
                            if row["status"] == "trashed":
                                row["status"] = "closed"
                        existing = self.connection.execute(
                            "SELECT * FROM sessions WHERE id=?", (row["id"],)
                        ).fetchone()
                        if existing is None:
                            columns = [
                                item[1]
                                for item in self.connection.execute("PRAGMA table_info(sessions)")
                            ]
                            self.connection.execute(
                                f"INSERT INTO sessions ({','.join(columns)}) "
                                f"VALUES ({','.join('?' for _ in columns)})",
                                tuple(row[key] for key in columns),
                            )
                            detail = {
                                "session": row["id"],
                                "correction": "session rebuilt from mirror",
                            }
                            differences.append(detail)
                            self._event(row["id"], "reconcile_correction", detail)
                        elif (
                            existing["status"] != "purged"
                            and existing["dataset_path"] != row["dataset_path"]
                        ):
                            self.connection.execute(
                                "UPDATE sessions SET dataset_path=?,status=? WHERE id=?",
                                (row["dataset_path"], row["status"], row["id"]),
                            )
                            detail = {
                                "session": row["id"],
                                "correction": "session location recovered",
                            }
                            differences.append(detail)
                            self._event(row["id"], "reconcile_correction", detail)
            selected = (
                [self.get_session(session_id)]
                if session_id
                else self.sessions(include_trashed=True)
            )
            for session in selected:
                if session["tombstone"].get("purge_pending"):
                    if not self._directory(session).exists():
                        self._finish_purge(session["id"], session["tombstone"])
                        detail = {
                            "session": session["id"],
                            "correction": "completed purge recovered",
                        }
                        differences.append(detail)
                        self._event(session["id"], "reconcile_correction", detail)
                    continue
                if session["status"] != "purged":
                    self.connection.execute("SAVEPOINT episode_reconcile")
                    try:
                        corrections = self._reconcile_episodes(session)
                    except (ValueError, RuntimeError, OSError) as error:
                        self.connection.execute("ROLLBACK TO episode_reconcile")
                        detail = {
                            "session": session["id"],
                            "correction": "reconcile refused",
                            "recovery_blocked": True,
                            "error": str(error),
                        }
                        differences.append(detail)
                        if self._recovery_blocks.get(session["id"]) != str(error):
                            self._event(session["id"], "reconcile_refused", detail)
                        self._recovery_blocks[session["id"]] = str(error)
                    else:
                        differences.extend(corrections)
                        self._recovery_blocks.pop(session["id"], None)
                    finally:
                        self.connection.execute("RELEASE episode_reconcile")
            return differences

    def record_saved_episode(
        self,
        session_id: str,
        episode_index: int,
        capture_run: int | None = None,
        gate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Catalog an already committed episode; failures here must never repeat its save."""
        with self._lock, self.connection:
            self._reconcile_episodes(self.get_session(session_id))
            current = next(
                row for row in self.episodes(session_id) if row["episode_index"] == episode_index
            )
            report = gate_report or {
                "passed": current["gate_passed"],
                "failed_gates": current["failed_gates"],
            }
            cursor = self.connection.execute(
                "UPDATE episodes SET capture_run=?,gate_passed=?,failed_gates_json=? "
                "WHERE session=? AND episode_index=?",
                (
                    capture_run,
                    report.get("passed"),
                    _json(report.get("failed_gates", [])),
                    session_id,
                    episode_index,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("saved episode is missing from committed sidecars")
            self._event(session_id, "episode_saved", report, episode_index)
            self.connection.execute(
                "UPDATE sessions SET updated_utc=? WHERE id=?", (utc_now(), session_id)
            )
            return next(
                row for row in self.episodes(session_id) if row["episode_index"] == episode_index
            )

    def exclude_episode(
        self,
        session_id: str,
        episode_index: int,
        *,
        action: str,
        reason_code: str,
        note: str,
        operator: str,
    ) -> dict[str, Any]:
        """Append curation outside the dataset before mirroring its replayed state."""
        with self._lock, self.connection:
            session = self.get_session(session_id)
            self._require_idle(session)
            if not any(row["episode_index"] == episode_index for row in self.episodes(session_id)):
                raise KeyError("unknown episode")
            append_exclusion(
                Path(session["dataset_path"]),
                episode_index,
                action=action,
                reason_code=reason_code,
                note=note,
                operator=operator,
            )
            self._reconcile_episodes(session)
            self._event(
                session_id,
                "episode_excluded" if action == "exclude" else "episode_restored",
                {"reason_code": reason_code, "note": note, "operator": operator},
                episode_index,
            )
            return next(
                row for row in self.episodes(session_id) if row["episode_index"] == episode_index
            )

    def note_episode(self, session_id: str, episode_index: int, note: str) -> None:
        """Add an audited annotation without editing a saved operator label."""
        with self._lock, self.connection:
            if self.get_session(session_id)["recovery_blocked"]:
                raise RuntimeError("session awaits recovery; episode annotations refused")
            cursor = self.connection.execute(
                "UPDATE episodes SET note=? WHERE session=? AND episode_index=?",
                (note, session_id, episode_index),
            )
            if cursor.rowcount != 1:
                raise KeyError("unknown episode")
            self._event(session_id, "episode_note_added", {"note": note}, episode_index)

    def trash_session(self, session_id: str, *, confirmation: str, reason: str) -> dict[str, Any]:
        """Move an idle session on the same filesystem, retaining evidence and a reason."""
        with self._lock:
            session = self.get_session(session_id)
            self._require_idle(session)
            if session["status"] not in {"active", "closed"}:
                raise ValueError("only an existing session can be trashed")
            if confirmation != session["name"] or not reason.strip():
                raise ValueError("type the exact session name and provide a deletion reason")
            source = self._directory(session)
            self._require_plain_tree(source)
            destination = (
                self.workspace
                / "trash"
                / f"{session['slug']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
            )
            destination.parent.mkdir(exist_ok=True)
            if destination.parent.is_symlink():
                raise ValueError("trash collection must not be a symlink")
            source.rename(destination)
            with self.connection:
                self.connection.execute(
                    "UPDATE sessions SET status='trashed',dataset_path=?,trashed_utc=?,"
                    "trash_reason=?,updated_utc=? WHERE id=?",
                    (str(destination / "dataset"), utc_now(), reason, utc_now(), session_id),
                )
                self._event(
                    session_id,
                    "session_trashed",
                    {"reason": reason, "evidence_warning": session["evidence_warning"]},
                )
            result = self.get_session(session_id)
            self._mirror(result)
            return self.get_session(session_id)

    def restore_session(self, session_id: str) -> dict[str, Any]:
        """Restore a trashed session to its original collection without overwriting paths."""
        with self._lock:
            session = self.get_session(session_id)
            self._require_idle(session)
            if session["status"] != "trashed":
                raise ValueError("only trashed sessions can be restored")
            if session["tombstone"].get("purge_pending"):
                raise RuntimeError("partial purge cannot be restored; inspect and retry the purge")
            source = self._directory(session)
            self._require_plain_tree(source)
            destination = (
                self.workspace
                / ("sessions-demo" if session["source_kind"] == "synthetic_demo" else "sessions")
                / session["slug"]
            )
            if destination.exists() or destination.is_symlink():
                raise FileExistsError("original session location is occupied")
            if (
                destination.parent.is_symlink()
                or destination.parent.resolve().parent != self.workspace
            ):
                raise ValueError("restore collection must remain confined and not be a symlink")
            destination.parent.mkdir(exist_ok=True)
            source.rename(destination)
            with self.connection:
                self.connection.execute(
                    "UPDATE sessions SET status='closed',dataset_path=?,updated_utc=? WHERE id=?",
                    (str(destination / "dataset"), utc_now(), session_id),
                )
                self._event(session_id, "session_restored", {})
            result = self.get_session(session_id)
            self._mirror(result)
            return self.get_session(session_id)

    def purge_session(self, session_id: str, *, confirmation: str, reason: str) -> dict[str, Any]:
        """Permanently remove a confined trash directory while preserving its audit tombstone."""
        with self._lock:
            session = self.get_session(session_id)
            self._require_idle(session)
            if (
                session["status"] != "trashed"
                or confirmation != session["name"]
                or not reason.strip()
            ):
                raise ValueError("purge requires a trashed session, exact name and reason")
            directory = self._directory(session)
            self._require_plain_tree(directory)
            tombstone = (
                session["tombstone"]
                if session["tombstone"].get("purge_pending")
                else {
                    "purge_pending": True,
                    "requested_utc": utc_now(),
                    "reason": reason,
                    "counts": session["counts"],
                    "size_bytes": session["disk_size_bytes"],
                    "evidence_warning": session["evidence_warning"],
                }
            )
            with self.connection:
                self.connection.execute(
                    "UPDATE sessions SET tombstone_json=?,updated_utc=? WHERE id=?",
                    (_json(tombstone), utc_now(), session_id),
                )
                self._event(session_id, "session_purge_requested", tombstone)
            try:
                if directory.exists():
                    shutil.rmtree(directory)
            except BaseException as error:
                with self.connection:
                    self._event(session_id, "session_purge_failed", {"error": str(error)})
                raise
            with self.connection:
                self._finish_purge(session_id, tombstone)
            return self.get_session(session_id)

    def _finish_purge(self, session_id: str, intent: dict[str, Any]) -> None:
        """Finalize the tombstone only after its confined session directory is gone."""
        tombstone = {**intent, "purge_pending": False, "purged_utc": utc_now()}
        self.connection.execute(
            "UPDATE sessions SET status='purged',tombstone_json=?,updated_utc=? WHERE id=?",
            (_json(tombstone), utc_now(), session_id),
        )
        self._event(session_id, "session_purged", tombstone)
