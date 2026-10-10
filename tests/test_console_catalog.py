"""Exercise catalog rebuild, immutable audits and curation using synthetic sidecars only."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from test_quality_gates import _baseline_episode

from synria_lerobot.console.catalog import Catalog
from synria_lerobot.recording_transaction import JOURNAL_NAME


def create(catalog: Catalog, name: str = "Bench synthetic") -> dict:
    """Create one catalog fixture with no source construction or dataset library import."""
    return catalog.create_session(
        {
            "name": name,
            "source_kind": "synthetic_demo",
            "recording_purpose": "synthetic_demo",
            "task_id": "die_into_cup",
            "task_definition_sha256": "a" * 64,
            "settings": {"fps": 1, "gripper_type": "50mm"},
            "repo_id": "local/fake",
            "operator": "fixture",
            "target_episodes": 2,
        }
    )


def sidecars(session: dict, count: int = 2) -> None:
    """Simulate successful dataset transactions preceding a catalog crash."""
    root = Path(session["dataset_path"])
    root.mkdir(parents=True, exist_ok=True)
    records = [
        replace(
            _baseline_episode(),
            episode_index=index,
            operator_label="success" if index == 0 else "failure",
        ).as_dict()
        for index in range(count)
    ]
    for name in (
        "physical_episode_metadata",
        "physical_quality_records",
        "physical_capture_provenance",
    ):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )


def test_catalog_schema_backup_and_immutable_events(tmp_path: Path) -> None:
    """An empty database migrates forward and retains five consistent backups."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "console.sqlite3").touch()
    catalog = Catalog(workspace)
    session = create(catalog)
    assert not Path(session["dataset_path"]).exists()
    assert catalog.connection.execute("PRAGMA user_version").fetchone()[0] == 1
    assert catalog.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert catalog.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    for statement in ("UPDATE events SET kind='changed'", "DELETE FROM events"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            catalog.connection.execute(statement)
    catalog.connection.rollback()
    for _ in range(6):
        catalog.backup()
    backups = list((workspace / "backups").glob("*.sqlite3"))
    assert len(backups) == 5
    with sqlite3.connect(backups[-1]) as backup:
        assert backup.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    catalog.close()


def test_reconcile_recovers_disk_episode_facts_and_missing_catalog(tmp_path: Path) -> None:
    """A crash after dataset save cannot lose catalog episodes or their original labels."""
    workspace = tmp_path / "workspace"
    catalog = Catalog(workspace)
    session = create(catalog)
    sidecars(session)
    assert len(catalog.reconcile()) == 2
    assert len(catalog.episodes(session["id"])) == 2
    assert catalog.get_session(session["id"])["counts"]["failure"] == 1
    with catalog.connection:
        catalog.connection.execute("UPDATE episodes SET operator_label='wrong'")
    assert len(catalog.reconcile()) == 2
    assert catalog.episodes(session["id"])[0]["operator_label"] == "success"
    assert catalog.episodes(session["id"])[0]["gate_passed"] == 1
    assert catalog.reconcile() == []
    catalog.close()
    (workspace / "console.sqlite3").unlink()
    rebuilt = Catalog(workspace)
    assert len(rebuilt.reconcile()) == 3
    assert rebuilt.get_session(session["id"])["settings"]["fps"] == 1
    assert rebuilt.get_session(session["id"])["counts"]["saved"] == 2
    rebuilt.close()


def test_catalog_lifecycle_exclusion_notes_and_tombstone(tmp_path: Path) -> None:
    """Curation retains failures and deletion keeps counts, size, reason and audit evidence."""
    catalog = Catalog(tmp_path / "workspace", repository=tmp_path / "repository")
    session = create(catalog)
    sidecars(session)
    catalog.reconcile()
    run = catalog.start_capture_run(
        session["id"],
        measured_state_rate=50,
        callback_count=100,
        maximum_gap=0.02,
        preflight={"guard": "passed"},
    )
    catalog.record_saved_episode(session["id"], 0, run, {"passed": True, "failed_gates": []})
    catalog.end_capture_run(run, "closed")
    catalog.note_episode(session["id"], 1, "Task failed; demonstration remains valid.")
    catalog.exclude_episode(
        session["id"],
        0,
        action="exclude",
        reason_code="capture_fault",
        note="synthetic interruption",
        operator="fixture",
    )
    assert catalog.get_session(session["id"])["counts"]["excluded"] == 1
    catalog.exclude_episode(
        session["id"],
        0,
        action="restore",
        reason_code="capture_fault",
        note="inspection completed",
        operator="fixture",
    )
    assert catalog.get_session(session["id"])["counts"]["excluded"] == 0
    evidence = tmp_path / "repository/reports/ludo_flagship/data" / session["slug"]
    evidence.mkdir(parents=True)
    with pytest.raises(ValueError, match="exact"):
        catalog.trash_session(session["id"], confirmation="wrong", reason="test")
    trashed = catalog.trash_session(session["id"], confirmation=session["name"], reason="fixture")
    assert "correction" in trashed["evidence_warning"]
    assert Path(trashed["dataset_path"]).parent.parent.name == "trash"
    restored = catalog.restore_session(session["id"])
    assert restored["status"] == "closed"
    assert Path(restored["dataset_path"]).parent.parent.name == "sessions-demo"
    catalog.trash_session(session["id"], confirmation=session["name"], reason="fixture")
    purged = catalog.purge_session(
        session["id"], confirmation=session["name"], reason="permanent fixture"
    )
    assert purged["status"] == "purged"
    assert purged["tombstone"]["counts"]["saved"] == 2
    assert purged["tombstone"]["size_bytes"] > 0
    assert not Path(purged["dataset_path"]).parent.exists()
    assert evidence.exists()
    kinds = {event["kind"] for event in catalog.events(session["id"])}
    assert {
        "session_created",
        "session_trashed",
        "session_restored",
        "session_purged",
        "capture_run_started",
        "capture_run_ended",
        "episode_saved",
        "episode_excluded",
        "episode_restored",
        "episode_note_added",
    } <= kinds
    catalog.close()


@pytest.mark.parametrize("guard", ["busy", "lock", "journal"])
def test_lifecycle_refuses_busy_locked_or_blocked(tmp_path: Path, guard: str) -> None:
    """No lifecycle request removes a recording lock, recovery journal or pending capture."""
    catalog = Catalog(tmp_path / "workspace")
    session = create(catalog)
    sidecars(session)
    root = Path(session["dataset_path"])
    if guard == "busy":
        catalog.set_busy(session["id"], True)
    else:
        path = root.parent / ".dataset.recording.lock" if guard == "lock" else root / JOURNAL_NAME
        path.write_text("owned guard", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        catalog.trash_session(session["id"], confirmation=session["name"], reason="fixture")
    assert root.exists()
    if guard != "busy":
        assert path.read_text(encoding="utf-8") == "owned guard"
    if guard == "journal":
        assert catalog.reconcile()[0]["recovery_blocked"]
        assert catalog.get_session(session["id"])["recovery_blocked"]
    catalog.close()


def test_reindex_recovers_crash_between_trash_rename_and_catalog_update(tmp_path: Path) -> None:
    """The scanned location wins over a pre-rename mirror after an interrupted trash move."""
    catalog = Catalog(tmp_path / "workspace")
    session = create(catalog)
    sidecars(session)
    destination = catalog.workspace / "trash" / (session["slug"] + "-20261010T010203123456")
    destination.parent.mkdir()
    Path(session["dataset_path"]).parent.rename(destination)
    changes = catalog.reconcile()
    assert any(change["correction"] == "session location recovered" for change in changes)
    assert catalog.get_session(session["id"])["status"] == "trashed"
    assert catalog.get_session(session["id"])["dataset_path"] == str(destination / "dataset")
    catalog.close()


def test_workspace_and_newer_schema_refused(tmp_path: Path) -> None:
    """Reject git-owned workspaces and unknown forward schemas before catalog mutation."""
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    with pytest.raises(ValueError, match="outside"):
        Catalog(repository / "console")
    workspace = tmp_path / "future"
    workspace.mkdir()
    with sqlite3.connect(workspace / "console.sqlite3") as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="newer"):
        Catalog(workspace)


@pytest.mark.parametrize("crash", [False, True])
def test_purge_failure_is_retryable_and_deleted_directory_crash_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash: bool,
) -> None:
    """Incomplete deletion never reports purged, and original tombstone counts survive retries."""
    import shutil

    catalog = Catalog(tmp_path / "workspace")
    session = create(catalog)
    sidecars(session)
    catalog.reconcile()
    trashed = catalog.trash_session(session["id"], confirmation=session["name"], reason="fixture")
    original_size = trashed["disk_size_bytes"]
    if crash:
        original_finish = catalog._finish_purge

        def fail_finish(*args: object) -> None:
            """Simulate process loss after deletion but before final catalog commit."""
            raise RuntimeError("synthetic finalization loss")

        monkeypatch.setattr(catalog, "_finish_purge", fail_finish)
        with pytest.raises(RuntimeError, match="finalization"):
            catalog.purge_session(session["id"], confirmation=session["name"], reason="fixture")
        assert catalog.get_session(session["id"])["status"] == "trashed"
        assert not Path(trashed["dataset_path"]).parent.exists()
        monkeypatch.setattr(catalog, "_finish_purge", original_finish)
        assert any(row["correction"] == "completed purge recovered" for row in catalog.reconcile())
    else:
        original_delete = shutil.rmtree

        def fail_delete(path: Path) -> None:
            """Remove one fixture sidecar before simulating an interrupted directory deletion."""
            (path / "dataset/physical_quality_records.jsonl").unlink()
            raise OSError("synthetic partial deletion")

        monkeypatch.setattr(shutil, "rmtree", fail_delete)
        with pytest.raises(OSError, match="partial"):
            catalog.purge_session(session["id"], confirmation=session["name"], reason="fixture")
        pending = catalog.get_session(session["id"])
        assert pending["status"] == "trashed"
        assert pending["tombstone"]["purge_pending"]
        assert catalog.reconcile() == []
        with pytest.raises(RuntimeError, match="partial purge"):
            catalog.restore_session(session["id"])
        monkeypatch.setattr(shutil, "rmtree", original_delete)
        catalog.purge_session(session["id"], confirmation=session["name"], reason="retry")
    final = catalog.get_session(session["id"])
    assert final["status"] == "purged"
    assert final["tombstone"]["size_bytes"] == original_size
    assert final["tombstone"]["counts"]["saved"] == 2
    assert len([row for row in catalog.events() if row["kind"] == "session_purged"]) == 1
    catalog.close()


def test_restore_and_sidecar_reads_refuse_symlink_redirects(tmp_path: Path) -> None:
    """Neither restoring directories nor reading evidence follows a link outside the workspace."""
    catalog = Catalog(tmp_path / "workspace")
    session = create(catalog)
    sidecars(session)
    source = Path(session["dataset_path"]) / "physical_quality_records.jsonl"
    external = tmp_path / "external.jsonl"
    source.rename(external)
    try:
        source.symlink_to(external)
    except OSError:
        catalog.close()
        pytest.skip("symbolic links are unavailable for this account")
    assert "symlink" in catalog.reconcile()[0]["error"]
    source.unlink()
    external.rename(source)
    catalog.reconcile()
    trashed = catalog.trash_session(session["id"], confirmation=session["name"], reason="fixture")
    collection = catalog.workspace / "sessions-demo"
    collection.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    collection.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="confined"):
        catalog.restore_session(session["id"])
    assert Path(trashed["dataset_path"]).exists()
    assert not list(outside.iterdir())
    catalog.close()


def test_restore_rename_crash_recovers_closed_session(tmp_path: Path) -> None:
    """A restored directory with its old trash mirror recovers without an active capture."""
    catalog = Catalog(tmp_path / "workspace")
    session = create(catalog)
    sidecars(session)
    trashed = catalog.trash_session(session["id"], confirmation=session["name"], reason="fixture")
    destination = catalog.workspace / "sessions-demo" / session["slug"]
    Path(trashed["dataset_path"]).parent.rename(destination)
    assert any(row["correction"] == "session location recovered" for row in catalog.reconcile())
    restored = catalog.get_session(session["id"])
    assert restored["status"] == "closed"
    assert restored["dataset_path"] == str(destination / "dataset")
    catalog.close()


def test_started_utc_uses_matching_audit_only(tmp_path: Path) -> None:
    """Historical timestamps remain unknown unless a matching capture-start event exists."""
    catalog = Catalog(tmp_path / "workspace")
    session = create(catalog)
    sidecars(session)
    catalog.record_event(session["id"], "episode_started", {}, episode_index=0)
    timestamp = catalog.events(session["id"])[-1]["utc"]
    catalog.reconcile()
    episodes = catalog.episodes(session["id"])
    assert episodes[0]["started_utc"] == timestamp
    assert episodes[1]["started_utc"] is None
    catalog.close()


def test_blocked_recovery_does_not_hide_other_sessions(tmp_path: Path) -> None:
    """A journal blocks its own writes while other sessions remain browsable and recoverable."""
    catalog = Catalog(tmp_path / "workspace")
    blocked = create(catalog, "Blocked")
    healthy = create(catalog, "Healthy")
    sidecars(blocked)
    sidecars(healthy)
    journal = Path(blocked["dataset_path"]) / JOURNAL_NAME
    journal.write_text("preserve this recovery evidence", encoding="utf-8")
    differences = catalog.reconcile()
    assert any(
        row.get("recovery_blocked") and row["session"] == blocked["id"] for row in differences
    )
    assert len(catalog.episodes(healthy["id"])) == 2
    assert len(catalog.sessions()) == 2
    assert catalog.get_session(blocked["id"])["recovery_blocked"]
    assert "journal" in catalog.get_session(blocked["id"])["recovery_error"]
    for mutate in (
        lambda: catalog.update_session(blocked["id"], name="Renamed"),
        lambda: catalog.note_episode(blocked["id"], 0, "note"),
        lambda: catalog.trash_session(blocked["id"], confirmation="Blocked", reason="fixture"),
    ):
        with pytest.raises(RuntimeError, match="recovery"):
            mutate()
    catalog.reconcile()
    assert len([event for event in catalog.events() if event["kind"] == "reconcile_refused"]) == 1
    assert journal.read_text(encoding="utf-8") == "preserve this recovery evidence"
    journal.unlink()
    catalog.reconcile()
    assert not catalog.get_session(blocked["id"])["recovery_blocked"]
    assert len(catalog.episodes(blocked["id"])) == 2
    catalog.close()
