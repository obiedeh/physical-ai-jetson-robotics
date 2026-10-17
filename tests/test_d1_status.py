"""Exercise the D1 status script against a synthetic catalog and a synthetic aggregate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.d1_status import main, render, status
from synria_lerobot.console.catalog import Catalog

SESSION_COLUMNS = (
    "id, name, slug, status, source_kind, task_id, task_definition_sha256, recording_purpose, "
    "dataset_path, repo_id, settings_json, operator, scene, follower_serial, leader_serial, "
    "power_state_start, power_state_end, target_episodes, notes, git_sha, host, created_utc, "
    "updated_utc"
)


def seed_catalog(workspace: Path) -> None:
    """Two physical qualifying sessions, one trashed, one demo; episodes with mixed verdicts."""
    catalog = Catalog(workspace, backup=False)
    try:
        connection = catalog.connection
        rows = [
            ("a", "First real", "first-real", "closed", "physical", "qualifying", "/data/a"),
            ("b", "Second real", "second-real", "active", "physical", "qualifying", "/data/b"),
            ("c", "Binned", "binned", "trashed", "physical", "qualifying", "/data/c"),
            ("d", "Demo", "demo", "closed", "synthetic_demo", "synthetic_demo", "/data/d"),
            ("e", "Smoke", "smoke-x", "closed", "physical", "disposable_smoke", "/data/e"),
        ]
        for index, (sid, name, slug, state, kind, purpose, dataset) in enumerate(rows):
            connection.execute(
                f"INSERT INTO sessions ({SESSION_COLUMNS}) VALUES "
                "(?,?,?,?,?,?,'sha','" + purpose + "',?,'local/x','{}','op','scene','','',"
                "'on','', 10, '', 'deadbeef', 'host', ?, ?)",
                (sid, name, slug, state, kind, "die_into_cup", dataset,
                 f"2026-10-18T0{index}:00:00+00:00", f"2026-10-18T0{index}:00:00+00:00"),
            )
        episodes = [
            ("a", 0, 1, 0), ("a", 1, 1, 0), ("a", 2, 0, 0), ("a", 3, 1, 1),
            ("b", 0, 1, 0), ("c", 0, 1, 0), ("d", 0, 1, 0), ("e", 0, 1, 0),
        ]
        for session, index, passed, excluded in episodes:
            connection.execute(
                "INSERT INTO episodes (session, episode_index, operator_label, duration_s, "
                "frame_count, requested_fps, gate_passed, failed_gates_json, excluded) "
                "VALUES (?,?,'success',20.0,300,15.0,?,'[]',?)",
                (session, index, passed, excluded),
            )
        connection.commit()
    finally:
        catalog.close()


def test_counts_come_from_the_committed_aggregate_and_the_catalog(tmp_path: Path) -> None:
    """Committed count leads; the catalog lists only physical qualifying live sessions."""
    workspace = tmp_path / "workspace"
    seed_catalog(workspace)
    summary = tmp_path / "D1_dataset_summary.json"
    summary.write_text(json.dumps({
        "qualifying_episode_count": 2, "quality_valid_episode_count": 3, "session_count": 1,
        "status": "planned", "limits_status": "verified",
    }))
    report = status(workspace, summary, Path("config/synria_limits.yaml"))
    assert report["qualifying_committed"] == 2 and report["remaining"] == 98
    assert [s["name"] for s in report["catalog_sessions"]] == ["First real", "Second real"]
    first = report["catalog_sessions"][0]
    assert first["saved"] == 4 and first["gate_passed_not_excluded"] == 2
    assert first["excluded"] == 1 and first["dataset_path"] == "/data/a"
    assert report["catalog_candidates"] == 3 and report["candidates_not_yet_committed"] == 1
    assert report["limits_verified"] is True
    text = render(report)
    assert text.startswith("D1 qualifying episodes: 2 of 100")
    assert "/data/b" in text and "/data/c" not in text and "Demo" not in text


def test_missing_catalog_and_summary_read_as_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """No workspace and no aggregate is 0 of 100, stated plainly, in text and JSON."""
    code = main(["--workspace", str(tmp_path / "nowhere"),
                 "--summary", str(tmp_path / "missing.json")])
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("D1 qualifying episodes: 0 of 100")
    assert "committed aggregate: none" in out and "0 physical qualifying-purpose sessions" in out
    main(["--workspace", str(tmp_path / "nowhere"), "--summary", str(tmp_path / "missing.json"),
          "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["qualifying_committed"] == 0 and payload["catalog_sessions"] == []


def test_the_committed_aggregate_in_the_repository_is_still_zero() -> None:
    """The README's D1 number must match the committed file; today that is zero."""
    report = status(Path("/nonexistent-workspace"),
                    Path("reports/ludo_flagship/data/D1_dataset_summary.json"),
                    Path("config/synria_limits.yaml"))
    assert report["committed_summary"] is not None
    assert report["qualifying_committed"] == 0
