"""Count qualifying D1 episodes from the data, so the README number never comes from memory.

Two sources are read, both read-only:

- the committed aggregate ``reports/ludo_flagship/data/D1_dataset_summary.json``, which is
  what the README cites and which ``physical-ai-lab d1-dataset-summary`` regenerates from
  the committed session summaries;
- the console catalog in the recording workspace, which holds every saved episode with its
  gate verdict and exclusion state, including sessions whose summary has not been
  committed yet.

The script prints ``N of 100`` from the committed aggregate, then the catalog's physical
qualifying-purpose sessions with their dataset paths and gate-passed, not-excluded episode
counts, and says how many of those are not yet in a committed summary. Qualifying also
requires operator-verified limits, so the limits file's verification fields are shown.

Usage::

    python scripts/d1_status.py
    python scripts/d1_status.py --workspace ~/synria-console --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.quality_gates import load_limits  # noqa: E402

DEFAULT_WORKSPACE = Path.home() / "synria-console"
DEFAULT_SUMMARY = REPOSITORY / "reports" / "ludo_flagship" / "data" / "D1_dataset_summary.json"
DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"
TARGET = 100


def committed_summary(path: Path) -> dict[str, Any] | None:
    """The committed D1 aggregate, or None when it does not exist."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "qualifying_episode_count": int(payload.get("qualifying_episode_count", 0)),
        "quality_valid_episode_count": int(payload.get("quality_valid_episode_count", 0)),
        "session_count": int(payload.get("session_count", 0)),
        "status": payload.get("status"),
        "limits_status": payload.get("limits_status"),
    }


def catalog_sessions(workspace: Path) -> list[dict[str, Any]]:
    """Physical, qualifying-purpose sessions in the console catalog with their episode counts.

    The database is opened read-only. Trashed and purged sessions are left out. An episode
    counts as a candidate when its gate passed and it is not excluded; whether it is
    qualifying also depends on verified limits and a committed session summary.
    """
    database = workspace / "console.sqlite3"
    if not database.is_file():
        return []
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        sessions = connection.execute(
            "SELECT id, name, status, task_id, dataset_path, target_episodes, created_utc "
            "FROM sessions WHERE source_kind='physical' AND recording_purpose='qualifying' "
            "AND status NOT IN ('trashed', 'purged') ORDER BY created_utc"
        ).fetchall()
        result = []
        for session in sessions:
            rows = connection.execute(
                "SELECT gate_passed, excluded FROM episodes WHERE session=?", (session["id"],)
            ).fetchall()
            result.append({
                "name": session["name"], "status": session["status"],
                "task_id": session["task_id"], "dataset_path": session["dataset_path"],
                "target_episodes": session["target_episodes"],
                "created_utc": session["created_utc"],
                "saved": len(rows),
                "gate_passed_not_excluded": sum(
                    1 for row in rows if row["gate_passed"] == 1 and not row["excluded"]
                ),
                "excluded": sum(1 for row in rows if row["excluded"]),
            })
        return result
    finally:
        connection.close()


def status(workspace: Path, summary_path: Path, limits_path: Path) -> dict[str, Any]:
    """Everything the printed report and the JSON output contain."""
    summary = committed_summary(summary_path)
    sessions = catalog_sessions(workspace)
    limits = load_limits(limits_path)
    committed = summary["qualifying_episode_count"] if summary else 0
    candidates = sum(session["gate_passed_not_excluded"] for session in sessions)
    return {
        "target": TARGET,
        "qualifying_committed": committed,
        "remaining": max(0, TARGET - committed),
        "committed_summary": summary,
        "catalog_workspace": str(workspace),
        "catalog_sessions": sessions,
        "catalog_candidates": candidates,
        "candidates_not_yet_committed": max(0, candidates - committed),
        "limits_verified": limits.verified,
        "limits_verified_by": limits.verified_by,
        "limits_verified_on": limits.verified_on,
    }


def render(report: dict[str, Any]) -> str:
    """The human-readable report."""
    lines = [
        f"D1 qualifying episodes: {report['qualifying_committed']} of {report['target']} "
        f"(committed summaries; {report['remaining']} remaining)",
    ]
    summary = report["committed_summary"]
    if summary is None:
        lines.append("  committed aggregate: none")
    else:
        lines.append(f"  committed aggregate: {summary['path']} "
                     f"(status {summary['status']}, {summary['session_count']} sessions)")
    if report["limits_verified"]:
        lines.append(f"  limits: verified by {report['limits_verified_by']} "
                     f"on {report['limits_verified_on']}")
    else:
        lines.append("  limits: unverified by operator; nothing can qualify")
    lines.append(f"Console catalog ({report['catalog_workspace']}): "
                 f"{len(report['catalog_sessions'])} physical qualifying-purpose sessions")
    for session in report["catalog_sessions"]:
        lines.append(
            f"  {session['name']} [{session['status']}] task {session['task_id']}: "
            f"{session['saved']} saved, {session['gate_passed_not_excluded']} gate-passed "
            f"and kept, {session['excluded']} excluded; dataset {session['dataset_path']}"
        )
    lines.append(f"  gate-passed and kept episodes in the catalog: {report['catalog_candidates']}"
                 f"; not yet in a committed summary: {report['candidates_not_yet_committed']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE,
                        help="Console workspace holding console.sqlite3.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help="Committed D1 aggregate summary.")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; always 0, the numbers are the output."""
    args = build_parser().parse_args(argv)
    report = status(args.workspace, args.summary, args.limits)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
