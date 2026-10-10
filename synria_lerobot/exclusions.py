"""Replay audited episode exclusions without modifying authoritative dataset files."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REASON_CODES = frozenset({"capture_fault", "scene_setup_error", "operator_interruption", "other"})


@dataclass(frozen=True)
class ExclusionState:
    """Expose the exact log hash and its latest per-episode curation decisions."""

    sha256: str | None
    entries: dict[int, dict[str, Any]]

    @property
    def excluded_ids(self) -> tuple[int, ...]:
        """Return sorted active exclusions, leaving restored demonstrations available."""
        return tuple(
            sorted(index for index, row in self.entries.items() if row["action"] == "exclude")
        )

    def evidence(self) -> dict[str, Any]:
        """Serialize curation identity for summaries and immutable training evidence."""
        return {
            "exclusion_log_sha256": self.sha256,
            "excluded_episode_ids": list(self.excluded_ids),
        }


def exclusion_path(dataset_root: Path) -> Path:
    """Locate the shared sibling log, including datasets created outside the console."""
    return dataset_root.parent / f"{dataset_root.name}.exclusions.jsonl"


def _validate(row: dict[str, Any]) -> None:
    """Reject ambiguous audit rows rather than silently counting unwanted demonstrations."""
    if type(row.get("episode_index")) is not int or row["episode_index"] < 0:
        raise ValueError("exclusion episode index must be a non-negative integer")
    if row.get("action") not in {"exclude", "restore"}:
        raise ValueError("exclusion action must be exclude or restore")
    if row.get("reason_code") not in REASON_CODES:
        raise ValueError("invalid exclusion reason code; task failure is not an exclusion reason")
    if not isinstance(row.get("note"), str) or (
        row["reason_code"] == "other" and not row["note"].strip()
    ):
        raise ValueError("other exclusions require a note")
    if not isinstance(row.get("operator"), str) or not row["operator"].strip():
        raise ValueError("exclusion operator is required")
    try:
        parsed = datetime.fromisoformat(row["utc"].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError("exclusion UTC timestamp is required") from error


def read_exclusions(dataset_root: Path) -> ExclusionState:
    """Replay a complete UTF-8 log and hash its bytes; a missing log changes nothing."""
    path = exclusion_path(dataset_root)
    if path.is_symlink():
        raise ValueError("exclusion log must not be a symlink")
    if not path.exists():
        return ExclusionState(None, {})
    encoded = path.read_bytes()
    if encoded and not encoded.endswith(b"\n"):
        raise ValueError("exclusion log has an incomplete trailing entry; inspect before appending")
    entries: dict[int, dict[str, Any]] = {}
    for line in encoded.decode("utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("exclusion log rows must be objects")
        _validate(row)
        entries[row["episode_index"]] = row
    return ExclusionState(hashlib.sha256(encoded).hexdigest(), entries)


def append_exclusion(
    dataset_root: Path,
    episode_index: int,
    *,
    action: str,
    reason_code: str,
    note: str,
    operator: str,
    utc: str | None = None,
) -> ExclusionState:
    """Durably append one operator decision before any rebuildable catalog update."""
    read_exclusions(dataset_root)
    row = {
        "episode_index": episode_index,
        "action": action,
        "reason_code": reason_code,
        "note": note,
        "operator": operator,
        "utc": utc or datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    _validate(row)
    path = exclusion_path(dataset_root)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return read_exclusions(dataset_root)


def require_training_exclusions(
    dataset_root: Path,
    held_out_episodes: list[int] | tuple[int, ...],
) -> ExclusionState:
    """Preserve frozen partition checks by refusing active exclusions before training."""
    state = read_exclusions(dataset_root)
    conflict = set(state.excluded_ids).intersection(held_out_episodes)
    if conflict:
        raise ValueError(f"frozen held-out probe episodes are excluded: {sorted(conflict)}")
    if state.excluded_ids:
        raise ValueError(
            "training refuses active episode exclusions to preserve the frozen hash-bound "
            "training partition; prospectively register a new curated dataset and probe set"
        )
    return state


def require_unchanged_exclusions(dataset_root: Path, expected: ExclusionState) -> None:
    """Refuse checkpoint completion when curation changed after training preflight."""
    if read_exclusions(dataset_root).sha256 != expected.sha256:
        raise ValueError("exclusion log changed during training; rerun after reviewing curation")
