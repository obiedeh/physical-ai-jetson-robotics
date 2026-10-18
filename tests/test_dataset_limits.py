"""Auditing a small parquet dataset against the operator limits."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from synria_lerobot.dataset_limits import audit_dataset, render_markdown, write_audit
from synria_lerobot.quality_gates import PhysicalLimits

JOINTS = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6")
LIMITS = PhysicalLimits(
    verified_by="an operator", verified_on="2026-10-17", joint_names=JOINTS,
    joint_limits_rad=tuple((-1.0, 1.0) for _ in JOINTS),
    gripper_limits_m={"50mm": (0.0, 0.025)}, recording_tolerance_deg=1.0,
)


def write_dataset(root: Path, episodes: dict[int, list[tuple[list[float], list[float]]]]) -> None:
    """A data/ tree with one parquet file per episode of (state, action) frames."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for index, frames in episodes.items():
        path = root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "episode_index": [index] * len(frames),
            "frame_index": list(range(len(frames))),
            "observation.state": [state for state, _ in frames],
            "action": [action for _, action in frames],
        })
        pq.write_table(table, path)


def frame(joint2_action: float = 0.0, joint2_state: float = 0.0,
          gripper: float = 0.01) -> tuple[list[float], list[float]]:
    """A frame whose Joint2 is set; every other joint rests at zero."""
    state = [0.0, joint2_state, 0.0, 0.0, 0.0, 0.0, gripper]
    action = [0.0, joint2_action, 0.0, 0.0, 0.0, 0.0, gripper]
    return state, action


def test_pressing_outside_and_beyond_tolerance_are_counted_per_joint(tmp_path: Path) -> None:
    """One frame inside, one pressing from inside, one outside, one beyond tolerance."""
    half_degree = math.radians(0.5)
    write_dataset(tmp_path, {
        0: [frame(0.5, 0.5), frame(1.0 - half_degree / 2, 0.5)],
        1: [frame(1.0 + half_degree / 2, 1.0), frame(0.0, 1.0 + math.radians(1.5))],
        2: [frame(-0.2, -0.2, gripper=0.03)],
    })
    audit = audit_dataset(tmp_path, LIMITS, "50mm")
    assert audit["episode_count"] == 3 and audit["frames"] == 5
    joint2 = audit["joints"]["Joint2"]
    assert joint2["episodes_pressing"] == 2
    assert joint2["frames_pressing"] == 3
    assert joint2["upper_presses"] == 3 and joint2["lower_presses"] == 0
    assert joint2["frames_outside"] == 2
    assert joint2["frames_beyond_tolerance"] == 1
    assert joint2["action_max"] == pytest.approx(1.0 + half_degree / 2)
    assert joint2["state_max"] == pytest.approx(1.0 + math.radians(1.5))
    gripper = audit["joints"]["Gripper"]
    assert gripper["frames_pressing"] == 1 and gripper["frames_outside"] == 1
    assert gripper["frames_beyond_tolerance"] == 1
    assert audit["episodes"]["0"]["pressing"] == 1 and audit["episodes"]["0"]["outside"] == 0
    assert audit["episodes"]["1"]["pressing"] == 2 and audit["episodes"]["1"]["outside"] == 2
    assert audit["episodes"]["1"]["beyond_tolerance"] == 1
    assert audit["episodes_pressing"] == 3 and audit["frames_pressing"] == 4
    assert audit["frames_outside"] == 3 and audit["frames_beyond_tolerance"] == 2


def test_table_names_the_pressed_side_and_the_files_are_written(tmp_path: Path) -> None:
    """The Markdown names each pressed bound; the JSON round-trips."""
    write_dataset(tmp_path / "ds", {4: [frame(-1.0, -1.0), frame(0.0, 0.0)]})
    audit = audit_dataset(tmp_path / "ds", LIMITS, "50mm")
    text = render_markdown(audit)
    assert "| Joint2 | -57.3 to 57.3 deg | 1 | 1 | 1 / 0 |" in text
    assert "| 4 | 2 | 1 | 0 | 0 | Joint2 lower |" in text
    json_path, markdown_path = write_audit(audit, tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8"))["kind"] == "dataset_limits_audit"
    assert markdown_path.read_text(encoding="utf-8") == text


def test_quiet_dataset_says_so(tmp_path: Path) -> None:
    """No pressing frame leaves the per-episode table out."""
    write_dataset(tmp_path, {0: [frame(0.1, 0.1)]})
    text = render_markdown(audit_dataset(tmp_path, LIMITS, "50mm"))
    assert "No episode pressed a bound." in text


def test_refuses_bad_inputs(tmp_path: Path) -> None:
    """Unknown gripper, bad margin, missing data and short frames are refused."""
    with pytest.raises(ValueError, match="gripper"):
        audit_dataset(tmp_path, LIMITS, "100mm")
    with pytest.raises(ValueError, match="press_deg"):
        audit_dataset(tmp_path, LIMITS, "50mm", press_deg=-1.0)
    with pytest.raises(ValueError, match="no data files"):
        audit_dataset(tmp_path, LIMITS, "50mm")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "data" / "episode_000000.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0], "frame_index": [0],
                             "observation.state": [[0.0] * 7], "action": [[0.0] * 3]}), path)
    with pytest.raises(ValueError, match="seven values"):
        audit_dataset(tmp_path, LIMITS, "50mm")
