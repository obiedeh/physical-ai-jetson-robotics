from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from physical_ai_lab.cli import app
from synria_lerobot.physical_contract import CONTRACT_VERSION
from synria_lerobot.quality_gates import (
    EpisodeQualityRecord,
    FrameQualityRecord,
    ImageDiagnostic,
    write_episode_records,
)

runner = CliRunner()


def _record() -> EpisodeQualityRecord:
    frames = tuple(
        FrameQualityRecord(
            state=(0.0,) * 6 + (0.01,),
            action=(0.0,) * 6 + (0.01,),
            timestamps={
                "state_monotonic_s": float(index),
                "state_ros_header_s": 1000.0 + index,
                "action_monotonic_s": float(index),
                "wrist_monotonic_s": float(index),
                "front_monotonic_s": float(index),
            },
            wrist=ImageDiagnostic(True, 50.0, f"wrist-{index}"),
            front=ImageDiagnostic(True, 60.0, f"front-{index}"),
        )
        for index in range(20)
    )
    return EpisodeQualityRecord(
        episode_index=0,
        fps=1.0,
        duration_s=20.0,
        operator_label="success",
        action_source="leader",
        contract_version=CONTRACT_VERSION,
        gripper_type="50mm",
        final_still="episode_000000_final.jpg",
        smoke=False,
        frames=frames,
    )


def test_d1_session_summary_command_writes_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "payload.bin").write_bytes(b"dataset")
    records = write_episode_records([_record()], dataset / "records.jsonl")
    data_root = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "d1-session-summary",
            "--session-id",
            "session-001",
            "--records",
            str(records),
            "--dataset-path",
            str(dataset),
            "--follower-serial",
            "ADF-unverified",
            "--leader-serial",
            "ADL-unverified",
            "--host",
            "fake-host",
            "--git-sha",
            "test-sha",
            "--operator",
            "tester",
            "--power-state-start",
            "test",
            "--power-state-end",
            "test",
            "--scene",
            "fake scene",
            "--wrist-camera-id",
            "wrist",
            "--front-camera-id",
            "front",
            "--width",
            "4",
            "--height",
            "4",
            "--rate-hz",
            "1",
            "--gripper-type",
            "50mm",
            "--action-source",
            "leader",
            "--utc-date",
            "2026-10-07",
            "--data-root",
            str(data_root),
            "--limits-path",
            str(Path("config/synria_limits.yaml")),
        ],
    )
    assert result.exit_code == 0, result.output
    session = data_root / "session-001"
    assert (session / "provenance.json").is_file()
    assert (session / "session_summary.json").is_file()
    assert (session / "session_notes.md").is_file()


def test_d1_dataset_summary_command_writes_aggregate_and_timeline(tmp_path: Path) -> None:
    data_root = tmp_path / "reports"
    session = data_root / "session-001"
    session.mkdir(parents=True)
    (session / "session_summary.json").write_text(
        json.dumps(
            {
                "quality_valid_episode_count": 1,
                "action_source": "leader",
                "limits_status": "limits unverified by operator",
            }
        )
    )
    output = data_root / "D1_dataset_summary.json"
    timeline = tmp_path / "timeline.jsonl"
    result = runner.invoke(
        app,
        [
            "d1-dataset-summary",
            "--data-root",
            str(data_root),
            "--output",
            str(output),
            "--timeline",
            str(timeline),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["quality_valid_episode_count"] == 1
    assert "d1_dataset_summary" in timeline.read_text()
