from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_task_registry import synthetic_task
from typer.testing import CliRunner

from physical_ai_lab.cli import app
from synria_lerobot.physical_contract import (
    CONTRACT_VERSION,
    ActionSource,
    PhysicalDatasetContract,
    StateRateMeasurement,
    StateSourceProvenance,
    action_timing_metadata,
)
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
                "sample_monotonic_s": index + 0.01,
                "state_ros_arrival_s": 1000.0 + index,
                "action_ros_header_s": 1000.0 + index,
                "action_ros_arrival_s": 1000.0 + index,
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
        achieved_sample_rate_hz=1.0,
        **action_timing_metadata(ActionSource.LEADER, 1, 1.0),
        state_rate_measurement=asdict(StateRateMeasurement(50, 100, 2, 0, 2, 0.02)),
        state_source_provenance=StateSourceProvenance(
            "standalone_driver", "/joint_states"
        ).as_dict(),
        task_definition=synthetic_task(20, 30),
    )


@pytest.mark.parametrize("mismatch", [None, "resolution", "camera_ids", "missing"])
def test_d1_session_summary_command_writes_artifacts(
    tmp_path: Path, mismatch: str | None
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "payload.bin").write_bytes(b"dataset")
    contract = PhysicalDatasetContract("50mm", ActionSource.LEADER, False,
        task_id="die_into_cup", task_definition=synthetic_task(20, 30),
    ).as_dict(fps=1)
    (dataset / "physical_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    capture = {
        "episode_index": 0,
        "camera_ids": {"wrist": "wrist", "front": "front"},
        "native_resolution": {
            "wrist": {"width": 640, "height": 480},
            "front": {"width": 1280, "height": 720},
        },
        "stored_resolution": {"width": 4, "height": 4},
        "stored_color_space": "RGB",
        "achieved_sample_rate_hz": 1.0,
        "state_rate_measurement": _record().state_rate_measurement,
        "state_source_provenance": _record().state_source_provenance,
        "action_source": "leader",
        "contract_version": CONTRACT_VERSION,
        "gripper_type": "50mm",
        **action_timing_metadata(ActionSource.LEADER, 1, 1.0),
        **synthetic_task(20, 30).metadata(),
    }
    if mismatch == "resolution":
        capture["stored_resolution"] = {"width": 8, "height": 4}
    elif mismatch == "camera_ids":
        capture["camera_ids"] = {"wrist": "wrong", "front": "front"}
    if mismatch != "missing":
        (dataset / "physical_capture_provenance.jsonl").write_text(
            json.dumps(capture) + "\n", encoding="utf-8"
        )
    records = write_episode_records([_record()], dataset / "records.jsonl")
    data_root = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "d1-session-summary",
            "--state-source", "standalone_driver",
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
    session = data_root / "session-001"
    if mismatch is not None:
        assert result.exit_code != 0
        assert not session.exists()
        return
    assert result.exit_code == 0, result.output
    assert (session / "provenance.json").is_file()
    assert (session / "session_summary.json").is_file()
    assert (session / "session_notes.md").is_file()
    provenance = json.loads((session / "provenance.json").read_text())
    assert provenance["native_resolution"] == capture["native_resolution"]
    assert provenance["stored_resolution"] == {"width": 4, "height": 4}
    assert provenance["capture_provenance"] == [capture]
    assert provenance["achieved_sample_rates_hz"] == [{"episode_index": 0, "rate_hz": 1.0}]
    assert provenance["freshness_thresholds_s"]["source_age"] == 0.2


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
                **synthetic_task(20, 30).metadata(),
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
