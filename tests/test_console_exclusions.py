"""Verify sibling exclusion logs affect physical summaries and training without device access."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_quality_gates import LIMITS_PATH, _baseline_episode, _provenance

from synria_lerobot.exclusions import (
    append_exclusion,
    exclusion_path,
    read_exclusions,
    require_training_exclusions,
    require_unchanged_exclusions,
)
from synria_lerobot.quality_gates import (
    GateConfig,
    load_limits,
    write_aggregate_summary,
    write_session_artifacts,
)


def test_exclusion_append_replay_and_training_fail_closed(tmp_path: Path) -> None:
    """Restore preserves history; active exclusions cannot bypass frozen training partitions."""
    root = tmp_path / "dataset"
    root.mkdir()
    assert read_exclusions(root).evidence() == {
        "exclusion_log_sha256": None,
        "excluded_episode_ids": [],
    }
    assert require_training_exclusions(root, [1]).excluded_ids == ()
    excluded = append_exclusion(
        root,
        0,
        action="exclude",
        reason_code="capture_fault",
        note="broken source",
        operator="fixture",
    )
    first = exclusion_path(root).read_bytes()
    assert excluded.excluded_ids == (0,)
    with pytest.raises(ValueError, match="hash-bound"):
        require_training_exclusions(root, [1])
    with pytest.raises(ValueError, match="held-out"):
        require_training_exclusions(root, [0])
    restored = append_exclusion(
        root, 0, action="restore", reason_code="capture_fault", note="reviewed", operator="fixture"
    )
    assert exclusion_path(root).read_bytes().startswith(first)
    assert restored.sha256 != excluded.sha256
    assert require_training_exclusions(root, [0]).excluded_ids == ()
    assert not list(root.iterdir())


@pytest.mark.parametrize("reason,note", [("task_failed", "genuine failure"), ("other", "")])
def test_exclusion_reason_cannot_erase_task_failure(tmp_path: Path, reason: str, note: str) -> None:
    """Task failures remain labeled data and miscellaneous exclusions require explanation."""
    with pytest.raises(ValueError, match="reason|note"):
        append_exclusion(
            tmp_path / "dataset",
            0,
            action="exclude",
            reason_code=reason,
            note=note,
            operator="fixture",
        )
    assert not exclusion_path(tmp_path / "dataset").exists()


def test_summaries_honor_exclusions_and_refuse_stale_curation(tmp_path: Path) -> None:
    """A stale pre-exclusion summary cannot inflate aggregate qualifying progress."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    reports = tmp_path / "reports"
    limits = replace(load_limits(LIMITS_PATH), verified_by="fixture", verified_on="2026-10-10")
    episodes = [
        _baseline_episode(),
        replace(_baseline_episode(), episode_index=1, operator_label="failure"),
    ]

    def summarize() -> dict:
        """Rebuild the same session using authoritative curation and unchanged quality gates."""
        return write_session_artifacts(
            session_dir=reports / "one",
            dataset_path=dataset,
            provenance=_provenance("leader"),
            episodes=episodes,
            limits=limits,
            gate_config=GateConfig(20, 30),
        )

    assert summarize()["qualifying_episode_count"] == 2
    append_exclusion(
        dataset,
        0,
        action="exclude",
        reason_code="scene_setup_error",
        note="fixture scene",
        operator="fixture",
    )
    with pytest.raises(ValueError, match="stale exclusion"):
        write_aggregate_summary(
            data_root=reports,
            output_path=reports / "aggregate.json",
            timeline_path=tmp_path / "timeline.jsonl",
        )
    summary = summarize()
    assert summary["qualifying_episode_count"] == 1
    assert summary["excluded_episode_count"] == 1
    assert summary["operator_labels"] == ["failure"]
    assert summary["demonstration_success_rate"] == 0
    aggregate = write_aggregate_summary(
        data_root=reports,
        output_path=reports / "aggregate.json",
        timeline_path=tmp_path / "timeline.jsonl",
    )
    assert aggregate["qualifying_episode_count"] == 1
    assert aggregate["excluded_episode_count"] == 1


def test_d1_summaries_refuse_synthetic_identity_before_writes(tmp_path: Path) -> None:
    """Synthetic purpose is ineligible at both summary entry points, even without episode rows."""
    provenance = {**_provenance("leader"), "recording_purpose": "synthetic_demo"}
    with pytest.raises(ValueError, match="synthetic demo"):
        write_session_artifacts(
            session_dir=tmp_path / "result",
            dataset_path=tmp_path / "dataset",
            provenance=provenance,
            episodes=[],
            limits=load_limits(LIMITS_PATH),
            gate_config=GateConfig(20, 30),
        )
    assert not (tmp_path / "result").exists()
    report = tmp_path / "reports/one"
    report.mkdir(parents=True)
    (report / "session_summary.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic demo"):
        write_aggregate_summary(
            data_root=report.parent,
            output_path=tmp_path / "aggregate.json",
            timeline_path=tmp_path / "timeline.jsonl",
        )
    assert not (tmp_path / "aggregate.json").exists()


@pytest.mark.parametrize("case", ["active", "held_out", "unknown_restored"])
def test_actual_training_entry_point_checks_curation_before_optional_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Training refuses excluded or nonexistent IDs before creating run output or a model."""
    from test_synria_act_training import input_files

    from synria_lerobot import act_training

    config, root, contract = input_files(tmp_path)
    index = 9 if case == "unknown_restored" else 0
    append_exclusion(
        root,
        index,
        action="exclude",
        reason_code="capture_fault",
        note="fixture",
        operator="fixture",
    )
    if case == "unknown_restored":
        append_exclusion(
            root,
            index,
            action="restore",
            reason_code="capture_fault",
            note="fixture",
            operator="fixture",
        )
    held_out = [0] if case == "held_out" else [1]
    monkeypatch.setattr(act_training, "load_probes", lambda *args: {"held_out_episodes": held_out})
    monkeypatch.setattr(act_training, "read_episode_metadata", lambda root: {0: {}, 1: {}})
    match = {"active": "hash-bound", "held_out": "held-out", "unknown_restored": "missing"}[case]
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=match):
        act_training.train(root, contract, output, 1, 2, config)
    assert not output.exists()


def test_truncated_logs_and_mid_training_changes_are_refused(tmp_path: Path) -> None:
    """Partial writes and later curation cannot silently retain outdated training evidence."""
    root = tmp_path / "dataset"
    initial = read_exclusions(root)
    append_exclusion(
        root, 0, action="exclude", reason_code="capture_fault", note="fixture", operator="fixture"
    )
    with pytest.raises(ValueError, match="changed during training"):
        require_unchanged_exclusions(root, initial)
    path = exclusion_path(root)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ValueError, match="incomplete"):
        read_exclusions(root)
