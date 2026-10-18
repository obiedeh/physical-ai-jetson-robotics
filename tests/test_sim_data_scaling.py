"""The data-scaling study driver: subset rules, the study record and the rendered curve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from synria_lerobot.sim_data_scaling import (
    STUDY_VERSION,
    render_markdown,
    run_config,
    run_study,
    subset_sizes,
    validate_study,
    write_study,
)


def test_subset_sizes_are_increasing_and_within_the_data() -> None:
    """Sizes must grow and the largest may be every available episode, never more."""
    assert subset_sizes([5, 10, 15], 15) == (5, 10, 15)
    for bad in ([], [0], [5, 5], [10, 5], [True, 2], [3, 16]):
        with pytest.raises(ValueError):
            subset_sizes(bad, 15)


def spec_for(tmp_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """A study specification over a written training configuration."""
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    return {
        "version": STUDY_VERSION, "study_id": "study-a", "dataset_root": str(tmp_path / "ds"),
        "contract": str(tmp_path / "ds" / "physical_contract.json"),
        "training_config": str(config_path), "raw_output_root": str(tmp_path / "raw"),
        "evidence_root": "reports/sim_policy/study-a", "policy_prefix": "simcup",
        "subset_sizes": [1, 2], "seed": 3, "steps": 1,
    }


def test_study_spec_requires_the_simulation_lane_and_an_open_limit(tmp_path: Path) -> None:
    """A physical config or a preset limit is refused before training."""
    with pytest.raises(ValueError, match="simulation lane"):
        validate_study(spec_for(tmp_path, {"lane": "physical"}))
    with pytest.raises(ValueError, match="training_episode_limit"):
        validate_study(spec_for(tmp_path, {"lane": "simulation", "training_episode_limit": 3}))
    spec = spec_for(tmp_path, {"lane": "simulation"})
    assert validate_study(spec)["trials"] is None
    with pytest.raises(ValueError, match="study_id"):
        validate_study({**spec, "study_id": " "})
    with pytest.raises(ValueError, match="steps"):
        validate_study({**spec, "steps": 0})
    config = run_config(spec, {"lane": "simulation", "x": 1}, 2, 5)
    assert config["policy_id"] == "simcup_n002" and config["training_episode_limit"] == 2
    assert config["evidence_directory"] == "reports/sim_policy/study-a/train_n002"
    assert run_config(spec, {}, 5, 5)["training_episode_limit"] is None


def test_study_records_each_row_and_renders_the_curve(tmp_path: Path) -> None:
    """A fake trainer per subset; the record, table and per-row write are checked."""
    repository = tmp_path / "repo"
    repository.mkdir()
    dataset = tmp_path / "ds"
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (dataset / "data").mkdir()
    probe = repository / "probes.json"
    probe.write_text(json.dumps({"held_out_episodes": [4]}))
    config = {"lane": "simulation", "repository": str(repository), "probe_set": "probes.json"}
    spec = spec_for(tmp_path, config)
    spec["subset_sizes"] = [1, 3]
    calls: list[tuple[int, dict[str, Any]]] = []

    def fake_train(root: Path, contract: Path, output: Path, seed: int, steps: int,
                   cfg: dict[str, Any]) -> dict[str, Any]:
        calls.append((steps, cfg))
        evidence = repository / cfg["evidence_directory"]
        (evidence / "evaluations").mkdir(parents=True)
        limit = cfg["training_episode_limit"]
        trained = list(range(limit if limit else 4))
        record = {"status": "evaluated", "metrics": {
            "held_out_frames": 30,
            "axes": {**{f"Joint{i}": {"mae": 0.1 / len(trained)} for i in range(1, 7)},
                     "Gripper": {"mae": 0.001}},
        }}
        (evidence / "evaluations" / f"{cfg['policy_id']}-step-000000001.json").write_text(
            json.dumps(record))
        return {
            "policy_id": cfg["policy_id"], "status": "completed",
            "training_episode_indices": tuple(trained),
            "omitted_episode_indices": tuple(range(len(trained), 4)),
            "checkpoints": [{"step": 1, "path": str(output / "step-1"),
                             "checkpoint_content_sha256": "c" * 64}],
        }

    import synria_lerobot.sim_data_scaling as module

    counts = {"available": 4}
    monkey = pytest.MonkeyPatch()
    monkey.setattr(module, "available_episode_count", lambda root, held: counts["available"])
    try:
        trials = {"simcup_n001": {"report": "r1.json", "trials": 4, "successes": 1,
                                  "success_rate": 0.25}}
        study = run_study(spec, train_fn=fake_train,
                          trial_fn=lambda row: trials.get(row["policy_id"]),
                          log=lambda text: None, write_to=tmp_path / "evidence")
    finally:
        monkey.undo()
    assert study["status"] == "completed" and study["physical_evidence"] is False
    assert [row["episodes_trained"] for row in study["rows"]] == [1, 3]
    assert [c["training_episode_limit"] for _, c in calls] == [1, 3]
    assert [c["policy_id"] for _, c in calls] == ["simcup_n001", "simcup_n003"]
    first, second = study["rows"]
    assert first["joint_mae_rad_mean"] == pytest.approx(0.1)
    assert second["joint_mae_rad_mean"] == pytest.approx(0.1 / 3)
    assert first["trials"]["success_rate"] == 0.25 and second["trials"] is None
    assert second["omitted_episode_indices"] == [3]
    assert first["checkpoint_record"].endswith("train_n001/checkpoint-step-000000001.json")
    record = json.loads((tmp_path / "evidence" / "study.json").read_text())
    assert record["status"] == "completed" and len(record["rows"]) == 2
    table = (tmp_path / "evidence" / "STUDY.md").read_text()
    assert "| 1 | simcup_n001 | 0.1000 | 0.1000 | 0.0010 | 25% | 4 | `r1.json` |" in table
    assert "| 3 | simcup_n003 | 0.0333 | 0.0333 | 0.0010 | not measured |  |  |" in table
    assert "never physical evidence" in table

    def failing_train(*args: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic training failure")

    spec["subset_sizes"] = [1]
    monkey.setattr(module, "available_episode_count", lambda root, held: 4)
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            run_study(spec, train_fn=failing_train, log=lambda text: None,
                      write_to=tmp_path / "failed")
    finally:
        monkey.undo()
    failed = json.loads((tmp_path / "failed" / "study.json").read_text())
    assert failed["status"] == "failed" and failed["rows"] == []
    assert "synthetic training failure" in failed["error"]
    with pytest.raises(ValueError, match="exceeds"):
        subset_sizes([5], 4)
    record, table_path = write_study(study, tmp_path / "again")
    assert record.exists() and table_path.read_text() == render_markdown(study)


def test_real_tiny_study_trains_two_nested_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two real sim-lane CPU runs of one step each, 1 then 2 episodes, through the study."""
    from test_synria_act_training_real import training_fixture

    config, dataset = training_fixture(tmp_path, monkeypatch, simulation=True)
    repository = Path(config["repository"])
    config_path = repository / "train.json"
    config_path.write_text(json.dumps(config))
    spec = {
        "version": STUDY_VERSION, "study_id": "tiny", "dataset_root": str(dataset),
        "contract": str(dataset / "physical_contract.json"),
        "training_config": str(config_path), "raw_output_root": str(tmp_path / "raw"),
        "evidence_root": "reports/study-tiny", "policy_prefix": "tiny",
        "subset_sizes": [1, 2], "seed": 7, "steps": 1,
    }
    study = run_study(spec, log=lambda text: None, write_to=repository / "reports/study-tiny")
    assert study["status"] == "completed" and study["available_episodes"] == 2
    rows = study["rows"]
    assert [row["episodes_trained"] for row in rows] == [1, 2]
    assert rows[0]["training_episode_indices"] == [0] and rows[0]["omitted_episode_indices"] == [1]
    assert rows[1]["training_episode_indices"] == [0, 1]
    assert rows[1]["omitted_episode_indices"] == []
    for row in rows:
        assert row["evaluation_status"] == "evaluated" and row["training_status"] == "completed"
        assert row["joint_mae_rad_mean"] is not None and row["gripper_mae_m"] is not None
        assert Path(row["checkpoint_record"]).exists()
    ledger = (repository / "POLICY_LEDGER.md").read_text()
    assert ledger.count("lane=simulation") == 2
    assert (repository / "reports/study-tiny/STUDY.md").exists()
