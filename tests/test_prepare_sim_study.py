"""Preparing a simulation study from a dataset: probes, configuration and spec."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_lerobot_dataset_writer import synthetic_episode, writer_config

from scripts.prepare_sim_study import choose_held_out, prepare
from synria_lerobot.checkpoint_eval import validate_probes
from synria_lerobot.exclusions import append_exclusion
from synria_lerobot.recorder import LeRobotDatasetWriter
from synria_lerobot.task_registry import SIMULATION


def test_held_out_takes_are_spread_and_bounded() -> None:
    """Evenly spread, never the first or last take, refused when too many."""
    assert choose_held_out(list(range(10)), 1) == [5]
    assert choose_held_out(list(range(10)), 3) == [2, 5, 8]
    assert choose_held_out([3, 4, 7, 9, 12], 2) == [7, 9]
    with pytest.raises(ValueError, match="fewer than all"):
        choose_held_out([1, 2], 2)


def test_prepare_writes_registered_probes_configuration_and_spec(tmp_path: Path) -> None:
    """The probe set spans the non-excluded takes and the configuration binds to it."""
    pytest.importorskip("lerobot")  # the readiness path and the writer need it
    dataset = tmp_path / "corpus" / "dataset"
    config = replace(writer_config(dataset), image_width=32, image_height=32)
    config = replace(config, contract=replace(config.contract, recording_purpose=SIMULATION))
    writer = LeRobotDatasetWriter(config)
    for index in range(6):
        writer.write_episode(replace(synthetic_episode(index, width=32, height=32),
                                     declared_recording_purpose=SIMULATION))
    writer.finalize()
    append_exclusion(dataset, 3, action="exclude", reason_code="other",
                     note="synthetic failure", operator="test")
    args = argparse.Namespace(
        dataset_root=dataset, study_id="synthetic_study", policy_prefix="synth",
        held_out=2, subset_sizes=[2, 4, 4], steps=10, seed=3, save_freq=5, chunk_size=6,
        command_period_s=0.4, response_timeout_s=0.3, dataloader_workers=0,
        registered_by="test", base_training_config=Path("config/synria_act_training.json"),
        evidence_root=tmp_path / "evidence", raw_output_root=tmp_path / "raw",
    )
    paths = prepare(args)
    probes = json.loads(paths["probes"].read_text())
    validate_probes(probes)
    assert probes["held_out_episodes"] == [2, 4], "excluded take 3 is never held out"
    assert probes["held_out_frame_counts"] == {"2": 3, "4": 3}
    assert probes["lane"] == "simulation" and probes["physical_trials"] == []
    train = json.loads(paths["train_config"].read_text())
    assert train["probe_sha256"] == probes["probe_sha256"]
    assert train["lane"] == "simulation" and train["chunk_size"] == 6
    assert train["expected_contract"]["recording_purpose"] == "simulation"
    assert train["probe_set"] == str(paths["probes"]), "outside the repository, absolute"
    spec = json.loads(paths["spec"].read_text())
    assert spec["subset_sizes"] == [2, 4] and spec["steps"] == 10 and spec["seed"] == 3
    assert spec["training_config"] == str(paths["train_config"])
    with pytest.raises(SystemExit, match="frozen once"):
        prepare(args)
