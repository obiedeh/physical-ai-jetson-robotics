"""Optional upstream metadata integration; synthetic numeric episodes only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_lerobot_dataset_writer import real_dataset_type
from test_synria_checkpoint_eval import batches, commit, fixed_probe_trial
from test_task_registry import synthetic_task

from synria_lerobot.checkpoint_eval import (
    CheckpointEvaluator,
    read_episode_metadata,
    register_probes,
    strict_content_hash,
)
from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract


def test_real_finalized_episode_metadata_binds_frozen_probe_partition(tmp_path: Path) -> None:
    library = real_dataset_type()
    root = tmp_path / "dataset"
    dataset = library.create(
        repo_id="local/checkpoint-probe-test",
        root=root,
        fps=10,
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (7,)},
            "action": {"dtype": "float32", "shape": (7,)},
        },
    )
    try:
        for _ in range(3):
            for _ in range(3):
                dataset.add_frame(
                    {
                        "observation.state": np.zeros(7, dtype=np.float32),
                        "action": np.zeros(7, dtype=np.float32),
                        "task": "synthetic metadata test",
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()
    contract = PhysicalDatasetContract(
        "50mm",
        ActionSource.NEXT_STATE,
        False,
        action_lookahead_steps=2,
        task_id="die_into_cup",
        task_definition=synthetic_task(0.1, 1),
    )
    (root / "physical_contract.json").write_text(json.dumps(contract.as_dict(fps=10)))
    assert read_episode_metadata(root) == {0: 3, 1: 3, 2: 3}
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    probe = repository / "probes.json"
    config = json.loads(Path("config/synria_checkpoint_probes.json").read_text())
    config.update(
        probe_set_id="real-library-synthetic",
        registered_by="synthetic fixture",
        registered_on="2026-10-08",
        dataset_content_sha256=strict_content_hash(root),
        task_id=contract.task_id,
        physical_contract=contract.as_dict(fps=10),
        held_out_episodes=[2],
        held_out_frame_counts={"2": 3},
        physical_trials=[fixed_probe_trial("a")],
        capture=dict(fps=15, max_duration_s=10, shutdown_timeout_s=1),
    )
    probe.write_text(json.dumps(config))
    digest = register_probes(probe)
    commit(repository, probe)
    evaluator = CheckpointEvaluator(probe, repository, digest, root, (0, 1))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"synthetic weights, not a trained policy")
    record = evaluator.evaluate(
        checkpoint,
        "synthetic-policy",
        1,
        batches(),
        lambda observations: np.zeros((3, 2, 7)),
        tmp_path / "evaluations",
        tmp_path / "timeline.jsonl",
    )
    assert record["status"] == "evaluated" and record["metrics"]["held_out_frames"] == 3
    with pytest.raises(ValueError, match="actual episodes"):
        CheckpointEvaluator(probe, repository, digest, root, (0, 999))
    info = json.loads((root / "meta/info.json").read_text())
    info["total_frames"] += 1
    (root / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="frame count"):
        read_episode_metadata(root)
