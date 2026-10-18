"""Per-take held-out split of a simulation study, on a real one-step CPU checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from synria_lerobot.act_training import train
from synria_lerobot.sim_held_out import held_out_by_take, render_markdown, write_result


def test_split_scores_each_held_out_take_with_the_saved_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One sim-lane run, then the split on its single held-out take, real and fake predictors."""
    from test_synria_act_training_real import training_fixture

    config, dataset = training_fixture(tmp_path, monkeypatch, simulation=True)
    manifest = train(dataset, dataset / "physical_contract.json", tmp_path / "raw", 1, 1, config)
    latest = manifest["checkpoints"][-1]
    study = {
        "study_id": "tiny", "lane": "simulation", "dataset_root": str(dataset),
        "rows": [{"episodes_trained": 2, "policy_id": manifest["policy_id"], "step": 1,
                  "checkpoint_path": latest["path"],
                  "checkpoint_content_sha256": latest["checkpoint_content_sha256"]}],
    }
    result = held_out_by_take(
        study, device="cpu", chunk_size=config["chunk_size"], repo_id=config["dataset_repo_id"],
        fps=15.0, held_out_lengths={2: 3},
    )
    assert result["physical_evidence"] is False and result["held_out_episodes"] == [2]
    [row] = result["rows"]
    take = row["takes"]["2"]
    assert take["frames"] == 3 and np.isfinite(take["joint_mae_rad_mean"])
    assert take["joint_mae_rad_max"] >= take["joint_mae_rad_mean"] >= 0
    record, table = write_result(result, tmp_path / "out", {"2": "randomized"})
    text = table.read_text()
    assert "take 2 (randomized)" in text and f"| 2 | {take['joint_mae_rad_mean']:.4f} |" in text
    assert json.loads(record.read_text())["rows"][0]["policy_id"] == manifest["policy_id"]

    def zeros(checkpoint: Path, device: str):
        return lambda observations: np.zeros(
            (len(observations["observation.state"]), config["chunk_size"], 7))

    fake = held_out_by_take(
        study, device="cpu", chunk_size=config["chunk_size"], repo_id=config["dataset_repo_id"],
        fps=15.0, held_out_lengths={2: 3}, predictor_factory=zeros,
    )
    assert fake["rows"][0]["takes"]["2"]["joint_mae_rad_mean"] == pytest.approx(0.9, abs=1e-6)
    assert render_markdown(fake).count("\n| 2 |") == 1
    with pytest.raises(ValueError, match="simulation-lane"):
        held_out_by_take({**study, "lane": "physical"}, device="cpu", chunk_size=6,
                         repo_id="local/x", fps=15.0, held_out_lengths={2: 3},
                         predictor_factory=zeros)
    (Path(latest["path"]) / "tampered.txt").write_text("x")
    with pytest.raises(ValueError, match="changed since the study"):
        held_out_by_take(study, device="cpu", chunk_size=6, repo_id="local/x", fps=15.0,
                         held_out_lengths={2: 3}, predictor_factory=zeros)
