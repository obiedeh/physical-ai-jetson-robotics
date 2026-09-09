import importlib.util

import pytest

from physical_ai_lab.rtx_training import TrainingRunConfig, train_synria_reach_policy


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_train_synria_reach_policy_writes_report_and_checkpoint(tmp_path) -> None:
    report = tmp_path / "report.json"
    checkpoint = tmp_path / "policy.pt"

    result = train_synria_reach_policy(
        TrainingRunConfig(
            samples=512,
            epochs=2,
            batch_size=128,
            device="cpu",
            report_path=report,
            checkpoint_path=checkpoint,
        )
    )

    assert result["workload"] == "synria-synthetic-reaching-policy"
    assert result["samples"] == 512
    assert report.exists()
    assert checkpoint.exists()
