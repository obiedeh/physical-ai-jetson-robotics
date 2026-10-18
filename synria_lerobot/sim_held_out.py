"""Held-out error per take for every row of a simulation data-scaling study.

The trainer's evaluation record averages the held-out error over every held-out frame.
When the held-out set mixes takes recorded under different conditions, such as fixed
and randomized object placements, that average can stay flat while the model gets
better on one kind and worse on the other. This module reloads each row's final
checkpoint, runs the same prediction metric on each held-out take separately, and
writes a table with one column per take. Simulation lane only; never physical evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .act_training import _evaluation_batches
from .checkpoint_eval import PredictionBatch, prediction_metrics, strict_content_hash
from .physical_contract import SIMULATION_LANE, training_lane

Predictor = Callable[[dict[str, Any]], np.ndarray]


def load_predictor(checkpoint: Path, device: str) -> Predictor:
    """The saved policy and its normalizers as one function from observations to chunks."""
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    local = checkpoint / "pretrained_model"
    saved = ACTConfig.from_pretrained(local, local_files_only=True)
    saved.device = device
    policy = ACTPolicy.from_pretrained(local, config=saved, local_files_only=True, strict=True)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        saved, pretrained_path=str(local),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    def predict(observations: dict[str, Any]) -> np.ndarray:
        """Predict one action chunk per observation row."""
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(preprocessor(observations))
            return np.asarray(postprocessor(prediction).detach().cpu().numpy())

    return predict


def episode_batches(
    dataset_root: Path, repo_id: str, episode: int, chunk_size: int, fps: float,
    batch_size: int = 8,
) -> Iterator[PredictionBatch]:
    """Held-out batches of one episode, built the way the trainer builds them."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id, root=dataset_root,
        delta_timestamps={"action": [index / fps for index in range(chunk_size)]},
        video_backend="pyav", download_videos=False, token=False, episodes=[episode],
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=0)
    return _evaluation_batches(loader)


def held_out_by_take(
    study: dict[str, Any], *, device: str, chunk_size: int, repo_id: str, fps: float,
    held_out_lengths: dict[int, int],
    predictor_factory: Callable[[Path, str], Predictor] | None = None,
) -> dict[str, Any]:
    """Per-take held-out metrics for every completed row of the study."""
    if training_lane(study) != SIMULATION_LANE:
        raise ValueError("held-out split is a simulation-lane tool")
    factory = load_predictor if predictor_factory is None else predictor_factory
    dataset_root = Path(study["dataset_root"])
    rows = []
    for row in study["rows"]:
        checkpoint = Path(row["checkpoint_path"])
        if strict_content_hash(checkpoint) != row["checkpoint_content_sha256"]:
            raise ValueError(f"checkpoint for {row['policy_id']} changed since the study")
        predict = factory(checkpoint, device)
        takes = {}
        for episode, length in sorted(held_out_lengths.items()):
            metrics = prediction_metrics(
                episode_batches(dataset_root, repo_id, episode, chunk_size, fps),
                predict, {episode: length},
            )
            joints = [metrics["axes"][f"Joint{i}"]["mae"] for i in range(1, 7)]
            takes[str(episode)] = {
                "frames": length,
                "joint_mae_rad_mean": sum(joints) / 6,
                "joint_mae_rad_max": max(joints),
                "gripper_mae_m": metrics["axes"]["Gripper"]["mae"],
            }
        rows.append({"episodes_trained": row["episodes_trained"], "policy_id": row["policy_id"],
                     "step": row["step"], "takes": takes})
    return {
        "kind": "simulation_held_out_by_take",
        "lane": SIMULATION_LANE,
        "physical_evidence": False,
        "study_id": study["study_id"],
        "utc": datetime.now(timezone.utc).isoformat(),
        "held_out_episodes": sorted(held_out_lengths),
        "rows": rows,
    }


def render_markdown(result: dict[str, Any], labels: dict[str, str] | None = None) -> str:
    """One row per subset, one column per held-out take, mean joint error in radians."""
    takes = [str(index) for index in result["held_out_episodes"]]
    names = labels or {}
    header = " | ".join(f"take {take}" + (f" ({names[take]})" if take in names else "")
                        for take in takes)
    lines = [
        f"# Held-out error by take, study `{result['study_id']}`",
        "",
        "Simulation lane; never physical evidence. Mean joint error in radians of each",
        "row's final checkpoint on each held-out take separately.",
        "",
        f"| episodes trained | {header} |",
        "|---|" + "---|" * len(takes),
    ]
    for row in result["rows"]:
        cells = " | ".join(f"{row['takes'][take]['joint_mae_rad_mean']:.4f}" for take in takes)
        lines.append(f"| {row['episodes_trained']} | {cells} |")
    lines.append("")
    return "\n".join(lines)


def write_result(result: dict[str, Any], directory: Path,
                 labels: dict[str, str] | None = None) -> tuple[Path, Path]:
    """Write the JSON record and the Markdown table into the study directory."""
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / "held_out_by_take.json"
    record.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    table = directory / "HELD_OUT_BY_TAKE.md"
    table.write_text(render_markdown(result, labels), encoding="utf-8")
    return record, table
