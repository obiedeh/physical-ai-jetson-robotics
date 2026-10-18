"""Average the last saved checkpoints of a run into one servable checkpoint.

A single checkpoint is a noisy draw: two checkpoints from the same run with the same
held-out error can behave quite differently in closed loop. Averaging the weights of
the last few checkpoints smooths that noise at no training cost. The averaged
checkpoint is written next to its sources, carries the run's manifest with the
averaged steps named, is evaluated on the frozen held-out set with the run's evaluator,
and gets a completion receipt so the policy server can serve it exactly like a saved
step. Only float tensors are averaged; any other tensor is taken from the newest
checkpoint. Nothing here trains, and the source checkpoints are left untouched.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .act_training import _evaluation_batches, _json
from .checkpoint_eval import CheckpointEvaluator, PredictionBatch, strict_content_hash
from .physical_contract import training_lane
from .sim_held_out import load_predictor

MODEL_FILE = "model.safetensors"


def average_state_dicts(states: list[dict[str, Any]]) -> dict[str, Any]:
    """The elementwise mean of float tensors across state dicts; other tensors from the last."""
    import torch

    if not states:
        raise ValueError("nothing to average")
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("checkpoints have different tensors")
    result = {}
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        if any(tuple(t.shape) != tuple(tensors[0].shape) for t in tensors):
            raise ValueError(f"tensor {key} changes shape between checkpoints")
        if tensors[0].is_floating_point():
            stacked = torch.stack([t.to(torch.float64) for t in tensors])
            result[key] = stacked.mean(dim=0).to(tensors[0].dtype).contiguous()
        else:
            result[key] = tensors[-1].clone().contiguous()
    return result


def held_out_batches(
    dataset_root: Path, repo_id: str, episodes: list[int], chunk_size: int, fps: float,
    batch_size: int = 8,
) -> Iterator[PredictionBatch]:
    """Held-out batches over every held-out episode, built as the trainer builds them."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id, root=dataset_root,
        delta_timestamps={"action": [index / fps for index in range(chunk_size)]},
        video_backend="pyav", download_videos=False, token=False, episodes=list(episodes),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=0)
    return _evaluation_batches(loader)


def average_checkpoints(
    evidence: Path, last_k: int, *, repository: Path, dataset_root: Path, device: str,
    predictor_factory: Callable[[Path, str], Any] = load_predictor,
) -> dict[str, Any]:
    """Average the run's last ``last_k`` evaluated checkpoints, evaluate and receipt the result.

    ``evidence`` is the run's evidence directory holding ``training_run.json``. The new
    checkpoint lands beside its sources as ``averaged-last<k>-step-<step>``, its receipt
    as ``checkpoint-averaged-last<k>.json`` in the evidence directory, and its held-out
    record under ``evaluations`` with the policy id suffixed ``_avg<k>``.
    """
    if type(last_k) is not int or last_k < 2:
        raise ValueError("average at least two checkpoints")
    run = json.loads((evidence / "training_run.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed":
        raise ValueError("average a completed run only")
    evaluated = [entry for entry in run["checkpoints"] if entry["evaluation_status"] == "evaluated"]
    if len(evaluated) < last_k:
        raise ValueError(f"the run has {len(evaluated)} evaluated checkpoints, fewer than {last_k}")
    sources = evaluated[-last_k:]
    paths = [Path(entry["path"]) for entry in sources]
    for entry, path in zip(sources, paths, strict=True):
        if strict_content_hash(path) != entry["checkpoint_content_sha256"]:
            raise ValueError(f"checkpoint {path} changed since the run")
    newest = paths[-1]
    step = sources[-1]["step"]
    target = newest.parent / f"averaged-last{last_k}-step-{step:09d}"
    receipt_path = evidence / f"checkpoint-averaged-last{last_k}.json"
    if target.exists() or receipt_path.exists():
        raise FileExistsError("averaged checkpoint or its receipt already exists")
    from safetensors.torch import load_file, save_file

    states = [load_file(path / "pretrained_model" / MODEL_FILE) for path in paths]
    averaged = average_state_dicts(states)
    shutil.copytree(newest / "pretrained_model", target / "pretrained_model")
    save_file(averaged, target / "pretrained_model" / MODEL_FILE)
    manifest = json.loads((newest / "physical_policy_manifest.json").read_text(encoding="utf-8"))
    manifest["averaged_from_steps"] = [entry["step"] for entry in sources]
    manifest["averaging"] = "uniform mean of float model weights; other tensors from the newest"
    (target / "physical_policy_manifest.json").write_text(_json(manifest), encoding="utf-8")
    digest = strict_content_hash(target)
    config = run["configuration"]
    contract = run["physical_contract"]
    probes = Path(config["probe_set"])
    probes = probes if probes.is_absolute() else repository / probes
    evaluator = CheckpointEvaluator(
        probes, repository, config["probe_sha256"], dataset_root,
        tuple(run["training_episode_indices"]), lane=training_lane(run),
        omitted_episode_indices=tuple(run.get("omitted_episode_indices", ())),
    )
    predict = predictor_factory(target, device)
    result = evaluator.evaluate(
        target, f"{run['policy_id']}_avg{last_k}", step,
        held_out_batches(dataset_root, config["dataset_repo_id"], run["held_out_episodes"],
                         config["chunk_size"], float(contract["requested_rate_hz"])),
        predict, evidence / "evaluations", repository / config["timeline"],
    )
    receipt = {
        **manifest, "path": str(target.resolve()), "checkpoint_content_sha256": digest,
        "evaluation_status": result["status"],
    }
    receipt_path.write_text(_json(receipt), encoding="utf-8")
    axes = (result.get("metrics") or {}).get("axes") or {}
    joints = [axes[f"Joint{i}"]["mae"] for i in range(1, 7) if f"Joint{i}" in axes]
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "policy_id": run["policy_id"],
        "averaged_from_steps": manifest["averaged_from_steps"],
        "checkpoint_path": str(target.resolve()),
        "checkpoint_content_sha256": digest,
        "checkpoint_record": str(receipt_path.resolve()),
        "evaluation_status": result["status"],
        "joint_mae_rad_mean": (sum(joints) / len(joints)) if joints else None,
        "gripper_mae_m": axes.get("Gripper", {}).get("mae"),
    }
