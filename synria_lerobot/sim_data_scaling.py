"""The simulation data-scaling study: train nested subsets, score each, render the curve.

A study trains one frozen simulation dataset at several episode counts with everything
else held fixed: the same held-out probe episodes, seed, step count and architecture.
Subsets are nested by recording order, so the run at 10 episodes trains on exactly the
first 10 of the run at 15. Each run goes through the ordinary simulation-lane trainer,
so it has its own evidence directory, ledger row and hash-bound checkpoint. Two scores
are read back per run: the held-out action error of the final checkpoint, which the
trainer measures itself, and, when a trial campaign is run against the served
checkpoint in the Isaac scene, the closed-loop success rate.

The study file and the rendered Markdown table are the document that says how much
each increment of data bought. Both say they are simulation evidence only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .act_training import train
from .checkpoint_eval import read_episode_metadata
from .exclusions import read_exclusions
from .physical_contract import SIMULATION_LANE, training_lane

STUDY_VERSION = "synria_sim_data_scaling_v1"
TrainFn = Callable[[Path, Path, Path, int, int, dict[str, Any]], dict[str, Any]]
TrialFn = Callable[[dict[str, Any]], dict[str, Any] | None]


def available_episode_count(dataset_root: Path, held_out: list[int] | tuple[int, ...]) -> int:
    """How many episodes a simulation-lane run can train on: all minus held out minus excluded."""
    metadata = read_episode_metadata(dataset_root)
    excluded = set(read_exclusions(dataset_root).excluded_ids)
    return len(set(metadata) - set(held_out) - excluded)


def subset_sizes(requested: list[int] | tuple[int, ...], available: int) -> tuple[int, ...]:
    """Strictly increasing positive sizes no larger than ``available``."""
    sizes = tuple(requested)
    if (
        not sizes
        or any(type(size) is not int or size < 1 for size in sizes)
        or any(b <= a for a, b in zip(sizes, sizes[1:], strict=False))
    ):
        raise ValueError("subset sizes must be strictly increasing positive integers")
    if sizes[-1] > available:
        raise ValueError(
            f"largest subset {sizes[-1]} exceeds the {available} episodes available for training"
        )
    return sizes


def validate_study(spec: dict[str, Any]) -> dict[str, Any]:
    """Check a study specification before any run starts; returns it with defaults filled."""
    if spec.get("version") != STUDY_VERSION:
        raise ValueError("unsupported study specification")
    for key in ("study_id", "dataset_root", "contract", "training_config", "raw_output_root",
                "evidence_root", "policy_prefix"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"study needs a nonempty {key}")
    for key in ("seed", "steps"):
        if type(spec.get(key)) is not int or spec[key] < (0 if key == "seed" else 1):
            raise ValueError(f"study {key} must be a nonnegative integer")
    if not isinstance(spec.get("subset_sizes"), list):
        raise ValueError("study needs a list of subset sizes")
    config = json.loads(Path(spec["training_config"]).read_text(encoding="utf-8"))
    if training_lane(config) != SIMULATION_LANE:
        raise ValueError("the study trains in the simulation lane only")
    if config.get("training_episode_limit") is not None:
        raise ValueError("the base training config must leave training_episode_limit unset")
    return {**spec, "trials": spec.get("trials")}


def run_config(spec: dict[str, Any], base: dict[str, Any], size: int, total: int) -> dict[str, Any]:
    """The trainer configuration for one subset: its own policy id, evidence and limit."""
    tag = f"n{size:03d}"
    return {
        **base,
        "policy_id": f"{spec['policy_prefix']}_{tag}",
        "training_episode_limit": None if size == total else size,
        "evidence_directory": str(Path(spec["evidence_root"]) / f"train_{tag}"),
    }


def held_out_summary(evidence: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """The final checkpoint's held-out metrics as the trainer recorded them."""
    latest = manifest["checkpoints"][-1]
    name = f"{manifest['policy_id']}-step-{latest['step']:09d}.json"
    record_path = evidence / "evaluations" / name
    record = json.loads(record_path.read_text(encoding="utf-8"))
    metrics = record.get("metrics") or {}
    axes = metrics.get("axes") or {}
    joints = [axes[f"Joint{i}"]["mae"] for i in range(1, 7) if f"Joint{i}" in axes]
    return {
        "step": latest["step"],
        "checkpoint_path": latest["path"],
        "checkpoint_content_sha256": latest["checkpoint_content_sha256"],
        "checkpoint_record": str(evidence / f"checkpoint-step-{latest['step']:09d}.json"),
        "evaluation_record": str(record_path),
        "evaluation_status": record.get("status"),
        "held_out_frames": metrics.get("held_out_frames"),
        "joint_mae_rad_mean": (sum(joints) / len(joints)) if joints else None,
        "joint_mae_rad_max": max(joints) if joints else None,
        "gripper_mae_m": axes.get("Gripper", {}).get("mae"),
    }


def run_study(
    spec: dict[str, Any], *, train_fn: TrainFn = train, trial_fn: TrialFn | None = None,
    log: Callable[[str], None] = print, write_to: Path | None = None,
) -> dict[str, Any]:
    """Train every subset in order, score each, and return the study record.

    ``trial_fn`` receives one finished row and returns a trial report (or None to skip);
    the script wires it to a served checkpoint and the scene. With ``write_to`` the
    record and table are written there after every row and once more at the end, so a
    failed run leaves the rows so far on disk marked failed, never a partial curve
    passed off as complete.
    """
    spec = validate_study(spec)
    dataset_root = Path(spec["dataset_root"])
    base = json.loads(Path(spec["training_config"]).read_text(encoding="utf-8"))
    probes = json.loads(Path(base["probe_set"]).read_text(encoding="utf-8")) if Path(
        base["probe_set"]).is_absolute() else json.loads(
        (Path(base["repository"]) / base["probe_set"]).read_text(encoding="utf-8"))
    held_out = list(probes["held_out_episodes"])
    available = available_episode_count(dataset_root, held_out)
    sizes = subset_sizes(spec["subset_sizes"], available)
    study: dict[str, Any] = {
        "version": STUDY_VERSION,
        "study_id": spec["study_id"],
        "lane": SIMULATION_LANE,
        "physical_evidence": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "held_out_episodes": held_out,
        "available_episodes": available,
        "subset_sizes": list(sizes),
        "seed": spec["seed"],
        "steps": spec["steps"],
        "status": "started",
        "rows": [],
    }
    try:
        for size in sizes:
            config = run_config(spec, base, size, available)
            output = Path(spec["raw_output_root"]) / config["policy_id"]
            log(f"study {spec['study_id']}: training {config['policy_id']} on {size} episodes")
            manifest = train_fn(
                dataset_root, Path(spec["contract"]), output, spec["seed"], spec["steps"], config,
            )
            evidence = Path(config["repository"]) / config["evidence_directory"]
            row: dict[str, Any] = {
                "episodes_trained": size,
                "policy_id": config["policy_id"],
                "training_episode_indices": list(manifest["training_episode_indices"]),
                "omitted_episode_indices": list(manifest["omitted_episode_indices"]),
                "training_status": manifest["status"],
                "evidence_directory": str(evidence),
                **held_out_summary(evidence, manifest),
                "trials": None,
            }
            if trial_fn is not None:
                row["trials"] = trial_fn(row)
            study["rows"].append(row)
            if write_to is not None:
                write_study(study, write_to)
        study["status"] = "completed"
    except BaseException as error:
        study["status"] = "failed"
        study["error"] = str(error) or type(error).__name__
        raise
    finally:
        study["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if write_to is not None:
            write_study(study, write_to)
    return study


def add_trials(
    study: dict[str, Any], trial_fn: TrialFn, *, write_to: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the trial campaign for every row of a completed study that has none yet.

    Training is never repeated: the rows' saved checkpoints are served as they are.
    The record is rewritten after every campaign when ``write_to`` is given.
    """
    if study.get("status") != "completed":
        raise ValueError("trials are added to a completed study only")
    if training_lane(study) != SIMULATION_LANE:
        raise ValueError("trial campaigns belong to the simulation lane")
    for row in study["rows"]:
        if row.get("trials") is not None:
            continue
        log(f"study {study['study_id']}: trials for {row['policy_id']}")
        row["trials"] = trial_fn(row)
        if write_to is not None:
            write_study(study, write_to)
    study["trials_utc"] = datetime.now(timezone.utc).isoformat()
    if write_to is not None:
        write_study(study, write_to)
    return study


def _number(value: Any, digits: int, unit: str = "") -> str:
    """A table cell for an optional number."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "not measured"
    return f"{value:.{digits}f}{unit}"


def render_markdown(study: dict[str, Any]) -> str:
    """The curve as a Markdown document: one row per subset, largest last."""
    lines = [
        f"# Simulation data-scaling study `{study['study_id']}`",
        "",
        "Simulation lane; never physical evidence. Every run trains the same frozen",
        "dataset with the same held-out episodes, seed, step count and architecture;",
        "subsets are nested by recording order, so each row contains every row above it.",
        "Held-out error is the trainer's own measure on the final checkpoint; the success",
        "rate is the closed-loop cup-lift score in the Isaac scene, when a campaign ran.",
        "",
        f"- dataset: `{study['dataset_root']}`",
        f"- held-out episodes: {study['held_out_episodes']}",
        f"- episodes available for training: {study['available_episodes']}",
        f"- seed {study['seed']}, {study['steps']} steps per run, status {study['status']}",
        "",
        "| episodes trained | policy id | held-out joint MAE, mean (rad) | max (rad) "
        "| gripper MAE (m) | sim success | trials | campaign report |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in study["rows"]:
        trials = row.get("trials") or {}
        rate = trials.get("success_rate")
        count = trials.get("trials")
        report = trials.get("report")
        lines.append(
            f"| {row['episodes_trained']} | {row['policy_id']} "
            f"| {_number(row.get('joint_mae_rad_mean'), 4)} "
            f"| {_number(row.get('joint_mae_rad_max'), 4)} "
            f"| {_number(row.get('gripper_mae_m'), 4)} "
            f"| {_number(rate * 100 if rate is not None else None, 0, '%')} "
            f"| {'' if count is None else count} "
            f"| {'' if report is None else '`' + str(report) + '`'} |"
        )
    lines += [
        "",
        "Caveats: recording order is not random, since the operator's picks improved",
        "across the session, so later episodes are not interchangeable with earlier ones;",
        "one seed per size gives no variance estimate; a simulation success says nothing",
        "about the physical arm.",
        "",
    ]
    return "\n".join(lines)


def write_study(study: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    """Write ``study.json`` and ``STUDY.md`` into the study's evidence directory."""
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / "study.json"
    record.write_text(json.dumps(study, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    table = directory / "STUDY.md"
    table.write_text(render_markdown(study), encoding="utf-8")
    return record, table
