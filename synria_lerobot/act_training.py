"""Local upstream ACT training with frozen diagnostic evidence after every save."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import re
import subprocess
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_eval import (
    CheckpointEvaluator,
    PredictionBatch,
    _artifact_destination,
    load_probes,
    read_episode_metadata,
    strict_content_hash,
)
from .evaluation import append_policy_row
from .physical_contract import PhysicalDatasetContract

TRAINING_VERSION = "synria_act_training_v1"
CADENCE_SEMANTICS = (
    "absolute target offered now and held until the next command tick; "
    "chunk slots 0,q,2q preserve recorded lookahead; physical cadence is unmeasured"
)
FEATURE_KEYS = ("observation.state", "observation.images.wrist", "observation.images.front")


def _save_upstream_checkpoint(
    save_checkpoint: Callable[..., None],
    checkpoint: Path,
    step: int,
    config: Any,
    policy: Any,
    optimizer: Any,
    *,
    preprocessor: Any,
    postprocessor: Any,
    accelerator: Any,
) -> None:
    """Adapt the named upstream 0.6 save API without retrying failed saves."""
    kwargs = {"preprocessor": preprocessor, "postprocessor": postprocessor}
    parameter = inspect.signature(save_checkpoint).parameters.get("accelerator")
    if parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    ):
        kwargs["accelerator"] = accelerator
    save_checkpoint(
        checkpoint, step, config, accelerator.unwrap_model(policy), optimizer, **kwargs
    )


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, not boolean")
    return value


def _number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value < 0 if allow_zero else value <= 0)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(value))


def _path(value: Any, repository: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("explicit nonempty path required")
    path = Path(value)
    return path if path.is_absolute() else repository / path


def validate_config(config: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if config.get("version") != TRAINING_VERSION:
        raise ValueError("unsupported training configuration")
    if (
        not isinstance(config.get("policy_id"), str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", config["policy_id"]) is None
    ):
        raise ValueError("explicit simple policy id required")
    if not isinstance(config.get("dataset_repo_id"), str) or "/" not in config["dataset_repo_id"]:
        raise ValueError("explicit local dataset repo id required")
    if _json(config.get("expected_contract")) != _json(contract):
        raise ValueError("expected contract, action source, lookahead and rate must match dataset")
    if type(contract.get("state_has_velocity")) is not bool:
        raise ValueError("contract state_has_velocity must be boolean")
    for key in ("action_lookahead_steps", "effective_action_lookahead_steps"):
        _integer(contract.get(key), key, minimum=0)
    for key in ("requested_rate_hz", "nominal_action_lookahead_s"):
        _number(contract.get(key), key, allow_zero=key != "requested_rate_hz")
    physical = PhysicalDatasetContract.from_dict(contract)
    physical.require_qualifying()
    if physical.as_dict(fps=contract["requested_rate_hz"]) != contract:
        raise ValueError("inconsistent physical contract metadata")
    period = _number(config.get("command_period_s"), "command_period_s")
    stride = period * contract["requested_rate_hz"]
    if (
        not math.isfinite(stride)
        or stride < 1
        or not math.isclose(stride, round(stride), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("command period times dataset fps must be a positive integer stride")
    chunk = _integer(config.get("chunk_size"), "chunk_size")
    if chunk < round(stride):
        raise ValueError("chunk_size must cover at least one command-period stride")
    for key in ("batch_size", "save_freq"):
        _integer(config.get(key), key)
    for key in ("response_timeout_s", "learning_rate", "backbone_learning_rate", "grad_clip_norm"):
        _number(config.get(key), key)
    _number(config.get("weight_decay"), "weight_decay", allow_zero=True)
    if config.get("device") not in {"cpu", "cuda"} or config.get("video_backend") != "pyav":
        raise ValueError("explicit cpu/cuda device and local pyav video backend required")
    architecture = config.get("architecture")
    keys = {
        "dim_model",
        "n_heads",
        "dim_feedforward",
        "n_encoder_layers",
        "n_decoder_layers",
        "latent_dim",
        "n_vae_encoder_layers",
        "dropout",
        "kl_weight",
    }
    if not isinstance(architecture, dict) or set(architecture) != keys:
        raise ValueError("explicit ACT architecture fields required")
    for key in keys - {"dropout", "kl_weight"}:
        _integer(architecture[key], key)
    dropout = _number(architecture["dropout"], "dropout", allow_zero=True)
    _number(architecture["kl_weight"], "kl_weight", allow_zero=True)
    if dropout >= 1 or architecture["dim_model"] % architecture["n_heads"]:
        raise ValueError("dropout must be below one and model dimension divisible by head count")
    return {
        "command_stride_steps": round(stride),
        "command_period_s": period,
        "dataset_rate_hz": contract["requested_rate_hz"],
        "action_lookahead_steps": contract["action_lookahead_steps"],
        "semantics": CADENCE_SEMANTICS,
    }


def training_stats(root: Path, episode_ids: tuple[int, ...], state_size: int) -> dict[str, Any]:
    """Aggregate only selected episode statistics, never the dataset-global statistics."""
    import pyarrow.parquet as parquet
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import cast_stats_to_numpy
    from lerobot.utils.utils import unflatten_dict

    selected = []
    seen = set()
    shapes = {
        "observation.state": (state_size,),
        "action": (7,),
        "observation.images.wrist": (3, 1, 1),
        "observation.images.front": (3, 1, 1),
    }
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        for row in parquet.read_table(path).to_pylist():
            episode = row["episode_index"]
            if episode not in episode_ids:
                continue
            if episode in seen:
                raise ValueError("duplicate training episode statistics")
            seen.add(episode)
            flat = {
                key.removeprefix("stats/"): value
                for key, value in row.items()
                if key.startswith("stats/")
            }
            stats = cast_stats_to_numpy(unflatten_dict(flat))
            for key, shape in shapes.items():
                if key not in stats:
                    raise ValueError("training episode lacks required feature statistics")
                for metric in ("mean", "std", "min", "max"):
                    values = np.asarray(stats[key].get(metric))
                    if (
                        values.shape != shape
                        or values.dtype.kind not in "fiu"
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError("invalid training feature statistics shape or value")
                count = np.asarray(stats[key].get("count"))
                if (
                    count.size != 1
                    or count.dtype.kind not in "fiu"
                    or not np.isfinite(count).all()
                    or np.any(count <= 0)
                ):
                    raise ValueError("positive training statistics count required")
                if np.any(stats[key]["std"] < 0):
                    raise ValueError("training standard deviation cannot be negative")
            selected.append({key: stats[key] for key in shapes})
    if seen != set(episode_ids) or not selected:
        raise ValueError("training statistics do not cover the exact training episode set")
    return dict(aggregate_stats(selected))


def _plain_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {metric: np.asarray(value).tolist() for metric, value in fields.items()}
        for key, fields in stats.items()
    }


def _policy_features(metadata: Any, state_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.configs.types import FeatureType, PolicyFeature

    features = metadata.features
    if tuple(features["observation.state"]["shape"]) != (state_size,) or tuple(
        features["action"]["shape"]
    ) != (7,):
        raise ValueError("dataset state/action shapes differ from the physical contract")
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_size,))}
    image_shapes = []
    for key in FEATURE_KEYS[1:]:
        feature = features[key]
        shape = tuple(feature["shape"])
        if (
            feature["dtype"] not in {"image", "video"}
            or len(shape) != 3
            or feature.get("names") != ["channels", "height", "width"]
            or shape[0] != 3
            or min(shape) <= 0
        ):
            raise ValueError("dataset images must use the D1 named CHW RGB feature declaration")
        image_shapes.append(shape)
        inputs[key] = PolicyFeature(FeatureType.VISUAL, shape)
    if image_shapes[0] != image_shapes[1]:
        raise ValueError("ACT requires equal stored camera shapes")
    return inputs, {"action": PolicyFeature(FeatureType.ACTION, (7,))}


def _evaluation_batches(loader: Any) -> Iterator[PredictionBatch]:
    for batch in loader:
        yield PredictionBatch(
            {key: batch[key] for key in FEATURE_KEYS},
            batch["action"].cpu().numpy(),
            batch["action_is_pad"].cpu().numpy(),
            tuple(int(value) for value in batch["episode_index"].reshape(-1).tolist()),
            tuple(int(value) for value in batch["frame_index"].reshape(-1).tolist()),
        )


def train(
    dataset_root: Path,
    contract_path: Path,
    output: Path,
    seed: int,
    steps: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run local single-process training; never import robot drivers or publish commands."""
    _integer(seed, "seed", minimum=0)
    _integer(steps, "steps")
    if seed >= 2**32:
        raise ValueError("seed must be below 2**32")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if _json(contract) != _json(
        json.loads((dataset_root / "physical_contract.json").read_text(encoding="utf-8"))
    ):
        raise ValueError("contract input differs from dataset contract")
    cadence = validate_config(config, contract)
    repository = Path(config["repository"]).resolve()
    probes = _path(config["probe_set"], repository)
    evidence = _path(config["evidence_directory"], repository)
    timeline = _path(config["timeline"], repository)
    ledger = _path(config["policy_ledger"], repository)
    if any(not path.resolve().is_relative_to(repository) for path in (timeline, ledger)):
        raise ValueError("timeline and policy ledger must remain in the repository")
    if timeline.resolve() == ledger.resolve() or any(
        path.resolve().is_relative_to(evidence.resolve()) for path in (timeline, ledger)
    ):
        raise ValueError("timeline, ledger and owned run evidence must be separate")
    if output.exists() or evidence.exists():
        raise FileExistsError("new unique raw output and evidence directories are required")
    if output.resolve().is_relative_to(repository) or not evidence.resolve().is_relative_to(
        repository
    ):
        raise ValueError(
            "raw training output stays outside git; small evidence stays in repository"
        )
    for destination in (output, evidence, timeline, ledger):
        _artifact_destination(destination, (dataset_root,), (probes, contract_path))
    if any(
        path.resolve().is_relative_to(output.resolve())
        for path in (probes, evidence, timeline, ledger, dataset_root)
    ):
        raise ValueError("training output must not contain inputs or evidence destinations")
    ledger_text = ledger.read_text(encoding="utf-8")
    if any(
        line.startswith("|")
        and len(line.split("|")) > 3
        and line.split("|")[2].strip() == config["policy_id"]
        for line in ledger_text.splitlines()
    ):
        raise ValueError("policy id already recorded")
    frozen = load_probes(probes, repository, config["probe_sha256"])
    metadata = read_episode_metadata(dataset_root)
    training_ids = tuple(sorted(set(metadata) - set(frozen["held_out_episodes"])))
    evaluator = CheckpointEvaluator(
        probes, repository, config["probe_sha256"], dataset_root, training_ids
    )
    if evaluator.contract != contract:
        raise ValueError("frozen dataset contract differs from requested contract")
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    # Apply offline settings before optional runtime imports. No pretrained assets or uploads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    if config["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from accelerate import Accelerator
    from lerobot.common.train_utils import save_checkpoint
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    installed = importlib.metadata.version("lerobot")
    if installed.split(".")[:2] != ["0", "6"]:
        raise ValueError("upstream LeRobot 0.6 is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_size = 13 if contract["state_has_velocity"] else 7
    stats = training_stats(dataset_root, training_ids, state_size)
    delta = {
        "action": [index / contract["requested_rate_hz"] for index in range(config["chunk_size"])]
    }
    dataset_args = dict(
        repo_id=config["dataset_repo_id"],
        root=dataset_root,
        delta_timestamps=delta,
        video_backend=config["video_backend"],
        download_videos=False,
        token=False,
    )
    training = LeRobotDataset(**dataset_args, episodes=list(training_ids))
    held_out = LeRobotDataset(**dataset_args, episodes=frozen["held_out_episodes"])
    if training.meta.fps != contract["requested_rate_hz"]:
        raise ValueError("library dataset rate differs from physical contract")
    inputs, outputs = _policy_features(training.meta, state_size)
    policy_config = ACTConfig(
        input_features=inputs,
        output_features=outputs,
        chunk_size=config["chunk_size"],
        n_action_steps=config["chunk_size"],
        device=config["device"],
        push_to_hub=False,
        pretrained_backbone_weights=None,
        optimizer_lr=config["learning_rate"],
        optimizer_lr_backbone=config["backbone_learning_rate"],
        optimizer_weight_decay=config["weight_decay"],
        **config["architecture"],
    )
    accelerator = Accelerator(cpu=config["device"] == "cpu", mixed_precision="no")
    if accelerator.num_processes != 1:
        raise ValueError("this evidence wrapper supports a single training process only")
    policy = ACTPolicy(policy_config)
    optimizer_config = policy_config.get_optimizer_preset()
    optimizer = optimizer_config.build(policy.get_optim_params())
    preprocessor, postprocessor = make_act_pre_post_processors(policy_config, dataset_stats=stats)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    loader = torch.utils.data.DataLoader(
        training,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    held_loader = torch.utils.data.DataLoader(
        held_out, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    pipeline = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=config["dataset_repo_id"],
            root=str(dataset_root),
            episodes=list(training_ids),
            use_imagenet_stats=False,
            video_backend=config["video_backend"],
        ),
        policy=policy_config,
        output_dir=output,
        job_name=config["policy_id"],
        seed=seed,
        batch_size=config["batch_size"],
        steps=steps,
        num_workers=0,
        env=None,
        env_eval_freq=0,
        eval_steps=0,
        save_checkpoint=True,
        save_freq=config["save_freq"],
        optimizer=optimizer_config,
        scheduler=None,
        wandb=WandBConfig(enable=False),
        save_checkpoint_to_hub=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "version": TRAINING_VERSION,
        "policy_id": config["policy_id"],
        "utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_root.resolve()),
        "dataset_content_sha256": evaluator.dataset_hash,
        "physical_contract": contract,
        "probe_sha256": evaluator.probe_hash,
        "probe_set": str(probes.resolve()),
        "training_episode_indices": training_ids,
        "held_out_episodes": frozen["held_out_episodes"],
        "cadence": cadence,
        "configuration": config,
        "seed": seed,
        "requested_steps": steps,
        "git_sha": git_sha,
        "host": platform.node(),
        "upstream_version": installed,
        "torch_version": torch.__version__,
        "normalization": "training episodes only",
        "task_text_conditioning": False,
        "declared_input_conditioning": list(FEATURE_KEYS),
        "visual_goal_cue_declared": False,
        "varying_target_motion_eligible": False,
        "training_stats": _plain_stats(stats),
        "pretrained_backbone_weights": None,
        "stage_status": "planned",
        "status": "started",
        "checkpoints": [],
        "partial_checkpoints": [],
        "completed_steps": 0,
    }
    _write(evidence / "started.json", manifest)
    failure: BaseException | None = None
    optimization_started = False
    metrics = MetricsTracker(
        config["batch_size"],
        len(training),
        len(training_ids),
        {key: AverageMeter(key) for key in ("loss", "grad_norm", "lr", "update_s")},
    )

    def predict(observations: dict[str, Any]) -> np.ndarray:
        model = accelerator.unwrap_model(policy)
        model.eval()
        with torch.inference_mode():
            prediction = model.predict_action_chunk(preprocessor(observations))
            return np.asarray(postprocessor(prediction).detach().cpu().numpy())

    try:
        iterator = iter(loader)
        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimization_started = True
            metrics, _ = update_policy(
                metrics,
                policy,
                preprocessor(batch),
                optimizer,
                config["grad_clip_norm"],
                accelerator,
            )
            metrics.step()
            current = metrics.to_dict()
            if not all(math.isfinite(float(current[key])) for key in ("loss", "grad_norm")) or any(
                not torch.isfinite(parameter).all().item() for parameter in policy.parameters()
            ):
                raise ValueError("nonfinite optimizer result; no checkpoint saved for this step")
            manifest["completed_steps"] = step
            manifest["last_training_metrics"] = current
            if step % config["save_freq"] and step != steps:
                continue
            checkpoint = output / "checkpoints" / f"step-{step:09d}"
            if checkpoint.exists():
                raise FileExistsError("checkpoint output must not exist")
            checkpoint_manifest = {
                key: value
                for key, value in manifest.items()
                if key
                not in {"checkpoints", "partial_checkpoints", "status", "last_training_metrics"}
            }
            checkpoint_manifest["step"] = step
            try:
                _write(checkpoint / "physical_policy_manifest.json", checkpoint_manifest)
                _save_upstream_checkpoint(
                    save_checkpoint,
                    checkpoint,
                    step,
                    pipeline,
                    policy,
                    optimizer,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    accelerator=accelerator,
                )
            except BaseException as error:
                manifest["partial_checkpoints"].append(
                    {
                        "step": step,
                        "path": str(checkpoint.resolve()),
                        "status": "incomplete; do not load or evaluate",
                        "error": str(error),
                    }
                )
                raise
            digest = strict_content_hash(checkpoint)
            entry = {
                "step": step,
                "path": str(checkpoint.resolve()),
                "checkpoint_content_sha256": digest,
                "evaluation_status": "pending",
            }
            manifest["checkpoints"].append(entry)
            evaluation_error: BaseException | None = None
            try:
                result = evaluator.evaluate(
                    checkpoint,
                    config["policy_id"],
                    step,
                    _evaluation_batches(held_loader),
                    predict,
                    evidence / "evaluations",
                    timeline,
                )
                entry["evaluation_status"] = result["status"]
            except BaseException as error:
                entry["evaluation_status"] = "failed"
                evaluation_error = error
            finally:
                try:
                    _write(
                        evidence / f"checkpoint-step-{step:09d}.json",
                        {**checkpoint_manifest, **entry},
                    )
                except BaseException as error:
                    entry["evidence_error"] = str(error)
                    evaluation_error = evaluation_error or error
            if evaluation_error is not None:
                raise evaluation_error
        manifest["status"] = "completed"
    except BaseException as error:
        failure = error
        manifest["status"] = "failed"
        manifest["error"] = str(error) or type(error).__name__
    finally:
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        finalization_errors = []
        try:
            _write(evidence / "training_run.json", manifest)
        except BaseException as error:
            failure = failure or error
            finalization_errors.append(f"run manifest: {error}")
        if optimization_started:
            latest = manifest["checkpoints"][-1] if manifest["checkpoints"] else None
            fields = [
                manifest["utc"],
                config["policy_id"],
                f"{config['dataset_repo_id']}; {len(metadata)} episodes; {evaluator.dataset_hash}",
                f"ACT; seed={seed}; steps={steps}; config={evidence / 'training_run.json'}",
                latest["checkpoint_content_sha256"] if latest else "N/A (no finalized checkpoint)",
                str(evidence / "training_run.json"),
                f"{manifest['status']}; {manifest['completed_steps']} completed steps",
                "Diagnostic only; no D2 policy selection or physical result",
            ]
            try:
                append_policy_row(
                    ledger,
                    [
                        field.replace("|", "&#124;").replace("\n", "\\n").replace("\r", "\\r")
                        for field in fields
                    ],
                    response_timeout_s=config["response_timeout_s"],
                )
            except BaseException as error:
                failure = failure or error
                finalization_errors.append(f"policy ledger: {error}")
        if finalization_errors:
            try:
                _write(
                    evidence / "finalization_errors.json",
                    {
                        "primary_error": str(failure),
                        "errors": finalization_errors,
                        "policy_id": config["policy_id"],
                        "stage_status": "planned",
                    },
                )
            except BaseException as error:
                finalization_errors.append(f"finalization error evidence: {error}")
    if failure is not None:
        detail = "; ".join(finalization_errors)
        raise RuntimeError(
            f"training failed; inspect evidence at {evidence}; {detail}"
        ) from failure
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "contract", "output", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    result = train(
        args.dataset_root,
        args.contract,
        args.output,
        args.seed,
        args.steps,
        json.loads(args.config.read_text(encoding="utf-8")),
    )
    print(
        _json(
            {
                "policy_id": result["policy_id"],
                "status": result["status"],
                "completed_steps": result["completed_steps"],
            }
        )
    )


if __name__ == "__main__":
    main()
