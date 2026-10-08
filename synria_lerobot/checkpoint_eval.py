"""Frozen diagnostic probes, per-save evidence and local checkpoint playback."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
import math
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .embodiment import SynriaObservation
from .evaluation import committed_bytes
from .physical_contract import DRIVER_JOINT_NAMES, ActionSource, PhysicalDatasetContract
from .policy_client import GuardedCommandPath, HttpPolicyTransport, PolicyClient, PolicyTransport
from .quality_gates import OBJECT_SUCCESS_LIMITATION, load_limits
from .task_registry import TaskDefinition

if TYPE_CHECKING:
    from .sessions import SessionIO

PROBE_VERSION = "synria_checkpoint_probes_v1"
DIAGNOSTIC_PURPOSE = "diagnostic_only_no_policy_selection"


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise ValueError("an explicit simple identifier is required")
    return value


def _artifact_destination(path: Path, roots: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
    absolute = path.absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError("artifact destination must not use symbolic links")
    resolved = absolute.resolve()
    if any(resolved.is_relative_to(root.resolve()) for root in roots) or any(
        resolved == source.resolve() for source in inputs
    ):
        raise ValueError("artifact destination must not modify frozen inputs")


def strict_content_hash(root: Path) -> str:
    """Hash a nonempty regular-file tree without following symbolic links."""
    if (
        any(part.is_symlink() for part in (root.absolute(), *root.absolute().parents))
        or not root.is_dir()
    ):
        raise ValueError("content root must be an existing non-symlink directory")
    files = []
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("content tree must contain only regular files and directories")
        if path.is_file():
            files.append(path)
    if not files or not any(path.stat().st_size for path in files):
        raise ValueError("content root contains no nonempty files")
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big") + relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        read = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                read += len(block)
        if read != size:
            raise ValueError("content changed while hashing")
    return digest.hexdigest()


def read_episode_metadata(root: Path) -> dict[int, int]:
    """Read only local version-three episode metadata, without a remote dataset lookup."""
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("episode metadata requires the optional dataset dependencies") from error
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not files or any(
        type(info.get(key)) is not int or info[key] <= 0
        for key in ("total_episodes", "total_frames")
    ):
        raise ValueError("nonempty finalized dataset episode metadata required")
    rows = [row for path in files for row in parquet.read_table(path).to_pylist()]
    rows.sort(key=lambda row: row["episode_index"])
    if len(rows) != info["total_episodes"]:
        raise ValueError("episode metadata count differs from dataset info")
    result, end = {}, 0
    for index, row in enumerate(rows):
        keys = ("episode_index", "length", "dataset_from_index", "dataset_to_index")
        if (
            any(type(row.get(key)) is not int for key in keys)
            or row["episode_index"] != index
            or row["length"] <= 0
            or row["dataset_from_index"] != end
            or row["dataset_to_index"] != end + row["length"]
        ):
            raise ValueError("episode metadata ranges must be complete and contiguous")
        result[index], end = row["length"], row["dataset_to_index"]
    if end != info["total_frames"]:
        raise ValueError("episode frame count differs from dataset info")
    return result


def probe_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(_json({**config, "probe_sha256": ""}).encode()).hexdigest()


def _episode_ids(values: Any) -> tuple[int, ...]:
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(type(value) is not int or value < 0 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("nonempty unique nonnegative episode indices required")
    return tuple(values)


def validate_probes(config: dict[str, Any]) -> None:
    if config.get("version") != PROBE_VERSION or config.get("purpose") != DIAGNOSTIC_PURPOSE:
        raise ValueError("frozen probes are diagnostic only, never D2 or policy selection")
    _identifier(config.get("probe_set_id"))
    for key in ("registered_by", "registered_on"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError("operator registration fields are required")
    if re.fullmatch(r"[a-f0-9]{64}", str(config.get("dataset_content_sha256", ""))) is None:
        raise ValueError("explicit dataset content hash required")
    ids = _episode_ids(config.get("held_out_episodes"))
    counts = config.get("held_out_frame_counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != {str(index) for index in ids}
        or any(type(value) is not int or value <= 0 for value in counts.values())
    ):
        raise ValueError("each held-out episode needs its frozen positive frame count")
    trials = config.get("physical_trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("fixed physical start/target/token trials required")
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("physical trial must be an object")
        identifier = _identifier(trial.get("trial_id"))
        if identifier in seen or any(
            not isinstance(trial.get(key), str) or not trial[key].strip()
            for key in ("start_square", "target_square", "token")
        ):
            raise ValueError("unique trial id and explicit start/target/token required")
        seen.add(identifier)
    capture = config.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("explicit fixed probe capture settings required")
    for key in ("fps", "max_duration_s", "shutdown_timeout_s"):
        _positive(capture.get(key), "probe capture " + key)


def register_probes(path: Path) -> str:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_probes(config)
    config["probe_sha256"] = probe_digest(config)
    path.write_text(_json(config), encoding="utf-8")
    return str(config["probe_sha256"])


def load_probes(path: Path, repository: Path, expected_hash: str) -> dict[str, Any]:
    committed_bytes(path, repository)
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    validate_probes(config)
    if config.get("probe_sha256") != expected_hash or probe_digest(config) != expected_hash:
        raise ValueError("frozen probe hash mismatch")
    return config


@dataclass(frozen=True)
class PredictionBatch:
    observations: dict[str, Any]
    actions: np.ndarray
    action_is_pad: np.ndarray
    episode_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]


def prediction_metrics(
    batches: Iterable[PredictionBatch],
    predict: Callable[[dict[str, Any]], Any],
    frame_counts: dict[int, int],
) -> dict[str, Any]:
    absolute, squared, maximum = np.zeros(7), np.zeros(7), np.zeros(7)
    seen: set[tuple[int, int]] = set()
    count = 0
    for batch in batches:
        labels, mask = np.asarray(batch.actions), np.asarray(batch.action_is_pad)
        if labels.ndim != 3 or labels.shape[-1] != 7 or labels.dtype.kind not in "fiu":
            raise ValueError("held-out actions must have numeric BxTx7 shape")
        if (
            not np.isfinite(labels).all()
            or mask.dtype != np.bool_
            or mask.shape != labels.shape[:2]
        ):
            raise ValueError("finite targets and a matching boolean padding mask required")
        if (
            len(batch.episode_indices) != len(labels)
            or len(batch.frame_indices) != len(labels)
            or not len(labels)
            or not labels.shape[1]
        ):
            raise ValueError("episode/frame indices must match the nonempty batch")
        for row, (episode, frame) in enumerate(
            zip(batch.episode_indices, batch.frame_indices, strict=True)
        ):
            if (
                type(episode) is not int
                or type(frame) is not int
                or episode not in frame_counts
                or not 0 <= frame < frame_counts[episode]
                or (episode, frame) in seen
            ):
                raise ValueError("unexpected or duplicate held-out frame")
            expected_pad = np.arange(labels.shape[1]) + frame >= frame_counts[episode]
            if not np.array_equal(mask[row], expected_pad):
                raise ValueError("padding differs from the frozen episode boundary")
            seen.add((episode, frame))
        if any(key.startswith("action") for key in batch.observations):
            raise ValueError("prediction inputs must not include target actions or padding labels")
        predictions = np.asarray(predict(batch.observations))
        if predictions.shape != labels.shape or predictions.dtype.kind not in "fiu":
            raise ValueError("predictions must match BxTx7 physical actions")
        if not np.isfinite(predictions).all():
            raise ValueError("predictions must be finite")
        error = predictions.astype(np.float64)[~mask] - labels.astype(np.float64)[~mask]
        absolute += np.abs(error).sum(axis=0)
        squared += np.square(error).sum(axis=0)
        maximum = np.maximum(maximum, np.abs(error).max(axis=0))
        count += len(error)
    expected = {(episode, frame) for episode, size in frame_counts.items() for frame in range(size)}
    if seen != expected or not count:
        raise ValueError("held-out evaluation must cover every frozen frame exactly once")
    metrics: dict[str, Any] = {
        "valid_action_count": count,
        "held_out_frames": len(seen),
        "axes": {},
    }
    for index, name in enumerate((*DRIVER_JOINT_NAMES, "Gripper")):
        metrics["axes"][name] = dict(
            unit="rad" if index < 6 else "m",
            mae=float(absolute[index] / count),
            rmse=float(np.sqrt(squared[index] / count)),
            max_error=float(maximum[index]),
        )
    _json(metrics)
    return metrics


class CheckpointEvaluator:
    """Construct before training, then call after every saved checkpoint, without selection."""

    def __init__(
        self,
        probes: Path,
        repository: Path,
        expected_probe_hash: str,
        dataset_root: Path,
        training_episode_indices: tuple[int, ...],
        *,
        metadata_reader: Callable[[Path], dict[int, int]] = read_episode_metadata,
    ) -> None:
        self.probe_path, self.repository, self.probe_hash = probes, repository, expected_probe_hash
        self.probes = load_probes(probes, repository, expected_probe_hash)
        self.training_ids = _episode_ids(training_episode_indices)
        if set(self.training_ids) & set(self.probes["held_out_episodes"]):
            raise ValueError("training and frozen held-out episodes must be disjoint")
        self.dataset_root = dataset_root
        self.dataset_hash = strict_content_hash(dataset_root)
        if self.dataset_hash != self.probes["dataset_content_sha256"]:
            raise ValueError("dataset does not match frozen probes")
        self.episode_lengths = metadata_reader(dataset_root)
        held_out = {int(key): value for key, value in self.probes["held_out_frame_counts"].items()}
        if (
            not self.episode_lengths
            or set(self.training_ids) != set(self.episode_lengths) - set(held_out)
            or any(self.episode_lengths.get(index) != count for index, count in held_out.items())
        ):
            raise ValueError(
                "training and held-out data must cover actual episodes with matching lengths"
            )
        self.contract = json.loads(
            (dataset_root / "physical_contract.json").read_text(encoding="utf-8")
        )
        contract = PhysicalDatasetContract.from_dict(self.contract)
        if contract.as_dict(fps=self.contract["requested_rate_hz"]) != self.contract:
            raise ValueError("dataset physical contract metadata is inconsistent")

    def evaluate(
        self,
        checkpoint: Path,
        policy_id: str,
        step: int,
        batches: Iterable[PredictionBatch],
        predict: Callable[[dict[str, Any]], Any],
        output_directory: Path,
        timeline: Path,
    ) -> dict[str, Any]:
        policy_id = _identifier(policy_id)
        if type(step) is not int or step < 0:
            raise ValueError("checkpoint step must be a nonnegative integer")
        checkpoint_hash = strict_content_hash(checkpoint)
        destination = output_directory / f"{policy_id}-step-{step:09d}.json"
        for artifact in (output_directory, destination, timeline):
            _artifact_destination(artifact, (self.dataset_root, checkpoint), (self.probe_path,))
        if timeline.resolve() == destination.resolve():
            raise ValueError("timeline must be separate from immutable checkpoint evidence")
        if destination.exists():
            raise FileExistsError("checkpoint evaluation evidence is immutable")
        record: dict[str, Any] = {
            "kind": "checkpoint_eval",
            "purpose": DIAGNOSTIC_PURPOSE,
            "policy_id": policy_id,
            "step": step,
            "utc": datetime.now(timezone.utc).isoformat(),
            "dataset_content_sha256": self.dataset_hash,
            "dataset_path": str(self.dataset_root.resolve()),
            "checkpoint_content_sha256": checkpoint_hash,
            "probe_sha256": self.probe_hash,
            "probe_set_id": self.probes["probe_set_id"],
            "physical_contract": self.contract,
            "training_episode_indices": self.training_ids,
            "held_out_episodes": self.probes["held_out_episodes"],
            "held_out_frame_counts": self.probes["held_out_frame_counts"],
            "checkpoint_path": str(checkpoint.resolve()),
            "status": "failed",
            "metrics": None,
            "physical_probe_attempts": [],
            "stage_status": "planned",
        }
        failure: BaseException | None = None
        try:
            load_probes(self.probe_path, self.repository, self.probe_hash)
            if strict_content_hash(self.dataset_root) != self.dataset_hash:
                raise ValueError("dataset changed after frozen preflight")
            metrics = prediction_metrics(
                batches,
                predict,
                {int(key): value for key, value in self.probes["held_out_frame_counts"].items()},
            )
            load_probes(self.probe_path, self.repository, self.probe_hash)
            if strict_content_hash(self.dataset_root) != self.dataset_hash:
                raise ValueError("dataset changed during checkpoint evaluation")
            if strict_content_hash(checkpoint) != checkpoint_hash:
                raise ValueError("checkpoint changed during evaluation")
            record["metrics"] = metrics
            record["status"] = "evaluated"
        except BaseException as error:
            failure = error
            record["metrics"] = None
            record["error"] = str(error) or type(error).__name__
        output_directory.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(_json(record))
        timeline.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "t_utc": record["utc"],
            "kind": "checkpoint_eval",
            "title": f"Diagnostic checkpoint evaluation: {policy_id}, step {step}",
            "numbers": {
                "step": step,
                "held_out_frames": sum(self.probes["held_out_frame_counts"].values()),
                "failed_evaluations": int(record["status"] == "failed"),
                "metrics": record["metrics"],
            },
            "status": record["status"],
            "policy_id": policy_id,
            "step": step,
            "artifact": str(destination),
            "probe_sha256": self.probe_hash,
            "error": record.get("error"),
        }
        with timeline.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        if failure is not None:
            raise RuntimeError(
                f"checkpoint evaluation failed; evidence retained at {destination}"
            ) from failure
        return record


def _positive(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def _file_hash(path: Path) -> str:
    if (
        not path.is_absolute()
        or not path.is_file()
        or not path.stat().st_size
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ValueError("media must be a nonempty absolute local regular-file path")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WebMWriter:
    """Lazy OpenCV encoder for RGB arrays only; never opens a capture device."""

    def __init__(self, path: Path, fps: float, shape: tuple[int, ...]) -> None:
        import cv2

        self.cv2 = cv2
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"VP80"), fps, (shape[1], shape[0])
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError("WebM encoder unavailable; no physical clip may be claimed")

    def write(self, image: np.ndarray) -> None:
        self.writer.write(self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()


class TrialClipCapture:
    """Sample both latest views through the entire trial, independently of policy requests."""

    def __init__(
        self,
        observe: Callable[[str], SynriaObservation],
        directory: Path,
        repository: Path,
        *,
        fps: float,
        max_duration_s: float,
        shutdown_timeout_s: float,
        writer_factory: Callable[..., Any] = WebMWriter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fps = _positive(fps, "capture fps")
        self.max_duration = _positive(max_duration_s, "capture duration cap")
        self.shutdown_timeout = _positive(shutdown_timeout_s, "capture shutdown timeout")
        absolute = directory.absolute()
        if absolute.resolve().is_relative_to(repository.resolve()) or any(
            part.is_symlink() for part in (absolute, *absolute.parents)
        ):
            raise ValueError("raw video directory must be outside the repository, without symlinks")
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.observe, self.clock = absolute, observe, clock
        self.writer_factory = writer_factory
        self._stop, self._ready, self._finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self._error: BaseException | None = None
        self._writers: dict[str, Any] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._count = 0
        self._first = self._last = 0.0
        self._final: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name="synria-probe-clips", daemon=True)

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(self.shutdown_timeout):
            raise TimeoutError("camera clip startup timed out; retain attempt evidence")
        self.check()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("trial camera capture failed") from self._error

    def _sample(self, stream: Any) -> None:
        observation = self.observe("fixed checkpoint probe camera recording")
        now = self.clock()
        stamps: dict[str, Any] = {"sample_index": self._count, "sample_monotonic_s": now}
        for name in ("wrist", "front"):
            frame = getattr(observation, name)
            image = frame.data
            if (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or not image.size
            ):
                raise ValueError("both clip views require nonempty RGB uint8 images")
            if not math.isfinite(frame.monotonic_timestamp_s) or frame.monotonic_timestamp_s > now:
                raise ValueError("invalid camera arrival timestamp")
            if name not in self._writers:
                self._writers[name] = self.writer_factory(
                    self.directory / f"{name}.webm", self.fps, image.shape
                )
                self._shapes[name] = image.shape
            if image.shape != self._shapes[name]:
                raise ValueError("camera dimensions changed during a fixed trial")
            self._writers[name].write(image)
            stamps[name + "_monotonic_s"] = frame.monotonic_timestamp_s
            self._final[name] = {
                "sample_index": self._count,
                "source_monotonic_s": frame.monotonic_timestamp_s,
                "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
        stream.write(json.dumps(stamps, allow_nan=False) + "\n")
        stream.flush()
        if not self._count:
            self._first = now
        self._last, self._count = now, self._count + 1

    def _run(self) -> None:
        started = self.clock()
        try:
            with (self.directory / "frame_timestamps.jsonl").open("x", encoding="utf-8") as stream:
                while not self._stop.is_set():
                    if self.clock() - started > self.max_duration:
                        raise TimeoutError("trial camera duration cap exceeded")
                    self._sample(stream)
                    self._ready.set()
                    self._stop.wait(max(0, started + self._count / self.fps - self.clock()))
                self._sample(stream)  # Explicit final frame after the operator's label.
        except BaseException as error:
            self._error = error
        finally:
            for writer in self._writers.values():
                try:
                    writer.close()
                except BaseException as error:
                    self._error = self._error or error
            self._ready.set()
            self._finished.set()

    @property
    def stopped(self) -> bool:
        return self._finished.is_set() or self._thread.ident is None

    def stop(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(self.shutdown_timeout)
            if self._thread.is_alive():
                raise RuntimeError("clip worker still active; retain writers and source resources")
        self.check()

    def evidence(self, label_stamp: float | None = None) -> dict[str, Any]:
        duration = self._last - self._first
        result: dict[str, Any] = {
            "requested_fps": self.fps,
            "frame_count": self._count,
            "achieved_sample_rate_hz": (self._count - 1) / duration
            if self._count > 1 and duration > 0
            else 0,
            "first_sample_monotonic_s": self._first,
            "last_sample_monotonic_s": self._last,
            "final_frames": dict(self._final) if self._finished.is_set() else {},
            "label_front_monotonic_s": label_stamp,
            "complete": self._finished.is_set() and self._error is None,
            "error": str(self._error)
            if self._error
            else (None if self.stopped else "worker still active"),
            "views": {},
        }
        for name in ("wrist", "front"):
            path = self.directory / f"{name}.webm"
            entry: dict[str, Any] = {"path": str(path), "sha256": None, "missing_reason": None}
            if self._finished.is_set():
                try:
                    entry["sha256"] = _file_hash(path)
                except ValueError as error:
                    entry["missing_reason"] = str(error)
            else:
                entry["missing_reason"] = "clip writer has not finalized"
            result["views"][name] = entry
            if entry["sha256"] is None:
                result["complete"] = False
        timestamps = self.directory / "frame_timestamps.jsonl"
        result["timestamps_path"] = str(timestamps)
        result["timestamps_sha256"] = (
            _file_hash(timestamps) if self._finished.is_set() and self._count else None
        )
        if label_stamp is not None and self._finished.is_set() and self._count:
            rows = [
                json.loads(line) for line in timestamps.read_text(encoding="utf-8").splitlines()
            ]
            nearest = min(rows, key=lambda row: abs(row["front_monotonic_s"] - label_stamp))
            result["label_nearest_sample_index"] = nearest["sample_index"]
            result["label_link_skew_s"] = abs(nearest["front_monotonic_s"] - label_stamp)
        return result


def run_physical_probes(
    probe_path: Path,
    expected_probe_hash: str,
    checkpoint_record: Path,
    dataset_root: Path,
    session_config: dict[str, Any],
    repository: Path,
    output: Path,
    media_root: Path,
    *,
    enable_motion: bool = False,
    factory: Callable[[dict[str, Any]], SessionIO] | None = None,
    transport: PolicyTransport | None = None,
    writer_factory: Callable[..., Any] = WebMWriter,
) -> dict[str, Any]:
    """Run operator-chosen diagnostic trials, never a D2 evaluation or selection loop."""
    from .embodiment import SynriaEmbodiment
    from .sessions import OperatorAbort, validate_session
    from .turn_executor import BoardCalibration, PolicyMoveExecutor, SquareMove

    probes = load_probes(probe_path, repository, expected_probe_hash)
    saved = json.loads(checkpoint_record.read_text(encoding="utf-8"))
    if (
        saved.get("purpose") != DIAGNOSTIC_PURPOSE
        or saved.get("status") != "evaluated"
        or saved.get("probe_sha256") != expected_probe_hash
        or strict_content_hash(dataset_root) != probes["dataset_content_sha256"]
        or saved.get("dataset_content_sha256") != probes["dataset_content_sha256"]
    ):
        raise ValueError(
            "physical probes require the matching frozen dataset and checkpoint evaluation"
        )
    checkpoint = Path(saved["checkpoint_path"])
    if strict_content_hash(checkpoint) != saved["checkpoint_content_sha256"]:
        raise ValueError("selected checkpoint content changed")
    for artifact in (output, media_root):
        _artifact_destination(artifact, (dataset_root, checkpoint), (probe_path, checkpoint_record))
    if (
        session_config["provenance"]["policy_id"] != saved["policy_id"]
        or session_config["provenance"]["checkpoint_sha256"] != saved["checkpoint_content_sha256"]
    ):
        raise ValueError("operator-selected policy/checkpoint differs from session configuration")
    safety = validate_session(session_config, "d3", repository, enable_motion=enable_motion)
    contract = PhysicalDatasetContract(
        session_config["gripper_type"],
        ActionSource(session_config["action_source"]),
        session_config["state_has_velocity"],
        action_lookahead_steps=session_config["action_lookahead_steps"],
        task_id=session_config["task_id"],
        task_definition=TaskDefinition.from_metadata(session_config),
    )
    if (
        contract.as_dict(fps=saved["physical_contract"]["requested_rate_hz"])
        != saved["physical_contract"]
    ):
        raise ValueError("session physical contract differs from evaluated checkpoint")
    if any(
        media_root.resolve().is_relative_to(root.resolve())
        for root in (repository, dataset_root, checkpoint)
    ):
        raise ValueError("probe videos must be outside git, the frozen dataset and the checkpoint")
    calibration = BoardCalibration(
        Path(session_config["calibration"]), Path(session_config["reachable"])
    )
    moves = []
    for trial in probes["physical_trials"]:
        move = calibration.task(
            SquareMove(trial["start_square"], trial["target_square"], trial["token"], "probe")
        )
        if move is None:
            raise ValueError("frozen physical trial is unreachable")
        moves.append(move)
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "kind": "physical_checkpoint_probe",
        "purpose": DIAGNOSTIC_PURPOSE,
        "policy_id": saved["policy_id"],
        "step": saved["step"],
        "probe_sha256": expected_probe_hash,
        "checkpoint_content_sha256": saved["checkpoint_content_sha256"],
        "dataset_content_sha256": saved["dataset_content_sha256"],
        "physical_contract": saved["physical_contract"],
        "session_provenance": session_config["provenance"],
        "policy_safety": asdict(safety),
        "endpoint_checkpoint_identity": "operator-declared; server attestation not yet available",
        "utc": datetime.now(timezone.utc).isoformat(),
        "motion_enabled": enable_motion,
        "status": "failed",
        "attempts": [],
        "errors": [],
        "stage_status": "planned",
        "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
    }
    io: SessionIO | None = None
    path: GuardedCommandPath | None = None
    client: PolicyClient | None = None
    captures: list[TrialClipCapture] = []
    active_capture: TrialClipCapture | None = None
    try:
        if factory is None:
            module, name = session_config["adapter"].split(":", 1)
            factory = getattr(importlib.import_module(module), name)
        io = factory(
            {**session_config, "enable_motion": enable_motion, "session_output": str(output)}
        )
        report["source_preflight"] = io.preflight()
        if not enable_motion:
            report["status"] = "read_only"
        else:
            report["source_preflight"] = io.authorize_motion()
            client = PolicyClient(
                SynriaEmbodiment(contract),
                transport or HttpPolicyTransport(session_config["policy_endpoint"]),
                load_limits(Path(session_config["limits"])),
                safety,
            )

            class CaptureGuardSink:
                def __init__(self) -> None:
                    assert io is not None
                    self.sink = io.command_sink()

                def offer(self, action: tuple[float, ...]) -> None:
                    if active_capture is None:
                        raise RuntimeError("probe cameras are not recording")
                    active_capture.check()
                    self.sink.offer(action)

                def hold(self) -> None:
                    self.sink.hold()

            path = GuardedCommandPath(client, CaptureGuardSink, enable_motion=True)
            abort = False
            terminal_status = "completed"
            for trial, move in zip(probes["physical_trials"], moves, strict=True):
                for number in range(1, session_config["max_attempts"] + 1):
                    identifier = f"{trial['trial_id']}-{number}-{uuid.uuid4().hex}"
                    attempt_path = output / f"{identifier}.json"
                    attempt: dict[str, Any] = {
                        "attempt_id": identifier,
                        "trial": trial,
                        "attempt_number": number,
                        "status": "failed",
                        "attempted": False,
                        "motion_completed": False,
                        "operator_grade": None,
                        "clips": None,
                        "errors": [],
                        "purpose": DIAGNOSTIC_PURPOSE,
                        "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
                        "policy_id": saved["policy_id"],
                        "step": saved["step"],
                        "checkpoint_content_sha256": saved["checkpoint_content_sha256"],
                        "probe_sha256": expected_probe_hash,
                    }
                    capture: TrialClipCapture | None = None
                    try:
                        if io.abort_requested():
                            raise OperatorAbort("operator aborted before trial capture")
                        capture = TrialClipCapture(
                            io.observe,
                            media_root / identifier,
                            repository,
                            fps=probes["capture"]["fps"],
                            max_duration_s=probes["capture"]["max_duration_s"],
                            shutdown_timeout_s=probes["capture"]["shutdown_timeout_s"],
                            writer_factory=writer_factory,
                        )
                        captures.append(capture)
                        active_capture = capture
                        capture.start()

                        def observe(
                            task: str, recording: TrialClipCapture = capture
                        ) -> SynriaObservation:
                            assert io is not None
                            recording.check()
                            return io.observe(task)

                        mover = PolicyMoveExecutor(
                            path,
                            observe,
                            io.completed,
                            max_steps=session_config["max_steps"],
                            period_s=safety.command_period_s,
                        )
                        attempt["attempted"] = True
                        mover.execute(move)
                        attempt["motion_completed"] = True
                        grade = io.grade(move)
                        attempt["operator_grade"] = grade.evidence()
                        attempt["status"] = "success" if grade.label == "success" else "failure"
                    except (OperatorAbort, KeyboardInterrupt, EOFError) as error:
                        abort = True
                        terminal_status = "aborted"
                        attempt["status"] = "aborted"
                        attempt["errors"].append(str(error) or type(error).__name__)
                    except BaseException as error:
                        abort = True
                        terminal_status = "failed"
                        attempt["errors"].append(str(error) or type(error).__name__)
                    finally:
                        for callback in (path.hold, capture.stop if capture else lambda: None):
                            try:
                                callback()
                            except BaseException as error:
                                abort = True
                                terminal_status = "failed"
                                attempt["status"] = "failed"
                                attempt["errors"].append(str(error) or type(error).__name__)
                        if capture is not None:
                            stamp = (
                                attempt["operator_grade"]["still_timestamp_s"]
                                if attempt["operator_grade"]
                                else None
                            )
                            try:
                                attempt["clips"] = capture.evidence(stamp)
                                if not attempt["clips"]["complete"]:
                                    abort = True
                                    terminal_status = attempt["status"] = "failed"
                                    attempt["errors"].append("both finalized camera clips required")
                            except BaseException as error:
                                abort = True
                                terminal_status = attempt["status"] = "failed"
                                attempt["missing_clips_reason"] = (
                                    "clip evidence could not be finalized"
                                )
                                attempt["errors"].append(str(error) or type(error).__name__)
                        else:
                            attempt["missing_clips_reason"] = "trial stopped before clip startup"
                        with attempt_path.open("x", encoding="utf-8") as stream:
                            stream.write(_json(attempt))
                        report["attempts"].append(
                            {
                                "path": str(attempt_path.resolve()),
                                "sha256": _file_hash(attempt_path.resolve()),
                            }
                        )
                    if (
                        abort
                        or attempt["status"] == "success"
                        or number == session_config["max_attempts"]
                    ):
                        break
                    if not io.recover(move):
                        break
                if abort:
                    break
            report["status"] = terminal_status
    except (OperatorAbort, KeyboardInterrupt, EOFError) as error:
        report["status"] = "aborted"
        report["errors"].append(str(error) or type(error).__name__)
    except BaseException as error:
        report["errors"].append(str(error) or type(error).__name__)
    finally:

        def close_io() -> None:
            if any(not capture.stopped for capture in captures):
                raise RuntimeError("clip worker still active; session source resources retained")
            if io is not None:
                io.close()

        cleanup = [close_io]
        if path is not None:
            cleanup.insert(0, path.hold)
        if client is not None:
            cleanup.append(lambda: client.write_latencies(output / "latencies.jsonl"))
        for callback in cleanup:
            try:
                callback()
            except BaseException as error:
                report["status"] = "failed"
                report["errors"].append(str(error) or type(error).__name__)
        if io is not None:
            try:
                report["source_preflight"] = io.session_evidence()
                report["power_state_end"] = io.power_state_end()
            except BaseException as error:
                report["status"] = "failed"
                report["errors"].append(str(error) or type(error).__name__)
        with (output / "probe_run.json").open("x", encoding="utf-8") as stream:
            stream.write(_json(report))
    return report


def render_report(evaluations: list[Path], physical_runs: list[Path], destination: Path) -> str:
    """Build escaped, self-contained HTML; verified videos remain external local files."""
    records = [json.loads(path.read_text(encoding="utf-8")) for path in evaluations]
    if not records or destination.suffix.lower() != ".html":
        raise ValueError("checkpoint records and an explicit HTML output path required")
    first = records[0]
    seen = set()
    for record in records:
        _artifact_destination(
            destination,
            (Path(record["dataset_path"]), Path(record["checkpoint_path"])),
            tuple((*evaluations, *physical_runs)),
        )
        if (
            record.get("purpose") != DIAGNOSTIC_PURPOSE
            or record.get("kind") != "checkpoint_eval"
            or any(
                record.get(key) != first.get(key)
                for key in (
                    "probe_sha256",
                    "dataset_content_sha256",
                    "policy_id",
                    "physical_contract",
                )
            )
            or type(record.get("step")) is not int
            or record["step"] < 0
            or record["step"] in seen
        ):
            raise ValueError(
                "comparison requires unique steps of one policy on the same frozen probes"
            )
        seen.add(record["step"])
        datetime.fromisoformat(record["utc"])
        if record["status"] == "evaluated":
            axes = record["metrics"]["axes"]
            for index, name in enumerate((*DRIVER_JOINT_NAMES, "Gripper")):
                if axes[name]["unit"] != ("rad" if index < 6 else "m") or any(
                    type(axes[name][key]) not in (int, float)
                    or not math.isfinite(axes[name][key])
                    or axes[name][key] < 0
                    for key in ("mae", "rmse", "max_error")
                ):
                    raise ValueError("finite per-axis metrics with physical units required")
        elif record["status"] != "failed" or record.get("metrics") is not None:
            raise ValueError("failed checkpoint records must not invent metrics")
        record["clips"] = []
    records.sort(key=lambda record: record["step"])
    by_step = {record["step"]: record for record in records}
    for run_path in physical_runs:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        record = by_step.get(run.get("step"))
        if record is None or any(
            run.get(key) != record.get(key)
            for key in (
                "purpose",
                "policy_id",
                "probe_sha256",
                "checkpoint_content_sha256",
                "dataset_content_sha256",
            )
        ):
            raise ValueError("physical probe run does not match a displayed checkpoint")
        for reference in run["attempts"]:
            path = Path(reference["path"])
            if _file_hash(path) != reference["sha256"]:
                raise ValueError("physical trial record content changed")
            attempt = json.loads(path.read_text(encoding="utf-8"))
            if any(
                attempt.get(key) != record.get(key)
                for key in (
                    "purpose",
                    "policy_id",
                    "step",
                    "probe_sha256",
                    "checkpoint_content_sha256",
                )
            ):
                raise ValueError("physical trial identity differs from checkpoint")
            grade = attempt.get("operator_grade")
            if grade and _file_hash(Path(grade["still"]).absolute()) != grade["still_sha256"]:
                raise ValueError("operator-label camera still changed")
            views = {}
            clips = attempt.get("clips")
            for name in ("wrist", "front"):
                media = clips["views"][name] if clips else None
                if media and media.get("sha256"):
                    path = Path(media["path"])
                    if _file_hash(path) != media["sha256"]:
                        raise ValueError("external probe media content changed")
                    views[name] = {"url": path.as_uri(), "sha256": media["sha256"]}
                else:
                    views[name] = {
                        "unavailable": (media or {}).get("missing_reason") or "not recorded"
                    }
            record["clips"].append(
                {
                    "trial_id": attempt["trial"]["trial_id"],
                    "attempt_number": attempt["attempt_number"],
                    "status": attempt["status"],
                    "operator_label": grade["label"] if grade else "unavailable",
                    "errors": attempt["errors"],
                    "views": views,
                }
            )
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(value)) + "</td>"
                for value in (
                    record["step"],
                    record["utc"],
                    record["status"],
                    record.get("error", ""),
                )
            )
            + "</tr>"
        )
    embedded = (
        json.dumps(records, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Frozen checkpoint diagnostics</title>
<style>
body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:0 1em;color:#17212b}
table{border-collapse:collapse;width:100%}
td,th{padding:.5em;border-bottom:1px solid #ccc;text-align:left}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:1em}
video{width:100%;max-height:300px;background:#111}
svg{width:100%;height:240px;border:1px solid #ccc}
pre{white-space:pre-wrap}.notice{padding:1em;background:#fff3d0}
</style>
<h1>Frozen checkpoint diagnostics</h1>
<p class="notice">Diagnostic only — never D2 evaluation or policy selection.
Software implemented, unmeasured; physical stages remain planned.
Prediction error is not object success.</p>
<p>@@LIMITATION@@</p><p>Policy: <strong>@@POLICY@@</strong>.
Missing and failed checkpoints remain visible.</p>
<label>Curve horizontal axis <select id="axis">
<option value="step">Training step</option><option value="time">UTC time</option></select></label>
<div class="pair"><section><h2>Mean six-joint action MAE (rad)</h2>
<svg id="joints" viewBox="0 0 600 240"></svg></section>
<section><h2>Gripper action MAE (m)</h2><svg id="gripper" viewBox="0 0 600 240"></svg>
</section></div>
<table><thead><tr><th>Step</th><th>UTC</th><th>Status</th><th>Error</th></tr></thead>
<tbody>@@ROWS@@</tbody></table>
<p><label>Checkpoint step <select id="step"></select></label>
<label>Fixed trial <select id="trial"></select></label></p>
<div class="pair"><section><h2>First saved checkpoint</h2><div id="baseline"></div></section>
<section><h2>Selected checkpoint</h2><div id="selected"></div></section></div>
<pre id="metrics"></pre>
<script id="records" type="application/json">@@RECORDS@@</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("records").textContent);
const select=document.getElementById("step");
const trial=document.getElementById("trial"), axis=document.getElementById("axis");
data.forEach((r,i)=>{
 const o=document.createElement("option");o.value=i;
 o.textContent=`${r.step}: ${r.status}`;select.append(o)
});
const ids=[...new Set(data.flatMap(r=>r.clips.map(c=>c.trial_id)))];
(ids.length?ids:["No physical trial collected"]).forEach(id=>{
 const o=document.createElement("option");o.value=id;o.textContent=id;trial.append(o)
});
function panel(node,r){
 node.replaceChildren();const title=document.createElement("p");
 title.textContent=`Step ${r.step}, ${r.status}`;node.append(title);
 const clips=r.clips.filter(c=>c.trial_id===trial.value);
 if(!clips.length){
  const p=document.createElement("p");
  p.textContent="Physical probe unavailable for this fixed trial/checkpoint.";node.append(p)
 }
 clips.forEach(c=>{
  const h=document.createElement("p");
  h.textContent=`Attempt ${c.attempt_number}: ${c.status}; operator label: ${c.operator_label}. `
   +c.errors.join("; ");node.append(h);
  ["wrist","front"].forEach(name=>{
   const p=document.createElement("p");p.textContent=name;node.append(p);const v=c.views[name];
   if(v.url){
    const video=document.createElement("video");video.controls=true;
    video.preload="metadata";video.src=v.url;node.append(video)
   }else{p.textContent+=`: ${v.unavailable}`}
  })
 })
}
function plots(){
 ["joints","gripper"].forEach(name=>{
  const svg=document.getElementById(name);svg.replaceChildren();
  const points=data.filter(r=>r.status==="evaluated").map(r=>({
   x:axis.value==="step"?r.step:Date.parse(r.utc),
   y:name==="gripper"?r.metrics.axes.Gripper.mae:Object.entries(r.metrics.axes)
    .filter(([k])=>k!=="Gripper").reduce((s,[k,v])=>s+v.mae,0)/6
  }));
  if(!points.length)return;
  const xs=points.map(p=>p.x),ys=points.map(p=>p.y),lo=Math.min(...xs);
  const span=Math.max(...xs)-lo||1,top=Math.max(...ys)||1;
  const line=document.createElementNS("http://www.w3.org/2000/svg","polyline");
  line.setAttribute("fill","none");line.setAttribute("stroke","#2357ad");
  line.setAttribute("stroke-width","3");
  line.setAttribute("points",points.map(p=>
   `${30+(p.x-lo)/span*540},${210-p.y/top*180}`).join(" "));
  svg.append(line);
  const label=document.createElementNS("http://www.w3.org/2000/svg","text");
  label.setAttribute("x","30");label.setAttribute("y","20");
  label.textContent=`Range 0–${top.toPrecision(3)}; horizontal: ${axis.value}`;svg.append(label)
 })
}
function update(){
 const r=data[Number(select.value)];panel(document.getElementById("baseline"),data[0]);
 panel(document.getElementById("selected"),r);
 document.getElementById("metrics").textContent=
  JSON.stringify(r.metrics||{unavailable:r.error},null,2)
}
select.addEventListener("change",update);trial.addEventListener("change",update);
axis.addEventListener("change",plots);plots();update();
</script></html>"""
    replacements = {
        "LIMITATION": html.escape(OBJECT_SUCCESS_LIMITATION),
        "POLICY": html.escape(str(first["policy_id"])),
        "ROWS": "".join(rows),
        "RECORDS": embedded,
    }
    page = re.sub(
        r"@@(LIMITATION|POLICY|ROWS|RECORDS)@@", lambda match: replacements[match[1]], page
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    return page


def report_main() -> None:
    parser = argparse.ArgumentParser(
        description="Render frozen checkpoint diagnostics and local clips"
    )
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--physical-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_report(args.evaluations, args.physical_runs, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register")
    register.add_argument("probe_set", type=Path)
    probe = commands.add_parser("probe")
    for name in (
        "probe-set",
        "checkpoint-evaluation",
        "dataset-root",
        "session-config",
        "repository",
        "output",
        "media-root",
    ):
        probe.add_argument("--" + name, type=Path, required=True)
    probe.add_argument("--probe-hash", required=True)
    probe.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args()
    if args.command == "register":
        print(register_probes(args.probe_set))
    else:
        result = run_physical_probes(
            args.probe_set,
            args.probe_hash,
            args.checkpoint_evaluation,
            args.dataset_root,
            json.loads(args.session_config.read_text(encoding="utf-8")),
            args.repository,
            args.output,
            args.media_root,
            enable_motion=args.enable_motion,
        )
        print(_json(result))
        if result["status"] in {"failed", "aborted"}:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
