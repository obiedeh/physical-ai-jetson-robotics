"""Quality gates and evidence artifacts for physical Synria D1 sessions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from synria_lerobot.recorder import OperatorLabel, RecordedPhysicalEpisode

OBJECT_SUCCESS_LIMITATION = (
    "Object success is the operator's label plus a camera still; "
    "no independent sensor confirms it."
)
UNVERIFIED_LIMITS_LINE = "limits unverified by operator"
GATE_NAMES = (
    "frame_count",
    "finite_state_action",
    "cameras_present",
    "state_action_limits",
    "timestamp_skew",
    "episode_length",
    "gripper_dimensionality",
    "visual_sanity",
)


@dataclass(frozen=True)
class ImageDiagnostic:
    present: bool
    mean_intensity: float
    content_hash: str


@dataclass(frozen=True)
class FrameQualityRecord:
    state: tuple[float, ...]
    action: tuple[float, ...]
    timestamps: dict[str, float]
    wrist: ImageDiagnostic
    front: ImageDiagnostic


@dataclass(frozen=True)
class EpisodeQualityRecord:
    episode_index: int
    fps: float
    duration_s: float
    operator_label: str
    action_source: str
    contract_version: str
    gripper_type: str
    final_still: str
    smoke: bool
    frames: tuple[FrameQualityRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeQualityRecord:
        if type(payload["episode_index"]) is not int or payload["episode_index"] < 0:
            raise ValueError("episode indices must be non-negative integers")
        return cls(
            episode_index=payload["episode_index"],
            fps=float(payload["fps"]),
            duration_s=float(payload["duration_s"]),
            operator_label=str(payload["operator_label"]),
            action_source=str(payload["action_source"]),
            contract_version=str(payload["contract_version"]),
            gripper_type=str(payload["gripper_type"]),
            final_still=str(payload["final_still"]),
            smoke=bool(payload["smoke"]),
            frames=tuple(
                FrameQualityRecord(
                    state=tuple(float(value) for value in frame["state"]),
                    action=tuple(float(value) for value in frame["action"]),
                    timestamps={
                        str(name): float(value)
                        for name, value in frame["timestamps"].items()
                    },
                    wrist=ImageDiagnostic(**frame["wrist"]),
                    front=ImageDiagnostic(**frame["front"]),
                )
                for frame in payload["frames"]
            ),
        )


@dataclass(frozen=True)
class PhysicalLimits:
    verified_by: str
    verified_on: str
    joint_names: tuple[str, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    gripper_limits_m: dict[str, tuple[float, float]]

    @property
    def verified(self) -> bool:
        return bool(self.verified_by.strip() and self.verified_on.strip())


@dataclass(frozen=True)
class GateConfig:
    frame_count_tolerance_fraction: float = 0.1
    max_timestamp_skew_s: float = 0.05
    min_episode_s: float = 20.0
    max_episode_s: float = 30.0
    black_mean_threshold: float = 1.0


DEFAULT_GATE_CONFIG = GateConfig()


@dataclass(frozen=True)
class GateReport:
    episode_index: int
    gates: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    @property
    def failed_gates(self) -> list[str]:
        return [name for name, passed in self.gates.items() if not passed]

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "passed": self.passed,
            "gates": self.gates,
            "failed_gates": self.failed_gates,
        }


def load_limits(path: Path) -> PhysicalLimits:
    """Load the JSON-compatible YAML limits file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(str(name) for name in payload["joint_names"])
    return PhysicalLimits(
        verified_by=str(payload.get("verified_by", "")),
        verified_on=str(payload.get("verified_on", "")),
        joint_names=names,
        joint_limits_rad=tuple(
            _bounds(payload["joint_limits_rad"][name]) for name in names
        ),
        gripper_limits_m={
            str(name): _bounds(bounds)
            for name, bounds in payload["gripper_limits_m"].items()
        },
    )


def _bounds(values: list[Any]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError("limit bounds must contain exactly two values")
    return float(values[0]), float(values[1])


def _image_diagnostic(data: Any) -> ImageDiagnostic:
    if data is None:
        return ImageDiagnostic(False, 0.0, "")
    array = np.asarray(data)
    if array.size == 0:
        return ImageDiagnostic(False, 0.0, "")
    encoded = array.tobytes()
    return ImageDiagnostic(
        present=True,
        mean_intensity=float(np.mean(array.astype(np.float64))),
        content_hash=hashlib.sha256(encoded).hexdigest(),
    )


def episode_quality_record(
    episode: RecordedPhysicalEpisode, *, fps: float
) -> EpisodeQualityRecord:
    return EpisodeQualityRecord(
        episode_index=episode.episode_index,
        fps=fps,
        duration_s=episode.duration_s,
        operator_label=episode.operator_label.value,
        action_source=episode.action_source.value,
        contract_version=episode.contract_version,
        gripper_type=episode.gripper_type,
        final_still=str(episode.final_still_path or ""),
        smoke=episode.smoke,
        frames=tuple(
            FrameQualityRecord(
                state=frame.state.observation_vector(),
                action=frame.action,
                timestamps=frame.timestamps(),
                wrist=_image_diagnostic(frame.wrist.data),
                front=_image_diagnostic(frame.front.data),
            )
            for frame in episode.frames
        ),
    )


def write_episode_records(records: list[EpisodeQualityRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.as_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def load_episode_records(path: Path) -> list[EpisodeQualityRecord]:
    return [
        EpisodeQualityRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_episode(
    episode: EpisodeQualityRecord,
    limits: PhysicalLimits,
    config: GateConfig = DEFAULT_GATE_CONFIG,
) -> GateReport:
    frames = episode.frames
    expected_frames = episode.fps * episode.duration_s
    frame_tolerance = max(1.0, expected_frames * config.frame_count_tolerance_fraction)
    frame_count_ok = abs(len(frames) - expected_frames) <= frame_tolerance

    finite_ok = all(
        all(math.isfinite(value) for value in (*frame.state, *frame.action))
        for frame in frames
    )
    cameras_ok = bool(frames) and all(
        frame.wrist.present and frame.front.present for frame in frames
    )

    gripper_bounds = limits.gripper_limits_m.get(episode.gripper_type)
    dimensions_ok = bool(frames) and gripper_bounds is not None and all(
        len(frame.state) in (7, 13) and len(frame.action) == 7 for frame in frames
    )
    limits_ok = bool(frames) and gripper_bounds is not None and all(
        _vector_within_limits(frame.state, limits, episode.gripper_type)
        and _vector_within_limits(frame.action, limits, episode.gripper_type)
        for frame in frames
    )

    timestamp_ok = bool(frames) and all(
        max(
            value
            for name, value in frame.timestamps.items()
            if name.endswith("monotonic_s")
        )
        - min(
            value
            for name, value in frame.timestamps.items()
            if name.endswith("monotonic_s")
        )
        <= config.max_timestamp_skew_s
        for frame in frames
    )
    length_ok = config.min_episode_s <= episode.duration_s <= config.max_episode_s
    visual_ok = _visual_sanity(frames, config.black_mean_threshold)
    return GateReport(
        episode_index=episode.episode_index,
        gates={
            "frame_count": frame_count_ok,
            "finite_state_action": finite_ok,
            "cameras_present": cameras_ok,
            "state_action_limits": limits_ok,
            "timestamp_skew": timestamp_ok,
            "episode_length": length_ok,
            "gripper_dimensionality": dimensions_ok,
            "visual_sanity": visual_ok,
        },
    )


def _vector_within_limits(
    values: tuple[float, ...], limits: PhysicalLimits, gripper_type: str
) -> bool:
    if len(values) < 7:
        return True
    if not all(math.isfinite(value) for value in values[:7]):
        return True
    if gripper_type not in limits.gripper_limits_m:
        return False
    bounded = zip(values[:6], limits.joint_limits_rad, strict=True)
    if not all(lower <= value <= upper for value, (lower, upper) in bounded):
        return False
    lower, upper = limits.gripper_limits_m[gripper_type]
    return lower <= values[6] <= upper


def _visual_sanity(
    frames: tuple[FrameQualityRecord, ...], black_threshold: float
) -> bool:
    if len(frames) < 2:
        return False
    wrist = [frame.wrist for frame in frames]
    front = [frame.front for frame in frames]
    if any(not image.present for image in (*wrist, *front)):
        return True
    if any(image.mean_intensity <= black_threshold for image in (*wrist, *front)):
        return False
    if len({image.content_hash for image in wrist}) == 1:
        return False
    if len({image.content_hash for image in front}) == 1:
        return False
    if all(w.content_hash == f.content_hash for w, f in zip(wrist, front, strict=True)):
        return False
    return True


def hash_dataset(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _merge_capture_provenance(
    dataset_path: Path,
    provenance: dict[str, object],
    episodes: list[EpisodeQualityRecord],
) -> dict[str, object]:
    path = dataset_path / "physical_capture_provenance.jsonl"
    if not path.is_file():
        if episodes and (dataset_path / "physical_contract.json").is_file():
            raise ValueError("physical dataset is missing recorded capture provenance")
        return provenance
    by_index: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        index = int(record["episode_index"])
        if index in by_index:
            raise ValueError("duplicate episode in capture provenance")
        by_index[index] = record
    selected = []
    for episode in episodes:
        if episode.episode_index not in by_index:
            raise ValueError("episode is missing recorded capture provenance")
        capture = by_index[episode.episode_index]
        if capture.get("stored_color_space") != "RGB":
            raise ValueError("recorded camera color space must be RGB")
        stored = capture.get("stored_resolution", {})
        native = capture.get("native_resolution", {})
        for resolution in (stored, native.get("wrist", {}), native.get("front", {})):
            if any(type(resolution.get(name)) is not int or resolution[name] <= 0 for name in (
                "width", "height"
            )):
                raise ValueError("capture provenance has invalid image resolution")
        if stored != provenance["resolution"]:
            raise ValueError("supplied resolution differs from recorded stored resolution")
        if capture.get("camera_ids") != provenance["camera_ids"]:
            raise ValueError("supplied camera ids differ from recorded capture provenance")
        selected.append(capture)
    if not selected:
        return provenance
    native = selected[0]["native_resolution"]
    if any(capture["native_resolution"] != native for capture in selected):
        raise ValueError("native camera resolution changed; summarize separate capture sessions")
    return {
        **provenance,
        "native_resolution": native,
        "stored_resolution": selected[0]["stored_resolution"],
        "stored_color_space": "RGB",
        "capture_provenance": selected,
    }


def write_session_artifacts(
    *,
    session_dir: Path,
    dataset_path: Path,
    provenance: dict[str, object],
    episodes: list[EpisodeQualityRecord],
    limits: PhysicalLimits,
    gate_config: GateConfig = DEFAULT_GATE_CONFIG,
) -> dict[str, object]:
    indices = [episode.episode_index for episode in episodes]
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("episode indices must be non-negative integers")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate episode indices in session records")
    required = {
        "follower_serial",
        "leader_serial",
        "host",
        "git_sha",
        "utc_date",
        "operator",
        "power_state_start",
        "power_state_end",
        "scene",
        "camera_ids",
        "resolution",
        "rate_hz",
        "contract_version",
        "gripper_type",
        "action_source",
    }
    missing = sorted(required - provenance.keys())
    if missing:
        raise ValueError(f"provenance missing required fields: {missing}")
    provenance = _merge_capture_provenance(dataset_path, provenance, episodes)
    dataset_root = dataset_path.resolve()
    for existing in session_dir.parent.glob("*/session_summary.json"):
        if existing.parent.resolve() == session_dir.resolve():
            continue
        existing_summary = json.loads(existing.read_text(encoding="utf-8"))
        if "dataset_path" in existing_summary and Path(
            existing_summary["dataset_path"]
        ).resolve() == dataset_root:
            raise ValueError(
                "dataset already summarized; reuse its session directory when resuming"
            )
    session_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = session_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reports = [evaluate_episode(episode, limits, gate_config) for episode in episodes]
    retained = [episode for episode in episodes if not episode.smoke]
    labels = [episode.operator_label for episode in retained]
    successes = sum(label == OperatorLabel.SUCCESS.value for label in labels)
    quality_valid = sum(
        report.passed and not episode.smoke
        for report, episode in zip(reports, episodes, strict=True)
    )
    summary: dict[str, object] = {
        "episode_count": len(retained),
        "recorded_episode_count": len(episodes),
        "smoke_episode_count": len(episodes) - len(retained),
        "episode_indices": [episode.episode_index for episode in retained],
        "quality_valid_episode_count": quality_valid,
        "qualifying_episode_count": quality_valid if limits.verified else 0,
        "gate_results": [report.as_dict() for report in reports],
        "operator_labels": labels,
        "demonstration_success_rate": successes / len(retained) if retained else 0.0,
        "dataset_path": str(dataset_root),
        "dataset_content_sha256": hash_dataset(dataset_path),
        "contract_version": provenance["contract_version"],
        "gripper_type": provenance["gripper_type"],
        "action_source": provenance["action_source"],
        "limits_status": "verified" if limits.verified else UNVERIFIED_LIMITS_LINE,
        "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
    }
    (session_dir / "session_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    notes = session_dir / "session_notes.md"
    if not notes.exists():
        notes.write_text(
            "# D1 session notes\n\n"
            "- Operator observations:\n"
            "- Resets or scene changes:\n"
            "- Incidents or discarded episodes:\n"
            "- Follow-up:\n\n"
            f"{OBJECT_SUCCESS_LIMITATION}\n",
            encoding="utf-8",
        )
    return summary


def write_aggregate_summary(
    *,
    data_root: Path,
    output_path: Path,
    timeline_path: Path,
    now_utc: str | None = None,
) -> dict[str, object]:
    session_summaries = []
    seen_datasets: set[Path] = set()
    for path in sorted(data_root.glob("*/session_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "dataset_path" in summary:
            dataset = Path(summary["dataset_path"]).resolve()
            if dataset in seen_datasets:
                raise ValueError("duplicate dataset in session summaries; refusing double counting")
            seen_datasets.add(dataset)
        session_summaries.append(summary)
    quality_valid_total = sum(
        int(summary["quality_valid_episode_count"]) for summary in session_summaries
    )
    qualifying_total = sum(
        int(
            summary.get(
                "qualifying_episode_count",
                summary["quality_valid_episode_count"]
                if summary["limits_status"] == "verified"
                else 0,
            )
        )
        for summary in session_summaries
    )
    action_sources = sorted({str(summary["action_source"]) for summary in session_summaries})
    aggregate: dict[str, object] = {
        "stage": "D1",
        "status": "planned" if qualifying_total < 100 else "measured",
        "quality_valid_episode_count": quality_valid_total,
        "qualifying_episode_count": qualifying_total,
        "session_count": len(session_summaries),
        "action_sources": action_sources,
        "progress": {
            str(target): {
                "target": target,
                "reached": qualifying_total >= target,
                "remaining": max(0, target - qualifying_total),
            }
            for target in (10, 50, 100)
        },
        "limits_status": (
            "verified"
            if session_summaries
            and all(summary["limits_status"] == "verified" for summary in session_summaries)
            else UNVERIFIED_LIMITS_LINE
        ),
        "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    timestamp = now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    with timeline_path.open("a", encoding="utf-8") as timeline:
        timeline.write(
            json.dumps(
                {
                    "t_utc": timestamp,
                    "kind": "d1_dataset_summary",
                    "title": (
                        f"D1 dataset progress: {qualifying_total}/100 "
                        "qualifying episodes"
                    ),
                    "artifact": str(output_path),
                    "numbers": {
                        "qualifying_episodes": qualifying_total,
                        "quality_valid_episodes": quality_valid_total,
                        "target": 100,
                    },
                },
                sort_keys=True,
            )
            + "\n"
        )
    return aggregate
