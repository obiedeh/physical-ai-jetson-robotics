"""Quality gates and evidence artifacts for physical Synria D1 sessions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from synria_lerobot.exclusions import read_exclusions
from synria_lerobot.physical_contract import (
    ACTION_TIMING_KEYS,
    EPISODE_WINDOW_KEYS,
    ActionSource,
    StateRateMeasurement,
    StateSourceProvenance,
    action_timing_metadata,
    episode_window_metadata,
)
from synria_lerobot.recorder import OperatorLabel, RecordedPhysicalEpisode
from synria_lerobot.task_registry import (
    DISPOSABLE_SMOKE,
    TASK_METADATA_KEYS,
    TaskDefinition,
    recording_purpose,
)

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
    "source_staleness",
    "episode_length",
    "gripper_dimensionality",
    "visual_sanity",
)


@dataclass(frozen=True)
class ImageDiagnostic:
    """Retain image presence and content diagnostics without storing frame pixels."""
    present: bool
    mean_intensity: float
    content_hash: str


@dataclass(frozen=True)
class FrameQualityRecord:
    """Bind one frame's state, action, source timing and dual-camera diagnostics."""
    state: tuple[float, ...]
    action: tuple[float, ...]
    timestamps: dict[str, float]
    wrist: ImageDiagnostic
    front: ImageDiagnostic


@dataclass(frozen=True)
class EpisodeQualityRecord:
    """Keep immutable per-episode evidence sufficient to repeat every quality gate."""
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
    achieved_sample_rate_hz: float | None = None
    action_lookahead_steps: int = 1
    effective_action_lookahead_steps: int | None = None
    nominal_action_lookahead_s: float | None = None
    requested_rate_hz: float | None = None
    state_rate_measurement: dict[str, Any] | None = None
    state_source_provenance: dict[str, Any] | None = None
    task_definition: TaskDefinition = field(kw_only=True)
    declared_recording_purpose: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Reject an unconfigured task window before accepting its episode evidence."""
        self.task_definition.recording_window(self.recording_purpose)

    @property
    def recording_purpose(self) -> str:
        """Preserve explicit synthetic identity while reading legacy physical records."""
        return recording_purpose(self.smoke, self.declared_recording_purpose)

    @property
    def min_episode_s(self) -> float:
        """Read the bound lower episode limit rather than guessing a collection window."""
        return self.task_definition.recording_window(self.recording_purpose)["min_episode_s"]

    @property
    def max_episode_s(self) -> float:
        """Read the bound hard cap used to validate this episode's duration."""
        return self.task_definition.recording_window(self.recording_purpose)["max_episode_s"]

    def as_dict(self) -> dict[str, object]:
        """Serialize frame diagnostics and their authoritative task-purpose metadata."""
        return {**asdict(self), **self.task_definition.metadata(self.recording_purpose)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeQualityRecord:
        """Validate saved task identity and reconstruct repeatable quality evidence."""
        if type(payload["episode_index"]) is not int or payload["episode_index"] < 0:
            raise ValueError("episode indices must be non-negative integers")
        return cls(
            episode_index=payload["episode_index"],
            task_definition=TaskDefinition.from_metadata(payload),
            fps=float(payload["fps"]),
            duration_s=float(payload["duration_s"]),
            operator_label=str(payload["operator_label"]),
            action_source=str(payload["action_source"]),
            contract_version=str(payload["contract_version"]),
            gripper_type=str(payload["gripper_type"]),
            final_still=str(payload["final_still"]),
            smoke=payload["smoke"],
            declared_recording_purpose=payload.get("recording_purpose"),
            achieved_sample_rate_hz=payload.get("achieved_sample_rate_hz"),
            action_lookahead_steps=payload.get("action_lookahead_steps", 1),
            effective_action_lookahead_steps=payload.get("effective_action_lookahead_steps"),
            nominal_action_lookahead_s=payload.get("nominal_action_lookahead_s"),
            requested_rate_hz=payload.get("requested_rate_hz"),
            state_rate_measurement=payload.get("state_rate_measurement"),
            state_source_provenance=payload.get("state_source_provenance"),
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
    """Represent candidate joint and gripper limits alongside operator verification."""
    verified_by: str
    verified_on: str
    joint_names: tuple[str, ...]
    joint_limits_rad: tuple[tuple[float, float], ...]
    gripper_limits_m: dict[str, tuple[float, float]]

    @property
    def verified(self) -> bool:
        """Require both operator and date before any episode can qualify."""
        return bool(self.verified_by.strip() and self.verified_on.strip())


@dataclass(frozen=True)
class GateConfig:
    """Freeze episode-window and freshness thresholds used to assess capture quality."""
    min_episode_s: float
    max_episode_s: float
    frame_count_tolerance_fraction: float = 0.1
    max_timestamp_skew_s: float = 0.05
    max_source_age_s: float = 0.2
    max_header_delay_s: float = 0.2
    max_header_future_s: float = 0.02
    black_mean_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Reject invalid durations or freshness tolerances instead of weakening gates."""
        episode_window_metadata(self.min_episode_s, self.max_episode_s)
        for value in (self.max_source_age_s, self.max_header_delay_s, self.max_header_future_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("freshness thresholds must be non-negative and finite")


@dataclass(frozen=True)
class GateReport:
    """Expose independent gate decisions without conflating quality and task success."""
    episode_index: int
    gates: dict[str, bool]

    @property
    def passed(self) -> bool:
        """Require every quality gate to pass for this episode."""
        return all(self.gates.values())

    @property
    def failed_gates(self) -> list[str]:
        """Name failed checks so the operator can investigate the original capture."""
        return [name for name, passed in self.gates.items() if not passed]

    def as_dict(self) -> dict[str, object]:
        """Produce a stable JSON-compatible report retaining each gate decision."""
        return {
            "episode_index": self.episode_index,
            "passed": self.passed,
            "gates": self.gates,
            "failed_gates": self.failed_gates,
        }


def load_limits(path: Path) -> PhysicalLimits:
    """Load the JSON-compatible YAML limits file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if any(not isinstance(payload.get(name, ""), str) for name in ("verified_by", "verified_on")):
        raise ValueError("limits verification fields must be strings")
    names = tuple(str(name) for name in payload["joint_names"])
    return PhysicalLimits(
        verified_by=payload.get("verified_by", ""),
        verified_on=payload.get("verified_on", ""),
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
    """Read one explicit lower/upper pair without inventing missing limits."""
    if len(values) != 2:
        raise ValueError("limit bounds must contain exactly two values")
    return float(values[0]), float(values[1])


def _image_diagnostic(data: Any) -> ImageDiagnostic:
    """Hash exact stored pixels and summarize brightness for repeatable image gates."""
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
    """Derive gate inputs from saved capture data without altering its operator label."""
    return EpisodeQualityRecord(
        episode_index=episode.episode_index,
        task_definition=episode.task_definition,
        fps=fps,
        duration_s=episode.duration_s,
        operator_label=episode.operator_label.value,
        action_source=episode.action_source.value,
        contract_version=episode.contract_version,
        gripper_type=episode.gripper_type,
        final_still=str(episode.final_still_path or ""),
        smoke=episode.smoke,
        declared_recording_purpose=episode.recording_purpose,
        achieved_sample_rate_hz=episode.achieved_sample_rate_hz,
        **action_timing_metadata(episode.action_source, episode.action_lookahead_steps, fps),
        state_rate_measurement=(
            asdict(episode.state_rate_measurement)
            if episode.state_rate_measurement is not None else None
        ),
        state_source_provenance=(
            episode.state_source_provenance.as_dict()
            if episode.state_source_provenance is not None else None
        ),
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
    """Write complete quality records for deterministic offline assessment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.as_dict(), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def load_episode_records(path: Path) -> list[EpisodeQualityRecord]:
    """Load and validate committed per-episode quality sidecars."""
    return [
        EpisodeQualityRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_episode(
    episode: EpisodeQualityRecord,
    limits: PhysicalLimits,
    config: GateConfig,
) -> GateReport:
    """Assess independent capture gates without changing task labels or thresholds."""
    if any(getattr(config, name) != getattr(episode, name) for name in EPISODE_WINDOW_KEYS):
        raise ValueError("gate episode window differs from recorded episode")
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
        _timestamp_skew_ok(frame, episode.action_source, config) for frame in frames
    )
    freshness_ok = _episode_sources_fresh(episode, config)
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
            "source_staleness": freshness_ok,
            "episode_length": length_ok,
            "gripper_dimensionality": dimensions_ok,
            "visual_sanity": visual_ok,
        },
    )


def _timestamp_skew_ok(frame: FrameQualityRecord, action_source: str, config: GateConfig) -> bool:
    """Compare only contemporaneous sources, excluding intentionally shifted actions."""
    names = ["state_monotonic_s", "wrist_monotonic_s", "front_monotonic_s"]
    if action_source == "leader":
        names.append("action_monotonic_s")
    values = [frame.timestamps.get(name, math.nan) for name in names]
    return all(math.isfinite(value) for value in values) and (
        max(values) - min(values) <= config.max_timestamp_skew_s
    )


def _sources_fresh(frame: FrameQualityRecord, action_source: str, config: GateConfig) -> bool:
    """Check original arrival ages and same-clock header delays at sampling time."""
    timestamps = frame.timestamps
    sample = timestamps.get("sample_monotonic_s", math.nan)
    if not math.isfinite(sample):
        return False
    sources = ["state", "wrist", "front"]
    ros_sources = ["state"]
    if action_source == "leader":
        sources.append("action")
        ros_sources.append("action")
    for source in sources:
        arrival = timestamps.get(f"{source}_monotonic_s", math.nan)
        if not math.isfinite(arrival) or not 0 <= sample - arrival <= config.max_source_age_s:
            return False
    for source in ros_sources:
        arrival = timestamps.get(f"{source}_ros_arrival_s", math.nan)
        header = timestamps.get(f"{source}_ros_header_s", math.nan)
        if not all(math.isfinite(value) for value in (arrival, header)):
            return False
        if not -config.max_header_future_s <= arrival - header <= config.max_header_delay_s:
            return False
    return True


def _episode_sources_fresh(episode: EpisodeQualityRecord, config: GateConfig) -> bool:
    """Check ordered sampling and exact target timestamps for derived next-state actions."""
    if not episode.frames or episode.action_source not in {"leader", "next_state"}:
        return False
    try:
        expected_timing = action_timing_metadata(
            ActionSource(episode.action_source), episode.action_lookahead_steps, episode.fps
        )
    except (TypeError, ValueError):
        return False
    if any(getattr(episode, name) != expected for name, expected in expected_timing.items()):
        return False
    samples = [frame.timestamps.get("sample_monotonic_s", math.nan) for frame in episode.frames]
    if not all(math.isfinite(sample) for sample in samples) or any(
        newer <= older for older, newer in zip(samples, samples[1:], strict=False)
    ):
        return False
    for index, frame in enumerate(episode.frames):
        if not _sources_fresh(frame, episode.action_source, config):
            return False
        if episode.action_source == "next_state":
            target = episode.frames[
                min(index + episode.action_lookahead_steps, len(episode.frames) - 1)
            ]
            for action_name, state_name in (
                ("action_monotonic_s", "state_monotonic_s"),
                ("action_ros_header_s", "state_ros_header_s"),
                ("action_ros_arrival_s", "state_ros_arrival_s"),
            ):
                actual = frame.timestamps.get(action_name, math.nan)
                expected = target.timestamps.get(state_name, math.nan)
                if (
                    not all(math.isfinite(value) for value in (actual, expected))
                    or actual != expected
                ):
                    return False
    return True


def _vector_within_limits(
    values: tuple[float, ...], limits: PhysicalLimits, gripper_type: str
) -> bool:
    """Check position and gripper bounds separately from dimensionality and NaN gates."""
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
    """Reject black, frozen or duplicated camera streams using recorded image diagnostics."""
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
    """Bind dataset-relative paths and file bytes without including sibling curation logs."""
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
    """Cross-check summary inputs against original camera and source capture facts."""
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
    final_stills = []
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
        if capture.get("achieved_sample_rate_hz") != episode.achieved_sample_rate_hz:
            raise ValueError("achieved sample rate differs from recorded capture provenance")
        if capture.get("state_rate_measurement") != episode.state_rate_measurement:
            raise ValueError("state rate measurement differs from recorded capture provenance")
        if capture.get("state_source_provenance") != episode.state_source_provenance:
            raise ValueError("state source differs from recorded capture provenance")
        still = capture.get("final_still_capture")
        if "final_still_capture" not in capture:
            final_stills.append({
                "episode_index": episode.episode_index,
                "status": "unknown: legacy capture has no native still evidence",
                "capture": None,
            })
        else:
            if (
                not isinstance(still, dict)
                or not isinstance(still.get("resolution"), dict)
                or any(
                    type(still["resolution"].get(name)) is not int
                    or still["resolution"][name] <= 0
                    for name in ("width", "height")
                )
                or still.get("resolution") != native["front"]
                or still.get("source_id") != capture["camera_ids"]["front"]
                or still.get("color_space") != "RGB"
                or not episode.frames
                or type(still.get("source_monotonic_s")) not in (int, float)
                or not math.isfinite(still["source_monotonic_s"])
                or still["source_monotonic_s"] != (
                    episode.frames[-1].timestamps.get("front_monotonic_s")
                )
            ):
                raise ValueError("final still capture differs from the last native front sample")
            final_stills.append({
                "episode_index": episode.episode_index, "status": "native",
                "capture": still,
            })
        _validate_timing_evidence(capture)
        TaskDefinition.from_metadata(capture)
        for name in (
            *ACTION_TIMING_KEYS, *TASK_METADATA_KEYS,
            "action_source", "gripper_type", "contract_version",
        ):
            if capture.get(name) != provenance[name]:
                raise ValueError("capture provenance differs from dataset contract")
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
        "final_still_captures": final_stills,
    }


def _validate_timing_evidence(payload: dict[str, Any]) -> None:
    """Reject missing or inconsistent action lookahead metadata."""
    try:
        expected = action_timing_metadata(
            ActionSource(payload["action_source"]), payload["action_lookahead_steps"],
            payload["requested_rate_hz"],
        )
        if any(payload.get(name) != value for name, value in expected.items()):
            raise ValueError("inconsistent action timing evidence")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid or missing action timing evidence") from error


def _validate_summary_contract(
    dataset_path: Path, provenance: dict[str, Any], episodes: list[EpisodeQualityRecord]
) -> dict[str, Any]:
    """Require supplied session identity to match recorded contract and episode facts."""
    expected: dict[str, Any] = {
        name: provenance[name] for name in ("action_source", "gripper_type", "contract_version")
    }
    expected.update(action_timing_metadata(
        ActionSource(provenance["action_source"]), provenance["action_lookahead_steps"],
        provenance["rate_hz"],
    ))
    expected.update(TaskDefinition.from_metadata(provenance).metadata(
        provenance["recording_purpose"]
    ))
    for name, value in expected.items():
        if name in provenance and provenance[name] != value:
            raise ValueError("supplied timing differs from requested rate and action source")
    contract_path = dataset_path / "physical_contract.json"
    source_evidence = provenance.get("state_source_provenance")
    if source_evidence is not None:
        source_evidence = StateSourceProvenance.from_dict(source_evidence).as_dict()
    if contract_path.is_file() and episodes and source_evidence is None:
        raise ValueError("physical summaries require operator-declared state source provenance")
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        TaskDefinition.from_metadata(contract)
        _validate_timing_evidence(contract)
        if any(contract.get(name) != value for name, value in expected.items()):
            raise ValueError("supplied metadata differs from dataset contract")
    for episode in episodes:
        if episode.state_source_provenance is not None:
            recorded_source = StateSourceProvenance.from_dict(
                episode.state_source_provenance
            ).as_dict()
        else:
            recorded_source = None
        if recorded_source != source_evidence:
            raise ValueError("episode state source differs from supplied provenance")
        if contract_path.is_file() and episode.state_rate_measurement is None:
            raise ValueError("physical summaries require recorded incoming state rate evidence")
        if episode.state_rate_measurement is not None:
            measurement = StateRateMeasurement(**episode.state_rate_measurement)
            if measurement.rate_hz < episode.fps:
                raise ValueError("requested rate exceeds recorded incoming state measurement")
        _validate_timing_evidence({
            "action_source": episode.action_source,
            **{name: getattr(episode, name) for name in ACTION_TIMING_KEYS},
        })
        recorded_metadata = {
            name: getattr(episode, name) for name in expected if name not in TASK_METADATA_KEYS
        }
        recorded_metadata.update(episode.task_definition.metadata(episode.recording_purpose))
        if episode.fps != expected["requested_rate_hz"] or any(
            recorded_metadata.get(name) != value for name, value in expected.items()
        ):
            raise ValueError("episode metadata differs from dataset contract")
    return {**provenance, **expected, "state_source_provenance": source_evidence}


def write_session_artifacts(
    *,
    session_dir: Path,
    dataset_path: Path,
    provenance: dict[str, object],
    episodes: list[EpisodeQualityRecord],
    limits: PhysicalLimits,
    gate_config: GateConfig,
) -> dict[str, object]:
    """Write a physical-only session summary with audited exclusions applied to counts."""
    if provenance.get("recording_purpose") == "synthetic_demo" or any(
        episode.recording_purpose == "synthetic_demo" for episode in episodes
    ):
        raise ValueError("synthetic demo datasets cannot produce D1 session summaries")
    contract_path = dataset_path / "physical_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text(encoding="utf-8")).get(
        "recording_purpose"
    ) == "synthetic_demo":
        raise ValueError("synthetic demo datasets cannot produce D1 session summaries")
    indices = [episode.episode_index for episode in episodes]
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("episode indices must be non-negative integers")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate episode indices in session records")
    exclusions = read_exclusions(dataset_path)
    if set(exclusions.entries) - set(indices):
        raise ValueError("exclusion log references episodes missing from session evidence")
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
        "action_lookahead_steps",
    }
    missing = sorted(required - provenance.keys())
    if missing:
        raise ValueError(f"provenance missing required fields: {missing}")
    provenance = _validate_summary_contract(dataset_path, provenance, episodes)
    if any(provenance[name] != getattr(gate_config, name) for name in EPISODE_WINDOW_KEYS):
        raise ValueError("gate episode window differs from dataset contract")
    provenance = _merge_capture_provenance(dataset_path, provenance, episodes)
    sample_rates = [
        {"episode_index": episode.episode_index, "rate_hz": episode.achieved_sample_rate_hz}
        for episode in episodes
    ]
    state_rate_measurements = [
        {"episode_index": episode.episode_index, "measurement": episode.state_rate_measurement}
        for episode in episodes
    ]
    provenance = {
        **provenance,
        "achieved_sample_rates_hz": sample_rates,
        "state_rate_measurements": state_rate_measurements,
        "freshness_thresholds_s": {
            "source_age": gate_config.max_source_age_s,
            "header_delay": gate_config.max_header_delay_s,
            "header_future": gate_config.max_header_future_s,
        },
    }
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
    retained = [episode for episode in episodes if not episode.smoke
                and episode.episode_index not in exclusions.excluded_ids]
    labels = [episode.operator_label for episode in retained]
    successes = sum(label == OperatorLabel.SUCCESS.value for label in labels)
    quality_valid = sum(
        report.passed and not episode.smoke and episode.episode_index not in exclusions.excluded_ids
        for report, episode in zip(reports, episodes, strict=True)
    )
    summary: dict[str, object] = {
        "episode_count": len(retained),
        "recorded_episode_count": len(episodes),
        "smoke_episode_count": sum(episode.smoke for episode in episodes),
        "excluded_episode_count": len(exclusions.excluded_ids),
        **exclusions.evidence(),
        "episode_indices": [episode.episode_index for episode in retained],
        "achieved_sample_rates_hz": sample_rates,
        "state_rate_measurements": state_rate_measurements,
        "quality_valid_episode_count": quality_valid,
        "final_still_captures": provenance.get("final_still_captures", []),
        "state_source_provenance": provenance["state_source_provenance"],
        "qualifying_episode_count": quality_valid if limits.verified else 0,
        "gate_results": [report.as_dict() for report in reports],
        "operator_labels": labels,
        "demonstration_success_rate": successes / len(retained) if retained else 0.0,
        "dataset_path": str(dataset_root),
        "dataset_content_sha256": hash_dataset(dataset_path),
        "contract_version": provenance["contract_version"],
        "gripper_type": provenance["gripper_type"],
        "action_source": provenance["action_source"],
        **{name: provenance[name] for name in ACTION_TIMING_KEYS},
        **{name: provenance[name] for name in TASK_METADATA_KEYS},
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
    """Aggregate current physical summaries, refusing stale curation or demo evidence."""
    session_summaries = []
    seen_datasets: set[Path] = set()
    for path in sorted(data_root.glob("*/session_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("recording_purpose") == "synthetic_demo":
            raise ValueError("synthetic demo datasets cannot contribute to D1 aggregates")
        TaskDefinition.from_metadata(summary)
        if summary["recording_purpose"] == DISPOSABLE_SMOKE and (
            any(type(summary.get(key)) is not int or summary[key] != 0 for key in (
                "episode_count", "quality_valid_episode_count", "qualifying_episode_count",
            )) or summary.get("episode_indices") != []
            or summary.get("operator_labels") != []
            or summary.get("demonstration_success_rate") != 0.0
        ):
            raise ValueError(
                "disposable smoke summary cannot claim retained or qualifying episodes"
            )
        if "dataset_path" in summary:
            dataset = Path(summary["dataset_path"]).resolve()
            exclusions = read_exclusions(dataset)
            if exclusions.sha256 != summary.get("exclusion_log_sha256") or (
                list(exclusions.excluded_ids) != summary.get("excluded_episode_ids", [])
            ):
                raise ValueError("session summary has stale exclusion evidence; regenerate it")
            contract_file = dataset / "physical_contract.json"
            if contract_file.is_file() and json.loads(
                contract_file.read_text(encoding="utf-8")
            ).get("recording_purpose") == "synthetic_demo":
                raise ValueError("synthetic demo datasets cannot contribute to D1 aggregates")
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
        "excluded_episode_count": sum(
            int(summary.get("excluded_episode_count", 0)) for summary in session_summaries
        ),
        "session_count": len(session_summaries),
        "action_sources": action_sources,
        "state_sources": [
            {"dataset_path": summary.get("dataset_path"),
             "provenance": summary.get("state_source_provenance")}
            for summary in session_summaries
        ],
        "state_rate_measurements": [
            {"dataset_path": summary.get("dataset_path"),
             "episodes": summary.get("state_rate_measurements", [])}
            for summary in session_summaries
        ],
        "recording_contracts": [
            {
                name: summary.get(name)
                for name in (
                    "dataset_path", "action_source", "contract_version", "gripper_type",
                    *ACTION_TIMING_KEYS, *TASK_METADATA_KEYS,
                )
            }
            for summary in session_summaries
        ],
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
