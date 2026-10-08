"""Episode recording interface for the Synria LeRobot track.

Provides the legacy simulation recorder plus an isolated physical recorder.
The physical path consumes state and image interfaces, never publishes robot
commands, and imports ROS, OpenCV, and the external dataset library lazily.

Usage::

    from synria_lerobot.recorder import EpisodeRecorder, RecordingSession

    recorder = EpisodeRecorder(zone="left", game="chess")
    episode = recorder.record(n_steps=60)

    session = RecordingSession(n_episodes=4, zones=("left", "right"))
    dataset = session.record_all()
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from edge_ai.camera_inference import CameraInferenceLoop, MockFrameSource
from synria_lerobot.dataset import SynriaEpisodeDataset, generate_synthetic_episode
from synria_lerobot.physical_contract import (
    ActionSource as ActionSourceKind,
)
from synria_lerobot.physical_contract import (
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalFrame,
    PhysicalState,
)
from synria_lerobot.recording_transaction import RecordingRecoveryError, RecordingTransaction
from synria_lerobot.schema import (
    DEFAULT_FPS,
    DEFAULT_ROBOT_ID,
    VALID_GAMES,
    VALID_STAGING_ZONES,
    VALID_TASK_VARIANTS,
    Episode,
)


class StateSource(Protocol):
    """Read the newest follower state without commanding either arm."""

    def read(self) -> PhysicalState: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ActionSample:
    values: tuple[float, ...]
    monotonic_timestamp_s: float
    ros_header_stamp_s: float | None = None
    ros_arrival_stamp_s: float | None = None


class ActionSource(Protocol):
    """Provide leader actions or identify next-state action derivation."""

    kind: ActionSourceKind

    def read(self, follower_state: PhysicalState) -> ActionSample | None: ...

    def close(self) -> None: ...


class FrameSource(Protocol):
    """Read camera frames without opening devices during module import."""

    def read(self) -> ImageFrame: ...

    def close(self) -> None: ...


class DatasetWriter(Protocol):
    """Persist one labeled episode and its final front-camera still."""

    @property
    def next_episode_index(self) -> int: ...

    recovery_blocked: bool

    def write_episode(self, episode: RecordedPhysicalEpisode) -> None: ...

    def finalize(self) -> None: ...

    def save_final_still(
        self, episode_index: int, frame: ImageFrame
    ) -> Path: ...


class RecorderState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    STOPPED = "stopped"


class OperatorLabel(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


def _close_all(*callbacks: Callable[[], None]) -> None:
    """Attempt every cleanup without hiding an exception already being handled."""
    active_error = sys.exc_info()[1]
    failures: list[BaseException] = []
    for callback in callbacks:
        try:
            callback()
        except BaseException as error:
            failures.append(error)
    if failures:
        if active_error is None:
            raise RuntimeError(f"session cleanup failed: {failures[0]}") from failures[0]
        for cleanup_error in failures:
            print(f"Session cleanup also failed: {cleanup_error}", file=sys.stderr)


@dataclass(frozen=True)
class PhysicalRecorderConfig:
    dataset_path: Path
    repo_id: str
    contract: PhysicalDatasetContract
    fps: float = 30.0
    image_width: int = 224
    image_height: int = 224
    min_episode_s: float = 20.0
    max_episode_s: float = 30.0
    hard_cap_s: float = 30.0
    task: str = "move the token from square A to square B"
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if any(type(value) is not int or value <= 0 for value in (
            self.image_width, self.image_height
        )):
            raise ValueError("image width and height must be positive integers")
        if not 0 < self.min_episode_s <= self.max_episode_s <= self.hard_cap_s:
            raise ValueError("episode bounds must satisfy 0 < min <= max <= hard cap")
        if not self.repo_id.strip():
            raise ValueError("repo_id is required")

    def require_outside_repository(self, repository_root: Path) -> None:
        root = repository_root.resolve()
        output = self.dataset_path.resolve()
        if output == root or root in output.parents:
            raise ValueError("physical datasets must be written outside the git tree")


@dataclass
class RecordedPhysicalEpisode:
    episode_index: int
    frames: list[PhysicalFrame]
    started_monotonic_s: float
    ended_monotonic_s: float
    operator_label: OperatorLabel
    action_source: ActionSourceKind
    contract_version: str
    gripper_type: str
    final_still_path: Path | None = None
    smoke: bool = False

    @property
    def duration_s(self) -> float:
        return self.ended_monotonic_s - self.started_monotonic_s

    @property
    def achieved_sample_rate_hz(self) -> float:
        samples = [frame.sample_monotonic_timestamp_s for frame in self.frames]
        if len(samples) < 2 or any(value is None for value in samples):
            return 0.0
        start, end = samples[0], samples[-1]
        assert start is not None and end is not None
        return (len(samples) - 1) / (end - start) if end > start else 0.0


@dataclass
class PhysicalSessionResult:
    saved_episode_count: int = 0
    discarded_episode_count: int = 0
    failed_save_count: int = 0
    last_episode_index: int | None = None
    smoke: bool = False
    last_achieved_sample_rate_hz: float | None = None


@dataclass(frozen=True)
class _PendingFrame:
    state: PhysicalState
    action: ActionSample | None
    wrist: ImageFrame
    front: ImageFrame
    sample_monotonic_s: float


class NextStateActionSource:
    kind = ActionSourceKind.NEXT_STATE

    def read(self, follower_state: PhysicalState) -> None:
        del follower_state
        return None

    def close(self) -> None:
        return None


class LeaderActionSource:
    kind = ActionSourceKind.LEADER

    def __init__(self, leader_state_source: StateSource) -> None:
        self._source = leader_state_source

    def read(self, follower_state: PhysicalState) -> ActionSample:
        del follower_state
        leader = self._source.read()
        return ActionSample(
            values=(*leader.joint_positions_rad, leader.gripper_m),
            monotonic_timestamp_s=leader.monotonic_timestamp_s,
            ros_header_stamp_s=leader.ros_header_stamp_s,
            ros_arrival_stamp_s=leader.ros_arrival_stamp_s,
        )

    def close(self) -> None:
        self._source.close()


_Sample = TypeVar("_Sample")


class _LatestSource(Generic[_Sample]):
    """A bounded latest-only cache whose worker failure is never hidden."""

    def __init__(self, *, timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("source timeout must be positive and finite")
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._latest: _Sample | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._received_count = 0

    def _store(self, sample: _Sample) -> None:
        with self._lock:
            self._latest = sample
            self._received_count += 1
        self._ready.set()

    def _start_worker(self, target: Callable[[], None], name: str) -> None:
        def run() -> None:
            try:
                target()
            except BaseException as error:
                with self._lock:
                    self._error = error
                self._ready.set()

        self._thread = threading.Thread(target=run, name=name, daemon=True)
        self._thread.start()

    def _read_latest(self) -> _Sample:
        if self._closed:
            raise RuntimeError("source is closed")
        self._ready.wait(self._timeout_s)
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"background source failed: {self._error}") from self._error
            if self._latest is None:
                raise RuntimeError("no source sample has been received")
            return self._latest

    def _join(self, timeout_s: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(self._timeout_s if timeout_s is None else timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("background source did not stop; resources retained")


class RosJointStateSource(_LatestSource[PhysicalState]):
    """Owned-context, depth-one subscriber; no command publisher is created."""

    def __init__(
        self, topic: str, *, node_name: str, state_has_velocity: bool = False,
        timeout_s: float = 1.0,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        try:
            import rclpy  # type: ignore[import-not-found]
            from rclpy.context import Context  # type: ignore[import-not-found]
            from rclpy.executors import (  # type: ignore[import-not-found]
                ExternalShutdownException,
                SingleThreadedExecutor,
            )
            from sensor_msgs.msg import JointState  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional ROS dependency
            raise RuntimeError("ROS 2 state dependencies are unavailable") from exc
        self._rclpy = rclpy
        self._context = Context()
        with ExitStack() as startup:
            rclpy.init(context=self._context)
            startup.callback(_close_all, lambda: rclpy.shutdown(context=self._context))
            self._node = rclpy.create_node(node_name, context=self._context)
            startup.callback(_close_all, self._node.destroy_node)
            self._executor = SingleThreadedExecutor(context=self._context)
            startup.callback(
                _close_all, lambda: self._executor.shutdown(timeout_sec=self._timeout_s)
            )

            def on_state(message: Any) -> None:
                arrived = time.monotonic()
                ros_arrival = self._node.get_clock().now().nanoseconds / 1_000_000_000
                values = dict(zip(message.name, message.position, strict=True))
                velocities = None
                if state_has_velocity:
                    if len(message.velocity) != len(message.name):
                        raise ValueError("physical contract requires six reported joint velocities")
                    by_name = dict(zip(message.name, message.velocity, strict=True))
                    velocities = tuple(float(by_name[f"Joint{i}"]) for i in range(1, 7))
                stamp = float(message.header.stamp.sec) + float(
                    message.header.stamp.nanosec
                ) / 1_000_000_000
                self._store(PhysicalState(
                    joint_positions_rad=tuple(float(values[f"Joint{i}"]) for i in range(1, 7)),
                    gripper_m=float(values["Gripper"]), monotonic_timestamp_s=arrived,
                    ros_header_stamp_s=stamp, joint_velocities_rad_s=velocities,
                    ros_arrival_stamp_s=ros_arrival,
                ))

            self._subscription = self._node.create_subscription(JointState, topic, on_state, 1)
            startup.callback(
                _close_all, lambda: self._node.destroy_subscription(self._subscription)
            )
            self._executor.add_node(self._node)

            def spin() -> None:
                try:
                    self._executor.spin()
                    if not self._stop.is_set():
                        raise RuntimeError("ROS executor stopped unexpectedly")
                except ExternalShutdownException:
                    if not self._stop.is_set():
                        raise

            self._start_worker(spin, node_name)
            startup.pop_all()

    def read(self) -> PhysicalState:
        return self._read_latest()

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        started = time.monotonic()
        shutdown_error: BaseException | None = None
        try:
            self._executor.shutdown(timeout_sec=self._timeout_s)
        except BaseException as error:
            shutdown_error = error
        self._join(max(0.0, self._timeout_s - (time.monotonic() - started)))
        self._closed = True
        _close_all(
            lambda: self._executor.remove_node(self._node),
            lambda: self._node.destroy_subscription(self._subscription),
            self._node.destroy_node,
            lambda: self._rclpy.shutdown(context=self._context),
        )
        if shutdown_error is not None:
            raise RuntimeError("ROS executor shutdown failed") from shutdown_error


class OpenCVFrameSource(_LatestSource[ImageFrame]):
    """Lazy camera adapter that requires a stable by-id device path."""

    def __init__(
        self, device_path: str, *, width: int = 224, height: int = 224, timeout_s: float = 1.0
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        if not device_path.startswith("/dev/v4l/by-id/"):
            raise ValueError("camera source must use a stable /dev/v4l/by-id path")
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("image width and height must be positive integers")
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on vision extra
            raise RuntimeError("OpenCV is unavailable") from exc
        self._cv2 = cv2
        self._device_path = device_path
        self._stored_resolution = (width, height)
        self._capture = cv2.VideoCapture(device_path)
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"cannot open camera source {device_path}")
            self._start_worker(self._grab, "synria-camera-grabber")
        except BaseException:
            _close_all(self._capture.release)
            raise

    def _grab(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            arrived = time.monotonic()
            if not ok:
                if self._stop.is_set():
                    return
                raise RuntimeError("camera frame read failed")
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError("camera must deliver a three-channel BGR image")
            native_resolution = (int(frame.shape[1]), int(frame.shape[0]))
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            stored = self._cv2.resize(
                rgb, self._stored_resolution, interpolation=self._cv2.INTER_AREA
            )
            self._store(ImageFrame(
                data=stored, monotonic_timestamp_s=arrived,
                native_resolution=native_resolution, source_id=self._device_path,
            ))

    def read(self) -> ImageFrame:
        return self._read_latest()

    def close(self) -> None:
        if not self._closed:
            self._stop.set()
            self._join()
            self._capture.release()
            self._closed = True


class LeRobotDatasetWriter:
    """Lazy adapter for the external LeRobot dataset library."""

    def __init__(self, config: PhysicalRecorderConfig) -> None:
        try:
            from lerobot.datasets import DEFAULT_EPISODES_PATH  # type: ignore[import-not-found]
            from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
                LeRobotDataset,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LeRobot dataset dependency is unavailable") from exc

        features: dict[str, dict[str, object]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (13 if config.contract.state_has_velocity else 7,),
            },
            "action": {"dtype": "float32", "shape": (7,)},
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (3, config.image_height, config.image_width),
                "names": ["channels", "height", "width"],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (3, config.image_height, config.image_width),
                "names": ["channels", "height", "width"],
            },
        }
        for timestamp_name in (
            "state_monotonic_s",
            "state_ros_header_s",
            "action_monotonic_s",
            "wrist_monotonic_s",
            "front_monotonic_s",
            "sample_monotonic_s",
            "state_ros_arrival_s",
            "action_ros_header_s",
            "action_ros_arrival_s",
        ):
            features[timestamp_name] = {"dtype": "float64", "shape": (1,)}
        config.require_outside_repository(Path(__file__).resolve().parents[1])
        if config.dataset_path.exists() and not all(
            (config.dataset_path / name).is_file()
            for name in ("physical_contract.json", "meta/info.json")
        ):
            raise ValueError("existing dataset root must contain a physical LeRobot dataset")
        self._transaction = RecordingTransaction(config.dataset_path)
        self._root = self._transaction.root
        contract_path = self._root / "physical_contract.json"
        contract_payload = config.contract.as_dict()
        self._finalized = False
        self._dataset_open = False
        self.recovery_blocked = False
        self._dataset: Any = None
        self._dataset_type = LeRobotDataset
        self._repo_id = config.repo_id
        self._contract = config.contract
        self._episode_path_template = DEFAULT_EPISODES_PATH
        try:
            if self._root.exists():
                self._transaction.files()
                if not contract_path.is_file() or not (self._root / "meta" / "info.json").is_file():
                    raise ValueError(
                        "existing dataset root must contain a physical LeRobot dataset"
                    )
                if json.loads(contract_path.read_text(encoding="utf-8")) != contract_payload:
                    raise ValueError("physical dataset contract differs from existing dataset")
                self._dataset = LeRobotDataset.resume(repo_id=config.repo_id, root=self._root)
                self._dataset_open = True
                for name in ("observation.images.wrist", "observation.images.front"):
                    stored = self._dataset.meta.features.get(name, {})
                    if stored.get("dtype") != features[name]["dtype"] or tuple(
                        stored.get("shape", ())
                    ) != features[name]["shape"]:
                        raise ValueError("image features differ from existing dataset")
                for name, expected in features.items():
                    stored = self._dataset.meta.features.get(name, {})
                    if stored.get("dtype") != expected["dtype"] or tuple(
                        stored.get("shape", ())
                    ) != expected["shape"]:
                        raise ValueError("physical frame features differ from existing dataset")
            else:
                self._dataset = LeRobotDataset.create(
                    repo_id=config.repo_id,
                    root=self._root,
                    fps=int(config.fps),
                    robot_type="synria_alicia_d",
                    features=features,
                )
                self._dataset_open = True
                contract_path.write_text(
                    json.dumps(contract_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            self._validate_segments()
            self._next_episode_index = int(self._dataset.meta.total_episodes)
            self._close_dataset()
        except BaseException:
            _close_all(self._close_dataset, self._transaction.close)
            raise
        self._task = config.task
        self._fps = config.fps
        self._image_width = config.image_width
        self._image_height = config.image_height
        self._metadata_path = self._root / "physical_episode_metadata.jsonl"
        self._quality_records_path = self._root / "physical_quality_records.jsonl"
        self._capture_records_path = self._root / "physical_capture_provenance.jsonl"

    @property
    def next_episode_index(self) -> int:
        return self._next_episode_index

    def _validate_segments(self) -> None:
        """Reject incomplete/orphan segments before upstream can reuse their names."""
        referenced: set[str] = set()
        meta = self._dataset.meta
        rows = list(meta.episodes) if meta.total_episodes else []
        if len(rows) != meta.total_episodes:
            raise RecordingRecoveryError("persisted episode metadata count is inconsistent")
        expected_frame = 0
        for index, row in enumerate(rows):
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            if row["episode_index"] != index or start != expected_frame or (
                end <= start or row["length"] != end - start
            ):
                raise RecordingRecoveryError("persisted episode frame ranges are inconsistent")
            expected_frame = end
        if expected_frame != meta.total_frames:
            raise RecordingRecoveryError("persisted frame metadata count is inconsistent")
        if meta.total_episodes:
            for row in rows:
                referenced.add(meta.data_path.format(
                    chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
                ))
                referenced.add(self._episode_path_template.format(
                    chunk_index=row["meta/episodes/chunk_index"],
                    file_index=row["meta/episodes/file_index"],
                ))
                for key in meta.video_keys:
                    referenced.add(meta.video_path.format(
                        video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                        file_index=row[f"videos/{key}/file_index"],
                    ))
        files = self._transaction.files()
        segments = {
            name for name in files
            if (name.startswith(("data/", "meta/episodes/")) and name.endswith(".parquet"))
            or (name.startswith("videos/") and name.endswith(".mp4"))
        }
        if segments != referenced:
            raise RecordingRecoveryError("dataset has missing or orphan recording segments")

    def _close_dataset(self) -> None:
        if self._dataset_open:
            self._dataset.finalize()
            self._dataset_open = False

    def write_episode(
        self, episode: RecordedPhysicalEpisode
    ) -> None:  # pragma: no cover - optional dependency
        if self._finalized:
            raise RuntimeError("dataset writer is finalized")
        if self.recovery_blocked:
            raise RecordingRecoveryError("dataset recovery is blocked; pending frames retained")
        if episode.episode_index != self._next_episode_index:
            raise ValueError("episode index differs from persisted dataset metadata")
        episode.frames = [
            replace(frame, state=self._contract.prepare_state(frame.state))
            for frame in episode.frames
        ]
        capture_provenance = self._capture_provenance(episode)
        try:
            self._transaction.begin()
        except BaseException:
            self.recovery_blocked = self._transaction.pending
            raise
        try:
            self._dataset = self._dataset_type.resume(repo_id=self._repo_id, root=self._root)
            self._dataset_open = True
            self._validate_segments()
            if self._dataset.meta.total_episodes != self._next_episode_index:
                raise RecordingRecoveryError(
                    "episode metadata changed during the recording session"
                )
            episode.final_still_path = self.save_final_still(
                episode.episode_index, episode.frames[-1].front
            )
            self._write_episode_data(episode, capture_provenance)
            if self._dataset.meta.total_episodes != self._next_episode_index + 1:
                raise RecordingRecoveryError("saved episode count differs from expected metadata")
            self._close_dataset()
            self._transaction.commit()
        except BaseException as save_error:
            try:
                self._close_dataset()
                self._transaction.rollback()
                episode.final_still_path = None
            except BaseException as recovery_error:
                self.recovery_blocked = True
                raise RecordingRecoveryError(
                    f"save failed ({save_error}); recovery blocked ({recovery_error}); "
                    "frames, journal, and lock retained"
                ) from recovery_error
            raise
        self._next_episode_index += 1

    def _write_episode_data(
        self, episode: RecordedPhysicalEpisode, capture_provenance: dict[str, object]
    ) -> None:
        import numpy as np

        for frame in episode.frames:
            timestamp_features = {
                name: np.asarray([value], dtype=np.float64)
                for name, value in frame.timestamps().items()
            }
            self._dataset.add_frame(
                {
                    "observation.state": np.asarray(
                        frame.state.observation_vector(), dtype=np.float32
                    ),
                    "action": np.asarray(frame.action, dtype=np.float32),
                    "observation.images.wrist": frame.wrist.data,
                    "observation.images.front": frame.front.data,
                    "task": self._task,
                    **timestamp_features,
                }
            )
        self._dataset.save_episode(parallel_encoding=False)
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metadata_path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(
                    {
                        "episode_index": episode.episode_index,
                        "operator_label": episode.operator_label.value,
                        "action_source": episode.action_source.value,
                        "contract_version": episode.contract_version,
                        "gripper_type": episode.gripper_type,
                        "final_still": str(episode.final_still_path),
                        "smoke": episode.smoke,
                        "achieved_sample_rate_hz": episode.achieved_sample_rate_hz,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        from synria_lerobot.quality_gates import episode_quality_record

        quality_record = episode_quality_record(episode, fps=self._fps)
        with self._quality_records_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(quality_record.as_dict(), sort_keys=True) + "\n")
        with self._capture_records_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(capture_provenance, sort_keys=True) + "\n")

    def _capture_provenance(self, episode: RecordedPhysicalEpisode) -> dict[str, object]:
        import numpy as np

        if not episode.frames:
            raise ValueError("cannot write an episode without frames")
        cameras: dict[str, str] = {}
        native: dict[str, dict[str, int]] = {}
        expected_shape = (self._image_height, self._image_width, 3)
        for name in ("wrist", "front"):
            images = [getattr(frame, name) for frame in episode.frames]
            first = images[0]
            if first.native_resolution is None or not isinstance(first.source_id, str) or not (
                first.source_id.strip()
            ):
                raise ValueError("camera frame must include native resolution and source id")
            for image in images:
                if not isinstance(image.data, np.ndarray) or image.data.shape != expected_shape:
                    raise ValueError("camera image shape differs from configured stored resolution")
                if image.data.dtype != np.uint8:
                    raise ValueError("camera frames must be RGB uint8")
                if image.native_resolution != first.native_resolution or image.source_id != (
                    first.source_id
                ):
                    raise ValueError("camera identity or native resolution changed during episode")
            cameras[name] = first.source_id
            width, height = first.native_resolution
            native[name] = {"width": width, "height": height}
        capture: dict[str, object] = {
            "episode_index": episode.episode_index,
            "camera_ids": cameras,
            "native_resolution": native,
            "stored_resolution": {"width": self._image_width, "height": self._image_height},
            "stored_color_space": "RGB",
            "achieved_sample_rate_hz": episode.achieved_sample_rate_hz,
        }
        if self._capture_records_path.is_file():
            with self._capture_records_path.open(encoding="utf-8") as source:
                previous = json.loads(next(source))
            for name in (
                "camera_ids", "native_resolution", "stored_resolution", "stored_color_space"
            ):
                if capture[name] != previous[name]:
                    raise ValueError("camera capture settings changed; start a separate dataset")
        return capture

    def finalize(self) -> None:
        if not self._finalized:
            self._close_dataset()
            self._transaction.close()
            self._finalized = True

    def save_final_still(
        self, episode_index: int, frame: ImageFrame
    ) -> Path:  # pragma: no cover - optional dependency
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("OpenCV is unavailable") from exc
        still_dir = self._root / "final_stills"
        still_dir.mkdir(parents=True, exist_ok=True)
        stamp_ns = int(frame.monotonic_timestamp_s * 1_000_000_000)
        path = still_dir / f"episode_{episode_index:06d}_{stamp_ns}.jpg"
        self._transaction.safe_path(path.relative_to(self._root).as_posix())
        if path.exists():
            raise FileExistsError("refusing to overwrite an existing episode still")
        bgr = cv2.cvtColor(frame.data, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"failed to write final still {path}")
        return path


class PhysicalEpisodeRecorder:
    """Operator-controlled recorder with a hard episode-duration cap."""

    def __init__(
        self,
        *,
        config: PhysicalRecorderConfig,
        state_source: StateSource,
        action_source: ActionSource,
        wrist_source: FrameSource,
        front_source: FrameSource,
        writer: DatasetWriter,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if action_source.kind is not config.contract.action_source:
            raise ValueError("action source does not match dataset contract")
        self.config = config
        self.state_source = state_source
        self.action_source = action_source
        self.wrist_source = wrist_source
        self.front_source = front_source
        self.writer = writer
        self.clock = clock
        self.state = RecorderState.IDLE
        self._pending: list[_PendingFrame] = []
        self._started = 0.0
        self._ended = 0.0
        self._next_episode_index = writer.next_episode_index
        self._closed = False
        self._pending_label: OperatorLabel | None = None

    def start(self) -> None:
        if self.writer.recovery_blocked:
            raise RecordingRecoveryError("recovery is blocked; pending frames must be retained")
        if self.state is not RecorderState.IDLE:
            raise RuntimeError(
                "save, retry, or discard the pending episode before starting another"
            )
        if self.config.contract.state_has_velocity:
            self.config.contract.prepare_state(self.state_source.read())
        self._pending = []
        self._pending_label = None
        self._started = self.clock()
        self._ended = self._started
        self.state = RecorderState.RECORDING

    def capture_once(self) -> bool:
        if self.state is not RecorderState.RECORDING:
            raise RuntimeError("start the recorder before capturing")
        now = self.clock()
        if now - self._started >= self.config.hard_cap_s:
            self.stop()
            return False
        follower = self.config.contract.prepare_state(self.state_source.read())
        action = self.action_source.read(follower)
        wrist = self.wrist_source.read()
        front = self.front_source.read()
        sampled = self.clock()
        if sampled - self._started >= self.config.hard_cap_s:
            self.stop()
            return False
        self._pending.append(
            _PendingFrame(
                state=follower,
                action=action,
                wrist=wrist,
                front=front,
                sample_monotonic_s=sampled,
            )
        )
        self._ended = sampled
        return True

    def stop(self) -> None:
        if self.state is not RecorderState.RECORDING:
            raise RuntimeError("no episode is recording")
        self._ended = min(self.clock(), self._started + self.config.hard_cap_s)
        self.state = RecorderState.STOPPED

    def mark_success(self) -> RecordedPhysicalEpisode:
        return self._finalize(OperatorLabel.SUCCESS)

    def mark_failure(self) -> RecordedPhysicalEpisode:
        return self._finalize(OperatorLabel.FAILURE)

    def discard(self) -> None:
        if self.writer.recovery_blocked:
            raise RecordingRecoveryError("recovery is blocked; pending frames must be retained")
        self._pending = []
        self._pending_label = None
        self.state = RecorderState.IDLE

    def retry(self) -> RecordedPhysicalEpisode:
        if self._pending_label is None:
            raise RuntimeError("no failed labeled episode is available to retry")
        return self._finalize(self._pending_label)

    def _finalize(self, label: OperatorLabel) -> RecordedPhysicalEpisode:
        if self.state is not RecorderState.STOPPED:
            raise RuntimeError("stop the episode before applying an operator label")
        if not self._pending:
            raise RuntimeError("cannot save an episode with no frames")
        self._pending_label = label
        frames = self._build_frames()
        episode = RecordedPhysicalEpisode(
            episode_index=self._next_episode_index,
            frames=frames,
            started_monotonic_s=self._started,
            ended_monotonic_s=self._ended,
            operator_label=label,
            action_source=self.config.contract.action_source,
            contract_version=self.config.contract.version,
            gripper_type=self.config.contract.gripper_type,
            smoke=self.config.smoke,
        )
        self.writer.write_episode(episode)
        self._next_episode_index += 1
        self._pending = []
        self._pending_label = None
        self.state = RecorderState.IDLE
        return episode

    def _build_frames(self) -> list[PhysicalFrame]:
        frames: list[PhysicalFrame] = []
        for index, pending in enumerate(self._pending):
            action = pending.action
            if self.config.contract.action_source is ActionSourceKind.NEXT_STATE:
                next_index = min(index + 1, len(self._pending) - 1)
                next_state = self._pending[next_index].state
                action = ActionSample(
                    values=(*next_state.joint_positions_rad, next_state.gripper_m),
                    monotonic_timestamp_s=next_state.monotonic_timestamp_s,
                    ros_header_stamp_s=next_state.ros_header_stamp_s,
                    ros_arrival_stamp_s=next_state.ros_arrival_stamp_s,
                )
            if action is None:
                raise RuntimeError("action source did not provide an action")
            frames.append(
                PhysicalFrame(
                    state=pending.state,
                    action=action.values,
                    action_monotonic_timestamp_s=action.monotonic_timestamp_s,
                    wrist=pending.wrist,
                    front=pending.front,
                    sample_monotonic_timestamp_s=pending.sample_monotonic_s,
                    action_ros_header_stamp_s=action.ros_header_stamp_s,
                    action_ros_arrival_stamp_s=action.ros_arrival_stamp_s,
                )
            )
        return frames

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            _close_all(
                self.state_source.close,
                self.action_source.close,
                self.wrist_source.close,
                self.front_source.close,
                self.writer.finalize,
            )

# ---------------------------------------------------------------------------
# Per-episode recorder
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecorder:
    """Simulate or perform one demonstration episode.

    In simulation mode (``mock=True``, the default), the recorder generates
    synthetic joint states from the demo trajectory and dummy camera frames
    from ``MockFrameSource``. Replace the source objects with real hardware
    handles to switch to live recording.

    Attributes:
        zone: Target staging zone.
        game: Board game context.
        task_variant: Pick-place task variant.
        fps: Recording frame rate in Hz.
        robot_id: Robot identifier stored in episode metadata.
        mock: Use synthetic motion + mock camera (default ``True``).
    """

    zone: str = "left"
    game: str = "chess"
    task_variant: str = "V1"
    fps: float = DEFAULT_FPS
    robot_id: str = DEFAULT_ROBOT_ID
    mock: bool = True
    hardware_recorder: PhysicalEpisodeRecorder | None = None

    def __post_init__(self) -> None:
        if self.zone not in VALID_STAGING_ZONES:
            raise ValueError(f"Unknown zone={self.zone!r}. Valid: {VALID_STAGING_ZONES}")
        if self.game not in VALID_GAMES:
            raise ValueError(f"Unknown game={self.game!r}. Valid: {VALID_GAMES}")
        if self.task_variant not in VALID_TASK_VARIANTS:
            raise ValueError(
                f"Unknown task_variant={self.task_variant!r}. "
                f"Valid: {VALID_TASK_VARIANTS}"
            )
        if self.fps <= 0.0:
            raise ValueError(f"fps={self.fps} must be positive.")

    def record(self, n_steps: int = 0) -> Episode | PhysicalSessionResult:
        """Record (or simulate) one demonstration episode.

        Args:
            n_steps: Limit episode to this many steps (0 = full trajectory).

        Returns:
            Populated :class:`~synria_lerobot.schema.Episode`.
        """
        if self.mock:
            return self._record_mock(n_steps=n_steps)
        return self._record_hardware(n_steps=n_steps)  # pragma: no cover

    # ------------------------------------------------------------------
    # Mock path
    # ------------------------------------------------------------------

    def _record_mock(self, n_steps: int) -> Episode:
        """Generate synthetic episode from arm_control demo trajectory."""
        return generate_synthetic_episode(
            zone=self.zone,
            game=self.game,
            task_variant=self.task_variant,
            fps=self.fps,
            robot_id=self.robot_id,
            n_steps=n_steps,
        )

    # ------------------------------------------------------------------
    # Hardware path (stub — filled in during Jetson bring-up)
    # ------------------------------------------------------------------

    def _record_hardware(self, n_steps: int) -> PhysicalSessionResult:
        """Run the operator-controlled physical recorder configured by the caller."""
        del n_steps
        if self.hardware_recorder is None:
            raise RuntimeError("hardware_recorder must be configured for physical capture")
        return run_operator_loop(self.hardware_recorder)

    # ------------------------------------------------------------------
    # Camera loop (used for timing / validation)
    # ------------------------------------------------------------------

    def camera_loop_stats(self, n_frames: int = 30) -> dict[str, Any]:
        """Run the camera inference loop for ``n_frames`` and return stats.

        Uses ``MockFrameSource`` in mock mode. Replace with a real capture
        source when hardware is available.
        """
        source = MockFrameSource(
            height=224, width=224, channels=3, max_frames=n_frames
        )
        loop = CameraInferenceLoop(
            source=source,
            inference_fn=lambda f: f,
            max_frames=n_frames,
        )
        loop.run()
        stats = loop.stats()
        return stats.to_dict() if stats else {}


# ---------------------------------------------------------------------------
# Multi-episode recording session
# ---------------------------------------------------------------------------


@dataclass
class RecordingSession:
    """Orchestrate recording of multiple episodes across zones and games.

    Attributes:
        n_episodes: Target number of episodes to record.
        zones: Staging zones to cycle through.
        games: Board games to cycle through.
        task_variant: Pick-place task variant.
        fps: Recording frame rate.
        robot_id: Robot identifier.
        mock: Use synthetic motion (default ``True``).
    """

    n_episodes: int = 4
    zones: tuple[str, ...] = ("left", "right", "top", "bottom")
    games: tuple[str, ...] = ("chess",)
    task_variant: str = "V1"
    fps: float = DEFAULT_FPS
    robot_id: str = DEFAULT_ROBOT_ID
    mock: bool = True
    _recorded: list[Episode] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.n_episodes < 1:
            raise ValueError(f"n_episodes={self.n_episodes} must be ≥ 1.")
        for z in self.zones:
            if z not in VALID_STAGING_ZONES:
                raise ValueError(f"Unknown zone={z!r}.")
        for g in self.games:
            if g not in VALID_GAMES:
                raise ValueError(f"Unknown game={g!r}.")

    def record_all(self, n_steps: int = 0) -> SynriaEpisodeDataset:
        """Record all planned episodes and return the resulting dataset.

        Episodes are assigned round-robin across (zone, game) combinations.
        When ``n_episodes`` exceeds the combination count, the cycle repeats.

        Args:
            n_steps: Per-episode step cap (0 = full trajectory).

        Returns:
            :class:`~synria_lerobot.dataset.SynriaEpisodeDataset` with all episodes.
        """
        combos = [(z, g) for g in self.games for z in self.zones]
        self._recorded = []
        for i in range(self.n_episodes):
            zone, game = combos[i % len(combos)]
            rec = EpisodeRecorder(
                zone=zone,
                game=game,
                task_variant=self.task_variant,
                fps=self.fps,
                robot_id=self.robot_id,
                mock=self.mock,
            )
            ep = rec.record(n_steps=n_steps)
            if not isinstance(ep, Episode):
                raise RuntimeError("RecordingSession only supports synthetic episodes")
            self._recorded.append(ep)
        return SynriaEpisodeDataset(episodes=list(self._recorded))

    @property
    def n_recorded(self) -> int:
        """Number of episodes recorded so far."""
        return len(self._recorded)


# ---------------------------------------------------------------------------
# Timing helper (standalone, no episode data)
# ---------------------------------------------------------------------------


def benchmark_recording_fps(n_frames: int = 60, fps: float = DEFAULT_FPS) -> dict[str, Any]:
    """Time the mock camera loop and return throughput statistics.

    This is a lightweight sanity check that the recording pipeline can
    sustain the target FPS without a real camera.
    """
    source = MockFrameSource(height=224, width=224, channels=3, max_frames=n_frames)
    loop = CameraInferenceLoop(
        source=source,
        inference_fn=lambda f: f,
        max_frames=n_frames,
    )
    t0 = time.perf_counter()
    loop.run()
    elapsed_s = time.perf_counter() - t0
    stats = loop.stats()
    return {
        "target_fps": fps,
        "n_frames": n_frames,
        "elapsed_s": round(elapsed_s, 4),
        "mock_fps": stats.mean_fps if stats else 0.0,
        "mean_inference_ms": stats.mean_inference_ms if stats else 0.0,
    }


def run_operator_loop(recorder: PhysicalEpisodeRecorder) -> PhysicalSessionResult:
    """Record many episodes while retaining only the currently pending frames."""
    import select
    import sys

    print("Commands: start | stop | success | failure | retry | discard | quit")
    result = PhysicalSessionResult(smoke=recorder.config.smoke)
    next_sample = recorder.clock()
    try:
        while True:
            timeout_s = 0.1
            if recorder.state is RecorderState.RECORDING:
                now = recorder.clock()
                if now >= next_sample:
                    if not recorder.capture_once():
                        print("Hard cap reached; episode stopped.")
                    next_sample = now + 1.0 / recorder.config.fps
                timeout_s = max(0.0, min(0.1, next_sample - now))
            readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if not readable:
                continue
            command_line = sys.stdin.readline()
            if command_line == "":
                raise RuntimeError("operator input closed")
            command = command_line.strip().lower()
            if command in {"quit", "q", "exit"}:
                if recorder.state is not RecorderState.IDLE:
                    print(
                        "Pending frames retained. Save, retry, or explicitly discard before quit."
                    )
                    continue
                return result
            try:
                if command == "start":
                    recorder.start()
                    next_sample = recorder.clock()
                elif command == "stop":
                    recorder.stop()
                elif command in {"success", "failure", "retry"}:
                    try:
                        episode = {
                            "success": recorder.mark_success,
                            "failure": recorder.mark_failure,
                            "retry": recorder.retry,
                        }[command]()
                    except Exception as error:
                        result.failed_save_count += 1
                        print(
                            f"Episode save failed: {error}. Frames retained; use retry or discard."
                        )
                        continue
                    result.saved_episode_count += 1
                    result.last_episode_index = episode.episode_index
                    result.last_achieved_sample_rate_hz = episode.achieved_sample_rate_hz
                    print(
                        f"Saved episode {episode.episode_index}: {episode.operator_label.value}; "
                        f"achieved {episode.achieved_sample_rate_hz:.2f} samples/s "
                        f"(requested {recorder.config.fps:g})."
                    )
                    del episode
                    if recorder.config.smoke:
                        return result
                elif command == "discard":
                    had_pending = recorder.state is not RecorderState.IDLE
                    recorder.discard()
                    result.discarded_episode_count += int(had_pending)
                elif command:
                    print(
                        "Unknown command. Use start, stop, success, failure, retry, discard, quit."
                    )
            except Exception as error:
                print(f"Command refused: {error}")
    finally:
        recorder.close()


def _parse_physical_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record physical Synria demonstrations")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--gripper-type", choices=("50mm", "100mm"), required=True)
    parser.add_argument(
        "--action-source",
        choices=tuple(source.value for source in ActionSourceKind),
        default=ActionSourceKind.NEXT_STATE.value,
    )
    parser.add_argument("--follower-topic", default="/joint_states")
    parser.add_argument("--leader-topic", default="/leader/joint_states")
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--front-camera", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--state-has-velocity", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def physical_main() -> int:  # pragma: no cover - hardware entry point
    args = _parse_physical_args()
    contract = PhysicalDatasetContract(
        gripper_type=args.gripper_type,
        action_source=ActionSourceKind(args.action_source),
        state_has_velocity=args.state_has_velocity,
    )
    with ExitStack() as session_cleanup:
        dataset_path: Path | None = args.dataset_path
        if args.smoke:
            temporary_path = session_cleanup.enter_context(
                tempfile.TemporaryDirectory(prefix="synria-d1-smoke-")
            )
            dataset_path = Path(temporary_path) / "dataset"
        elif dataset_path is None:
            raise SystemExit("--dataset-path is required unless --smoke is used")
        assert dataset_path is not None
        config = PhysicalRecorderConfig(
            dataset_path=dataset_path,
            repo_id=args.repo_id,
            contract=contract,
            fps=args.fps,
            image_width=args.image_width,
            image_height=args.image_height,
            min_episode_s=20.0,
            max_episode_s=20.0 if args.smoke else 30.0,
            hard_cap_s=20.0 if args.smoke else 30.0,
            smoke=args.smoke,
        )
        config.require_outside_repository(Path(__file__).resolve().parents[1])
        with ExitStack() as startup_cleanup:
            writer = LeRobotDatasetWriter(config)
            startup_cleanup.callback(_close_all, writer.finalize)
            follower = RosJointStateSource(
                args.follower_topic, node_name="synria_d1_follower_state",
                state_has_velocity=contract.state_has_velocity,
            )
            startup_cleanup.callback(_close_all, follower.close)
            if contract.action_source is ActionSourceKind.LEADER:
                leader_state = RosJointStateSource(
                    args.leader_topic, node_name="synria_d1_leader_state", state_has_velocity=False,
                )
                startup_cleanup.callback(_close_all, leader_state.close)
                action_source: ActionSource = LeaderActionSource(leader_state)
            else:
                action_source = NextStateActionSource()
            wrist = OpenCVFrameSource(
                args.wrist_camera, width=config.image_width, height=config.image_height
            )
            startup_cleanup.callback(_close_all, wrist.close)
            front = OpenCVFrameSource(
                args.front_camera, width=config.image_width, height=config.image_height
            )
            startup_cleanup.callback(_close_all, front.close)
            recorder = PhysicalEpisodeRecorder(
                config=config,
                state_source=follower,
                action_source=action_source,
                wrist_source=wrist,
                front_source=front,
                writer=writer,
            )
            session_cleanup.callback(_close_all, recorder.close)
            startup_cleanup.pop_all()
        result = run_operator_loop(recorder)
        print(
            json.dumps(
                {
                    **asdict(result),
                    "dataset_path": str(dataset_path),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(physical_main())
