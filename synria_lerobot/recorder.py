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
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

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

    def write_episode(self, episode: RecordedPhysicalEpisode) -> None: ...

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


@dataclass(frozen=True)
class PhysicalRecorderConfig:
    dataset_path: Path
    repo_id: str
    contract: PhysicalDatasetContract
    fps: float = 30.0
    min_episode_s: float = 20.0
    max_episode_s: float = 30.0
    hard_cap_s: float = 30.0
    task: str = "move the token from square A to square B"
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
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


@dataclass(frozen=True)
class _PendingFrame:
    state: PhysicalState
    action: ActionSample | None
    wrist: ImageFrame
    front: ImageFrame


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
        )

    def close(self) -> None:
        self._source.close()


class RosJointStateSource:
    """Lazy ROS subscriber adapter; it never creates a publisher."""

    def __init__(self, topic: str, *, node_name: str) -> None:
        try:
            import rclpy  # type: ignore[import-not-found]
            from sensor_msgs.msg import JointState  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on ROS installation
            raise RuntimeError("ROS 2 state dependencies are unavailable") from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._latest: PhysicalState | None = None

        def on_state(message: Any) -> None:
            values = dict(zip(message.name, message.position, strict=True))
            velocities = None
            if len(message.velocity) == len(message.name):
                by_name = dict(zip(message.name, message.velocity, strict=True))
                velocities = tuple(float(by_name[f"Joint{i}"]) for i in range(1, 7))
            stamp = float(message.header.stamp.sec) + float(
                message.header.stamp.nanosec
            ) / 1_000_000_000
            self._latest = PhysicalState(
                joint_positions_rad=tuple(
                    float(values[f"Joint{i}"]) for i in range(1, 7)
                ),
                gripper_m=float(values["Gripper"]),
                monotonic_timestamp_s=time.monotonic(),
                ros_header_stamp_s=stamp,
                joint_velocities_rad_s=velocities,
            )

        self._subscription = self._node.create_subscription(
            JointState, topic, on_state, 20
        )

    def read(self) -> PhysicalState:  # pragma: no cover - depends on ROS installation
        self._rclpy.spin_once(self._node, timeout_sec=0.1)
        if self._latest is None:
            raise RuntimeError("no joint state has been received")
        return self._latest

    def close(self) -> None:  # pragma: no cover - depends on ROS installation
        self._node.destroy_subscription(self._subscription)
        self._node.destroy_node()


class OpenCVFrameSource:
    """Lazy camera adapter that requires a stable by-id device path."""

    def __init__(self, device_path: str) -> None:
        if not device_path.startswith("/dev/v4l/by-id/"):
            raise ValueError("camera source must use a stable /dev/v4l/by-id path")
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on vision extra
            raise RuntimeError("OpenCV is unavailable") from exc
        self._cv2 = cv2
        self._capture = cv2.VideoCapture(device_path)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open camera source {device_path}")

    def read(self) -> ImageFrame:  # pragma: no cover - depends on camera
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")
        return ImageFrame(data=frame, monotonic_timestamp_s=time.monotonic())

    def close(self) -> None:  # pragma: no cover - depends on camera
        self._capture.release()


class LeRobotDatasetWriter:
    """Lazy adapter for the external LeRobot dataset library."""

    def __init__(self, config: PhysicalRecorderConfig) -> None:
        try:
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
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (3, 224, 224),
                "names": ["channels", "height", "width"],
            },
        }
        for timestamp_name in (
            "state_monotonic_s",
            "state_ros_header_s",
            "action_monotonic_s",
            "wrist_monotonic_s",
            "front_monotonic_s",
        ):
            features[timestamp_name] = {"dtype": "float64", "shape": (1,)}
        self._root = config.dataset_path
        contract_path = self._root / "physical_contract.json"
        contract_payload = config.contract.as_dict()
        if self._root.exists():
            if not contract_path.is_file() or not (self._root / "meta" / "info.json").is_file():
                raise ValueError("existing dataset root must contain a physical LeRobot dataset")
            if json.loads(contract_path.read_text(encoding="utf-8")) != contract_payload:
                raise ValueError("physical dataset contract differs from existing dataset")
            self._dataset = LeRobotDataset.resume(
                repo_id=config.repo_id,
                root=config.dataset_path,
            )
        else:
            self._dataset = LeRobotDataset.create(
                repo_id=config.repo_id,
                root=config.dataset_path,
                fps=int(config.fps),
                robot_type="synria_alicia_d",
                features=features,
            )
            contract_path.write_text(
                json.dumps(contract_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self._task = config.task
        self._fps = config.fps
        self._metadata_path = self._root / "physical_episode_metadata.jsonl"
        self._quality_records_path = self._root / "physical_quality_records.jsonl"

    @property
    def next_episode_index(self) -> int:
        return int(self._dataset.meta.total_episodes)

    def write_episode(
        self, episode: RecordedPhysicalEpisode
    ) -> None:  # pragma: no cover - optional dependency
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
                    **timestamp_features,
                },
                task=self._task,
            )
        self._dataset.save_episode()
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
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        from synria_lerobot.quality_gates import episode_quality_record

        quality_record = episode_quality_record(episode, fps=self._fps)
        with self._quality_records_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(quality_record.as_dict(), sort_keys=True) + "\n")

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
        if not cv2.imwrite(str(path), frame.data):
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

    def start(self) -> None:
        if self.state is RecorderState.RECORDING:
            raise RuntimeError("an episode is already recording")
        self._pending = []
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
        follower = self.state_source.read()
        self._pending.append(
            _PendingFrame(
                state=follower,
                action=self.action_source.read(follower),
                wrist=self.wrist_source.read(),
                front=self.front_source.read(),
            )
        )
        self._ended = now
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
        self._pending = []
        self.state = RecorderState.IDLE

    def _finalize(self, label: OperatorLabel) -> RecordedPhysicalEpisode:
        if self.state is not RecorderState.STOPPED:
            raise RuntimeError("stop the episode before applying an operator label")
        if not self._pending:
            raise RuntimeError("cannot save an episode with no frames")
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
        episode.final_still_path = self.writer.save_final_still(
            episode.episode_index, frames[-1].front
        )
        self.writer.write_episode(episode)
        if not self.config.smoke:
            self._next_episode_index += 1
        self._pending = []
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
                    monotonic_timestamp_s=pending.state.monotonic_timestamp_s,
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
                )
            )
        return frames

    def close(self) -> None:
        self.state_source.close()
        self.action_source.close()
        self.wrist_source.close()
        self.front_source.close()

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

    def record(self, n_steps: int = 0) -> Episode | RecordedPhysicalEpisode:
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

    def _record_hardware(self, n_steps: int) -> RecordedPhysicalEpisode:
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


def run_operator_loop(recorder: PhysicalEpisodeRecorder) -> RecordedPhysicalEpisode:
    """Run the start/stop/label/discard keyboard state machine."""
    import select
    import sys

    print("Commands: start | stop | success | failure | discard | quit")
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
            if command == "start":
                recorder.start()
                next_sample = recorder.clock()
            elif command == "stop":
                recorder.stop()
            elif command == "success":
                return recorder.mark_success()
            elif command == "failure":
                return recorder.mark_failure()
            elif command == "discard":
                recorder.discard()
            elif command in {"quit", "q", "exit"}:
                recorder.discard()
                raise RuntimeError("recording cancelled by operator")
            elif command:
                print("Unknown command. Use: start | stop | success | failure | discard | quit")
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
        required=True,
    )
    parser.add_argument("--follower-topic", default="/joint_states")
    parser.add_argument("--leader-topic", default="/leader/joint_states")
    parser.add_argument("--wrist-camera", required=True)
    parser.add_argument("--front-camera", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
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
    temporary: tempfile.TemporaryDirectory[str] | None = None
    dataset_path: Path | None = args.dataset_path
    if args.smoke:
        temporary = tempfile.TemporaryDirectory(prefix="synria-d1-smoke-")
        dataset_path = Path(temporary.name) / "dataset"
    elif dataset_path is None:
        raise SystemExit("--dataset-path is required unless --smoke is used")
    assert dataset_path is not None
    config = PhysicalRecorderConfig(
        dataset_path=dataset_path,
        repo_id=args.repo_id,
        contract=contract,
        fps=args.fps,
        min_episode_s=20.0,
        max_episode_s=20.0 if args.smoke else 30.0,
        hard_cap_s=20.0 if args.smoke else 30.0,
        smoke=args.smoke,
    )
    config.require_outside_repository(Path(__file__).resolve().parents[1])
    follower = RosJointStateSource(
        args.follower_topic, node_name="synria_d1_follower_state"
    )
    if contract.action_source is ActionSourceKind.LEADER:
        leader_state = RosJointStateSource(
            args.leader_topic, node_name="synria_d1_leader_state"
        )
        action_source: ActionSource = LeaderActionSource(leader_state)
    else:
        action_source = NextStateActionSource()
    recorder = PhysicalEpisodeRecorder(
        config=config,
        state_source=follower,
        action_source=action_source,
        wrist_source=OpenCVFrameSource(args.wrist_camera),
        front_source=OpenCVFrameSource(args.front_camera),
        writer=LeRobotDatasetWriter(config),
    )
    episode = run_operator_loop(recorder)
    print(
        json.dumps(
            {
                "episode_index": episode.episode_index,
                "operator_label": episode.operator_label.value,
                "duration_s": episode.duration_s,
                "smoke": episode.smoke,
                "dataset_path": str(dataset_path),
            },
            sort_keys=True,
        )
    )
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(physical_main())
