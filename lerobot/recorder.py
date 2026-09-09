"""Episode recording interface for the Synria LeRobot track.

Simulates the LeRobot demonstration recording workflow without requiring
a physical arm or camera. The recording interface is designed to be a
drop-in replacement for real hardware recording once the Synria arm and
C10 camera are live.

Hardware path (hardware-later):
    Replace ``MockFrameSource`` with ``cv2.VideoCapture(0)`` and replace
    the synthetic joint-position generator with the real ROS 2 ``/joint_states``
    subscriber. The ``Episode`` schema and observation contract are identical
    for both paths.

Usage::

    from lerobot.recorder import EpisodeRecorder, RecordingSession

    recorder = EpisodeRecorder(zone="left", game="chess")
    episode = recorder.record(n_steps=60)

    session = RecordingSession(n_episodes=4, zones=("left", "right"))
    dataset = session.record_all()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from edge_ai.camera_inference import CameraInferenceLoop, MockFrameSource
from lerobot.dataset import SynriaEpisodeDataset, generate_synthetic_episode
from lerobot.schema import (
    DEFAULT_FPS,
    DEFAULT_ROBOT_ID,
    VALID_GAMES,
    VALID_STAGING_ZONES,
    VALID_TASK_VARIANTS,
    Episode,
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

    def record(self, n_steps: int = 0) -> Episode:
        """Record (or simulate) one demonstration episode.

        Args:
            n_steps: Limit episode to this many steps (0 = full trajectory).

        Returns:
            Populated :class:`~lerobot.schema.Episode`.
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

    def _record_hardware(self, n_steps: int) -> Episode:  # pragma: no cover
        """Live recording from ROS 2 joint states + C10 camera.

        TODO(jetson): Subscribe to /joint_states, /c10_camera/image_raw, and TF.
        Replace MockFrameSource with cv2.VideoCapture(0) (or a ROS image bridge).
        Replace synthetic joint positions with real sensor readings.
        """
        raise NotImplementedError(
            "Hardware recording is not yet implemented. "
            "Set mock=True for simulation, or implement this method with "
            "real ROS 2 subscribers and cv2.VideoCapture."
        )

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
            :class:`~lerobot.dataset.SynriaEpisodeDataset` with all episodes.
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
