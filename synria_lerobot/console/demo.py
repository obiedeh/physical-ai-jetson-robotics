"""Clock-indexed synthetic sources that cannot be mistaken for physical capture.

Preview reads do not advance the trajectory. The frame number and synthetic
timestamp are rendered into each image, making recorded synchronization visible.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from synria_lerobot.physical_contract import ImageFrame, PhysicalState
from synria_lerobot.task_registry import SYNTHETIC_DEMO

if TYPE_CHECKING:
    from synria_lerobot.recorder import PhysicalRecorderConfig


class SyntheticStateSource:
    """Generate a deterministic bounded joint trajectory without a runtime connection."""

    def __init__(
        self, config: PhysicalRecorderConfig, *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Accept only explicitly nonqualifying demo configuration."""
        if config.contract.recording_purpose != SYNTHETIC_DEMO:
            raise ValueError("synthetic sources require the synthetic_demo recording purpose")
        self.config = config
        self.clock = clock
        self.closed = False

    def read(self) -> PhysicalState:
        """Return the trajectory at this clock tick, independent of preview read count."""
        if self.closed:
            raise RuntimeError("synthetic state source is closed")
        now = self.clock()
        sampled = math.floor(now * self.config.fps) / self.config.fps
        phase = sampled % 20.0
        positions = tuple(0.1 * math.sin(phase + joint) for joint in range(6))
        velocities = tuple(0.1 * math.cos(phase + joint) for joint in range(6))
        return PhysicalState(
            joint_positions_rad=positions,
            gripper_m=self.config.contract.gripper_stroke_m * (0.5 + 0.2 * math.sin(phase)),
            monotonic_timestamp_s=sampled,
            ros_header_stamp_s=sampled,
            ros_arrival_stamp_s=now,
            joint_velocities_rad_s=velocities if self.config.contract.state_has_velocity else None,
        )

    def close(self) -> None:
        """Mark the synthetic source closed without any device cleanup."""
        self.closed = True


class SyntheticFrameSource:
    """Cache one generated RGB sample per clock tick, never opening a camera."""

    def __init__(
        self, name: str, config: PhysicalRecorderConfig,
        *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Keep generated native and stored resolutions distinct for still-image testing."""
        if config.contract.recording_purpose != SYNTHETIC_DEMO:
            raise ValueError("synthetic sources require the synthetic_demo recording purpose")
        if name not in {"wrist", "front"}:
            raise ValueError("unknown synthetic camera")
        self.name = name
        self.config = config
        self.clock = clock
        self.closed = False
        self._lock = threading.Lock()
        self._index: int | None = None
        self._frame: ImageFrame | None = None

    def read(self) -> ImageFrame:
        """Generate outside the cache lock and return immutable latest-size RGB pixels."""
        if self.closed:
            raise RuntimeError("synthetic camera source is closed")
        now = self.clock()
        index = math.floor(now * self.config.fps)
        with self._lock:
            if index == self._index and self._frame is not None:
                return self._frame
        import cv2
        import numpy as np

        width, height = 640, 480
        native: Any = np.empty((height, width, 3), dtype=np.uint8)
        native[:, :, 0] = 45 + index % 100
        native[:, :, 1] = 90 if self.name == "wrist" else 140
        native[:, :, 2] = 150 if self.name == "wrist" else 70
        center = (40 + index * 7 % (width - 80), height // 2)
        cv2.circle(native, center, 30, (220, 210, 50), -1)
        cv2.putText(
            native, f"SYNTHETIC DEMO - {self.name}", (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            native, f"frame {index}  time {index / self.config.fps:.3f}s", (15, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        stored = cv2.resize(native, (self.config.image_width, self.config.image_height))
        native.setflags(write=False)
        stored.setflags(write=False)
        frame = ImageFrame(
            stored, index / self.config.fps, source_id=f"synthetic-demo:{self.name}",
            native_resolution=(width, height), native_data=native,
        )
        with self._lock:
            if self._index is None or index >= self._index:
                self._frame, self._index = frame, index
            return frame

    def close(self) -> None:
        """Release the cached synthetic pixels without accessing a device."""
        self.closed = True
        with self._lock:
            self._frame = None
