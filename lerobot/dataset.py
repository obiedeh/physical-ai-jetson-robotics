"""Synthetic episode dataset generator for the Synria LeRobot track.

Generates demonstration episodes by sampling the canned arm-control demo
trajectories defined in ``arm_control.demo``. The resulting episodes match
the observation/action contract in ``lerobot.schema`` and can be written
to ``reports/lerobot/`` as evidence of the recording schema.

No real hardware is required; all motion data comes from the validated
demo waypoints that already pass the ``ArmSafetyGate``.

Usage::

    from lerobot.dataset import generate_synthetic_episode, SynriaEpisodeDataset

    ep = generate_synthetic_episode(zone="left", game="chess", n_steps=60)
    ds = SynriaEpisodeDataset([ep])
    ds.write_json(Path("reports/lerobot/demo_dataset.json"))
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from arm_control.demo import full_pick_place_sequence
from arm_control.kinematics import LINK_BASE_HEIGHT_M, forward_kinematics_planar
from arm_control.trajectory import JOINT_NAMES, JointTrajectory, JointWaypoint
from lerobot.schema import (
    DEFAULT_FPS,
    DEFAULT_ROBOT_ID,
    GRIPPER_CLOSED_M,
    GRIPPER_OPEN_M,
    VALID_GAMES,
    VALID_STAGING_ZONES,
    VALID_TASK_VARIANTS,
    ActionFrame,
    EEPose,
    Episode,
    EpisodeMetadata,
    EpisodeStep,
    JointState,
    ObservationFrame,
    now_iso,
)

# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

_ZERO_VELOCITIES: tuple[float, ...] = (0.0,) * 6


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at fraction t ∈ [0, 1]."""
    return a + (b - a) * t


def _lerp_joint_positions(
    start: dict[str, float],
    end: dict[str, float],
    t: float,
) -> dict[str, float]:
    """Linearly interpolate between two joint-position dicts."""
    return {name: _lerp(start[name], end[name], t) for name in JOINT_NAMES}


def _estimate_velocities(
    prev: dict[str, float],
    curr: dict[str, float],
    dt_s: float,
) -> tuple[float, ...]:
    """First-order finite-difference velocity estimate (rad/s)."""
    if dt_s <= 0.0:
        return _ZERO_VELOCITIES
    return tuple(
        (curr[name] - prev[name]) / dt_s for name in JOINT_NAMES
    )


def _compute_ee_pose(
    positions: dict[str, float],
    timestamp_s: float,
) -> EEPose:
    """Stub 3-D end-effector pose using planar FK + joint_1 azimuth.

    Projects the shoulder/elbow plane (q2, q3) forward, then rotates the
    reach vector by joint_1 around the vertical axis. Wrist joints (4–6)
    contribute identity orientation for now; the stub quaternion is set
    to the identity (w=1, x=y=z=0).

    Full 3-D FK should be added once the vendor DH table is confirmed
    (see ``docs/SYNRIA_UPSTREAM_TODO.md`` activity 1).
    """
    q1 = positions.get("joint_1", 0.0)
    q2 = positions.get("joint_2", 0.0)
    q3 = positions.get("joint_3", 0.0)

    fk = forward_kinematics_planar(q2_rad=q2, q3_rad=q3)
    # Rotate the sagittal-plane reach by joint_1 around Z.
    x_m = fk.x_m * math.cos(q1)
    y_m = fk.x_m * math.sin(q1)
    z_m = fk.z_m + LINK_BASE_HEIGHT_M  # absolute height above table

    return EEPose(
        x_m=round(x_m, 5),
        y_m=round(y_m, 5),
        z_m=round(z_m, 5),
        qw=1.0,
        qx=0.0,
        qy=0.0,
        qz=0.0,
        timestamp_s=timestamp_s,
    )


# ---------------------------------------------------------------------------
# Trajectory → per-frame sample list
# ---------------------------------------------------------------------------


def _sample_trajectory(
    trajectory: JointTrajectory,
    fps: float,
    t_offset_s: float = 0.0,
) -> list[tuple[float, dict[str, float]]]:
    """Sample a JointTrajectory at ``fps`` Hz, returning (time_s, positions).

    Linear interpolation is used between waypoints.  The first and last
    waypoints are always included regardless of the FPS grid.

    Args:
        trajectory: Trajectory with at least one waypoint.
        fps: Desired sample rate in Hz.
        t_offset_s: Add this offset to all returned timestamps (for multi-
            trajectory concatenation).

    Returns:
        List of ``(timestamp_s, positions_dict)`` tuples in chronological order.
    """
    wps: list[JointWaypoint] = trajectory.waypoints
    if not wps:
        return []

    duration = trajectory.duration_s()
    dt = 1.0 / fps

    samples: list[tuple[float, dict[str, float]]] = []
    t = 0.0
    while t <= duration + dt * 0.5:
        t_clamped = min(t, duration)
        # Find surrounding waypoints.
        if len(wps) == 1 or t_clamped <= wps[0].time_from_start_s:
            pos = dict(wps[0].positions_rad)
        elif t_clamped >= wps[-1].time_from_start_s:
            pos = dict(wps[-1].positions_rad)
        else:
            # Binary-search for the segment.
            lo, hi = 0, len(wps) - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if wps[mid].time_from_start_s <= t_clamped:
                    lo = mid
                else:
                    hi = mid
            t_lo = wps[lo].time_from_start_s
            t_hi = wps[hi].time_from_start_s
            alpha = (t_clamped - t_lo) / (t_hi - t_lo) if t_hi > t_lo else 0.0
            pos = _lerp_joint_positions(
                dict(wps[lo].positions_rad),
                dict(wps[hi].positions_rad),
                alpha,
            )
        samples.append((round(t_clamped + t_offset_s, 5), pos))
        if t >= duration:
            break
        t = min(t + dt, duration)

    return samples


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------


def generate_synthetic_episode(
    zone: str = "left",
    game: str = "chess",
    task_variant: str = "V1",
    fps: float = DEFAULT_FPS,
    robot_id: str = DEFAULT_ROBOT_ID,
    episode_id: str = "",
    n_steps: int = 0,
) -> Episode:
    """Generate a synthetic Synria pick-place demonstration episode.

    Samples the four-trajectory pick-place sequence from
    ``arm_control.demo.full_pick_place_sequence`` at ``fps`` Hz, producing
    per-step :class:`~lerobot.schema.EpisodeStep` objects that match the
    LeRobot observation/action contract.

    Args:
        zone: Staging zone — ``"left"``, ``"right"``, ``"top"``, or ``"bottom"``.
        game: Board game context — ``"ludo"``, ``"chess"``, or ``"checkers"``.
        task_variant: Task variant — ``"V1"``, ``"V2"``, or ``"V3"``.
        fps: Sample rate in Hz (default 30).
        robot_id: Robot identifier stored in metadata.
        episode_id: Unique episode ID (auto-generated UUID if empty).
        n_steps: Cap the episode at this many steps (0 = no cap — use full
            trajectory).

    Returns:
        A fully populated :class:`~lerobot.schema.Episode`.

    Raises:
        ValueError: For invalid zone / game / task_variant.
    """
    if zone not in VALID_STAGING_ZONES:
        raise ValueError(f"Unknown zone={zone!r}. Valid: {VALID_STAGING_ZONES}")
    if game not in VALID_GAMES:
        raise ValueError(f"Unknown game={game!r}. Valid: {VALID_GAMES}")
    if task_variant not in VALID_TASK_VARIANTS:
        raise ValueError(f"Unknown task_variant={task_variant!r}. Valid: {VALID_TASK_VARIANTS}")

    eid = episode_id or str(uuid.uuid4())

    # ------------------------------------------------------------------ #
    # Build the full timeline by concatenating all 4 trajectories.
    # ------------------------------------------------------------------ #
    trajectories = full_pick_place_sequence(zone)
    all_samples: list[tuple[float, dict[str, float]]] = []
    t_cursor = 0.0
    for traj in trajectories:
        seg = _sample_trajectory(traj, fps=fps, t_offset_s=t_cursor)
        # Avoid duplicating the shared boundary point between trajectories.
        if all_samples and seg:
            seg = seg[1:]
        all_samples.extend(seg)
        t_cursor = all_samples[-1][0] if all_samples else 0.0

    # Apply n_steps cap.
    if n_steps > 0:
        all_samples = all_samples[:n_steps]

    # ------------------------------------------------------------------ #
    # Convert samples → EpisodeStep objects.
    # ------------------------------------------------------------------ #
    steps: list[EpisodeStep] = []

    # Gripper: open during approach, close at board, open at staging, close return.
    total = len(all_samples)

    def _gripper_state(i: int) -> float:
        """Heuristic gripper width based on phase (pick = close, move = open)."""
        phase_frac = i / max(total - 1, 1)
        # Close gripper in the middle 20% of the episode (pick phase).
        if 0.35 < phase_frac < 0.55:
            return GRIPPER_CLOSED_M
        return GRIPPER_OPEN_M

    for i, (t_s, pos) in enumerate(all_samples):
        # Velocities via finite difference.
        if i > 0:
            prev_pos = all_samples[i - 1][1]
            actual_dt = t_s - all_samples[i - 1][0]
            vels = _estimate_velocities(prev_pos, pos, actual_dt)
        else:
            vels = _ZERO_VELOCITIES

        grip_m = _gripper_state(i)
        js = JointState(
            positions_rad=tuple(pos[j] for j in JOINT_NAMES),
            velocities_rad_s=vels,
            gripper_open_m=round(grip_m, 4),
            timestamp_s=t_s,
        )
        ee = _compute_ee_pose(pos, timestamp_s=t_s)
        obs = ObservationFrame(joint_state=js, ee_pose=ee, frame_index=i, timestamp_s=t_s)

        # Action: delta to next waypoint (zero on last step).
        if i < len(all_samples) - 1:
            next_pos = all_samples[i + 1][1]
            deltas = tuple(next_pos[j] - pos[j] for j in JOINT_NAMES)
            next_grip = _gripper_state(i + 1)
            grip_cmd = (next_grip - grip_m) / GRIPPER_OPEN_M  # normalise to [−1, +1]
            grip_cmd = max(-1.0, min(1.0, grip_cmd))
        else:
            deltas = (0.0,) * 6
            grip_cmd = 0.0

        act = ActionFrame(
            joint_deltas_rad=deltas,
            gripper_command=round(grip_cmd, 4),
            timestamp_s=t_s,
        )
        steps.append(EpisodeStep(observation=obs, action=act, step_index=i))

    meta = EpisodeMetadata(
        episode_id=eid,
        task_variant=task_variant,
        game=game,
        staging_zone=zone,
        robot_id=robot_id,
        fps=fps,
        n_steps=len(steps),
        captured_at=now_iso(),
        synthetic=True,
        notes=(
            f"Generated from arm_control.demo.full_pick_place_sequence(zone={zone!r}). "
            "All waypoints validated against ArmSafetyGate."
        ),
    )
    return Episode(metadata=meta, steps=steps)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


@dataclass
class SynriaEpisodeDataset:
    """Collection of Synria demonstration episodes.

    Attributes:
        episodes: List of :class:`~lerobot.schema.Episode` objects.
    """

    episodes: list[Episode]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Episode:
        return self.episodes[idx]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.episodes)

    @property
    def total_steps(self) -> int:
        """Total number of (obs, action) steps across all episodes."""
        return sum(len(ep) for ep in self.episodes)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_episodes": len(self),
            "total_steps": self.total_steps,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    def write_json(self, path: Path) -> Path:
        """Write all episodes to a single JSON file and return the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def write_dataset_summary(
    dataset: SynriaEpisodeDataset,
    output: Path,
) -> Path:
    """Write a compact dataset summary JSON (no per-step data) and return path."""
    episodes_meta = [ep.metadata.as_dict() for ep in dataset.episodes]
    zones = sorted({ep.metadata.staging_zone for ep in dataset.episodes})
    games = sorted({ep.metadata.game for ep in dataset.episodes})
    fps_values = sorted({ep.metadata.fps for ep in dataset.episodes})

    summary: dict[str, object] = {
        "n_episodes": len(dataset),
        "total_steps": dataset.total_steps,
        "staging_zones": zones,
        "games": games,
        "fps_values": fps_values,
        "episodes_meta": episodes_meta,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return output
