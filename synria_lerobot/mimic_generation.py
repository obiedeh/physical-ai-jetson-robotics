"""Generate new cup-pick demonstrations from recorded teleoperated takes (MimicGen method).

The method follows MimicGen (Mandlekar et al., 2023, the algorithm behind Isaac Lab
Mimic): a recorded human take is expressed relative to the object it handles, moved to
the object's new pose, converted back to joint targets with inverse kinematics and
replayed in the scene; only replays the scene scores as successful are kept. It runs
here, in the Isaac scene that records and scores every other simulation take, rather
than in Isaac Lab Mimic, because that package needs an Isaac Lab environment and the
scored scene is not one (its arm model, cameras and placements differ from the Isaac Lab
task in ``isaac/isaaclab_tasks/synria_pickplace``).

Which part is moved. As in MimicGen, only the object-relative segment is moved: the
final approach from about 8 cm out, the grasp and the lift. The arm reaches the moved
segment's first frame by a joint-space walk from wherever the reset left it, and holds
the segment's last frame (cup lifted) afterwards. Frames far from the cup (the operator
moving away from and back to the rest pose) are not moved; moving them pushed the arm
out of reach on most placements in the offline check of 2026-11-23.

How a take is moved. The teleoperated takes do not record the cup's pose, so each
source take's cup position is taken as the gripper centre at its grasp: the frame where
the close command rises above the gripper's stalled position at a low height, which is
the fingers meeting the cup. The cup is a cylinder, so its yaw is free; it is chosen so
the move is a rotation about the base's vertical axis by the bearing change followed by
a radial shift, which keeps the human's approach direction relative to the arm. Every
frame's gripper pose (position, approach axis, finger axis) gets that transform and is
solved back to six joints, seeded from the previous frame and pulled toward the source
take's own joints so the arm keeps the human's elbow branch. The gripper command is the
source take's own. A take whose frames cannot all be solved, or whose solved path jumps,
is marked infeasible and recorded as a failure.

What survives the move is the point of the exercise: the human's hesitations, off-target
arrivals and corrections are replayed as they were, relative to the cup, unlike a
planner's clean approach. Simulation lane only; nothing here touches the physical arm.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synria_lerobot.arm_kinematics import KinematicChain

# The close command crosses this value on its way to the grasp (metres of finger travel;
# 0 is open, 0.025 fully closed on the 50 mm gripper).
CLOSE_COMMAND_M = 0.01
# Below this gripper-centre height (metres above the base plane) a close can be a grasp.
GRASP_HEIGHT_M = 0.06
# After a grasp the fingers stall on the 40 mm cup inside this band while the command
# stays at least STALL_GAP_M above them.
STALL_BAND_M = (0.001, 0.010)
STALL_GAP_M = 0.004
STALL_WINDOW = (5, 15)
# A replayed step larger than this on any joint means the solution changed branch.
MAX_REPLAY_STEP_RAD = 0.35
# A wrist re-split is accepted within these pose errors.
RESPLIT_POSITION_M = 0.003
RESPLIT_AXIS_RAD = math.radians(3.0)
# The object-relative segment starts at the last frame before the grasp where the
# gripper centre is farther than this from the cup's axis (the final approach) ...
SEGMENT_START_DISTANCE_M = 0.08
# ... and ends this many frames after the gripper centre has risen this far above its
# grasp height (the cup is lifted; the scorer then needs it held, which the held last
# frame does).
SEGMENT_LIFT_M = 0.06
SEGMENT_TAIL_FRAMES = 5
# A moved segment may end early once the source has lifted the cup this far (above the
# scorer's 3 cm): a later frame that cannot be solved, such as Joint5 crossing the wrist
# singularity, then ends the replay there with the cup held up instead of failing it.
SAFE_END_LIFT_M = 0.04


@dataclass(frozen=True)
class SourceTake:
    """The object-relative segment of one teleoperated take: final approach, grasp, lift."""

    episode_index: int
    actions: np.ndarray  # (frames, 7): six reported joint targets and the gripper command
    grasp_frame: int  # index into ``actions``
    cup_xy: tuple[float, float]
    source_frames: tuple[int, int]  # the segment's [start, end) in the recorded episode
    lifted_frame: int  # first index where the gripper centre is SAFE_END_LIFT_M above the grasp

    @property
    def radius_m(self) -> float:
        """The cup's distance from the base axis."""
        return math.hypot(*self.cup_xy)

    @property
    def bearing_rad(self) -> float:
        """The cup's bearing about the base axis."""
        return math.atan2(self.cup_xy[1], self.cup_xy[0])


def find_grasp_frame(states: np.ndarray, actions: np.ndarray,
                     heights_m: np.ndarray) -> int | None:
    """The last frame where a close command starts at a low height and the fingers stall.

    A stall (the measured opening held inside ``STALL_BAND_M`` while the command sits at
    least ``STALL_GAP_M`` beyond it, over ``STALL_WINDOW`` frames after the edge) is the
    fingers meeting the cup; a close in free air runs the fingers to the command instead.
    The last such frame is used because operators sometimes closed, reopened and grasped
    again. None when the take holds no grasp.
    """
    found: int | None = None
    first, last = STALL_WINDOW
    for i in range(1, len(actions) - last):
        if not (actions[i - 1, 6] <= CLOSE_COMMAND_M < actions[i, 6]):
            continue
        if heights_m[i] >= GRASP_HEIGHT_M:
            continue
        measured = states[i + first:i + last, 6]
        commanded = actions[i + first:i + last, 6]
        low, high = STALL_BAND_M
        if (np.all(measured > low) and np.all(measured < high)
                and np.all(commanded - measured > STALL_GAP_M)):
            found = i
    return found


def object_segment(positions_m: np.ndarray, grasp: int,
                   cup_xy: tuple[float, float]) -> tuple[int, int] | None:
    """The [start, end) frames of the take that are tied to the cup, or None.

    Start: the frame after the last one before the grasp whose gripper centre is more
    than ``SEGMENT_START_DISTANCE_M`` from the cup's axis, so every hesitation and
    correction of the final approach is inside. End: ``SEGMENT_TAIL_FRAMES`` after the
    gripper centre first rises ``SEGMENT_LIFT_M`` above its grasp height; None when the
    take never lifts that far.
    """
    distance = np.hypot(positions_m[:, 0] - cup_xy[0], positions_m[:, 1] - cup_xy[1])
    far = np.flatnonzero(distance[:grasp] > SEGMENT_START_DISTANCE_M)
    start = int(far[-1]) + 1 if len(far) else 0
    risen = np.flatnonzero(positions_m[grasp:, 2] >= positions_m[grasp, 2] + SEGMENT_LIFT_M)
    if not len(risen):
        return None
    end = min(len(positions_m), grasp + int(risen[0]) + SEGMENT_TAIL_FRAMES)
    return start, end


def load_source_takes(root: Path, chain: KinematicChain, lane: str, *,
                      exclude: Sequence[int] = (),
                      label: str = "success") -> tuple[list[SourceTake], dict[int, str]]:
    """Every usable take of a LeRobot dataset, and why each other episode was skipped.

    A take is used when its operator label is ``label``, it is not in ``exclude`` (the
    trainer's held-out episodes, so generation never sees them), a grasp is found and
    the cup is lifted after it. Only the object-relative segment is kept.
    """
    from pyarrow import parquet

    labels: dict[int, str] = {}
    metadata = root / "physical_episode_metadata.jsonl"
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            labels[int(record["episode_index"])] = str(record.get("operator_label"))
    frames: dict[int, list[dict]] = {}
    for path in sorted((root / "data").rglob("*.parquet")):
        columns = ["observation.state", "action", "episode_index", "frame_index"]
        for row in parquet.read_table(path, columns=columns).to_pylist():
            frames.setdefault(int(row["episode_index"]), []).append(row)
    takes: list[SourceTake] = []
    skipped: dict[int, str] = {}
    excluded = set(int(e) for e in exclude)
    for episode in sorted(frames):
        if episode in excluded:
            skipped[episode] = "held out"
            continue
        if labels.get(episode) != label:
            skipped[episode] = f"label {labels.get(episode)!r}"
            continue
        rows = sorted(frames[episode], key=lambda row: row["frame_index"])
        states = np.array([row["observation.state"][:7] for row in rows], dtype=float)
        actions = np.array([row["action"][:7] for row in rows], dtype=float)
        positions = np.array([chain.position(s[:6], lane) for s in states])
        grasp = find_grasp_frame(states, actions, positions[:, 2])
        if grasp is None:
            skipped[episode] = "no grasp found"
            continue
        cup_xy = (float(positions[grasp, 0]), float(positions[grasp, 1]))
        segment = object_segment(positions, grasp, cup_xy)
        if segment is None:
            skipped[episode] = "no lift after the grasp"
            continue
        start, end = segment
        lifted = np.flatnonzero(
            positions[grasp:end, 2] >= positions[grasp, 2] + SAFE_END_LIFT_M)
        lifted_frame = grasp - start + int(lifted[0]) if len(lifted) else end - start
        takes.append(SourceTake(episode, actions[start:end], grasp - start, cup_xy,
                                (start, end), lifted_frame))
    return takes, skipped


def choose_source(takes: Sequence[SourceTake], cup_xy: tuple[float, float],
                  rng: random.Random, nearest: int = 3) -> SourceTake:
    """A seeded pick among the ``nearest`` takes by cup radius (the bearing is free)."""
    if not takes:
        raise ValueError("no source takes")
    radius = math.hypot(*cup_xy)
    ranked = sorted(takes, key=lambda take: abs(take.radius_m - radius))
    return rng.choice(ranked[:max(1, nearest)])


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose_matches(chain: KinematicChain, joints: np.ndarray, lane: str, target: np.ndarray,
                  approach: np.ndarray, finger: np.ndarray) -> bool:
    """Whether six joints put the gripper centre at ``target`` with both axes aligned."""
    pose = chain.forward(joints, lane)
    if float(np.linalg.norm(pose[:3, 3] - target)) > RESPLIT_POSITION_M:
        return False
    for column, axis in ((2, approach), (1, finger)):
        cosine = float(np.dot(pose[:3, column], axis) / np.linalg.norm(axis))
        if math.acos(max(-1.0, min(1.0, cosine))) > RESPLIT_AXIS_RAD:
            return False
    return True


@dataclass
class TransformResult:
    """A moved take's joint targets, or why it could not be moved."""

    commands: np.ndarray | None
    delta_bearing_rad: float
    radial_shift_m: float
    failed_frame: int | None = None
    reason: str = ""
    ended_early: bool = False


def _stop(commands: np.ndarray, take: SourceTake, delta: float, shift: float, index: int,
          reason: str) -> TransformResult:
    """A failure at ``index``: the frames so far when the cup is already up, else none."""
    if index > take.lifted_frame:
        return TransformResult(commands[:index].copy(), delta, shift, index, reason,
                               ended_early=True)
    return TransformResult(None, delta, shift, index, reason)


def transform_take(take: SourceTake, cup_xy: tuple[float, float], chain: KinematicChain,
                   lane: str, bounds_rad: tuple[tuple[float, float], ...], *,
                   margin_rad: float) -> TransformResult:
    """The source take's joint targets moved to a cup at ``cup_xy``."""
    delta = math.atan2(cup_xy[1], cup_xy[0]) - take.bearing_rad
    delta = math.atan2(math.sin(delta), math.cos(delta))
    shift = math.hypot(*cup_xy) - take.radius_m
    rotation = _rotation_z(delta)
    outward = np.array([math.cos(math.atan2(cup_xy[1], cup_xy[0])),
                        math.sin(math.atan2(cup_xy[1], cup_xy[0])), 0.0])
    commands = np.empty_like(take.actions)
    previous: np.ndarray | None = None
    for index, frame in enumerate(take.actions):
        pose = chain.forward(frame[:6], lane)
        target = rotation @ pose[:3, 3] + shift * outward
        approach = rotation @ pose[:3, 2]
        finger = rotation @ pose[:3, 1]
        posture = frame[:6].copy()
        posture[0] += delta
        # The first frame takes the source's own joints (bearing added) as seed and
        # posture, which picks the human's elbow and wrist branch; every later frame is
        # pulled toward the frame before it, so the solved path stays continuous.
        seed = posture if previous is None else previous
        solution = chain.solve_ik(target, approach, seed, lane, bounds_rad,
                                  finger_axis=finger, posture_rad=seed,
                                  margin_rad=margin_rad)
        if solution is None:
            return _stop(commands, take, delta, shift, index, "inverse kinematics failed")
        joints = np.asarray(solution.joints_rad, dtype=float)
        if previous is not None and float(np.abs(joints - previous).max()) > MAX_REPLAY_STEP_RAD:
            # Near the wrist singularity (Joint5 near zero) Joint4 and Joint6 trade equal
            # and opposite turns for nearly the same pose. Keep Joint4 where it was, give
            # its turn to Joint6, and accept that only if the pose still matches.
            turn = joints[3] - previous[3]
            candidate = joints.copy()
            candidate[3], candidate[5] = previous[3], joints[5] + turn
            if _pose_matches(chain, candidate, lane, target, approach, finger):
                joints = candidate
            if float(np.abs(joints - previous).max()) > MAX_REPLAY_STEP_RAD:
                return _stop(commands, take, delta, shift, index, "solution jumped branch")
        commands[index, :6] = joints
        commands[index, 6] = frame[6]
        previous = joints
    return TransformResult(commands, delta, shift)


@dataclass
class MimicRecord:
    """What one generated take was built from; stored with the take's summary."""

    feasible: bool = False
    source_episode: int | None = None
    source_cup_xy: tuple[float, float] | None = None
    cup_xy: tuple[float, float] | None = None
    delta_bearing_deg: float | None = None
    radial_shift_m: float | None = None
    frames: int = 0
    failed_frame: int | None = None
    reason: str = ""
    ended_early: bool = False
    walk_steps: int = 0
    replayed_frames: int = 0


class MimicReplayExecutor:
    """Executor interface of the staged executor, replaying a moved human take.

    Planned on construction (the scene has already been reset, so the cup's pose is
    final): read the cup, pick a source take, move it. ``step`` then walks from the
    sampled state to the moved take's first frame at the step limits and replays one
    frame per call; after the last frame it holds that frame (phase ``done``).
    """

    def __init__(self, takes: Sequence[SourceTake], cup_pose: Callable[[], object],
                 chain: KinematicChain, lane: str,
                 bounds_rad: tuple[tuple[float, float], ...], rng: random.Random, *,
                 margin_rad: float, max_step_rad: float, max_gripper_step_m: float,
                 nearest: int = 3) -> None:
        """Plan the take; an unsolvable move leaves the executor ``infeasible``."""
        self.max_step_rad, self.max_gripper_step_m = max_step_rad, max_gripper_step_m
        self.record = MimicRecord()
        position = cup_pose().position_m  # type: ignore[attr-defined]
        cup_xy = (float(position[0]), float(position[1]))
        source = choose_source(takes, cup_xy, rng, nearest)
        result = transform_take(source, cup_xy, chain, lane, bounds_rad, margin_rad=margin_rad)
        self.record.source_episode = source.episode_index
        self.record.source_cup_xy = source.cup_xy
        self.record.cup_xy = cup_xy
        self.record.delta_bearing_deg = math.degrees(result.delta_bearing_rad)
        self.record.radial_shift_m = result.radial_shift_m
        self.record.frames = len(source.actions)
        self.record.failed_frame = result.failed_frame
        self.record.reason = result.reason
        self.record.ended_early = result.ended_early
        self.commands = result.commands
        self.record.feasible = result.commands is not None
        self.phase = "walk" if self.record.feasible else "infeasible"
        self._index = 0
        self._last: tuple[float, ...] | None = None

    def step(self, state: tuple[float, ...]) -> tuple[tuple[float, ...], str]:
        """One command for the sampled seven-value state."""
        if self.phase == "infeasible" or self.commands is None:
            return tuple(float(v) for v in state), self.phase
        if self.phase == "walk":
            first = self.commands[0]
            current = np.asarray(state, dtype=float) if self._last is None else np.asarray(
                self._last)
            limits = np.array([self.max_step_rad] * 6 + [self.max_gripper_step_m])
            delta = np.clip(first - current, -limits, limits)
            command = current + delta
            self.record.walk_steps += 1
            self._last = tuple(float(v) for v in command)
            if np.all(np.abs(first - command) <= 1e-9):
                self.phase = "replay"
            return self._last, self.phase
        if self.phase == "replay":
            command = self.commands[self._index]
            self._index += 1
            self.record.replayed_frames = self._index
            if self._index >= len(self.commands):
                self.phase = "done"
            self._last = tuple(float(v) for v in command)
            return self._last, self.phase
        assert self._last is not None
        return self._last, self.phase
