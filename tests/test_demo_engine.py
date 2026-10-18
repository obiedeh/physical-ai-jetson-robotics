"""The demonstration engine on a fake scene: executor-driven takes, labels and exclusions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_physical_recorder import FakeWriter
from test_task_registry import synthetic_task

from synria_lerobot.arm_kinematics import KinematicChain
from synria_lerobot.demo_engine import (
    DemonstrationConfig,
    DemonstrationSession,
    ExecutorActionSource,
    write_summary,
)
from synria_lerobot.exclusions import read_exclusions
from synria_lerobot.physical_contract import (
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.quality_gates import load_limits
from synria_lerobot.recorder import GoalSample, PhysicalEpisodeRecorder, PhysicalRecorderConfig
from synria_lerobot.sim_policy_runner import LIFT_SCORE, ObjectPose
from synria_lerobot.staged_executor import CUP_SIDE_PITCHED, StagedExecutor
from synria_lerobot.task_registry import SIMULATION

REPOSITORY = Path(__file__).resolve().parents[1]
CHAIN = KinematicChain.load(REPOSITORY / "config" / "synria_kinematics.json")
LIMITS = load_limits(REPOSITORY / "config" / "synria_limits.yaml")
START = (-0.0088, 1.5918, -0.582, 0.0341, -0.7888, -0.0499, 0.0143)


class FakeClock:
    """Advances only when slept."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


class FakeScene:
    """An arm that tracks its targets and a cup that follows a closed gripper."""

    def __init__(self, cup_xy: tuple[float, float], clock: FakeClock) -> None:
        self.state = list(START)
        self.cup = [cup_xy[0], cup_xy[1], 0.0]
        self.clock = clock
        self.holding = False
        self.targets = 0
        self.randomized: list[tuple[int | None, float]] = []

    def offer(self, action: tuple[float, ...]) -> None:
        self.targets += 1
        for index in range(6):
            self.state[index] += 0.7 * (action[index] - self.state[index])
        gripper = max(0.0, min(0.025, action[6]))
        centre = CHAIN.position(self.state[:6], "simulation")
        near = (math.hypot(centre[0] - self.cup[0], centre[1] - self.cup[1]) < 0.03
                and centre[2] < 0.06)
        if near and gripper > 0.005:
            gripper, self.holding = 0.005, True
        elif gripper < 0.005:
            self.holding = False
        self.state[6] = gripper
        if self.holding:
            self.cup = [centre[0], centre[1], max(0.0, centre[2] - 0.03)]

    def hold(self) -> None:
        pass

    def read(self) -> PhysicalState:
        now = self.clock()
        return PhysicalState(tuple(self.state[:6]), self.state[6], now, now + 1000.0,
                             ros_arrival_stamp_s=now + 1000.0)

    def cup_pose(self) -> ObjectPose:
        return ObjectPose(tuple(self.cup), self.clock())

    def randomize(self, seed: int | None, scale: float) -> None:
        self.randomized.append((seed, scale))


class FakeFrames:
    def __init__(self, name: str, clock: FakeClock) -> None:
        self.name, self.clock = name, clock

    def read(self) -> ImageFrame:
        return ImageFrame(np.zeros((8, 8, 3), np.uint8), self.clock(), (8, 8), self.name)

    def close(self) -> None:
        pass


class FakeGoal:
    def __init__(self, scene: FakeScene) -> None:
        self.scene = scene

    def read(self) -> GoalSample:
        return GoalSample(tuple(self.scene.cup), self.scene.clock())

    def close(self) -> None:
        pass


def session(tmp_path: Path, cup_xy: tuple[float, float], takes: int = 1,
            max_take_s: float = 20.0) -> tuple[DemonstrationSession, FakeScene, FakeWriter]:
    clock = FakeClock()
    scene = FakeScene(cup_xy, clock)
    contract = PhysicalDatasetContract(
        "50mm", ActionSource.LEADER, False, task_id="pick_cup",
        task_definition=synthetic_task(1, 60, "pick_cup"), recording_purpose=SIMULATION,
        goal_channels=("object",),
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    config = PhysicalRecorderConfig(dataset_path=dataset, repo_id="local/demo",
                                    contract=contract, fps=15.0)
    writer = FakeWriter(tmp_path)
    fps = 15.0
    actions = ExecutorActionSource(lambda: StagedExecutor(
        CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, LIFT_SCORE, scene.cup_pose,
        None, max_step_rad=0.8 / fps, max_gripper_step_m=0.04 / fps,
        margin_rad=math.radians(3),
    ), scene)
    recorder = PhysicalEpisodeRecorder(
        config=config, state_source=scene, action_source=actions,
        wrist_source=FakeFrames("wrist", clock), front_source=FakeFrames("front", clock),
        writer=writer, clock=clock, goal_source=FakeGoal(scene),
    )
    demo = DemonstrationSession(
        recorder, actions, scene, scene, scene.cup_pose, None, None, scene.randomize,
        DemonstrationConfig(takes=takes, fps=fps, start_pose=START, score=LIFT_SCORE,
                            placement_seed=3026, max_take_s=max_take_s, lift_m=0.03,
                            hold_s=0.5),
        dataset_root=dataset, clock=clock, sleep=clock.sleep, log=lambda line: None,
    )
    return demo, scene, writer


def test_a_reachable_cup_is_lifted_recorded_and_labelled_success(tmp_path: Path) -> None:
    """The executor's commands are the frames' actions; the goal is the cup; success is kept."""
    demo, scene, writer = session(tmp_path, (-0.32, -0.06))
    summary = demo.run()
    assert summary["successes"] == 1 and summary["failures_excluded"] == 0
    take = demo.takes[0]
    assert take.success and take.final_phase == "done" and take.feasible is True
    assert take.max_lift_m >= 0.03 and scene.randomized == [(3026, 1.0)]
    episode = writer.episodes[0]
    assert episode.operator_label.value == "success"
    assert len(episode.frames) == take.steps
    first = episode.frames[0]
    assert first.goal == (pytest.approx(-0.32), pytest.approx(-0.06), 0.0)
    assert first.action_ros_header_stamp_s == first.state.ros_header_stamp_s
    assert first.action_monotonic_timestamp_s == first.state.monotonic_timestamp_s
    assert all(len(frame.action) == 7 for frame in episode.frames)
    closed = [frame for frame in episode.frames if frame.action[6] > 0.02]
    assert closed, "the squeeze is on record"
    assert read_exclusions(tmp_path / "dataset").excluded_ids == ()
    write_summary(summary, tmp_path / "report" / "demo.json")
    assert json.loads((tmp_path / "report" / "demo.json").read_text())["successes"] == 1
    with pytest.raises(FileExistsError):
        write_summary(summary, tmp_path / "report" / "demo.json")


def test_an_unreachable_cup_is_a_labelled_failure_and_excluded(tmp_path: Path) -> None:
    """An infeasible plan ends the take after the grace, labelled failure and excluded."""
    demo, scene, writer = session(tmp_path, (-0.9, 0.0))
    summary = demo.run()
    assert summary["successes"] == 0 and summary["failures_excluded"] == 1
    take = demo.takes[0]
    assert not take.success and take.final_phase == "infeasible" and take.feasible is False
    assert writer.episodes[0].operator_label.value == "failure"
    assert read_exclusions(tmp_path / "dataset").excluded_ids == (0,)
    assert take.duration_s < 5.0, "the grace ended the take, not the limit"


def test_action_source_needs_a_reset_and_config_refuses_nonsense(tmp_path: Path) -> None:
    """No executor before reset; bad counts and timing are refused."""
    clock = FakeClock()
    scene = FakeScene((-0.3, 0.0), clock)
    actions = ExecutorActionSource(lambda: None, scene)  # type: ignore[arg-type,return-value]
    assert actions.phase == "plan"
    with pytest.raises(RuntimeError, match="reset"):
        actions.read(scene.read())
    with pytest.raises(ValueError, match="takes"):
        DemonstrationConfig(takes=0, fps=15, start_pose=START, score=LIFT_SCORE,
                            placement_seed=1)
    with pytest.raises(ValueError, match="score"):
        DemonstrationConfig(takes=1, fps=15, start_pose=START, score="x", placement_seed=1)
    with pytest.raises(ValueError, match="positive"):
        DemonstrationConfig(takes=1, fps=0, start_pose=START, score=LIFT_SCORE,
                            placement_seed=1)
