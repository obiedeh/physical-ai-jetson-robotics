"""The simulation trial runner, exercised with fake sources, a fake policy and a fake clock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_physical_recorder import FakeClock
from test_task_registry import synthetic_task

from synria_lerobot.embodiment import SynriaEmbodiment
from synria_lerobot.physical_contract import (
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.policy_client import FakePolicy, PolicyClient, PolicySafetyConfig
from synria_lerobot.quality_gates import load_limits
from synria_lerobot.sim_policy_runner import (
    REAL_ARM_TOPICS,
    LiftScorer,
    ObjectPose,
    SimulationTrialRunner,
    TrialConfig,
    approach_targets,
    require_sim_only_topic,
    write_report,
)

LIMITS = Path("config/synria_limits.yaml")
START = (0.0, 0.3, -0.2, 0.0, -0.5, 0.0, 0.02)


def contract(purpose: str = "simulation") -> PhysicalDatasetContract:
    """A pick-cup style contract in the given purpose."""
    return PhysicalDatasetContract(
        "50mm", ActionSource.LEADER, False, task_id="pick_cup",
        task_definition=synthetic_task(1, 120, "pick_cup"), recording_purpose=purpose,
    )


def safety(period: float = 0.2) -> PolicySafetyConfig:
    """The committed simulation safety file at a given command period."""
    return PolicySafetyConfig.load(
        Path("config/synria_sim_policy.json"), command_period_s=period,
        response_timeout_s=0.5, lane="simulation",
    )


class Scene:
    """A fake scene: arm state follows targets, the cup rises once the gripper closes on it."""

    def __init__(self, clock: FakeClock, *, liftable: bool = True) -> None:
        self.clock = clock
        self.liftable = liftable
        self.pose = (0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.02)
        self.targets: list[tuple[float, ...]] = []
        self.holds = 0
        self.randomized = 0
        self.cup_rest = 0.0
        self.cup_z = 0.0

    def offer(self, action: tuple[float, ...]) -> None:
        self.targets.append(tuple(action))
        self.pose = tuple(action)
        if self.liftable and action[6] < 0.005:
            self.cup_z = self.cup_rest + 0.05
        else:
            self.cup_z = self.cup_rest

    def hold(self) -> None:
        self.holds += 1

    def randomize(self) -> None:
        self.randomized += 1
        self.cup_rest = 0.001 * self.randomized
        self.cup_z = self.cup_rest

    def read_state(self) -> PhysicalState:
        now = self.clock()
        return PhysicalState(self.pose[:6], self.pose[6], now, now, ros_arrival_stamp_s=now)

    def read_image(self) -> ImageFrame:
        return ImageFrame(np.zeros((4, 4, 3), dtype=np.uint8), self.clock(), (4, 4), "sim")

    def read_cup(self) -> ObjectPose:
        return ObjectPose((0.3, 0.0, self.cup_z), self.clock())


class Source:
    """Adapt one scene reader to the source protocol."""

    def __init__(self, read: Any) -> None:
        self.read = read

    def close(self) -> None:
        pass


def runner(scene: Scene, clock: FakeClock, action: tuple[float, ...], *, trials: int = 2,
           purpose: str = "simulation", period: float = 0.2) -> SimulationTrialRunner:
    """A runner over the fake scene with a policy that always answers ``action``."""
    client = PolicyClient(SynriaEmbodiment(contract(purpose)), FakePolicy(action),
                          load_limits(LIMITS), safety(period), clock=clock)
    config = TrialConfig(trials=trials, command_period_s=period, start_pose=START,
                         max_trial_s=6.0, lift_m=0.03, hold_s=1.0, settle_s=0.4,
                         approach_s=1.0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    return SimulationTrialRunner(
        client, scene, Source(scene.read_state), Source(scene.read_image),
        Source(scene.read_image), Source(scene.read_cup), scene.randomize, config,
        clock=clock, sleep=sleep, log=lambda text: None,
    )


def test_lift_scorer_needs_an_unbroken_hold() -> None:
    """A dip below the lift height restarts the hold; success stamps the time it was met."""
    scorer = LiftScorer(0.01, 0.03, 1.0)
    assert not scorer.feed(0.05, 0.0) and not scorer.feed(0.05, 0.9)
    assert not scorer.feed(0.02, 1.0) and not scorer.feed(0.05, 1.1)
    assert not scorer.feed(0.05, 2.0) and scorer.feed(0.05, 2.1)
    assert scorer.success_at_s == 2.1 and scorer.max_lift_m == pytest.approx(0.04)
    assert scorer.feed(0.0, 5.0), "a success is never withdrawn"


def test_approach_walks_within_the_step_bound_and_refuses_a_hopeless_budget() -> None:
    """Each target moves at most max_delta per value and the last one is the goal."""
    current = (0.0,) * 7
    goal = (0.5, -0.5, 0.1, 0.0, 0.0, 0.0, 0.02)
    delta = (0.2,) * 6 + (0.01,)
    targets = approach_targets(current, goal, delta, 5)
    assert targets[-1] == pytest.approx(goal)
    previous = current
    for target in targets:
        assert all(abs(t - p) <= d + 1e-12
                   for t, p, d in zip(target, previous, delta, strict=True))
        previous = target
    assert len(targets) == 5
    with pytest.raises(ValueError, match="too far"):
        approach_targets(current, (5.0,) * 6 + (0.02,), delta, 2)
    with pytest.raises(ValueError, match="same length"):
        approach_targets(current, goal[:6], delta, 5)


def test_runner_scores_a_lift_and_a_timeout_and_reports_both(tmp_path: Path) -> None:
    """A policy that closes the gripper lifts the cup; one that never closes times out."""
    clock = FakeClock()
    scene = Scene(clock)
    closing = (*START[:6], 0.0)
    report = runner(scene, clock, closing).run({"lane": "simulation", "policy_id": "p"})
    assert report["kind"] == "simulation_policy_trials" and report["lane"] == "simulation"
    assert report["physical_evidence"] is False and report["successes"] == 2
    assert report["success_rate"] == 1.0 and report["policy"]["policy_id"] == "p"
    assert scene.randomized == 2
    first = report["trials"][0]
    assert first["success"] and first["time_to_success_s"] == pytest.approx(1.0 + 0.2, abs=0.21)
    assert first["approach_steps"] >= 1 and first["max_lift_m"] == pytest.approx(0.05)
    assert first["holds"] == 0 and first["cup_rest_z_m"] == pytest.approx(0.001)

    clock = FakeClock()
    idle = Scene(clock, liftable=False)
    report = runner(idle, clock, START, trials=1).run({"lane": "simulation"})
    [trial] = report["trials"]
    assert not trial["success"] and trial["time_to_success_s"] is None
    assert trial["duration_s"] >= 6.0 and trial["steps"] == pytest.approx(30, abs=1)
    assert report["success_rate"] == 0.0
    path = tmp_path / "trials.json"
    write_report(report, path)
    assert json.loads(path.read_text())["successes"] == 0
    with pytest.raises(FileExistsError):
        write_report(report, path)


def test_policy_holds_are_counted_and_the_sink_holds(tmp_path: Path) -> None:
    """A stale observation makes the client hold; the runner records why and holds the sink."""
    clock = FakeClock()
    scene = Scene(clock, liftable=False)

    class StaleScene(Scene):
        def read_state(self) -> PhysicalState:
            return PhysicalState(self.pose[:6], self.pose[6], -5.0, -5.0, ros_arrival_stamp_s=-5.0)

    stale = StaleScene(clock, liftable=False)
    stale.pose = scene.pose
    report = runner(stale, clock, START, trials=1).run({"lane": "simulation"})
    [trial] = report["trials"]
    assert trial["holds"] == trial["steps"] > 0
    assert list(trial["hold_reasons"]) == ["stale or future observation"]
    assert stale.holds >= trial["holds"]


def test_runner_refuses_physical_contracts_servers_and_topics() -> None:
    """Only a simulation contract, a simulation-lane server and sim-only topics are accepted."""
    clock = FakeClock()
    scene = Scene(clock)
    with pytest.raises(ValueError, match="simulation lane"):
        runner(scene, clock, START, purpose="qualifying")
    with pytest.raises(ValueError, match="not serving the simulation lane"):
        runner(scene, clock, START).run({"lane": "physical"})
    with pytest.raises(ValueError, match="not serving the simulation lane"):
        runner(scene, clock, START).run({})
    for topic in sorted(REAL_ARM_TOPICS):
        with pytest.raises(ValueError, match="belongs to the real arm"):
            require_sim_only_topic(topic)
    assert require_sim_only_topic("/isaac/policy_joint_targets") == "/isaac/policy_joint_targets"
    with pytest.raises(ValueError):
        require_sim_only_topic("policy_joint_targets")


def test_safety_config_files_belong_to_one_lane() -> None:
    """The sim speed file is refused on the physical path and the physical file on the sim path."""
    with pytest.raises(ValueError, match="lane differs"):
        PolicySafetyConfig.load(Path("config/synria_sim_policy.json"), command_period_s=0.4,
                                response_timeout_s=0.3)
    with pytest.raises(ValueError, match="lane differs"):
        PolicySafetyConfig.load(Path("config/synria_policy.json"), command_period_s=0.4,
                                response_timeout_s=0.3, lane="simulation")
    physical = PolicySafetyConfig.load(Path("config/synria_policy.json"), command_period_s=0.4,
                                       response_timeout_s=0.3)
    assert not physical.verified and safety().max_delta[0] == pytest.approx(0.2)


def test_trial_config_rejects_unusable_rules() -> None:
    """Counts, periods and the start pose are validated before any trial."""
    with pytest.raises(ValueError, match="trials"):
        TrialConfig(trials=0, command_period_s=0.2, start_pose=START)
    with pytest.raises(ValueError, match="start_pose"):
        TrialConfig(trials=1, command_period_s=0.2, start_pose=START[:6])
    with pytest.raises(ValueError, match="hold_s"):
        TrialConfig(trials=1, command_period_s=0.2, start_pose=START, hold_s=5.0, max_trial_s=4.0)
    with pytest.raises(ValueError, match="lift_m"):
        TrialConfig(trials=1, command_period_s=0.2, start_pose=START, lift_m=0.0)
