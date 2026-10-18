"""The simulation trial runner, exercised with fake sources, a fake policy and a fake clock."""

from __future__ import annotations

import json
import math
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
    PLACE_SCORE,
    REAL_ARM_TOPICS,
    LiftScorer,
    ObjectPose,
    PlaceScorer,
    SimulationTrialRunner,
    TrialConfig,
    approach_targets,
    recovery_target,
    require_sim_only_topic,
    write_report,
)

LIMITS = Path("config/synria_limits.yaml")
START = (0.0, 0.3, -0.2, 0.0, -0.5, 0.0, 0.02)


def contract(purpose: str = "simulation", task_id: str = "pick_cup") -> PhysicalDatasetContract:
    """A contract for one registry task in the given purpose."""
    return PhysicalDatasetContract(
        "50mm", ActionSource.LEADER, False, task_id=task_id,
        task_definition=synthetic_task(1, 120, task_id), recording_purpose=purpose,
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

    def read_gripper(self) -> ObjectPose:
        """The gripper sits 10 cm from the cup until it closes, then on it."""
        offset = 0.0 if self.pose[6] < 0.005 else 0.1
        return ObjectPose((0.3 + offset, 0.0, self.cup_z), self.clock())


MARK = (0.2, -0.1, 0.001)


class PlacingScene(Scene):
    """A fake scene with a mark: a closed gripper carries the cup over it; opening sets it down."""

    def __init__(self, clock: FakeClock, *, releases: bool = True) -> None:
        super().__init__(clock)
        self.releases = releases
        self.cup_xy = (0.3, 0.0)

    def offer(self, action: tuple[float, ...]) -> None:
        self.targets.append(tuple(action))
        closed = action[6] < 0.005
        carried = self.cup_xy == MARK[:2]
        self.pose = tuple(action)
        if closed:
            self.cup_xy = MARK[:2]
            self.cup_z = self.cup_rest + 0.05
        elif carried and self.releases:
            self.cup_z = self.cup_rest
        else:
            self.cup_z = self.cup_rest + (0.05 if carried else 0.0)

    def randomize(self) -> None:
        super().randomize()
        self.cup_xy = (0.3, 0.0)

    def read_cup(self) -> ObjectPose:
        return ObjectPose((*self.cup_xy, self.cup_z), self.clock())

    def read_gripper(self) -> ObjectPose:
        """On the cup while closed; 10 cm away once open."""
        offset = 0.0 if self.pose[6] < 0.005 else 0.1
        return ObjectPose((self.cup_xy[0] + offset, self.cup_xy[1], self.cup_z), self.clock())

    def read_mark(self) -> ObjectPose:
        return ObjectPose(MARK, self.clock())


class SequencePolicy:
    """A fake transport answering each action in turn, then the last one forever."""

    def __init__(self, actions: list[tuple[float, ...]]) -> None:
        self.actions = list(actions)

    def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        action = self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        return {"request_id": payload["request_id"],
                "observation_timestamp_s": payload["observation_timestamp_s"],
                "action": action, "inference_s": 0.0}


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
        gripper_source=Source(scene.read_gripper),
    )


def place_runner(scene: PlacingScene, clock: FakeClock, actions: list[tuple[float, ...]], *,
                 trials: int = 1, mark: bool = True, gripper: bool = True,
                 score: str = PLACE_SCORE, task_id: str = "cup_pick_and_place",
                 ) -> SimulationTrialRunner:
    """A place-scored runner over the placing scene with a scripted policy."""
    client = PolicyClient(SynriaEmbodiment(contract(task_id=task_id)), SequencePolicy(actions),
                          load_limits(LIMITS), safety(0.2), clock=clock)
    config = TrialConfig(trials=trials, command_period_s=0.2, start_pose=START, max_trial_s=6.0,
                         lift_m=0.03, hold_s=1.0, settle_s=0.4, approach_s=1.0, score=score)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    return SimulationTrialRunner(
        client, scene, Source(scene.read_state), Source(scene.read_image),
        Source(scene.read_image), Source(scene.read_cup), scene.randomize, config,
        clock=clock, sleep=sleep, log=lambda text: None,
        gripper_source=Source(scene.read_gripper) if gripper else None,
        target_source=Source(scene.read_mark) if mark else None,
    )


def test_place_scorer_needs_a_lift_then_a_resting_cup_in_the_mark_with_the_gripper_clear() -> None:
    """Nudged into the mark fails; lifted, set down inside, gripper clear, held a second passes."""
    scorer = PlaceScorer(0.0, (0.2, -0.1), 0.03, 0.03, 0.01, 0.06, 1.0)
    on_mark = (0.21, -0.1, 0.0)
    far = (0.5, -0.1, 0.0)
    assert not scorer.feed(on_mark, far, 0.0) and not scorer.feed(on_mark, far, 1.5)
    assert not scorer.lifted and scorer.min_target_distance_m == pytest.approx(0.01)
    assert not scorer.feed((0.3, 0.0, 0.05), (0.3, 0.0, 0.05), 2.0) and scorer.lifted
    assert not scorer.feed((0.2, -0.1, 0.04), (0.2, -0.1, 0.04), 2.2)   # held above the mark
    assert not scorer.feed(on_mark, (0.21, -0.1, 0.0), 2.4)            # gripper still on it
    assert not scorer.feed(on_mark, far, 2.6) and not scorer.feed(on_mark, far, 3.5)
    assert not scorer.feed((0.26, -0.1, 0.0), far, 3.6)                # knocked out of the mark
    assert not scorer.feed(on_mark, far, 3.8) and not scorer.feed(on_mark, far, 4.7)
    assert scorer.feed(on_mark, far, 4.8) and scorer.success_at_s == 4.8
    assert scorer.final_target_distance_m == pytest.approx(0.01)
    assert scorer.max_lift_m == pytest.approx(0.05)
    with pytest.raises(ValueError, match="finite"):
        PlaceScorer(0.0, (0.2, float("nan")), 0.03, 0.03, 0.01, 0.06, 1.0)


def test_runner_scores_a_placement_and_a_cup_never_released(tmp_path: Path) -> None:
    """Close, carry, open over the mark: success with distances recorded; never opening fails."""
    clock = FakeClock()
    scene = PlacingScene(clock)
    closing, opening = (*START[:6], 0.0), (*START[:6], 0.02)
    report = place_runner(scene, clock, [closing] * 4 + [opening]).run(
        {"lane": "simulation", "policy_id": "p"})
    assert report["score"] == "place" and "mark" in report["score_rule"]
    assert report["successes"] == 1
    [trial] = report["trials"]
    assert trial["success"] and trial["target_xy_m"] == pytest.approx(MARK[:2])
    assert trial["cup_start_xy_m"] == pytest.approx((0.3, 0.0))
    assert trial["max_lift_m"] == pytest.approx(0.05)
    assert trial["final_cup_target_distance_m"] == pytest.approx(0.0)
    assert trial["min_cup_target_distance_m"] == pytest.approx(0.0)
    assert trial["final_gripper_cup_distance_m"] == pytest.approx(0.1)
    write_report(report, tmp_path / "place.json")
    assert json.loads((tmp_path / "place.json").read_text())["score"] == "place"

    clock = FakeClock()
    held = PlacingScene(clock)
    report = place_runner(held, clock, [closing]).run({"lane": "simulation"})
    [trial] = report["trials"]
    assert not trial["success"] and trial["min_cup_target_distance_m"] == pytest.approx(0.0)
    assert trial["final_gripper_cup_distance_m"] == pytest.approx(0.0)

    lift = runner(Scene(clock), clock, closing, trials=1).run({"lane": "simulation"})
    assert lift["score"] == "lift" and lift["trials"][0]["target_xy_m"] is None
    assert lift["trials"][0]["cup_start_xy_m"] == pytest.approx((0.3, 0.0))


def test_runner_refuses_a_score_the_task_is_not_judged_by_or_a_missing_mark() -> None:
    """The place task refuses the lift score and runs only with the mark and gripper poses."""
    clock = FakeClock()
    scene = PlacingScene(clock)
    with pytest.raises(ValueError, match="scored by place, not lift"):
        place_runner(scene, clock, [START], score="lift")
    with pytest.raises(ValueError, match="scored by lift, not place"):
        place_runner(scene, clock, [START], task_id="pick_cup")
    with pytest.raises(ValueError, match="mark pose"):
        place_runner(scene, clock, [START], mark=False)
    with pytest.raises(ValueError, match="gripper pose"):
        place_runner(scene, clock, [START], gripper=False)
    with pytest.raises(ValueError, match="no scene score"):
        place_runner(scene, clock, [START], task_id="cup_return", score="lift")
    with pytest.raises(ValueError, match="score must be"):
        TrialConfig(trials=1, command_period_s=0.2, start_pose=START, score="drop")


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
    assert first["recoveries"] == 0 and "recovery_rule" in report
    assert first["min_gripper_cup_distance_m"] == pytest.approx(0.0)
    assert first["final_gripper_cup_distance_m"] == pytest.approx(0.0)
    assert len(first["final_state"]) == 7 and first["final_state"][6] == 0.0

    clock = FakeClock()
    idle = Scene(clock, liftable=False)
    report = runner(idle, clock, START, trials=1).run({"lane": "simulation"})
    [trial] = report["trials"]
    assert not trial["success"] and trial["time_to_success_s"] is None
    assert trial["duration_s"] >= 6.0 and trial["steps"] == pytest.approx(30, abs=1)
    assert trial["min_gripper_cup_distance_m"] == pytest.approx(0.1)
    assert trial["final_state"][:6] == pytest.approx(START[:6])
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


def test_recovery_pulls_a_clipped_joint_back_inside_and_the_policy_resumes() -> None:
    """A drive that overshoots past a bound would hold forever; the sim runner nudges it back."""
    bounds = ((-1.0, 1.0),) * 6 + ((0.0, 0.025),)
    nudged = recovery_target((0.0, 1.02, -1.5, 0.3, 0.0, 0.0, 0.0262), bounds,
                             math.radians(0.5), 0.001)
    assert nudged[0] == 0.0 and nudged[3] == 0.3
    assert nudged[1] == pytest.approx(1.0 - math.radians(0.5))
    assert nudged[2] == pytest.approx(-1.0 + math.radians(0.5))
    assert nudged[6] == pytest.approx(0.024)
    with pytest.raises(ValueError, match="wider than the bound"):
        recovery_target((0.0,) * 7, ((0.0, 0.001),) * 7, 0.01, 0.0005)

    clock = FakeClock()

    class Overshoot(Scene):
        """Commanding Joint3 near its bound lands the physics 0.3 degrees past it."""

        def offer(self, action: tuple[float, ...]) -> None:
            super().offer(action)
            if action[2] < -0.60:
                self.pose = (*action[:2], -0.6108 - math.radians(0.3), *action[3:])

    scene = Overshoot(clock)
    low_joint3 = (*START[:2], -0.605, *START[3:6], 0.0)
    report = runner(scene, clock, low_joint3, trials=1).run({"lane": "simulation"})
    [trial] = report["trials"]
    assert trial["recoveries"] >= 1 and trial["holds"] == trial["recoveries"]
    assert trial["hold_reasons"] == {"current state outside limits": trial["holds"]}
    assert trial["success"], "after each recovery the policy acts again and closes the gripper"
    assert max(t[2] for t in scene.targets) >= -0.6108 + math.radians(0.5) - 1e-9
