"""The staged executor on a fake arm: answers, plan, phases, infeasibility and serving."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from synria_lerobot.arm_kinematics import KinematicChain
from synria_lerobot.physical_contract import ActionSource, PhysicalDatasetContract
from synria_lerobot.policy_codec import decode_policy_response, encode_policy_request
from synria_lerobot.quality_gates import load_limits
from synria_lerobot.sim_policy_runner import LIFT_SCORE, PLACE_SCORE, ObjectPose
from synria_lerobot.staged_executor import (
    CUP_SIDE_PITCHED,
    PHASES,
    GraspStrategy,
    StagedExecutor,
    StagedPolicyService,
)
from synria_lerobot.task_registry import load_task_registry

REPOSITORY = Path(__file__).resolve().parents[1]
CHAIN = KinematicChain.load(REPOSITORY / "config" / "synria_kinematics.json")
LIMITS = load_limits(REPOSITORY / "config" / "synria_limits.yaml")
START = (-0.0088, 1.5918, -0.582, 0.0341, -0.7888, -0.0499, 0.0143)
MAX_STEP = 0.16


class FakeArm:
    """Tracks each command with a first-order lag; the gripper stops on a 41 mm cup."""

    def __init__(self, state: tuple[float, ...], cup_contact_m: float = 0.005) -> None:
        self.state = list(state)
        self.cup_contact_m = cup_contact_m
        self.holding = False

    def apply(self, action: tuple[float, ...], at_cup: bool) -> tuple[float, ...]:
        for index in range(6):
            self.state[index] += 0.7 * (action[index] - self.state[index])
        gripper = max(0.0, min(0.025, action[6]))
        if at_cup and gripper > self.cup_contact_m:
            gripper = self.cup_contact_m
            self.holding = True
        elif gripper < self.cup_contact_m:
            self.holding = False
        self.state[6] = gripper
        return tuple(self.state)


def pose(x: float, y: float, z: float = 0.0) -> ObjectPose:
    return ObjectPose((x, y, z), 0.0)


def run(executor: StagedExecutor, arm: FakeArm, cup_xy: tuple[float, float],
        steps: int = 400) -> list[str]:
    """Step until done or infeasible; returns the phase sequence."""
    phases: list[str] = []
    state = tuple(arm.state)
    for _ in range(steps):
        action, phase = executor.step(state)
        if not phases or phases[-1] != phase:
            phases.append(phase)
        centre = CHAIN.position(state[:6], "simulation")
        at_cup = (math.hypot(centre[0] - cup_xy[0], centre[1] - cup_xy[1]) < 0.03
                  and centre[2] < 0.06)
        state = arm.apply(action, at_cup)
        if phase in ("done", "infeasible"):
            break
    return phases


def test_lift_trial_runs_every_phase_and_squeezes_at_the_cup() -> None:
    """Plan, approach, descend, squeeze, lift, done; the squeeze happens at the cup."""
    cup = (-0.32, -0.06)
    executor = StagedExecutor(CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, LIFT_SCORE,
                              lambda: pose(*cup), None, max_step_rad=MAX_STEP,
                              max_gripper_step_m=0.008, margin_rad=math.radians(3))
    arm = FakeArm(START)
    phases = run(executor, arm, cup)
    assert phases == ["approach", "descend", "squeeze", "lift", "done"]
    record = executor.record
    assert record.feasible is True and record.final_phase == "done"
    assert record.what["object"] == "cup" and record.where["object_m"][:2] == list(cup)
    assert record.how["strategy"] == "cup_side_pitched"
    assert set(record.plan) >= {"pregrasp", "grasp", "lift"}
    squeeze = next(entry for entry in record.phases if entry["phase"] == "squeeze")
    centre = squeeze["gripper_centre_m"]
    assert math.hypot(centre[0] - cup[0], centre[1] - cup[1]) < 0.02
    assert abs(centre[2] - 0.03) < 0.015
    assert arm.holding, "the gripper closed on the cup"
    final = CHAIN.position(arm.state[:6], "simulation")
    assert final[2] > 0.03 + 0.06, "the cup was lifted"
    assert record.stalls == []
    assert [entry["phase"] for entry in record.phases][-1] == "done"


def test_place_trial_carries_to_the_mark_releases_and_retreats() -> None:
    """The place score adds carry, lower, release and retreat and ends open above the mark."""
    cup, mark = (-0.33, -0.05), (-0.24, 0.07)
    executor = StagedExecutor(CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, PLACE_SCORE,
                              lambda: pose(*cup), lambda: pose(*mark), max_step_rad=MAX_STEP,
                              max_gripper_step_m=0.008, margin_rad=math.radians(3))
    arm = FakeArm(START)
    phases = run(executor, arm, cup, steps=600)
    assert phases == ["approach", "descend", "squeeze", "lift", "carry", "lower", "release",
                      "retreat", "done"]
    release = next(entry for entry in executor.record.phases if entry["phase"] == "release")
    centre = release["gripper_centre_m"]
    assert math.hypot(centre[0] - mark[0], centre[1] - mark[1]) < 0.02
    assert arm.state[6] < 0.001, "released"
    final = CHAIN.position(arm.state[:6], "simulation")
    assert final[2] > 0.08 and math.hypot(final[0] - mark[0], final[1] - mark[1]) < 0.03


def test_unreachable_cup_is_infeasible_and_the_arm_holds() -> None:
    """A cup beyond the reach band makes the plan infeasible; the action is the state."""
    executor = StagedExecutor(CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, LIFT_SCORE,
                              lambda: pose(-0.9, 0.0), None, max_step_rad=MAX_STEP,
                              max_gripper_step_m=0.008, margin_rad=math.radians(3))
    action, phase = executor.step(START)
    assert phase == "infeasible" and action[:6] == START[:6]
    assert executor.record.feasible is False
    assert any(entry.get("feasible") is False for entry in executor.record.plan.values())
    action, phase = executor.step(START)
    assert phase == "infeasible" and action[:6] == START[:6]
    over_base = StagedExecutor(CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, LIFT_SCORE,
                               lambda: pose(0.01, 0.0), None, max_step_rad=MAX_STEP,
                               max_gripper_step_m=0.008, margin_rad=math.radians(3))
    assert over_base.step(START)[1] == "infeasible"
    assert "error" in over_base.record.how


def test_strategy_and_executor_refuse_nonsense() -> None:
    """Bad geometry, a place score without a mark and bad step limits are refused."""
    with pytest.raises(ValueError, match="pitch"):
        GraspStrategy("x", "cup", 0.03, math.radians(95), 0.08, 0.02, 0.0, 0.08, 6, 4,
                      (0.0,) * 6, "test")
    with pytest.raises(ValueError, match="target"):
        StagedExecutor(CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, PLACE_SCORE,
                       lambda: pose(-0.3, 0.0), None, max_step_rad=MAX_STEP,
                       max_gripper_step_m=0.008, margin_rad=0.05)
    with pytest.raises(ValueError, match="step limits"):
        StagedExecutor(CHAIN, "simulation", LIMITS, "50mm", CUP_SIDE_PITCHED, LIFT_SCORE,
                       lambda: pose(-0.3, 0.0), None, max_step_rad=0.0,
                       max_gripper_step_m=0.008, margin_rad=0.05)
    assert PHASES[0] == "plan" and PHASES[-2:] == ("done", "infeasible")


def contract(task_id: str) -> PhysicalDatasetContract:
    tasks = load_task_registry(REPOSITORY / "config" / "synria_tasks.json")
    return PhysicalDatasetContract("50mm", ActionSource.LEADER, False, task_id=task_id,
                                   task_definition=tasks[task_id], recording_purpose="simulation")


def request(state: tuple[float, ...], reset: bool, request_id: int = 1) -> bytes:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return encode_policy_request({
        "observation.state": list(state), "observation.images.wrist": image,
        "observation.images.front": image, "request_id": request_id, "reset": reset,
        "observation_timestamp_s": 1.5, "task": "x",
    })


def test_service_metadata_round_trips_and_logs_stages(tmp_path: Path) -> None:
    """The served contract reloads, resets start trials, and the stage log fills."""
    log = tmp_path / "stages.jsonl"
    service = StagedPolicyService(
        contract("pick_cup"), CHAIN, LIMITS, lambda: pose(-0.32, -0.06), None,
        command_period_s=0.2, max_joint_speed_rad_s=1.0, max_gripper_speed_m_s=0.05,
        stage_log=log,
    )
    metadata = service.metadata()
    assert metadata["lane"] == "simulation" and metadata["policy_id"] == "staged_expert_v1"
    assert metadata["physical_eligible"] is False and metadata["checkpoint_content_sha256"] is None
    reloaded = PhysicalDatasetContract.from_dict(metadata["physical_contract"])
    assert reloaded.task_id == "pick_cup"
    assert metadata["executor"]["kind"] == "staged_expert"
    assert metadata["executor"]["max_step_rad"] == pytest.approx(0.16)
    arm = FakeArm(START)
    state = tuple(arm.state)
    for index in range(40):
        response = decode_policy_response(service.request(request(state, index == 0, index)))
        assert response["request_id"] == index and len(response["action"]) == 7
        assert response["stage"]["trial"] == 1
        state = arm.apply(tuple(response["action"]), False)
    decode_policy_response(service.request(request(state, True, 99)))
    assert service.trial == 2
    service.close()
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    trials = [line for line in lines if line["kind"] == "trial"]
    assert [line["trial"] for line in trials] == [1, 2]
    assert trials[0]["feasible"] is True and trials[0]["final_phase"] in PHASES
    assert any(line["kind"] == "phase" and line["phase"] == "approach" for line in lines)
    with pytest.raises(ValueError, match="simulation lane"):
        StagedPolicyService(contract("pick_cup"), CHAIN, LIMITS, lambda: pose(-0.3, 0.0), None,
                            lane="physical", command_period_s=0.2, max_joint_speed_rad_s=1.0,
                            max_gripper_speed_m_s=0.05)
    with pytest.raises(ValueError, match="no scene score"):
        StagedPolicyService(contract("die_into_cup"), CHAIN, LIMITS, lambda: pose(-0.3, 0.0),
                            None, command_period_s=0.2, max_joint_speed_rad_s=1.0,
                            max_gripper_speed_m_s=0.05)
