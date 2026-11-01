"""Send real planning requests to move_group for the Synria arm and record what comes back.

The earlier smoke test launched the MoveIt demo and counted a timeout as a
pass. This campaign asks the planner to do its job: with move_group running
(ros2_ws/src/synria_arm_moveit_config/launch/planning.launch.py), it sends N
joint-space goals through the MoveGroup action with ``plan_only`` set, each
goal drawn uniformly inside the URDF position limits of the six arm joints
from a seeded generator and checked for validity (self-collision and limits)
with move_group's state validity service before it is used, so that a goal
the arm cannot occupy is counted as a rejected sample, not as a planning
failure, and records per request the planner's error code,
its reported planning time, and the length and duration of the returned
trajectory. A request counts as a success only when the error code is
SUCCESS and a trajectory with at least two points came back.

Output: one JSON record with the per-request rows, the success count, the
median and p95 planning time over successful requests, the planner id, the
planning time budget, the seed, the joint limits used, and provenance (git
commit, ROS distro, moveit_core version). Collision checking is whatever the
SRDF and URDF provide: self-collision of the arm links; no world objects.
This is planning in an empty simulated scene, not motion on a robot.

Run (after colcon build of synria_arm_description and synria_arm_moveit_config):
    scripts/linux_rtx/run_synria_moveit_planning.sh
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    RobotState,
)
from moveit_msgs.srv import GetStateValidity
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

ROOT = Path(__file__).resolve().parents[2]
ARM_JOINTS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
ERROR_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -4: "CONTROL_FAILED",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -21: "NO_IK_SOLUTION",
}


def joint_limits(urdf_xml: str) -> dict[str, tuple[float, float]]:
    """Position limits of the arm joints from the URDF text."""
    root = ET.fromstring(urdf_xml)
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        lim = joint.find("limit")
        if name in ARM_JOINTS and lim is not None and lim.get("lower") is not None:
            limits[name] = (float(lim.get("lower")), float(lim.get("upper")))
    missing = [j for j in ARM_JOINTS if j not in limits]
    if missing:
        raise RuntimeError(f"no position limits in the URDF for {missing}")
    return limits


def urdf_from_xacro() -> str:
    share = subprocess.run(
        ["ros2", "pkg", "prefix", "--share", "synria_arm_description"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return subprocess.run(
        ["xacro", f"{share}/urdf/synria_6dof_arm.urdf.xacro"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class Campaign(Node):
    def __init__(self, group: str, planner_id: str, planning_time: float, attempts: int) -> None:
        super().__init__("synria_moveit_planning_campaign")
        self.client = ActionClient(self, MoveGroup, "move_action")
        self.validity = self.create_client(GetStateValidity, "check_state_validity")
        self.group, self.planner_id, self.planning_time, self.attempts = (
            group,
            planner_id,
            planning_time,
            attempts,
        )

    def wait(self, timeout_s: float) -> bool:
        return self.client.wait_for_server(
            timeout_sec=timeout_s
        ) and self.validity.wait_for_service(timeout_sec=timeout_s)

    def goal_is_valid(self, goal_positions: dict[str, float]) -> tuple[bool, list[str]]:
        """Ask move_group whether the goal state is collision free and within bounds."""
        req = GetStateValidity.Request()
        req.group_name = self.group
        rs = RobotState()
        js = JointState()
        js.name = list(goal_positions.keys())
        js.position = [float(v) for v in goal_positions.values()]
        rs.joint_state = js
        req.robot_state = rs
        fut = self.validity.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None:
            return False, ["no reply from check_state_validity"]
        contacts = [f"{c.contact_body_1}:{c.contact_body_2}" for c in res.contacts]
        return bool(res.valid), contacts

    def plan(self, goal_positions: dict[str, float]) -> dict:
        req = MotionPlanRequest()
        req.group_name = self.group
        req.planner_id = self.planner_id
        req.allowed_planning_time = self.planning_time
        req.num_planning_attempts = self.attempts
        req.max_velocity_scaling_factor = 1.0
        req.max_acceleration_scaling_factor = 1.0
        c = Constraints()
        for name, pos in goal_positions.items():
            jc = JointConstraint()
            jc.joint_name, jc.position = name, float(pos)
            jc.tolerance_above = jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        t0 = time.perf_counter()
        send = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=self.planning_time + 10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            return {
                "error_code": None,
                "error": "goal rejected or no handle",
                "wall_s": time.perf_counter() - t0,
            }
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=self.planning_time + 30.0)
        res = res_fut.result()
        wall = time.perf_counter() - t0
        if res is None:
            return {"error_code": None, "error": "no result before timeout", "wall_s": wall}
        r = res.result
        traj = r.planned_trajectory.joint_trajectory
        points = len(traj.points)
        duration = (
            (traj.points[-1].time_from_start.sec + traj.points[-1].time_from_start.nanosec * 1e-9)
            if points
            else None
        )
        code = int(r.error_code.val)
        return {
            "error_code": code,
            "error_name": ERROR_NAMES.get(code, str(code)),
            "planning_time_s": float(r.planning_time),
            "trajectory_points": points,
            "trajectory_duration_s": duration,
            "wall_s": wall,
            "success": code == 1 and points >= 2,
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--group", default="synria_arm")
    ap.add_argument("--planner-id", default="RRTConnectkConfigDefault")
    ap.add_argument("--planning-time", type=float, default=5.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="fraction of each joint range kept away from the limits",
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "reports" / "simulation" / "synria_moveit_planning.json"
    )
    args = ap.parse_args(argv)

    urdf = urdf_from_xacro()
    limits = joint_limits(urdf)
    rng = random.Random(args.seed)

    def sample() -> dict[str, float]:
        g = {}
        for j in ARM_JOINTS:
            lo, hi = limits[j]
            span = hi - lo
            g[j] = rng.uniform(lo + args.margin * span, hi - args.margin * span)
        return g

    rclpy.init()
    node = Campaign(args.group, args.planner_id, args.planning_time, args.attempts)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not node.wait(60.0):
        print("move_action server or check_state_validity not available", file=sys.stderr)
        rclpy.shutdown()
        return 2
    goals = []
    rejected = []
    while len(goals) < args.requests:
        g = sample()
        valid, contacts = node.goal_is_valid(g)
        if valid:
            goals.append(g)
        else:
            rejected.append(
                {"goal_rad": {k: round(v, 4) for k, v in g.items()}, "contacts": contacts}
            )
    print(
        f"[goals] {len(goals)} valid goals after rejecting {len(rejected)} samples "
        "in collision or out of bounds"
    )
    rows = []
    for i, g in enumerate(goals):
        row = node.plan(g)
        row["request"] = i
        row["goal_rad"] = {k: round(v, 4) for k, v in g.items()}
        rows.append(row)
        print(
            f"[plan] {i:2d} {row.get('error_name')} planning {row.get('planning_time_s')} s "
            f"points {row.get('trajectory_points')}"
        )
    node.destroy_node()
    rclpy.shutdown()

    ok = [r for r in rows if r.get("success")]
    times = sorted(r["planning_time_s"] for r in ok)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip()
        )
    except Exception:
        commit, dirty = None, None
    try:
        moveit_version = subprocess.run(
            ["ros2", "pkg", "xml", "moveit_core", "-t", "version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        moveit_version = None
    import os

    record = {
        "workload": "synria-moveit-planning-campaign",
        "lane": "simulation",
        "physical_evidence": False,
        "note": "Planning only, in an empty scene with the SRDF self-collision model; "
        "no robot moved.",
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {
            "requests": args.requests,
            "seed": args.seed,
            "group": args.group,
            "planner_id": args.planner_id,
            "allowed_planning_time_s": args.planning_time,
            "num_planning_attempts": args.attempts,
            "goal_sampling": "uniform in joint space inside the URDF position limits with a "
            "margin; goals rejected by check_state_validity are resampled and counted",
            "margin": args.margin,
            "goals_rejected": len(rejected),
            "goal_tolerance_rad": 0.001,
            "start_state": "joint_state_publisher defaults (zeros)",
            "success_rule": "error code SUCCESS and a trajectory with at least two points",
        },
        "joint_limits_rad": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in limits.items()},
        "summary": {
            "successes": len(ok),
            "requests": len(rows),
            "success_rate": len(ok) / len(rows) if rows else None,
            "planning_time_s_median": statistics.median(times) if times else None,
            "planning_time_s_p95": times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
            if times
            else None,
            "planning_time_s_max": times[-1] if times else None,
            "trajectory_points_median": statistics.median(r["trajectory_points"] for r in ok)
            if ok
            else None,
            "trajectory_duration_s_median": statistics.median(
                r["trajectory_duration_s"] for r in ok
            )
            if ok
            else None,
            "failures_by_code": {
                str(r.get("error_name")): sum(
                    1 for q in rows if q.get("error_name") == r.get("error_name")
                )
                for r in rows
                if not r.get("success")
            },
        },
        "requests_detail": rows,
        "rejected_goals": rejected,
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "moveit_core_version": moveit_version,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[campaign] {len(ok)}/{len(rows)} planned; wrote {args.out}")
    return 0 if ok and len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
