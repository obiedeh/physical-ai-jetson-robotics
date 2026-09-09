"""Scripted grasp feasibility test — can the gripper physically hold the piece?

No RL. Answers the one question training telemetry cannot: whether the
Alicia-D parallel fingers (5 N effort, 25 mm travel each) can mechanically
hold the 40 mm training cylinder at all. If this fails, no reward design
can make the pick-place task learnable and the fix is physical (friction,
finger force, piece size).

Protocol (single env, headless):
  1. Settle the arm at its default pose, gripper open.
  2. Read the two finger body positions -> grasp centre (also printed as an
     env-local constant for the pre-grasped curriculum reset).
  3. Teleport the piece to the grasp centre, HOLDING it in place each step
     (rewriting the root state) while the fingers close — the fingers close
     at 0.05 m/s and free-fall would drop the piece 1 m before they arrive.
  4. Release the hold (stop rewriting) with fingers closed: does the piece
     stay at grasp height?  -> STATIC HOLD
  5. Raise the shoulder joint: does the piece rise with the hand?
     -> LIFT HOLD (the answer we care about)

Usage:
    $ISAAC_PYTHON isaac/scripts/diag_grasp_feasibility.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(prog="diag_grasp_feasibility")
    parser.add_argument(
        "--no-raise",
        action="store_true",
        help="Skip raising the hand — measure the grasp centre at the default "
        "reset pose (used to derive the pre-grasped curriculum constant).",
    )
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    app = app_launcher.app

    try:
        return _run(args)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]

    # Same shutdown-hang guard as the train script.
    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
        PIECE_HEIGHT_M,
        TABLE_SURFACE_Z,
    )

    env_cfg = parse_env_cfg("Synria-Chess-PickPlace-v0", device="cuda", num_envs=1)
    env_cfg.seed = 42
    # Same tabletop -> ground plane swap as the train script.
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)

    env = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    piece = env.scene["piece"]
    dev = env.device
    _TOOL0_NAME = "tool0"

    # JointPositionAction uses use_default_offset=True: the applied target is
    # action*scale + DEFAULT joint pos. So zeros = hold the default pose, and
    # closing the fingers needs the NEGATIVE of their default offsets:
    #   left default +0.025, scale 0.025 -> action -1 targets 0 (closed)
    #   right default -0.025, scale 0.025 -> action +1 targets 0 (closed)
    arm_hold = torch.zeros((1, 6), device=dev)          # hold default pose
    grip_open = torch.tensor([[0.0, 0.0]], device=dev)   # default = open
    grip_closed = torch.tensor([[-1.0, 1.0]], device=dev)

    def act(arm: torch.Tensor, grip: torch.Tensor, steps: int) -> None:
        a = torch.cat([arm, grip], dim=-1)
        for _ in range(steps):
            env.step(a)

    def piece_z() -> float:
        return float(piece.data.root_pos_w[0, 2])

    # 1. settle
    act(arm_hold, grip_open, 60)

    # 1b. raise the hand into free air — at the default pose the fingertips
    # sit at table height, so the piece rests on the ground mid-test and the
    # ground plane confounds the grasp. Probe Joint2's sign adaptively.
    tool_id = robot.find_bodies([_TOOL0_NAME])[0][0]

    def tool_z() -> float:
        return float(robot.data.body_pos_w[0, tool_id, 2])

    z0 = tool_z()
    directions = () if args.no_raise else (-0.5, 0.5)
    for direction in directions:
        arm_try = arm_hold.clone()
        arm_try[0, 1] = direction
        act(arm_try, grip_open, 60)
        if tool_z() > z0 + 0.05:
            arm_hold = arm_try
            break
        act(arm_hold, grip_open, 60)  # back to neutral
    print(f"[diag] hand raised: tool0 z {z0:.3f} -> {tool_z():.3f} "
          f"(Joint2 target {float(arm_hold[0, 1]):+.2f})")

    # 2. grasp centre BETWEEN THE FINGER PADS (not the body origins — those
    # sit at the knuckles; placing the piece there floats it above the pads
    # and the fingers sweep shut underneath it). Pad-centre offsets in each
    # finger's local frame measured from the v2 asset's collision AABBs.
    from isaaclab.utils.math import quat_apply

    lid = robot.find_bodies(["left_gripper"])[0][0]
    rid = robot.find_bodies(["right_gripper"])[0][0]
    off_l = torch.tensor([[0.0, -0.030, 0.0255]], device=dev)
    off_r = torch.tensor([[0.0, 0.030, -0.0255]], device=dev)
    pl = robot.data.body_pos_w[0:1, lid] + quat_apply(robot.data.body_quat_w[0:1, lid], off_l)
    pr = robot.data.body_pos_w[0:1, rid] + quat_apply(robot.data.body_quat_w[0:1, rid], off_r)
    finger_ids = [lid, rid]
    grasp_center_w = ((pl + pr) / 2.0)[0]
    grasp_center_local = grasp_center_w - env.scene.env_origins[0]
    print(f"[diag] finger bodies         : {finger_ids}")
    print(f"[diag] grasp centre (world)  : {grasp_center_w.tolist()}")
    print(f"[diag] grasp centre (env)    : {grasp_center_local.tolist()}  "
          "<- curriculum constant")

    rest_z = TABLE_SURFACE_Z + PIECE_HEIGHT_M / 2.0
    lf = robot.find_joints(["left_finger"])[0][0]
    rf = robot.find_joints(["right_finger"])[0][0]

    def width_m() -> float:
        return float(robot.data.joint_pos[0, lf] - robot.data.joint_pos[0, rf])

    # 2b. PRESS TEST — global collider sanity. Drive the hand down into the
    # ground plane; inert colliders let the fingertips sink through it.
    arm_press = arm_hold.clone()
    arm_press[0, 1] = -float(arm_hold[0, 1]) if float(arm_hold[0, 1]) != 0 else -0.5
    act(arm_press, grip_open, 90)
    tip_z = float(robot.data.body_pos_w[0, finger_ids, 2].min())
    press_blocked = tip_z > TABLE_SURFACE_Z - 0.03
    print(f"[diag] press test: fingertip min z {tip_z:.3f} vs table {TABLE_SURFACE_Z:.3f} "
          f"-> {'BLOCKED (colliders live)' if press_blocked else 'SANK THROUGH'}")
    act(arm_hold, grip_open, 90)  # back up
    # refresh grasp centre after returning
    pl = robot.data.body_pos_w[0:1, lid] + quat_apply(robot.data.body_quat_w[0:1, lid], off_l)
    pr = robot.data.body_pos_w[0:1, rid] + quat_apply(robot.data.body_quat_w[0:1, rid], off_r)
    grasp_center_w = ((pl + pr) / 2.0)[0]

    # 3. gentle grasp: pin the piece between the pads only until the fingers
    # are just short of contact, then release the pin and let the squeeze
    # (position target under piece diameter -> constant 5 N drive force)
    # hold it. Rigid pinning through the whole close pops the piece out of
    # the thin pad sweep instead.
    root = piece.data.default_root_state[0:1].clone()
    root[0, 0:3] = grasp_center_w
    root[0, 7:] = 0.0
    ids = torch.tensor([0], device=dev)
    # width target 30 mm (< 40 mm piece): action = (0.015 - 0.025)/0.025 = -0.4
    grip_squeeze = torch.tensor([[-0.4, 0.4]], device=dev)
    a_squeeze = torch.cat([arm_hold, grip_squeeze], dim=-1)
    for _ in range(120):
        if width_m() > 0.046:  # not yet at the piece — keep it pinned
            piece.write_root_pose_to_sim(root[:, :7], env_ids=ids)
            piece.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
        env.step(a_squeeze)

    lp = float(robot.data.joint_pos[0, lf])
    rp = float(robot.data.joint_pos[0, rf])
    print(f"[diag] finger joints         : left {lp * 1000:+.1f} mm, right {rp * 1000:+.1f} mm")
    print(f"[diag] gripper width on piece: {width_m() * 1000:.1f} mm")
    piece_err = (piece.data.root_pos_w[0] - grasp_center_w).norm()
    print(f"[diag] piece offset from grasp centre: {float(piece_err) * 1000:.1f} mm")
    grip_closed = grip_squeeze  # keep squeezing through hold + lift phases

    # 4. release the hold — static hold?
    act(arm_hold, grip_closed, 60)
    z_static = piece_z()
    static_hold = z_static > rest_z + 0.02
    print(f"[diag] piece z after release : {z_static:.4f} (rest {rest_z:.4f}) "
          f"-> {'STATIC HOLD' if static_hold else 'DROPPED'}")

    # 5. lift the shoulder further — does the piece come along?
    z_before_lift = piece_z()
    arm_lift = arm_hold.clone()
    # push Joint2 further in whichever direction raised the hand (0 if the
    # probe never fired — fall back to -0.35)
    j2 = float(arm_hold[0, 1])
    arm_lift[0, 1] = j2 + (0.3 if j2 > 0 else -0.3 if j2 < 0 else -0.35)
    act(arm_lift, grip_closed, 90)
    z_lifted = piece_z()
    lift_hold = z_lifted > z_before_lift + 0.03
    print(f"[diag] piece z after lift    : {z_lifted:.4f} "
          f"(+{z_lifted - z_before_lift:.4f}) -> "
          f"{'LIFT HOLD' if lift_hold else 'SLIPPED'}")

    print()
    if static_hold and lift_hold:
        print("[diag] VERDICT: GRASP FEASIBLE — gripper can hold and lift the piece.")
        return 0
    print("[diag] VERDICT: GRASP INFEASIBLE — fix physics before reward tuning.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
