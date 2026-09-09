"""R2: does the VALIDATED Robotiq 2F-85 (Isaac Lab Franka variant) lift our cup?

Minimal scene: Franka + 2F-85 (NVIDIA-tuned drives), ground plane, and the
task's exact cup (40 mm dia, 50 g, G30 friction material). Sequence:
teleport the cup between the open fingers at pad height -> close the
gripper drive -> lift the arm joint-space -> measure cup rise.

Cup rises >= 5 cm and tracks the hand => a validated gripper CAN grasp
this cup in our PhysX stack (the wall is the Synria asset, not PhysX).

    ~/.venv/isaacsim5/bin/python isaac/scripts/diag_robotiq_grasp.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        app.close()


def _run() -> int:
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.assets import RigidObject, RigidObjectCfg  # type: ignore
    from isaaclab.assets import Articulation  # type: ignore
    from isaaclab.sim import SimulationContext, SimulationCfg  # type: ignore
    from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG  # type: ignore

    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/light", light)

    robot_cfg = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    robot = Articulation(robot_cfg)

    cup_cfg = RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.02,
            height=0.04,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                angular_damping=5.0, linear_damping=0.5
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.3,
                dynamic_friction=1.1,
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.021)),
    )
    cup = RigidObject(cup_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())
    print("[r2] bodies:", robot.body_names, flush=True)
    print("[r2] joints:", robot.joint_names, flush=True)

    def steps(n: int) -> None:
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            cup.update(sim.get_physics_dt())

    # settle at default pose, gripper open
    jt = robot.data.default_joint_pos.clone()
    robot.set_joint_position_target(jt)
    steps(240)

    # find the grasp frame: use the hand body
    hand_candidates = [b for b in robot.body_names if "hand" in b.lower() or "base_link" in b.lower()]
    hand_idx = robot.body_names.index(hand_candidates[0]) if hand_candidates else -1
    hand_pos = robot.data.body_pos_w[0, hand_idx]
    print(f"[r2] hand body '{robot.body_names[hand_idx]}' at {hand_pos.tolist()}", flush=True)

    # place the cup ON THE GROUND directly under the hand XY, then
    # descend the arm onto it (mid-air teleports just fall: probe 1).
    root = cup.data.default_root_state.clone()
    root[0, 0] = hand_pos[0]
    root[0, 1] = hand_pos[1]
    root[0, 2] = 0.021
    root[0, 7:] = 0.0
    cup.write_root_pose_to_sim(root[:, :7])
    cup.write_root_velocity_to_sim(root[:, 7:])
    steps(60)
    z0 = float(cup.data.root_pos_w[0, 2])

    # descend with height feedback; pads are ~0.16 below panda_hand.
    j2 = robot.joint_names.index("panda_joint2")
    j4 = robot.joint_names.index("panda_joint4")
    target_hand_z = 0.021 + 0.16
    sign = 1.0
    hz_prev = float(robot.data.body_pos_w[0, hand_idx, 2])
    jt[0, j2] += 0.02 * sign
    robot.set_joint_position_target(jt)
    steps(20)
    if float(robot.data.body_pos_w[0, hand_idx, 2]) > hz_prev:
        sign = -1.0  # wrong way — flip
    for _ in range(600):
        hz = float(robot.data.body_pos_w[0, hand_idx, 2])
        if hz <= target_hand_z:
            break
        jt[0, j2] += 0.004 * sign
        robot.set_joint_position_target(jt)
        steps(4)
    hz = float(robot.data.body_pos_w[0, hand_idx, 2])
    dxy = float(torch.norm(robot.data.body_pos_w[0, hand_idx, :2] - cup.data.root_pos_w[0, :2]))
    print(f"[r2] descended: hand_z={hz:.3f} (target {target_hand_z:.3f}) xy-offset={dxy:.3f}",
          flush=True)

    # close the gripper drive
    fj = robot.joint_names.index("finger_joint")
    jt[0, fj] = 0.79
    robot.set_joint_position_target(jt)
    steps(300)
    print(f"[r2] closed: finger_joint={float(robot.data.joint_pos[0, fj]):.3f} "
          f"cup z={float(cup.data.root_pos_w[0, 2]):.3f}", flush=True)

    # lift: reverse the descent ramp
    for _ in range(300):
        hz = float(robot.data.body_pos_w[0, hand_idx, 2])
        if hz >= target_hand_z + 0.15:
            break
        jt[0, j2] -= 0.004 * sign
        robot.set_joint_position_target(jt)
        steps(4)
    z1 = float(cup.data.root_pos_w[0, 2])
    hand_z1 = float(robot.data.body_pos_w[0, hand_idx, 2])
    print(f"[r2] after lift: cup z {z0:.3f} -> {z1:.3f}, hand z {hand_z1:.3f}", flush=True)
    lifted = (z1 - z0) > 0.05
    print(f"[r2] RESULT: {'LIFTED — validated gripper grasps our cup' if lifted else 'not lifted'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
