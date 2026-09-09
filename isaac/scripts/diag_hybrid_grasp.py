"""R4-lite: does the SYNRIA + Robotiq 2F-85 hybrid arm lift the cup?

Same protocol as diag_robotiq_grasp (feedback descent, close, lift) but
with the composed hybrid asset and Synria G30 arm drives + NVIDIA 2F-85
gripper drives.

    ~/.venv/isaacsim5/bin/python isaac/scripts/diag_hybrid_grasp.py --headless
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
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import (  # type: ignore
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from isaaclab.sim import SimulationCfg, SimulationContext  # type: ignore

    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    g = sim_utils.GroundPlaneCfg()
    g.func("/World/ground", g)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/light", light)

    usd = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2/synria_robotiq_assembled.usd"
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd), copy_from_source=False),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "Joint1": 0.0, "Joint2": 0.8, "Joint3": 1.2,
                "Joint4": 0.0, "Joint5": 0.6, "Joint6": 0.0,
                "finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
                ".*_inner_finger_knuckle_joint": 0.0, ".*_outer_.*_joint": 0.0,
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["Joint[1-3]"], effort_limit_sim=12.0,
                velocity_limit_sim=3.0, stiffness=400.0, damping=40.0),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["Joint[4-6]"], effort_limit_sim=4.0,
                velocity_limit_sim=3.0, stiffness=400.0, damping=40.0),
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"], effort_limit_sim=1650,
                velocity_limit_sim=10.0, stiffness=17, damping=0.02),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"], effort_limit_sim=50,
                velocity_limit_sim=10.0, stiffness=0.2, damping=0.001),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_knuckle_joint",
                                  "right_outer_knuckle_joint"],
                effort_limit_sim=1.0, velocity_limit_sim=10.0,
                stiffness=0.0, damping=0.0),
        },
    )
    robot = Articulation(robot_cfg)

    cup = RigidObject(RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.02, height=0.04,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                angular_damping=5.0, linear_damping=0.5),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.3, dynamic_friction=1.1,
                friction_combine_mode="max"),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.021)),
    ))

    sim.reset()
    robot.update(sim.get_physics_dt())
    print("[r4] bodies:", robot.body_names, flush=True)
    print("[r4] joints:", robot.joint_names, flush=True)

    def steps(n: int) -> None:
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            cup.update(sim.get_physics_dt())

    jt = robot.data.default_joint_pos.clone()
    robot.set_joint_position_target(jt)
    steps(240)

    base_idx = robot.body_names.index("base_link_0")

    # boot-pose search: guessed poses draped the gripper low/backwards
    # (probe v5: base at x=-0.46, z=0.09). Scan a small joint grid and
    # keep the pose with the gripper forward (x>0.15) and high.
    j = {n: robot.joint_names.index(f"Joint{i}") for i, n in
         zip(range(1, 7), ["J1", "J2", "J3", "J4", "J5", "J6"])}
    best, best_pose = None, None
    for a2 in (-0.8, -0.4, 0.4, 0.8):
        for a3 in (0.6, 1.0, 1.4):
            for a5 in (-1.2, -0.6, 0.6, 1.2):
                jt[0, j["J1"]] = 0.0
                jt[0, j["J2"]] = a2
                jt[0, j["J3"]] = a3
                jt[0, j["J4"]] = 0.0
                jt[0, j["J5"]] = a5
                jt[0, j["J6"]] = 0.0
                robot.set_joint_position_target(jt)
                steps(50)
                b = robot.data.body_pos_w[0, base_idx]
                x, z = float(b[0]), float(b[2])
                score = -abs(z - 0.195) if x > 0.15 and 0.14 < z < 0.30 else -99
                if best is None or score > best:
                    best = score
                    best_pose = (a2, a3, a5, x, z)
    a2, a3, a5, bx, bz = best_pose
    print(f"[r4] boot pose J2={a2} J3={a3} J5={a5} -> base x={bx:.3f} z={bz:.3f}",
          flush=True)
    jt[0, j["J2"]], jt[0, j["J3"]], jt[0, j["J5"]] = a2, a3, a5
    robot.set_joint_position_target(jt)
    steps(100)
    hand = robot.data.body_pos_w[0, base_idx]

    root = cup.data.default_root_state.clone()
    root[0, 0], root[0, 1], root[0, 2] = hand[0], hand[1], 0.021
    root[0, 7:] = 0.0
    cup.write_root_pose_to_sim(root[:, :7])
    cup.write_root_velocity_to_sim(root[:, 7:])
    steps(60)
    z0 = float(cup.data.root_pos_w[0, 2])

    # 6-DOF DLS servo on the gripper base (attitude locked to boot quat) —
    # single-joint ramps overshoot into the ground on this arm (probe v2).
    from isaaclab.utils.math import (
        axis_angle_from_quat, quat_inv, quat_mul,
    )

    arm_ids = [robot.joint_names.index(f"Joint{i}") for i in range(1, 7)]
    lock_quat = robot.data.body_quat_w[:, base_idx].clone()

    # Empirical jacobian-row selection: perturb J2, compare measured base
    # displacement against each row's prediction (the -1 convention broke
    # on the welded hybrid).
    j2i = arm_ids[1]
    p_before = robot.data.body_pos_w[0, base_idx].clone()
    jac_all = robot.root_physx_view.get_jacobians()[0]  # (nrows, 6, dof)
    dq_test = 0.05
    jt[0, j2i] += dq_test
    robot.set_joint_position_target(jt)
    steps(60)
    d_meas = (robot.data.body_pos_w[0, base_idx] - p_before)
    jt[0, j2i] -= dq_test
    robot.set_joint_position_target(jt)
    steps(60)
    scores = []
    for r in range(jac_all.shape[0]):
        pred = jac_all[r, :3, j2i] * dq_test
        denom = (pred.norm() * d_meas.norm()).clamp(min=1e-9)
        scores.append(float((pred @ d_meas) / denom))
    jac_row = int(torch.tensor(scores).argmax())
    print(f"[r4] jac row for gripper base: {jac_row} (cos={scores[jac_row]:.2f}; "
          f"-1 convention would be {base_idx - 1})", flush=True)

    print("[r4] jac shape:", tuple(robot.root_physx_view.get_jacobians().shape),
          "n_bodies:", len(robot.body_names), "n_joints:", len(robot.joint_names),
          "base_idx:", base_idx, flush=True)

    def ik_to(target, iters):
        nonlocal jt
        for it in range(iters):
            jac = robot.root_physx_view.get_jacobians()[:, jac_row, :, :][:, :, arm_ids]
            pos = robot.data.body_pos_w[:, base_idx]
            quat = robot.data.body_quat_w[:, base_idx]
            perr = target - pos
            rerr = axis_angle_from_quat(quat_mul(lock_quat, quat_inv(quat)))
            err = torch.cat([perr, 0.5 * rerr], dim=1)
            if float(perr[0].norm()) < 0.008:
                break
            jtT = jac.transpose(1, 2)
            a = jac @ jtT + 0.0025 * torch.eye(6, device=jac.device).expand(1, 6, 6)
            dq = (jtT @ torch.linalg.solve(a, err.unsqueeze(-1))).squeeze(-1)
            jt[0, arm_ids] = jt[0, arm_ids] + dq[0].clamp(-0.03, 0.03)
            robot.set_joint_position_target(jt)
            steps(3)
            if it < 6:
                print(f"[r4-it] it={it} perr={float(perr[0].norm()):.3f} "
                      f"dq={[round(float(x),3) for x in dq[0]]} "
                      f"base={[round(float(x),3) for x in robot.data.body_pos_w[0, base_idx]]}",
                      flush=True)

    # no-descent protocol: boot stance is already at grasp height.
    b0 = robot.data.body_pos_w[0, base_idx]
    print(f"[r4] grasp stance base: {[round(float(v),3) for v in b0]}", flush=True)
    root = cup.data.default_root_state.clone()
    root[0, 0], root[0, 1], root[0, 2] = float(b0[0]), float(b0[1]), 0.021
    root[0, 7:] = 0.0
    cup.write_root_pose_to_sim(root[:, :7])
    cup.write_root_velocity_to_sim(root[:, 7:])
    steps(90)
    z0 = float(cup.data.root_pos_w[0, 2])
    print(f"[r4] cup under gripper: z={z0:.3f} "
          f"dist={float(torch.norm(cup.data.root_pos_w[0]-robot.data.body_pos_w[0,base_idx])):.3f}",
          flush=True)

    fj = robot.joint_names.index("finger_joint")
    jt[0, fj] = 0.79
    robot.set_joint_position_target(jt)
    steps(300)
    print(f"[r4] closed: finger_joint={float(robot.data.joint_pos[0, fj]):.3f} "
          f"cup z={float(cup.data.root_pos_w[0, 2]):.3f}", flush=True)

    z0 = float(cup.data.root_pos_w[0, 2])
    j2 = robot.joint_names.index("Joint2")
    hz0 = float(robot.data.body_pos_w[0, base_idx, 2])
    jt[0, j2] += 0.05
    robot.set_joint_position_target(jt)
    steps(40)
    sign = 1.0 if float(robot.data.body_pos_w[0, base_idx, 2]) > hz0 else -1.0
    for _ in range(150):
        if float(robot.data.body_pos_w[0, base_idx, 2]) >= hz0 + 0.15:
            break
        jt[0, j2] += 0.004 * sign
        robot.set_joint_position_target(jt)
        steps(4)
    z1 = float(cup.data.root_pos_w[0, 2])
    print(f"[r4] after lift: cup z {z0:.3f} -> {z1:.3f} "
          f"(base z {float(robot.data.body_pos_w[0, base_idx, 2]):.3f})", flush=True)
    lifted = (z1 - z0) > 0.05
    print(f"[r4] RESULT: {'LIFTED — hybrid Synria+2F-85 grasps the cup' if z1 - z0 > 0.05 else 'not lifted'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
