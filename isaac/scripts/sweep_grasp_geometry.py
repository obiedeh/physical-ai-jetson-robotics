"""Scripted grasp-geometry sweep: which grasp height and wrist pose PLACE?

The RL programme has spent two experiments failing to fix placement by
reward shaping (E29 made orientation 4x worse; E31 pending). This asks the
prior question directly, with no policy involved: **is there a grasp
geometry from which a scripted arm can pick this cup up AND set it back
down upright?** If none exists, no reward will find one.

Three ideas under test, per the operator's prescription:

  1. GRASP HEIGHT — grasp just below the cup's rim, sweeping DOWNWARD until
     one works. Bounded below by geometry: the pad collider extends 36.1 mm
     beneath its own centre (measured, reports/synria_grasp_audit.md), so a
     pad centre under surface+36.1 stands the pads ON the table and the
     fingers jam at zero travel. The cup is 50 mm tall. The whole feasible
     window is therefore ~36-50 mm and this sweep walks it.

  2. WRIST PITCH — approach at 45 / 135 degrees rather than straight down,
     so the pads meet the cup wall rather than its rim.

  3. POSE CONSISTENCY — "the way you grasp is the way you descend and drop".
     The wrist orientation at the moment of grasp is LATCHED and held through
     lift, transport and set-down via full 6-DOF Jacobian control. The E27
     takeover controller violated exactly this: it drove position only (3
     Jacobian rows) and let the redundant DOFs rotate the wrist during the
     descent, after which upright held in just 58 of 134 releases.

Writes its own report and NEVER touches reports/synria_grasp_trials_production.json
— that file is the Phase 9 gate the training launcher reads, and overwriting
it mid-experiment would block the running seeds.

    ~/.venv/isaacsim5/bin/python isaac/scripts/sweep_grasp_geometry.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CUP_R, CUP_H, CUP_MASS = 0.02, 0.05, 0.05   # CUP_R overridden by --cup_d
TABLE_H = 0.05
PAD_BELOW_CENTRE_M = 0.0361      # measured pad-collider underhang
UPRIGHT_COS = 0.966              # cos(15 deg)


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    p = argparse.ArgumentParser(prog="sweep_grasp_geometry")
    p.add_argument("--grasp_z", type=str, default="46,44,42,40,38,37",
                   help="comma list of pad-centre heights above the work "
                        "surface (mm) to sweep, high to low")
    p.add_argument("--pitch_deg", type=str, default="0,45,135",
                   help="comma list of wrist pitch offsets (deg) to try")
    p.add_argument("--cup_d", type=float, default=40.0,
                   help="cup diameter (mm). The gripper opens to 50mm, so the "
                        "40mm default leaves only 5mm per side and the "
                        "withdrawing jaws clip the cup after release. The "
                        "original audit's recommendation #5 was a 25-30mm "
                        "object (suggested 28x60mm); it was never acted on.")
    p.add_argument("--retract", choices=("up", "lateral", "lateral_slow"),
                   default="up",
                   help="how the hand leaves after release. 'up' (current) "
                        "withdraws straight through the cup's swept volume, "
                        "which knocks a correctly-placed cup (2.1 deg, 25.7mm) "
                        "to 90 deg. 'lateral' slides the open mouth of the "
                        "jaws off the cup in -x first, then lifts clear.")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--slow", type=float, default=1.0,
                   help="divide servo step size by this (>1 = gentler motion). "
                        "Tests whether tipping is acceleration-driven or a "
                        "fundamental limit of the grip.")
    p.add_argument("--grip_n", type=float, default=5.0,
                   help="finger effort limit (N). The default 5.0 is the "
                        "harness value; raising it tests whether the grip is "
                        "simply too weak to resist pitching.")
    p.add_argument("--out", type=Path, default=None)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        import os

        os._exit(1)
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import os
    import math

    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore
    from isaaclab.assets import RigidObject, RigidObjectCfg  # type: ignore
    from isaaclab.sim import SimulationCfg, SimulationContext  # type: ignore
    from isaaclab.utils.math import (  # type: ignore
        axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul,
    )

    global CUP_R
    CUP_R = args.cup_d / 2000.0
    print(f"[sweep] cup diameter {args.cup_d:.0f}mm -> clearance per side "
          f"{(50.0 - args.cup_d) / 2:.1f}mm at full open", flush=True)
    usd = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm.usd"
    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    g = sim_utils.GroundPlaneCfg()
    g.func("/World/ground", g)
    lt = sim_utils.DomeLightCfg(intensity=2000.0)
    lt.func("/World/light", lt)

    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd), copy_from_source=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["Joint[1-6]"],
                                       effort_limit_sim=200.0,
                                       stiffness=2000.0, damping=200.0),
            "fingers": ImplicitActuatorCfg(joint_names_expr=[".*_finger"],
                                           effort_limit_sim=args.grip_n,
                                           stiffness=2e3, damping=1e2),
        },
    ))
    table = RigidObject(RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.50, TABLE_H),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.3, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.0, TABLE_H / 2)),
    ))
    cup = RigidObject(RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=CUP_R, height=CUP_H,
            mass_props=sim_utils.MassPropertiesCfg(mass=CUP_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                angular_damping=5.0, linear_damping=0.5),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.3, dynamic_friction=1.1,
                friction_combine_mode="max"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, .1, .1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.24, 0.0, TABLE_H + CUP_H / 2 + 0.001)),
    ))

    sim.reset()
    robot.update(sim.get_physics_dt())
    lf = robot.joint_names.index("left_finger")
    rf = robot.joint_names.index("right_finger")
    j1 = robot.joint_names.index("Joint1")
    arm_ids = [robot.joint_names.index(f"Joint{i}") for i in range(1, 7)]
    lgb = robot.body_names.index("left_gripper")
    rgb = robot.body_names.index("right_gripper")
    tool = robot.body_names.index("tool0")
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import PAD_LOCAL_OFFSET_M

    def pad_centre():
        out = []
        for b, side in ((lgb, "left"), (rgb, "right")):
            pos, q = robot.data.body_pos_w[:, b], robot.data.body_quat_w[:, b]
            loc = torch.tensor([PAD_LOCAL_OFFSET_M[side]], device=pos.device, dtype=pos.dtype)
            out.append(pos + quat_apply(q, loc))
        return 0.5 * (out[0][0] + out[1][0])

    def cup_tilt_deg():
        q = cup.data.root_quat_w[0]
        up = (1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)).clamp(-1.0, 1.0)
        return float(torch.rad2deg(torch.arccos(up)))

    jt = robot.data.default_joint_pos.clone()
    lock = {"on": True}

    def step(n=1):
        for _ in range(n):
            if lock["on"]:
                robot.write_joint_state_to_sim(
                    jt[:, arm_ids], torch.zeros_like(jt[:, arm_ids]),
                    joint_ids=arm_ids)
            robot.set_joint_position_target(jt)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            cup.update(sim.get_physics_dt())

    def servo(target_pos, hold_quat=None, iters=1200, tol=8e-4):
        """Returns True only if it CONVERGED. With --slow the per-step motion
        shrinks, so a fixed iteration budget silently under-runs: at slow=8
        the set-down stopped 8 mm high, the cup was released perched on the
        pads, and a loose tolerance scored it as placed. Budgets now scale."""
        iters = int(iters * max(args.slow, 1.0))
        """6-DOF DLS servo of the pad centre; holds wrist orientation if given."""
        for _ in range(iters):
            err = target_pos - pad_centre()
            if float(err.norm()) < tol and hold_quat is None:
                return True
            jac = robot.root_physx_view.get_jacobians()[:, tool - 1, :, :6]
            if hold_quat is None:
                twist = torch.cat([err, torch.zeros(3, device=err.device)]).view(1, 6, 1)
            else:
                cq = robot.data.body_quat_w[:, tool]
                ae = axis_angle_from_quat(quat_mul(hold_quat, quat_conjugate(cq)))[0]
                if float(err.norm()) < tol and float(ae.norm()) < 0.02:
                    return True
                twist = torch.cat([err, 0.8 * ae]).view(1, 6, 1)
            jt_ = jac.transpose(1, 2)
            reg = (0.05 ** 2) * torch.eye(6, device=jac.device).unsqueeze(0)
            dq = (jt_ @ torch.linalg.solve(jac @ jt_ + reg, twist)).squeeze(-1)
            _sc = 1.0 / max(args.slow, 1e-6)
            jt[:, arm_ids] += torch.clamp(dq * 0.5 * _sc, -0.01 * _sc, 0.01 * _sc)
            step(1)
        return False

    def snap(name, store):
        """Cup tilt, height above table, and slip relative to the pads."""
        cp = cup.data.root_pos_w[0]
        pc = pad_centre()
        store.append({
            "at": name,
            "tilt_deg": round(cup_tilt_deg(), 1),
            "cup_above_table_mm": round((float(cp[2]) - TABLE_H) * 1000, 1),
            "cup_minus_pad_z_mm": round((float(cp[2]) - float(pc[2])) * 1000, 1),
            "cup_xy_vs_pad_mm": round(float((cp[:2] - pc[:2]).norm()) * 1000, 1),
        })

    results = []
    heights = [float(x) / 1000.0 for x in args.grasp_z.split(",")]
    pitches = [float(x) for x in args.pitch_deg.split(",")]

    for pitch in pitches:
        for gz in heights:
            stages: list = []
            rec = dict(pitch_deg=pitch, grasp_z_mm=round(gz * 1000, 1),
                       pad_bottom_clear_mm=round((gz - PAD_BELOW_CENTRE_M) * 1000, 1),
                       grasped=0, upright_at_grasp=0, placed=0, upright_at_place=0,
                       tipped=0, setdown_not_converged=0, released_free=0,
                       release_confirmed=0, gap_at_release_mm=0.0,
                       pad_floor_hit=0, exit_cleared=0,
                       trials=args.trials)
            for t in range(args.trials):
                # reset
                lock["on"] = True
                jt[:] = robot.data.default_joint_pos.clone()
                robot.write_joint_state_to_sim(jt, torch.zeros_like(jt))
                jt[0, lf], jt[0, rf] = 0.0, 0.0
                cx = 0.24 + (t - args.trials / 2) * 0.004
                root = cup.data.default_root_state.clone()
                root[0, 0], root[0, 1] = cx, 0.0
                root[0, 2] = TABLE_H + CUP_H / 2 + 0.001
                root[0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=root.device)
                root[0, 7:] = 0.0
                cup.write_root_pose_to_sim(root[:, :7])
                cup.write_root_velocity_to_sim(root[:, 7:])
                step(60)

                # approach: pad centre over the cup at the swept height
                cpos = cup.data.root_pos_w[0].clone()
                tgt = cpos.clone()
                tgt[2] = TABLE_H + gz
                if not servo(tgt):
                    continue
                step(30)

                # pitch the wrist, then re-servo to keep the pads on target
                if abs(pitch) > 1e-6:
                    jt[0, arm_ids[4]] += math.radians(pitch) * 0.5
                    step(30)
                    servo(tgt, iters=600)

                grasp_quat = robot.data.body_quat_w[:, tool].clone()  # LATCH

                # close
                for _ in range(140):
                    jt[0, lf] = min(0.025, float(jt[0, lf]) + 0.00025)
                    jt[0, rf] = max(-0.025, float(jt[0, rf]) - 0.00025)
                    step(2)
                step(60)
                width = 50.0 - float(jt[0, lf]) * 1000 + float(jt[0, rf]) * 1000
                w_real = (50.0 - float(robot.data.joint_pos[0, lf]) * 1000
                          + float(robot.data.joint_pos[0, rf]) * 1000)
                if not (args.cup_d - 6 <= w_real <= args.cup_d + 6):
                    continue
                rec["grasped"] += 1
                if cup_tilt_deg() <= 15.0:
                    rec["upright_at_grasp"] += 1
                tr: list = []
                snap("after_grasp", tr)

                # hand to real dynamics; friction must hold from here
                jt[:, arm_ids] = robot.data.joint_pos[:, arm_ids].clone()
                lock["on"] = False
                step(60)

                # lift, holding the grasp pose
                up = pad_centre().clone()
                up[2] += 0.10
                servo(up, hold_quat=grasp_quat, iters=900)
                step(60)
                snap("after_lift", tr)

                # transport: yaw only, pose preserved
                for _ in range(120):
                    jt[0, j1] += 0.004
                    step(2)
                step(90)
                snap("after_transport", tr)

                # ---- SET DOWN: OPEN WHILE DESCENDING -------------------
                # Prescribed after watching the failure. Neither a
                # cup-referenced set-down (drives the pad bottom 1.4 mm INTO
                # the table, since the pads hang 36.1 mm below their centre)
                # nor a pad-referenced one (leaves the cup 5.3 mm high) works,
                # because the cup slips ~5.8 mm up in the jaws during the lift
                # and the two references disagree by exactly that.
                #
                # Easing the jaws open DURING the descent dissolves the
                # conflict instead of trading it: once the grip is loose the
                # cup is no longer clamped, so it seats on the table under its
                # own weight while the pads travel past it. Pad-tip contact
                # then becomes the natural stop rather than a computed height.
                # Then leave STRAIGHT UP -- lateral motion is what clips the
                # rim -- and only home after clearing the cup.
                grip_closed = float(jt[0, lf])
                pad_stop_z = TABLE_H + PAD_BELOW_CENTRE_M + 0.001
                conv = False
                for k in range(260):
                    pz = float(pad_centre()[2])
                    if pz <= pad_stop_z:
                        conv = True
                        break
                    frac = min(1.0, k / 180.0)
                    loose = grip_closed * (1.0 - 0.55 * frac)
                    jt[0, lf], jt[0, rf] = loose, -loose
                    tgt = pad_centre().clone()
                    tgt[2] = max(pad_stop_z, pz - 0.002)
                    servo(tgt, hold_quat=grasp_quat, iters=40)
                if conv:
                    rec["pad_floor_hit"] += 1
                step(90)
                snap("after_setdown", tr)

                jt[0, lf], jt[0, rf] = 0.0, 0.0
                rel_ok = False
                for _ in range(900):
                    step(1)
                    gap = (50.0 - float(robot.data.joint_pos[0, lf]) * 1000
                           + float(robot.data.joint_pos[0, rf]) * 1000)
                    if gap >= args.cup_d + 4.0:
                        rel_ok = True
                        break
                rec["gap_at_release_mm"] = round(
                    50.0 - float(robot.data.joint_pos[0, lf]) * 1000
                    + float(robot.data.joint_pos[0, rf]) * 1000, 1)
                step(90)
                if rel_ok:
                    rec["release_confirmed"] += 1
                snap("after_release", tr)

                # ITERATION 2 (E33). Iteration 1 released perfectly (0.0 deg,
                # 25.0 mm, clean after_clear) but PLACED only 3/12: the
                # incremental +4mm-at-a-time climb stalled at 105 mm instead
                # of the intended 160, and the joint-space homing then swept
                # the arm back through the cup (after_home 20 deg). One
                # change: climb to an ABSOLUTE safe height, verify it, and
                # gate homing on having actually cleared.
                clear_z = TABLE_H + 0.20
                up_t = pad_centre().clone()
                up_t[2] = clear_z
                servo(up_t, hold_quat=grasp_quat, iters=2500)
                step(120)
                cleared = float(pad_centre()[2]) >= TABLE_H + CUP_H + 0.08
                if cleared:
                    rec["exit_cleared"] = rec.get("exit_cleared", 0) + 1
                snap("after_clear", tr)

                # ITERATION 3 (E33). Iteration 2 verified the climb
                # (exit_cleared 12/12, cup perfect at after_clear) and the cup
                # was STILL hit at homing: the home pose is the task-ready
                # hover with the pad centre only ~46 mm above the table, so a
                # plain joint-space interpolation DESCENDS while still yawed
                # over the cup's sector -- after_home showed the pad 21 mm
                # above the cup. ORDERED homing: yaw Joint1 back FIRST at the
                # verified-clear altitude (rotating out of the cup's sector
                # while high), and only then lower the rest into home, which
                # sits ~112 mm from the cup in xy.
                home = robot.data.default_joint_pos.clone()
                if not cleared:
                    up_t[2] = clear_z
                    servo(up_t, hold_quat=grasp_quat, iters=2500)
                for _ in range(400):        # phase 1: yaw only, at altitude
                    jt[0, j1] += (float(home[0, j1]) - float(jt[0, j1])) * 0.03
                    step(2)
                    if abs(float(jt[0, j1]) - float(home[0, j1])) < 0.01:
                        break
                for _ in range(500):        # phase 2: everything else
                    jt[:, arm_ids] += (home[:, arm_ids] - jt[:, arm_ids]) * 0.02
                    step(2)
                step(180)
                snap("after_home", tr)
                stages.append(tr)

                cz = float(cup.data.root_pos_w[0, 2])
                # Free = the cup did NOT ride up with the retracting hand.
                free = abs(float(cup.data.root_pos_w[0, 2]) - (TABLE_H + CUP_H / 2)) < 0.05
                if free:
                    rec["released_free"] += 1
                resting = abs(cz - (TABLE_H + CUP_H / 2)) <= 0.005
                tilt = cup_tilt_deg()
                if tilt > 15.0:
                    rec["tipped"] += 1
                if resting and tilt <= 15.0:
                    rec["placed"] += 1
                    rec["upright_at_place"] += 1
            rec["stage_trace"] = stages
            results.append(rec)
            print(f"[sweep] pitch={pitch:>5.0f}deg z={rec['grasp_z_mm']:>5.1f}mm "
                  f"(pad clears table by {rec['pad_bottom_clear_mm']:>5.1f}mm) -> "
                  f"grasp {rec['grasped']}/{rec['trials']} "
                  f"upright@grasp {rec['upright_at_grasp']} "
                  f"PLACED {rec['placed']} tipped {rec['tipped']}", flush=True)

    print("[trace] cup state through the scripted cycle (median over trials):", flush=True)
    for r in results:
        if not r.get("stage_trace"):
            continue
        names = [x["at"] for x in r["stage_trace"][0]]
        print(f"[trace] pitch={r['pitch_deg']:.0f} z={r['grasp_z_mm']:.0f}mm", flush=True)
        for i, nm in enumerate(names):
            def med(key):
                v = sorted(t[i][key] for t in r["stage_trace"] if len(t) > i)
                return v[len(v) // 2] if v else float("nan")
            print(f"[trace]    {nm:<16} tilt={med('tilt_deg'):>6.1f}deg "
                  f"cup_above_table={med('cup_above_table_mm'):>7.1f}mm "
                  f"cup-pad_z={med('cup_minus_pad_z_mm'):>7.1f}mm "
                  f"xy_off={med('cup_xy_vs_pad_mm'):>6.1f}mm", flush=True)

    best = max(results, key=lambda r: (r["placed"], r["grasped"]))
    print(f"[sweep] ======== BEST: pitch={best['pitch_deg']}deg "
          f"z={best['grasp_z_mm']}mm placed {best['placed']}/{best['trials']} "
          f"========", flush=True)
    out = args.out or (_REPO_ROOT / "reports/synria_grasp_geometry_sweep.json")
    out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"[sweep] wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
