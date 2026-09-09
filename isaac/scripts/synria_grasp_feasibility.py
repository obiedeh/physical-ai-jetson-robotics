"""Deterministic Synria gripper-vs-cup grasp feasibility trials.

Mission (2026-07-31): measure, do not assume. Fixed cup pose, scripted
joint commands only (no policy): descend -> close -> hold 2 s -> lift ->
transport (Joint1 yaw ramp) -> hold 2 s. Reports per-trial: fits,
symmetric contact, lifted, slips, ejected, falls-in-transport.

    ~/.venv/isaacsim5/bin/python isaac/scripts/synria_grasp_feasibility.py \
        --headless [--trials 10] [--usd g30|production]

Geometry context (measured from assets, see reports/synria_grasp_audit.md):
cup 40 mm dia x 50 mm; pads at +/-25.0 mm from centerline at joint 0
(usable opening 50.0 mm), full travel closes to 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CUP_R, CUP_H, CUP_MASS = 0.02, 0.05, 0.05


def main() -> int:
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--usd", choices=("production", "g30", "nopad", "padfix", "repaired", "padfix_l", "padfix2"), default="production")
    parser.add_argument("--raise_base", type=float, default=0.0,
                        help="elevate the robot base (m) — gripper-only phases "
                             "need no floor contact")
    parser.add_argument("--no_world", action="store_true",
                        help="spawn NO ground and NO cup — pure robot world "
                             "(isolates collision partners)")
    parser.add_argument("--block_test", action="store_true",
                        help="Phase 5: rigid-fixture width gauge. Arm locked, "
                             "kinematic blocks (20/28/30/40mm) pinned to the "
                             "re-measured pad centre; measured closed width must "
                             "match block width within 2mm. 40mm is a reported "
                             "stress case and does not gate. Exit 5 = the rig "
                             "self-test failed (cannot close on empty air)")
    parser.add_argument("--travel_test", action="store_true",
                        help="Phase 4: no-object full-travel test at the DEFAULT "
                             "pose (no descend): 0/25/50/75/100%%/0 targets, 10 "
                             "cycles, logs positions + clearances")
    parser.add_argument("--hover_test", action="store_true",
                        help="hold the cup between the pads at the default pose "
                             "(no descend), close, then release to dynamics: "
                             "clean contact-geometry + friction-hold test")
    parser.add_argument("--no_self_collision", action="store_true",
                        help="spawn with enabled_self_collisions=False — tests "
                             "whether the 0.4mm finger self-jam is self-collision")
    parser.add_argument("--table_h", type=float, default=0.05,
                        help="height (m) of the work surface the cup stands "
                             "on. The production task picks off a TABLE, not "
                             "the floor. With the cup on the ground the pads "
                             "-- which hang well below the finger frames -- "
                             "are driven to within ~3 mm of the ground plane, "
                             "so 'is the hand resting on the floor?' cannot be "
                             "answered cleanly. 0 = legacy floor-standing cup. "
                             "MUST stay low enough that the hand starts CLEAR: "
                             "at the rest pose the pad bottom is at ~92 mm, so "
                             "a 100 mm table is already 7.5 mm inside the pads "
                             "and every descent begins jammed.")
    parser.add_argument("--grasp_z", type=float, default=0.045,
                        help="height (m) ABOVE THE WORK SURFACE that the PAD "
                             "CENTRE descends to before closing. Measured "
                             "constraint: the pad collider box extends 36.1 mm "
                             "BELOW its own centre, so any pad centre lower "
                             "than surface+36.1 stands the pads ON the surface "
                             "and the fingers jam at zero travel. Default 0.045 "
                             "leaves ~9 mm of pad-to-surface clearance while "
                             "still putting 41 mm of pad face on the 50 mm cup")
    parser.add_argument("--jitter", type=float, default=0.003,
                        help="+/- cup placement noise (m) applied per trial in "
                             "x and y. Without it the harness is fully "
                             "deterministic and 10 trials are one trial run "
                             "ten times -- repeatability, not reliability. "
                             "0.003 is meaningful against the 40 mm cup's "
                             "5 mm/side clearance. 0 = legacy exact placement")
    parser.add_argument("--free_air", action="store_true",
                        help="teleport the cup 1 m away and close on nothing — "
                             "isolates finger self-jam from cup contact")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    app = app_launcher.app

    # Pad offsets, extents, and the rotations that turn them into world points.
    # MUST be imported after AppLauncher: this pulls in the synria_pickplace
    # package, whose __init__ imports env_cfg -> isaaclab. Importing that
    # before the app boots initialises USD's Python bindings against the wrong
    # runtime and Kit dies with a std::vector<SdfPath> converter error and a
    # segfault, several extensions into startup and nowhere near the cause.
    global gripper_geometry  # noqa: PLW0603 - deferred module binding
    from isaac.isaaclab_tasks.synria_pickplace import (  # noqa: PLC0415
        gripper_geometry,
    )

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
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore
    from isaaclab.assets import RigidObject, RigidObjectCfg  # type: ignore
    from isaaclab.sim import SimulationCfg, SimulationContext  # type: ignore

    usd_name = {
        "production": "synria_6dof_arm.usd",
        "g30": "synria_6dof_arm_g30.usd",
        "nopad": "synria_6dof_arm_nopad.usd",
        "padfix": "synria_6dof_arm_padfix.usd",
        "repaired": "synria_6dof_arm_repaired.usd",
        "padfix_l": "synria_6dof_arm_padfix_l.usd",
        "padfix2": "synria_6dof_arm_padfix2.usd",
    }[args.usd]
    usd_path = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2" / usd_name
    print(f"[feas] asset: {usd_path.name}", flush=True)

    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    if not args.no_world:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/light", light)

    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                copy_from_source=False,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                )
                if args.no_self_collision
                else None,
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, args.raise_base)),
            actuators={},
        ).replace(
            actuators={
                "arm": __import__(
                    "isaaclab.actuators", fromlist=["ImplicitActuatorCfg"]
                ).ImplicitActuatorCfg(
                    # TEST-HARNESS gains: the real 5 Nm / k=80 cannot hold
                    # the authored pose against gravity — the arm sagged to
                    # the floor and finger colliders dragged (the source of
                    # every "jam" in this audit). The harness tests the
                    # GRIPPER, not arm strength.
                    joint_names_expr=["Joint[1-6]"],
                    effort_limit_sim=200.0,
                    stiffness=2000.0,
                    damping=200.0,
                ),
                "fingers": __import__(
                    "isaaclab.actuators", fromlist=["ImplicitActuatorCfg"]
                ).ImplicitActuatorCfg(
                    joint_names_expr=[".*_finger"],
                    effort_limit_sim=5.0,
                    stiffness=2e3,
                    damping=1e2,
                ),
            }
        )
    )
    table = None
    if not args.no_world and args.table_h > 0.0:
        table = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(0.30, 0.40, args.table_h),
                    mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.3, 0.25)
                    ),
                ),
                # in front of the base, clear of the robot's own footprint
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.24, 0.0, args.table_h / 2)
                ),
            )
        )
    cup = None if args.no_world else RigidObject(
        RigidObjectCfg(
            prim_path="/World/Cup",
            spawn=sim_utils.CylinderCfg(
                radius=CUP_R,
                height=CUP_H,
                mass_props=sim_utils.MassPropertiesCfg(mass=CUP_MASS),
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
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.24, 0.0, args.table_h + CUP_H / 2 + 0.001)
            ),
        )
    )

    sim.reset()
    robot.update(sim.get_physics_dt())
    print("[feas] joints:", robot.joint_names, flush=True)
    lf = robot.joint_names.index("left_finger")
    rf = robot.joint_names.index("right_finger")
    j1 = robot.joint_names.index("Joint1")
    j2 = robot.joint_names.index("Joint2")
    lim = robot.data.joint_pos_limits[0]
    print(f"[feas] left_finger limits: {[round(v*1000,2) for v in lim[lf].tolist()]} mm, "
          f"right: {[round(v*1000,2) for v in lim[rf].tolist()]} mm", flush=True)
    print(f"[feas] finger defaults: "
          f"{round(float(robot.data.default_joint_pos[0, lf])*1000,2)} / "
          f"{round(float(robot.data.default_joint_pos[0, rf])*1000,2)} mm", flush=True)
    # teleport test: write joint state to 20 mm travel — a limit clamps it back
    jp0 = robot.data.default_joint_pos.clone()
    jp0[0, lf], jp0[0, rf] = 0.020, -0.020
    robot.write_joint_state_to_sim(jp0, torch.zeros_like(jp0))
    robot.set_joint_position_target(jp0)
    for _ in range(60):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    print(f"[feas] teleport-to-20mm test: fingers now at "
          f"{round(float(robot.data.joint_pos[0, lf])*1000,2)} / "
          f"{round(float(robot.data.joint_pos[0, rf])*1000,2)} mm", flush=True)
    grip_bodies = [i for i, b in enumerate(robot.body_names) if "gripper" in b.lower()]
    print("[feas] gripper bodies:", [robot.body_names[i] for i in grip_bodies], flush=True)

    def steps(n: int) -> None:
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            if cup is not None:
                cup.update(sim.get_physics_dt())

    # The pad offsets and the rotation that turns them into world points live
    # in ONE place — isaaclab_tasks/synria_pickplace/gripper_geometry.py — and
    # this harness must read them from there. It used to keep a private copy
    # of the constant, which is how a "single source of truth" fix shipped
    # while two sources were still live. The finger extends DIAGONALLY from
    # its frame, so "frame minus 30 mm in world z" lands outside the collider;
    # that mistake invalidated the first round of block/hover tests.

    def pad_center_world() -> torch.Tensor:
        """Midpoint between the two pad collider centres, in world coords."""
        return gripper_geometry.pad_center_world(robot)[0]

    def grip_center() -> torch.Tensor:
        return robot.data.body_pos_w[0, grip_bodies].mean(dim=0)

    results = []
    if args.block_test:
        import torch as _t
        import json as _json
        import os as _os

        # ------------------------------------------------------------------
        # Phase 5 RIGID FIXTURE (rewritten 2026-08-04).
        #
        # The first implementation scored 0/12 and the widths were declared
        # "unmeasurable because the arm drifts under load". Both symptoms had
        # a single cause in the RIG, not the asset -- the third instance of
        # this session's standing lesson:
        #
        #   The old harness stood each block FREE on a 120x120x20 mm kinematic
        #   pedestal whose top sat at pad_center_z - 25 mm. But the pad
        #   collider is a 43.3 x 60 x 28.5 mm box mounted DIAGONALLY, so it
        #   spans ~73 mm in world z: from pad_center_z - 36.5 mm to
        #   pad_center_z + 36.5 mm. The pedestal top was therefore ~11 mm
        #   INSIDE the pads, and 120 mm wide in xy so it could not miss them.
        #   The fingers were jamming on the PEDESTAL. That is why 20 mm blocks
        #   reported 0.06 mm of travel and a 49.86 mm "measured width" -- the
        #   fingers never moved at all.
        #
        # The fixture below removes every degree of freedom that was polluting
        # the measurement:
        #   1. No pedestal. Blocks are KINEMATIC, so they need no support and
        #      cannot be knocked off-centre; a kinematic body is still a fully
        #      contactful collider, which is exactly what a width gauge needs.
        #   2. The arm is LOCKED by writing Joint1-6 state every physics step.
        #      Contact load cannot drift the pads, so the object's pose is
        #      fixed relative to the PADS, as the handoff requires.
        #   3. The pad centre is re-measured immediately before each close and
        #      the block is re-pinned to it; the offset is recorded again at
        #      stall, so fixture rigidity is proven per trial rather than
        #      assumed.
        #   4. A no-block CONTROL close runs first and must reach full travel.
        #      If the rig cannot close on empty air it has no business
        #      reporting widths, and it exits 5 instead of emitting numbers.
        # ------------------------------------------------------------------
        lgb = robot.body_names.index("left_gripper")
        rgb = robot.body_names.index("right_gripper")
        arm_ids = [robot.joint_names.index(f"Joint{i}") for i in range(1, 7)]
        arm_hold = robot.data.default_joint_pos[:, arm_ids].clone()
        arm_zero = _t.zeros_like(arm_hold)

        BLOCK_WIDTHS = (20, 28, 30, 40)
        STRESS_WIDTHS = (40,)  # reported, not required to pass (40 mm cup case)
        blocks = {}
        for w_mm in BLOCK_WIDTHS:
            blocks[w_mm] = RigidObject(
                RigidObjectCfg(
                    prim_path=f"/World/Block{w_mm}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.03, w_mm / 1000.0, 0.05),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(1.0 + 0.2 * w_mm / 10, 0.0, 0.03)
                    ),
                )
            )
        sim.reset()
        robot.update(sim.get_physics_dt())
        for b in blocks.values():
            b.update(sim.get_physics_dt())

        _pin: dict = {"blk": None, "pose": None}

        def steps_b(n):
            """Step with the arm rigidly held and the fixture block re-pinned."""
            for _ in range(n):
                robot.write_joint_state_to_sim(arm_hold, arm_zero, joint_ids=arm_ids)
                robot.write_data_to_sim()
                if _pin["blk"] is not None:
                    _pin["blk"].write_root_pose_to_sim(_pin["pose"][:, :7])
                    _pin["blk"].write_root_velocity_to_sim(_pin["pose"][:, 7:])
                sim.step()
                robot.update(sim.get_physics_dt())
                for b in blocks.values():
                    b.update(sim.get_physics_dt())

        def open_fingers(jt) -> None:
            jt[0, lf], jt[0, rf] = 0.0, 0.0
            robot.set_joint_position_target(jt)
            steps_b(90)

        def ramp_close(jt) -> tuple:
            """Ramp both fingers shut; return (left_mm, right_mm) at stall."""
            for _ in range(140):
                jt[0, lf] = min(0.025, float(jt[0, lf]) + 0.00025)
                jt[0, rf] = max(-0.025, float(jt[0, rf]) - 0.00025)
                robot.set_joint_position_target(jt)
                steps_b(2)
            steps_b(60)  # let the stall settle before reading
            return (float(robot.data.joint_pos[0, lf]) * 1000,
                    -float(robot.data.joint_pos[0, rf]) * 1000)

        def pad_axis() -> tuple:
            """Unit vector from the left pad centre to the right pad centre."""
            left, right = gripper_geometry.pad_centers_world(
                robot, body_index={"left": lgb, "right": rgb}
            )
            v = right[0] - left[0]
            return v / v.norm(), 0.5 * (left[0] + right[0])

        # --- rig self-test: close on empty air, arm locked, no block ---------
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, _t.zeros_like(jp))
        jt = jp.clone()
        open_fingers(jt)
        ctl_l, ctl_r = ramp_close(jt)
        rig_ok = abs(ctl_l - 25.0) < 1.5 and abs(ctl_r - 25.0) < 1.5
        print(f"[block] rig self-test (no block, arm locked): "
              f"L{ctl_l:.2f} R{ctl_r:.2f} mm -> "
              f"{'PASS' if rig_ok else 'FAIL'}", flush=True)
        if not rig_ok:
            print("[block] ABORT: the fixture cannot close on empty air, so no "
                  "width it reports would mean anything. Fix the rig before "
                  "reading any block number.", flush=True)
            _os._exit(5)

        axis, _c = pad_axis()
        axis_deg = float(_t.rad2deg(_t.arccos(abs(axis[1]).clamp(max=1.0))))
        print(f"[block] pad closing axis {[round(float(v),4) for v in axis]} "
              f"({axis_deg:.2f} deg off world-Y; blocks are Y-aligned)", flush=True)

        recs = []
        for w_mm, blk in blocks.items():
            for trial in range(3):
                jp = robot.data.default_joint_pos.clone()
                robot.write_joint_state_to_sim(jp, _t.zeros_like(jp))
                jt = jp.clone()
                _pin["blk"] = None
                open_fingers(jt)

                # re-measure the pad centre immediately before the close and
                # pin the block to it -- this is the fixture
                mid = pad_center_world()
                root = blk.data.default_root_state.clone()
                root[0, :3] = mid
                root[0, 3:7] = _t.tensor([1.0, 0.0, 0.0, 0.0], device=root.device)
                root[0, 7:] = 0.0
                _pin["blk"], _pin["pose"] = blk, root
                steps_b(30)

                pre = pad_center_world()
                off_pre = [round(float(blk.data.root_pos_w[0, i] - pre[i]) * 1000, 2)
                           for i in range(3)]
                lq, rq = ramp_close(jt)
                post = pad_center_world()
                off_post = [round(float(blk.data.root_pos_w[0, i] - post[i]) * 1000, 2)
                            for i in range(3)]

                width = 50.0 - lq - rq
                expected = (50.0 - w_mm) / 2.0
                drift = max(abs(a - b) for a, b in zip(off_pre, off_post))
                rec = dict(
                    block_mm=w_mm, trial=trial,
                    expected_travel_mm=round(expected, 2),
                    left_travel_mm=round(lq, 2), right_travel_mm=round(rq, 2),
                    measured_width_mm=round(width, 2),
                    width_err_mm=round(width - w_mm, 2),
                    block_off_pre_mm=off_pre,
                    block_off_post_mm=off_post,
                    fixture_drift_mm=round(drift, 2),
                    symmetric=bool(abs(lq - rq) < 2.0),
                    stress_case=bool(w_mm in STRESS_WIDTHS),
                    ok=bool(abs(width - w_mm) <= 2.0 and abs(lq - rq) < 2.0),
                )
                recs.append(rec)
                print(f"[block] {rec}", flush=True)

                _pin["blk"] = None
                root[0, 0] = 1.0 + 0.2 * w_mm / 10
                root[0, 1], root[0, 2] = 0.0, 0.03
                blk.write_root_pose_to_sim(root[:, :7])
                blk.write_root_velocity_to_sim(root[:, 7:])
                open_fingers(jt)

        gate = [r for r in recs if not r["stress_case"]]
        stress = [r for r in recs if r["stress_case"]]
        n_ok = sum(1 for r in gate if r["ok"])
        s_ok = sum(1 for r in stress if r["ok"])
        worst = max((r["fixture_drift_mm"] for r in recs), default=0.0)
        print(f"[block] fixture rigidity: worst block-vs-pad drift "
              f"{worst:.2f} mm across {len(recs)} trials", flush=True)
        print(f"[block] ======== BLOCKS {n_ok}/{len(gate)} OK "
              f"(+ {s_ok}/{len(stress)} on the 40 mm stress case, not gating) "
              f"(asset={args.usd}) ========", flush=True)
        out = _REPO_ROOT / f"reports/synria_blocks_{args.usd}.json"
        out.write_text(_json.dumps(recs, indent=1))
        print(f"[feas] wrote {out}", flush=True)

        _os._exit(0 if n_ok == len(gate) else 4)

    if args.travel_test:
        import torch as _t
        import json as _json

        lgb = robot.body_names.index("left_gripper")
        rgb = robot.body_names.index("right_gripper")
        recs = []
        for cycle in range(10):
            jp = robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(jp, _t.zeros_like(jp))
            jt = jp.clone()
            cyc = {"cycle": cycle, "steps": []}
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0, 0.0):
                tgt = 0.025 * frac
                # ramp to target
                for _ in range(200):
                    cur = float(jt[0, lf])
                    if abs(cur - tgt) < 1e-5:
                        break
                    step_amt = max(-0.00035, min(0.00035, tgt - cur))
                    jt[0, lf] = cur + step_amt
                    jt[0, rf] = -float(jt[0, lf])
                    robot.set_joint_position_target(jt)
                    steps(2)
                steps(60)
                lq = float(robot.data.joint_pos[0, lf]) * 1000
                rq = float(robot.data.joint_pos[0, rf]) * 1000
                lw = robot.data.body_pos_w[0, lgb]
                rw = robot.data.body_pos_w[0, rgb]
                sep = float((lw - rw).norm()) * 1000
                cyc["steps"].append({
                    "target_mm": round(tgt * 1000, 1),
                    "left_mm": round(lq, 2),
                    "right_mm": round(rq, 2),
                    "frame_sep_mm": round(sep, 2),
                })
            recs.append(cyc)
            last = cyc["steps"]
            print(f"[travel] cycle {cycle}: "
                  + " | ".join(f"tgt {st['target_mm']}: L{st['left_mm']} R{st['right_mm']}"
                               for st in last), flush=True)
        full = [c["steps"][4] for c in recs]
        ok = all(abs(st["left_mm"] - 25.0) < 1.5 and abs(st["right_mm"] + 25.0) < 1.5
                 for st in full)
        print(f"[travel] ======== FULL TRAVEL {'PASS' if ok else 'FAIL'} "
              f"(asset={args.usd}) ========", flush=True)
        out = _REPO_ROOT / f"reports/synria_travel_{args.usd}.json"
        out.write_text(_json.dumps(recs, indent=1))
        print(f"[feas] wrote {out}", flush=True)
        import os as _os

        _os._exit(0 if ok else 3)

    if args.hover_test:
        import torch as _t

        for trial in range(args.trials):
            jp = robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(jp, _t.zeros_like(jp))
            jt = jp.clone()
            jt[0, lf], jt[0, rf] = 0.0, 0.0
            robot.set_joint_position_target(jt)
            steps(120)
            lgb = robot.body_names.index("left_gripper")
            rgb = robot.body_names.index("right_gripper")
            mid = 0.5 * (robot.data.body_pos_w[0, lgb] + robot.data.body_pos_w[0, rgb])
            # pads span 60mm below the frames: center the cup on the pads
            hold = mid.clone()
            hold[2] -= 0.030
            root = cup.data.default_root_state.clone()
            root[0, :3] = hold
            root[0, 7:] = 0.0

            def pin() -> None:
                cup.write_root_pose_to_sim(root[:, :7])
                cup.write_root_velocity_to_sim(root[:, 7:])

            pin()
            steps(10)
            # ramped close with the cup pinned in place
            for k in range(140):
                jt[0, lf] = min(0.025, float(jt[0, lf]) + 0.00025)
                jt[0, rf] = max(-0.025, float(jt[0, rf]) - 0.00025)
                robot.set_joint_position_target(jt)
                pin()
                steps(2)
            lq = float(robot.data.joint_pos[0, lf])
            rq = float(robot.data.joint_pos[0, rf])
            contacted = (lq < 0.024) and (-rq < 0.024)
            symmetric = abs(lq - (-rq)) < 0.004
            # release to dynamics: does friction hold the 50 g cup?
            steps(240)
            drop = float((cup.data.root_pos_w[0] - hold).norm())
            held = drop < 0.05
            rec = dict(
                trial=trial, mode="hover",
                stall_left_mm=round(lq * 1000, 1),
                stall_right_mm=round(-rq * 1000, 1),
                contacted=contacted, symmetric=symmetric,
                drop_mm=round(drop * 1000, 1), held=held,
                success=bool(contacted and held),
            )
            results.append(rec)
            print(f"[feas] hover trial {trial}: {rec}", flush=True)
        n_ok = sum(1 for r in results if r["success"])
        print(f"[feas] ======== HOVER SUCCESS {n_ok}/{len(results)} "
              f"(asset={args.usd}) ========", flush=True)
        import json as _json

        out = _REPO_ROOT / f"reports/synria_grasp_hover_{args.usd}.json"
        out.write_text(_json.dumps(results, indent=1))
        print(f"[feas] wrote {out}", flush=True)
        import os as _os

        _os._exit(0)

    for trial in range(args.trials):
        # reset arm + cup to identical states (deterministic)
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        jt = jp.clone()
        jt[0, lf], jt[0, rf] = 0.0, 0.0  # fingers fully open (50 mm)
        robot.set_joint_position_target(jt)

        # KINEMATIC ARM (2026-08-04). PD control at these gains sags: the
        # descent overshot its target by ~8 mm and parked the pad colliders
        # exactly on the work surface (pad bottom == table top, clearance
        # -0.0 mm, fingers jamming at 0 travel). That is the same arm-sag
        # artefact that produced this project's FIRST false "the asset is
        # broken" verdict, just landing on a table instead of the floor.
        # Phase 5 solved it by writing the arm's joint state every physics
        # step; do the same here. The arm is then a rigid positioner and the
        # only things left to physics are the finger DOFs and the free cup --
        # which is exactly what this gate is meant to measure. It tests the
        # GRIPPER, not arm strength (the real actuator gains are the RL
        # environment's concern).
        # ...but ONLY for positioning. write_joint_state_to_sim also writes the
        # joint VELOCITY, and this harness writes zeros: the solver then sees
        # perfectly stationary pads every step and generates no tangential
        # friction impulse, so a correctly-gripped cup is simply left behind
        # while the hand teleports away. That produced lifted=False on 10/10
        # trials whose grip was in fact textbook (5.0/5.0 mm stall, 40.0 mm
        # width) -- a false NEGATIVE manufactured by the rig, the mirror image
        # of the scoop false-positive earlier in this same session.
        # So: teleport to place the hand, then hand over to real PD dynamics
        # for hold/lift/transport, where friction has to do the work.
        _arm_ids = [robot.joint_names.index(f"Joint{i}") for i in range(1, 7)]
        _lock = {"on": True}

        def steps_k(n: int) -> None:
            for _ in range(n):
                if _lock["on"]:
                    robot.write_joint_state_to_sim(
                        jt[:, _arm_ids], torch.zeros_like(jt[:, _arm_ids]),
                        joint_ids=_arm_ids,
                    )
                robot.set_joint_position_target(jt)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())
                if cup is not None:
                    cup.update(sim.get_physics_dt())

        steps_k(120)

        # CORRECTED 2026-08-04 (grasp-pose audit). This loop used to place the
        # cup at grip_center() -- the midpoint of the two finger LINK FRAMES --
        # and descend until that frame midpoint reached z = 61 mm. Both are the
        # wrong point. The fingers extend diagonally, so the pad centre sits
        # 21.2 mm behind (-x) and 21.2 mm above (+z) the frame midpoint. The
        # old script therefore put the cup 21.2 mm off-axis in x and left the
        # pads ~57 mm above the cup centre: the cup was never between the pads,
        # which is the whole of the 0/10 scripted-trial result. Same 30 mm bug
        # as mdp._grasp_center_local -- see reports/synria_grasp_audit.md.
        #
        # Order is now: descend the PAD CENTRE to cup mid-height FIRST (the
        # pads sweep in x as Joint2 rotates, so their xy is only known after
        # the descent settles), then stand the cup on the ground at that xy.
        # Park the cup clear of the arm meanwhile, so the descending hand
        # cannot knock it and silently invalidate the trial.
        park = cup.data.default_root_state.clone()
        park[0, 0], park[0, 1], park[0, 2] = 1.5, 0.0, CUP_H / 2 + 0.001
        park[0, 7:] = 0.0
        cup.write_root_pose_to_sim(park[:, :7])
        cup.write_root_velocity_to_sim(park[:, 7:])
        steps_k(30)

        # Grasp height, PAD CENTRE frame. Not the cup mid-height (25 mm): the
        # pad centre rides 21.2 mm ABOVE the finger frame, so aiming the pads
        # at 25 mm forces the finger links down to ~4 mm and they rest on the
        # floor -- re-creating the arm-on-the-ground artefact that produced
        # this project's first false "the asset is broken" verdict. A 50 mm
        # cup is grasped near its top instead; the pad face spans ~73 mm in
        # world z, so an upper-third grip is still fully on the pad.
        target_z = args.table_h + args.grasp_z
        sign = 1.0
        z_prev = float(pad_center_world()[2])
        jt[0, j2] += 0.02
        steps_k(10)
        if float(pad_center_world()[2]) > z_prev:
            sign = -1.0
        jt[0, j2] -= 0.02
        steps_k(10)
        # With the arm driven kinematically the descent converges exactly, so
        # the old overshoot-then-correct dance is gone.
        descended = False
        for _ in range(4000):
            z = float(pad_center_world()[2])
            if abs(z - target_z) < 0.0005:
                descended = True
                break
            jt[0, j2] += 0.0004 * (sign if z > target_z else -sign)
            steps_k(1)
        steps_k(60)
        pc = pad_center_world()
        print(f"[feas] descend settled: PAD CENTRE z = "
              f"{round(float(pc[2])*1000,1)} mm (target {target_z*1000:.0f}); "
              f"finger-frame z = {round(float(grip_center()[2])*1000,1)} mm "
              f"(the old, wrong reference)", flush=True)

        # World-z footprint of the pad COLLIDER boxes at this pose. The pads are
        # mounted diagonally, so they sweep a large vertical span and it is the
        # pad BOTTOM -- not the finger frame -- that decides how close the hand
        # can get to a work surface. Guessing this number is how the pedestal
        # (Phase 5) and the table both ended up inside the pads. The extents and
        # the projection now live in gripper_geometry so this harness and the
        # env cannot disagree about where the hand ends.
        _pad_lo = [float(gripper_geometry.pad_lowest_z_world(robot)[0])]
        print(f"[feas] pad collider bottom z = {round(min(_pad_lo)*1000,1)} mm "
              f"(work surface at {args.table_h*1000:.0f} mm -> clearance "
              f"{round((min(_pad_lo)-args.table_h)*1000,1)} mm)", flush=True)

        # NOW stand the cup on the work surface under the settled pad centre,
        # with per-trial placement noise so the ten trials are independent.
        # Deterministic in the trial index -- reproducible, but not identical.
        import math as _math

        _jx = args.jitter * _math.sin(2.399963 * (trial + 1))
        _jy = args.jitter * _math.cos(2.399963 * (trial + 1))
        root = cup.data.default_root_state.clone()
        root[0, 0], root[0, 1] = float(pc[0]) + _jx, float(pc[1]) + _jy
        if args.free_air:
            root[0, 0] += 1.0
        root[0, 2] = args.table_h + CUP_H / 2 + 0.001
        root[0, 7:] = 0.0
        cup.write_root_pose_to_sim(root[:, :7])
        cup.write_root_velocity_to_sim(root[:, 7:])
        steps_k(90)
        cup0 = cup.data.root_pos_w[0].clone()

        cup_pre = cup.data.root_pos_w[0].clone()
        # "fits" now means what it should: the cup is actually BETWEEN the pads
        # when the close begins (within the 43.3 mm pad face, half-width 21.7 mm).
        pc_pre = pad_center_world()
        align = float((cup_pre - pc_pre)[:2].norm())
        fits = bool(align < 0.010)
        print(f"[feas] pre-close alignment: cup is {round(align*1000,1)} mm from "
              f"the pad centre in xy (jitter {round(_jx*1000,1)},{round(_jy*1000,1)} mm; "
              f"fits={fits})", flush=True)

        # close: ramp both fingers toward full travel (25 mm each)
        for k in range(120):
            jt[0, lf] = min(0.025, jt[0, lf] + 0.00025)
            jt[0, rf] = max(-0.025, jt[0, rf] - 0.00025)
            robot.set_joint_position_target(jt)
            steps_k(2)
        lq = float(robot.data.joint_pos[0, lf])
        rq = float(robot.data.joint_pos[0, rf])
        lgb = robot.body_names.index("left_gripper")
        rgb = robot.body_names.index("right_gripper")
        lp = robot.data.body_pos_w[0, lgb]
        rp = robot.data.body_pos_w[0, rgb]
        cp = cup.data.root_pos_w[0]
        print(f"[feas] close-end geometry: left_frame={[round(float(v)*1000,1) for v in lp]} "
              f"right_frame={[round(float(v)*1000,1) for v in rp]} "
              f"cup={[round(float(v)*1000,1) for v in cp]} (mm world)", flush=True)
        # expected stall at 25.0 - 20.0 = 5 mm travel per side for a
        # centered 40 mm cup
        symmetric = abs(lq - (-rq)) < 0.004
        contacted = (lq < 0.024) and (-rq < 0.024)  # stalled before full close
        cup_close = cup.data.root_pos_w[0].clone()
        ejected = bool((cup_close - cup0)[:2].norm() > 0.05)

        # WIDTH GATE (added 2026-08-04). `contacted` above is far too weak: it
        # only asks that the fingers stopped short of full close, which an
        # empty jam also satisfies. The first corrected run scored a bogus
        # 10/10 because trial 0 closed to 49.8 mm -- fingers effectively OPEN
        # -- scooped the cup on the way up, and passed on `lifted` alone.
        # A real grip on the 40 mm cup closes to the cup diameter; the other
        # nine trials all read 40.0-40.1 mm. Total travel is the honest
        # measure (the cup slides in y, so the per-finger split is uneven
        # while the SUM still equals the diameter), validated by Phase 5.
        closed_width_mm = 50.0 - lq * 1000 - (-rq * 1000)
        width_ok = bool(abs(closed_width_mm - 2 * CUP_R * 1000) <= 4.0)

        # FLOOR-CONTACT GATE. The finger links resting on the ground was the
        # source of this project's first false "asset is broken" verdict; a
        # grasp completed with the hand propped on the floor is not evidence
        # the gripper works. Fail the trial rather than quietly scoring it.
        # Proximity is NOT the fault. The pads hang ~35 mm below the finger
        # frames at the descended pose, so gripping a 50 mm cup leaves the
        # frames only ~3 mm above whatever surface it stands on -- that is
        # intrinsic geometry, not a defect, and thresholding on it produced a
        # false NEGATIVE (0/10 with every closed width at a correct 40.0 mm).
        # The real fault is the hand being SUPPORTED BY or driven THROUGH the
        # work surface, which is what created this project's first false
        # "asset is broken" verdict. Test that instead, and let the width gate
        # independently prove the stall is on the cup rather than the surface.
        min_frame_z = min(float(lp[2]), float(rp[2]))
        surface_clearance = min_frame_z - args.table_h
        floor_contact = bool(surface_clearance < 0.0)

        # Hand over to real dynamics now that the grip is established: from
        # here the cup must be held by friction, not by a teleporting hand.
        jt[:, _arm_ids] = robot.data.joint_pos[:, _arm_ids].clone()
        _lock["on"] = False
        robot.set_joint_position_target(jt)

        # hold 2 s
        steps_k(240)

        # lift: reverse joint2 ramp until the PAD CENTRE rises 12 cm. The cup
        # is a free dynamic body throughout -- if the grip cannot hold it, it
        # slips out here, which is the point of the test.
        z_start = float(pad_center_world()[2])
        for _ in range(3000):
            if float(pad_center_world()[2]) >= z_start + 0.12:
                break
            jt[0, j2] -= 0.0004 * sign
            steps_k(1)
        cup_lift = cup.data.root_pos_w[0].clone()
        lifted = bool(cup_lift[2] > cup0[2] + 0.05)
        print(f"[feas] after lift: pad centre z = "
              f"{round(float(pad_center_world()[2])*1000,1)} mm "
              f"(rose {round((float(pad_center_world()[2])-z_start)*1000,1)} mm); "
              f"cup z = {round(float(cup_lift[2])*1000,1)} mm "
              f"(started {round(float(cup0[2])*1000,1)}); fingers now "
              f"L{round(float(robot.data.joint_pos[0,lf])*1000,2)} "
              f"R{round(-float(robot.data.joint_pos[0,rf])*1000,2)} mm",
              flush=True)

        # transport: Joint1 yaw ramp 30 degrees
        for _ in range(300):
            jt[0, j1] += 0.0017
            steps_k(1)
        steps_k(240)  # hold 2 s
        cup_end = cup.data.root_pos_w[0].clone()
        held_through = bool(cup_end[2] > cup0[2] + 0.05)
        slipped = lifted and not held_through

        rec = dict(
            trial=trial,
            descended=descended,
            fits=fits,
            contacted=contacted,
            symmetric=symmetric,
            jitter_mm=[round(_jx * 1000, 2), round(_jy * 1000, 2)],
            pre_close_align_mm=round(align * 1000, 2),
            stall_left_mm=round(lq * 1000, 1),
            stall_right_mm=round(-rq * 1000, 1),
            closed_width_mm=round(closed_width_mm, 1),
            width_ok=width_ok,
            min_frame_z_mm=round(min_frame_z * 1000, 1),
            surface_clearance_mm=round(surface_clearance * 1000, 1),
            floor_contact=floor_contact,
            ejected=ejected,
            lifted=lifted,
            slipped_in_transport=slipped,
            success=bool(lifted and held_through and not ejected
                         and width_ok and not floor_contact),
        )
        results.append(rec)
        print(f"[feas] trial {trial}: {rec}", flush=True)

    n_ok = sum(1 for r in results if r["success"])
    print(f"[feas] ======== SUCCESS {n_ok}/{len(results)} "
          f"(asset={args.usd}) ========", flush=True)
    import json

    out = _REPO_ROOT / f"reports/synria_grasp_trials_{args.usd}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[feas] wrote {out}", flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
