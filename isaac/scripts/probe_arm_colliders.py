"""Runtime proof that the Synria ARM LINKS have live collision geometry.

Phase 6 context (2026-08-04). The audit recorded "arm links have no
collision geometry -- only 4 collision prims exist in the whole robot, all
on the fingers" and carried it as the last real asset defect. That was a
TRAVERSAL artefact: the arm links are `instanceable = True` (the production
wrapper sets `instanceable = false` only on the two grippers), so their
children live in a prototype and a plain Usd.Stage.Traverse() never
descends into them. Traversing with Usd.TraverseInstanceProxies() finds
**11** colliders, and base_link..link6 all report collisionEnabled=True.

USD says they are there. This script makes PHYSICS say it, which is the
only claim worth carrying forward -- the project's standing lesson is that
every "the asset is broken" verdict so far came from the test rig.

Method: drop a kinematic slab into link2's sweep path and command Joint2
through it. A live collider stalls the joint short of its target; a dead
one lets the link pass straight through. Runs the same test with the slab
parked far away as a control, so "it stalled" cannot be confused with
"it never got there".

    ~/.venv/isaacsim5/bin/python isaac/scripts/probe_arm_colliders.py --headless

Exit 0 = arm colliders verified live, 6 = arm passed through the obstacle.
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
    parser.add_argument("--usd", default="production")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
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
    import json

    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
    from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore
    from isaaclab.assets import RigidObject, RigidObjectCfg  # type: ignore
    from isaaclab.sim import SimulationCfg, SimulationContext  # type: ignore
    from pxr import Usd, UsdPhysics  # type: ignore

    usd_path = (_REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2"
                / "synria_6dof_arm.usd")

    # ---- static audit, this time descending into instance prototypes ------
    stage = Usd.Stage.Open(str(usd_path))
    found = []
    for p in stage.Traverse(Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            a = p.GetAttribute("physics:collisionEnabled")
            found.append((p.GetPath().pathString, bool(a.Get()) if a and a.IsValid() else True))
    print(f"[arm] USD colliders (with instance proxies): {len(found)}", flush=True)
    for path, en in found:
        print(f"[arm]    {'ON ' if en else 'off'} {path}", flush=True)
    arm_live = [p for p, en in found
                if en and any(f"/{l}/" in p for l in
                              ("base_link", "link1", "link2", "link3",
                               "link4", "link5", "link6"))]
    print(f"[arm] arm-link colliders reporting enabled: {len(arm_live)}", flush=True)

    # ---- runtime proof ----------------------------------------------------
    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/light", light)

    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path), copy_from_source=False),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["Joint[1-6]"],
                    effort_limit_sim=200.0, stiffness=2000.0, damping=200.0,
                ),
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=[".*_finger"],
                    effort_limit_sim=5.0, stiffness=2e3, damping=1e2,
                ),
            },
        )
    )
    # kinematic slab: infinite mass, so a live collider MUST stall the arm
    slab = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Slab",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.30, 0.06),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.8)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 0.15)),
        )
    )
    sim.reset()
    robot.update(sim.get_physics_dt())
    j2 = robot.joint_names.index("Joint2")

    def steps(n: int) -> None:
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            slab.update(sim.get_physics_dt())

    # Where does the arm actually GO? Placing an obstacle without measuring
    # the swept volume is exactly how the Phase 5 pedestal and the Phase 9
    # table ended up inside/outside the geometry they were meant to test.
    body_ids = {n: robot.body_names.index(n) for n in
                ("link1", "link2", "link3", "link4", "link5")}
    jp0 = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(jp0, torch.zeros_like(jp0))
    steps(60)
    start = {n: robot.data.body_pos_w[0, i].clone() for n, i in body_ids.items()}
    jt0 = jp0.clone()
    jt0[0, j2] = float(jp0[0, j2]) + 0.9
    robot.set_joint_position_target(jt0)
    steps(400)
    end = {n: robot.data.body_pos_w[0, i].clone() for n, i in body_ids.items()}
    print("[arm] link travel during a free +0.9 rad Joint2 sweep:", flush=True)
    best, best_d = None, 0.0
    for n in body_ids:
        d = float((end[n] - start[n]).norm())
        print(f"[arm]    {n}: {[round(float(v)*1000,1) for v in start[n]]} -> "
              f"{[round(float(v)*1000,1) for v in end[n]]} mm  (moved {d*1000:.1f} mm)",
              flush=True)
        if d > best_d:
            best, best_d = n, d
    mid = 0.5 * (start[best] + end[best])
    print(f"[arm] obstacle target: {best} mid-sweep at "
          f"{[round(float(v)*1000,1) for v in mid]} mm", flush=True)

    def sweep(slab_x: float, label: str) -> float:
        """Command Joint2 to +0.9 rad; return the angle actually reached."""
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        st = slab.data.default_root_state.clone()
        if slab_x > 1.0:
            st[0, 0], st[0, 1], st[0, 2] = slab_x, 0.0, 0.15
        else:
            st[0, 0], st[0, 1], st[0, 2] = (float(mid[0]), float(mid[1]),
                                            float(mid[2]))
        st[0, 7:] = 0.0
        slab.write_root_pose_to_sim(st[:, :7])
        slab.write_root_velocity_to_sim(st[:, 7:])
        steps(60)
        jt = jp.clone()
        jt[0, j2] = float(jp[0, j2]) + 0.9
        robot.set_joint_position_target(jt)
        steps(400)
        reached = float(robot.data.joint_pos[0, j2]) - float(jp[0, j2])
        print(f"[arm] {label}: Joint2 commanded +0.900 rad, reached "
              f"{reached:+.3f} rad", flush=True)
        return reached

    free = sweep(5.0, "CONTROL  (slab parked 5 m away)")
    blocked = sweep(0.0, f"OBSTACLE (slab at measured {best} mid-sweep)")

    stalled = blocked < free - 0.15
    print(f"[arm] control {free:+.3f} vs blocked {blocked:+.3f} rad "
          f"-> arm links {'STALLED on the obstacle' if stalled else 'PASSED THROUGH'}",
          flush=True)
    verdict = bool(stalled and len(arm_live) >= 7)
    print(f"[arm] ======== ARM COLLIDERS {'LIVE' if verdict else 'NOT VERIFIED'} "
          f"========", flush=True)

    out = _REPO_ROOT / "reports/synria_arm_collider_probe.json"
    out.write_text(json.dumps({
        "usd_colliders_total": len(found),
        "arm_link_colliders_enabled": len(arm_live),
        "control_rad": round(free, 4),
        "blocked_rad": round(blocked, 4),
        "stalled_on_obstacle": stalled,
        "verdict_live": verdict,
    }, indent=1))
    print(f"[arm] wrote {out}", flush=True)
    os._exit(0 if verdict else 6)


if __name__ == "__main__":
    raise SystemExit(main())
