"""Does the SPAWNED articulation actually carry finger collision shapes?

Queries the live PhysX view (not USD attributes) for per-link shape
counts, then drives a direct contact experiment: park a block exactly
between the pads and check whether closing produces any reaction force.
"""
import sys
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[2]

_REPO = _REPO_ROOT
sys.path.insert(0, str(_REPO))

from isaaclab.app import AppLauncher
import argparse

p = argparse.ArgumentParser()
p.add_argument("--usd", default="production")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.actuators import ImplicitActuatorCfg

    usd = {
        "production": "synria_6dof_arm.usd",
        "padfix2": "synria_6dof_arm_padfix2.usd",
    }[args.usd]
    sim = SimulationContext(SimulationCfg(dt=1 / 120))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_REPO / "isaac/usd/robots/synria_6dof_arm_v2" / usd),
                copy_from_source=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=50.0,
                    stiffness=400.0,
                    damping=40.0,
                )
            },
        )
    )
    blk = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Blk",
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.020, 0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.5)),
        )
    )
    sim.reset()
    robot.update(sim.get_physics_dt())

    view = robot.root_physx_view
    print("[probe] link names:", robot.body_names, flush=True)
    # PhysX view exposes per-body shape info in newer APIs; try several
    for attr in ("get_link_shape_count", "shape_count", "max_shapes"):
        if hasattr(view, attr):
            try:
                print(f"[probe] view.{attr} ->", getattr(view, attr)()
                      if callable(getattr(view, attr)) else getattr(view, attr),
                      flush=True)
            except Exception as e:
                print(f"[probe] view.{attr} raised {e}", flush=True)

    # authoritative: walk the SIMULATED stage under the spawned prim
    from pxr import Usd, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    n_col = 0
    for prim in stage.Traverse():
        pth = str(prim.GetPath())
        if pth.startswith("/World/Robot") and prim.HasAPI(UsdPhysics.CollisionAPI):
            en = prim.GetAttribute("physics:collisionEnabled")
            print(f"[probe] SIM collider: {pth} active={prim.IsActive()} "
                  f"enabled={en.Get() if en else 'unset'}", flush=True)
            n_col += 1
    print(f"[probe] total collision prims under spawned robot: {n_col}", flush=True)

    lf = robot.joint_names.index("left_finger")
    rf = robot.joint_names.index("right_finger")
    lgb = robot.body_names.index("left_gripper")
    rgb = robot.body_names.index("right_gripper")

    def step(n):
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            blk.update(sim.get_physics_dt())

    jt = robot.data.default_joint_pos.clone()
    jt[0, lf], jt[0, rf] = 0.0, 0.0
    robot.set_joint_position_target(jt)
    step(180)
    mid = 0.5 * (robot.data.body_pos_w[0, lgb] + robot.data.body_pos_w[0, rgb])
    root = blk.data.default_root_state.clone()
    root[0, :3] = mid
    root[0, 2] -= 0.030
    root[0, 7:] = 0.0
    blk.write_root_pose_to_sim(root[:, :7])
    blk.write_root_velocity_to_sim(root[:, 7:])
    step(60)
    print(f"[probe] KINEMATIC block pinned at {[round(float(v)*1000,1) for v in root[0,:3]]} mm; "
          f"finger frames L{[round(float(v)*1000,1) for v in robot.data.body_pos_w[0,lgb]]} "
          f"R{[round(float(v)*1000,1) for v in robot.data.body_pos_w[0,rgb]]}", flush=True)
    # close hard onto an IMMOVABLE block: any working collider must stall
    jt[0, lf], jt[0, rf] = 0.025, -0.025
    robot.set_joint_position_target(jt)
    step(300)
    print(f"[probe] after closing on immovable 20mm block: "
          f"L={float(robot.data.joint_pos[0,lf])*1000:.2f}mm "
          f"R={float(robot.data.joint_pos[0,rf])*1000:.2f}mm "
          f"(expect ~14.6mm each if colliders work; 25mm = no contact)",
          flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
finally:
    import os

    os._exit(0)
