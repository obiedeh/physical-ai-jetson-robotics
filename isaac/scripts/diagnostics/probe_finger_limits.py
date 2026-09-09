import sys
from pathlib import Path
_REPO_ROOT = _REPO_ROOT
sys.path.insert(0, str(_REPO_ROOT))
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.actuators import ImplicitActuatorCfg
    sim = SimulationContext(SimulationCfg(dt=1/120))
    g = sim_utils.GroundPlaneCfg(); g.func("/World/ground", g)
    robot = Articulation(ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_REPO_ROOT/"isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm.usd"),
            copy_from_source=False),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"],
                   effort_limit_sim=50.0, stiffness=2e3, damping=1e2)}))
    sim.reset(); robot.update(sim.get_physics_dt())
    lf = robot.joint_names.index("left_finger"); rf = robot.joint_names.index("right_finger")
    lim = robot.data.joint_pos_limits[0]
    print("[probe] left_finger limits:", lim[lf].tolist())
    print("[probe] right_finger limits:", lim[rf].tolist())
    print("[probe] default:", float(robot.data.default_joint_pos[0, lf]),
          float(robot.data.default_joint_pos[0, rf]))
    # teleport joint state to 20mm travel — physics clamps back if limited
    jp = robot.data.default_joint_pos.clone(); jp[0, lf] = 0.020; jp[0, rf] = -0.020
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    robot.set_joint_position_target(jp)
    for _ in range(60):
        robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
    print("[probe] after teleport to 20mm:",
          round(float(robot.data.joint_pos[0, lf])*1000, 2), "mm /",
          round(float(robot.data.joint_pos[0, rf])*1000, 2), "mm")
    # drive from 0 with strong effort
    jp[0, lf] = 0.0; jp[0, rf] = 0.0
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    tgt = jp.clone(); tgt[0, lf] = 0.025; tgt[0, rf] = -0.025
    robot.set_joint_position_target(tgt)
    for _ in range(240):
        robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
    print("[probe] drive 50N from 0 to 25mm ->",
          round(float(robot.data.joint_pos[0, lf])*1000, 2), "mm /",
          round(float(robot.data.joint_pos[0, rf])*1000, 2), "mm")
except BaseException:
    import traceback; traceback.print_exc(); sys.stdout.flush()
finally:
    import os; os._exit(0)
