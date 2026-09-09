"""R5: proper assembly via isaacsim.robot_setup.assembler."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app
try:
    import traceback as _tb
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.robot_setup.assembler")
    import carb.settings
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.robot_setup.assembler import RobotAssembler
    from pxr import Usd, UsdPhysics, Sdf
    import shutil

    # finger-free Synria arm
    src = "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm_g30.usd"
    arm_usd = "isaac/usd/robots/synria_6dof_arm_v2/synria_arm_nogripper.usd"
    shutil.copy2(src, arm_usd)
    st = Usd.Stage.Open(arm_usd)
    rp = st.GetDefaultPrim().GetPath().pathString
    for path in (f"{rp}/left_gripper", f"{rp}/right_gripper",
                 f"{rp}/joints/left_finger", f"{rp}/joints/right_finger"):
        prim = st.GetPrimAtPath(path)
        if prim:
            prim.SetActive(False)
    st.Save()

    cloud = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    grip_url = f"{cloud}/Isaac/Robots/Robotiq/2F-85/configuration/Robotiq_2F_85_config.usd"

    import os
    add_reference_to_stage(os.path.abspath(arm_usd), "/World/arm")
    add_reference_to_stage(grip_url, "/World/gripper")
    stage = get_current_stage()
    # discover gripper root structure
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.startswith("/World/gripper") and p.count("/") <= 3:
            print("[r5]", p, prim.GetTypeName(), flush=True)

    ra = RobotAssembler()
    assembled = ra.assemble_rigid_bodies(
        base_path="/World/arm",
        attach_path="/World/gripper/Robotiq_2F_85",
        base_mount_frame="/World/arm/tool0",
        attach_mount_frame="/World/gripper/Robotiq_2F_85/base_link",
        mask_all_collisions=True,
    )
    print("[r5] assembled:", assembled, flush=True)

    # single articulation: strip the gripper's own articulation root
    from pxr import UsdPhysics as _UP
    stage = get_current_stage()
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.startswith("/World/gripper") and prim.HasAPI(_UP.ArticulationRootAPI):
            prim.RemoveAPI(_UP.ArticulationRootAPI)
            print("[r5] stripped ArticulationRootAPI:", p, flush=True)

    out = "isaac/usd/robots/synria_6dof_arm_v2/synria_robotiq_assembled.usd"
    stage.Export(out)
    print("[r5] exported", out, flush=True)

    s2 = Usd.Stage.Open(out)
    joints = [p.GetName() for p in s2.Traverse()
              if "Joint" in p.GetTypeName() and "Fixed" not in p.GetTypeName()
              and p.IsActive()]
    print("[r5] movable joints:", joints, flush=True)
except BaseException:
    import sys
    _tb.print_exc()
    sys.stdout.flush()
finally:
    app.close()
