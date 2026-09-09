"""R3: compose Synria arm (fingers removed) + Robotiq 2F-85 at tool0."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app
try:
    import carb.settings
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    import shutil

    src = "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm_g30.usd"
    dst = "isaac/usd/robots/synria_6dof_arm_v2/synria_arm_robotiq.usd"
    shutil.copy2(src, dst)
    stage = Usd.Stage.Open(dst)
    root = stage.GetDefaultPrim()
    rp = root.GetPath().pathString
    print("[r3] root:", rp, flush=True)

    # 1. remove Synria fingers (links + their joints live under the links)
    removed = []
    for prim in list(stage.Traverse()):
        name = prim.GetName().lower()
        if ("gripper" in name) and prim.GetParent() == root:
            removed.append(prim.GetPath().pathString)
    for path in removed:
        stage.RemovePrim(path)
    print("[r3] removed:", removed, flush=True)
    # also remove finger joints authored elsewhere (search by name)
    for prim in list(stage.Traverse()):
        n = prim.GetName()
        if n in ("left_finger", "right_finger") and "Joint" in prim.GetTypeName():
            print("[r3] removing joint", prim.GetPath(), flush=True)
            stage.RemovePrim(prim.GetPath())

    # 2. reference the Robotiq subtree under the articulation root
    cloud = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    franka_url = f"{cloud}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    grip_path = Sdf.Path("/panda/Robotiq_2F_85_edit/Robotiq_2F_85")
    grip = stage.DefinePrim(f"{rp}/Robotiq_2F_85", "Xform")
    grip.GetReferences().AddReference(franka_url, grip_path)

    # find tool0 world-frame transform to place the gripper
    tool0 = stage.GetPrimAtPath(f"{rp}/tool0")
    print("[r3] tool0 valid:", bool(tool0), flush=True)
    xf = UsdGeom.Xformable(grip)
    cache = UsdGeom.XformCache()
    m = cache.GetLocalToWorldTransform(tool0)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(m))

    # 3. fixed joint tool0 -> gripper base_link
    base_link = stage.GetPrimAtPath(f"{rp}/Robotiq_2F_85/base_link")
    print("[r3] gripper base_link composed:", bool(base_link), flush=True)
    fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{rp}/tool0/robotiq_mount"))
    fj.CreateBody0Rel().SetTargets([Sdf.Path(f"{rp}/tool0")])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(f"{rp}/Robotiq_2F_85/base_link")])
    stage.Save()
    print("[r3] saved", dst, flush=True)

    # 4. verify composition: list gripper joints
    stage2 = Usd.Stage.Open(dst)
    joints = [p.GetPath().pathString.split("/")[-1] for p in stage2.Traverse()
              if "Joint" in p.GetTypeName() and "Fixed" not in p.GetTypeName()]
    print("[r3] movable joints in composed asset:", joints, flush=True)
finally:
    app.close()
