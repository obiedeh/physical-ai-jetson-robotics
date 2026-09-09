"""R3 v2: flatten the variant-selected 2F-85 to a local USD, compose hybrid."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app
try:
    import carb.settings
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
    import shutil

    # 1. flatten variant-selected gripper subtree to a local file
    cloud = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    fstage = Usd.Stage.Open(f"{cloud}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")
    fstage.GetDefaultPrim().GetVariantSets().GetVariantSet("Gripper").SetVariantSelection("Robotiq_2F_85")
    flat = fstage.Flatten()
    grip_file = "isaac/usd/robots/synria_6dof_arm_v2/robotiq_2f85_flat.usd"
    out = Sdf.Layer.CreateNew(grip_file)
    Sdf.CopySpec(flat, Sdf.Path("/panda/Robotiq_2F_85_edit/Robotiq_2F_85"),
                 out, Sdf.Path("/Robotiq_2F_85"))
    # instanced geometry lives in layer-root prototypes — copy them too
    for spec in flat.rootPrims:
        if spec.name.startswith("Flattened_Prototype"):
            Sdf.CopySpec(flat, Sdf.Path(f"/{spec.name}"), out, Sdf.Path(f"/{spec.name}"))
            print("[r3] copied prototype", spec.name, flush=True)
    # dangling joint to /panda/panda_hand — poison; drop it
    asm = out.GetPrimAtPath("/Robotiq_2F_85/base_link/AssemblerFixedJoint")
    if asm:
        out.ScheduleRemoveIfInert(asm)
    ja = Sdf.Path("/Robotiq_2F_85/base_link/AssemblerFixedJoint")
    if out.GetPrimAtPath(ja):
        import pxr
        pxr.Sdf.CreatePrimInLayer(out, ja)
        out.GetPrimAtPath(ja).active = False
    out.defaultPrim = "Robotiq_2F_85"
    out.Save()
    g = Usd.Stage.Open(grip_file)
    gj = [p.GetName() for p in g.Traverse() if "Joint" in p.GetTypeName()]
    print("[r3] flat gripper joints:", gj, flush=True)

    # 2. hybrid: copy G30 arm, deactivate Synria fingers, reference gripper
    src = "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm_g30.usd"
    dst = "isaac/usd/robots/synria_6dof_arm_v2/synria_arm_robotiq.usd"
    shutil.copy2(src, dst)
    stage = Usd.Stage.Open(dst)
    root = stage.GetDefaultPrim()
    rp = root.GetPath().pathString
    for path in (f"{rp}/left_gripper", f"{rp}/right_gripper",
                 f"{rp}/joints/left_finger", f"{rp}/joints/right_finger"):
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.SetActive(False)
            print("[r3] deactivated", path, flush=True)

    grip = stage.DefinePrim(f"{rp}/Robotiq_2F_85", "Xform")
    grip.GetReferences().AddReference("./robotiq_2f85_flat.usd")
    tool0 = stage.GetPrimAtPath(f"{rp}/tool0")
    cache = UsdGeom.XformCache()
    m = cache.GetLocalToWorldTransform(tool0)
    xf = UsdGeom.Xformable(grip)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(m))
    fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{rp}/link6/robotiq_mount"))
    fj.CreateBody0Rel().SetTargets([Sdf.Path(f"{rp}/link6")])
    fj.CreateBody1Rel().SetTargets([Sdf.Path(f"{rp}/Robotiq_2F_85/base_link")])
    stage.Save()

    s2 = Usd.Stage.Open(dst)
    joints = [p.GetName() for p in s2.Traverse()
              if "Joint" in p.GetTypeName() and "Fixed" not in p.GetTypeName()
              and p.IsActive()]
    print("[r3] hybrid movable joints:", joints, flush=True)
finally:
    app.close()
