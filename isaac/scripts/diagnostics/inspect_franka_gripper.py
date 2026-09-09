import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app
try:
    import carb.settings
    from pxr import Usd
    root = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    url = f"{root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    print("[insp] url:", url, flush=True)
    stage = Usd.Stage.Open(url)
    dp = stage.GetDefaultPrim()
    print("[insp] default prim:", dp.GetPath(), flush=True)
    vs = dp.GetVariantSets()
    for name in vs.GetNames():
        v = vs.GetVariantSet(name)
        print(f"[insp] variantSet {name}: {v.GetVariantNames()} (sel={v.GetVariantSelection()})", flush=True)
    if "Gripper" in vs.GetNames():
        vs.GetVariantSet("Gripper").SetVariantSelection("Robotiq_2F_85")
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if p.count("/") <= 3:
            print("[insp]", p, prim.GetTypeName(), flush=True)
finally:
    app.close()
