import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app
try:
    import carb.settings
    from pxr import Usd, UsdPhysics
    cloud = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
    st = Usd.Stage.Open(f"{cloud}/Isaac/Robots/Robotiq/2F-85/configuration/Robotiq_2F_85_config.usd")
    dp = st.GetDefaultPrim()
    print("[i] default:", dp.GetPath() if dp else "NONE", flush=True)
    if not dp:
        for r in st.GetPseudoRoot().GetChildren():
            print("[i] root child:", r.GetPath(), r.GetTypeName(), flush=True)
        dp = st.GetPseudoRoot().GetChildren()[0]
    for name in dp.GetVariantSets().GetNames():
        v = dp.GetVariantSets().GetVariantSet(name)
        print("[i] variantSet", name, v.GetVariantNames(), "sel:", v.GetVariantSelection())
    ncol = 0
    for p in st.Traverse():
        if p.GetPath().pathString.count("/") <= 2:
            print("[i]", p.GetPath(), p.GetTypeName(),
                  "ROOT" if p.HasAPI(UsdPhysics.ArticulationRootAPI) else "")
        if p.HasAPI(UsdPhysics.CollisionAPI):
            ncol += 1
    print("[i] collision prims:", ncol)
except BaseException:
    import traceback, sys
    traceback.print_exc(); sys.stdout.flush()
finally:
    app.close()
