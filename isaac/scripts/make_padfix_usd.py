from pxr import Usd, UsdGeom, UsdPhysics, Gf
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[2]

root = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2"
dst = root / "synria_6dof_arm_padfix.usd"
if dst.exists():
    dst.unlink()
new = Usd.Stage.CreateNew(str(dst))
ref = new.OverridePrim("/Alicia_D_v5_6_gripper_50mm")
ref.GetReferences().AddReference("./synria_6dof_arm.usd")
new.SetDefaultPrim(ref)
for side in ("left", "right"):
    over = new.OverridePrim(f"/Alicia_D_v5_6_gripper_50mm/{side}_gripper/pad_collider")
    over.SetActive(False)
# correctly-parented box pads in each finger link's LOCAL frame; mesh inner
# faces at local z = +0.5mm (left) / -0.5mm (right); boxes 4mm thick ending
# flush with those faces so the 50.0mm opening is preserved.
specs = {
    "left": (0.0, -0.030, -0.0015),
    "right": (0.0, 0.030, 0.0015),
}
size = (0.0433, 0.060, 0.004)
for side, center in specs.items():
    path = f"/Alicia_D_v5_6_gripper_50mm/{side}_gripper/pad_fix"
    cube = UsdGeom.Cube.Define(new, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
new.GetRootLayer().Save()
print("wrote", dst)
