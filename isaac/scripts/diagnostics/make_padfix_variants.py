from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[3]

root = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2"

SPECS = {"left": (0.0, -0.030, -0.0015), "right": (0.0, 0.030, 0.0015)}
SIZE = (0.0433, 0.060, 0.004)


def build(name: str, sides, explicit_offsets: bool) -> None:
    dst = root / name
    if dst.exists():
        dst.unlink()
    new = Usd.Stage.CreateNew(str(dst))
    ref = new.OverridePrim("/Alicia_D_v5_6_gripper_50mm")
    ref.GetReferences().AddReference("./synria_6dof_arm.usd")
    new.SetDefaultPrim(ref)
    for side in ("left", "right"):
        over = new.OverridePrim(
            f"/Alicia_D_v5_6_gripper_50mm/{side}_gripper/pad_collider"
        )
        over.SetActive(False)
        over2 = new.OverridePrim(
            f"/Alicia_D_v5_6_gripper_50mm/{side}_gripper/collisions/{side}_gripper_50mm"
        )
        over2.SetActive(False)
    for side in sides:
        path = f"/Alicia_D_v5_6_gripper_50mm/{side}_gripper/pad_fix"
        cube = UsdGeom.Cube.Define(new, path)
        cube.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*SPECS[side]))
        xf.AddScaleOp().Set(Gf.Vec3f(*SIZE))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if explicit_offsets:
            prim = cube.GetPrim()
            prim.CreateAttribute(
                "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
            ).Set(0.001)
            prim.CreateAttribute(
                "physxCollision:restOffset", Sdf.ValueTypeNames.Float
            ).Set(0.0)
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    new.GetRootLayer().Save()
    print("wrote", dst)


build("synria_6dof_arm_padfix_l.usd", ("left",), False)
build("synria_6dof_arm_padfix2.usd", ("left", "right"), True)
