from pxr import Usd, UsdGeom, UsdPhysics, Gf
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[3]
stage = Usd.Stage.Open(str(_REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm.usd"))
cache = UsdGeom.XformCache()
print("=== prims with physics collision ===")
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        path = str(prim.GetPath())
        # world transform + extent
        try:
            bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default")
            box = bbox.ComputeAlignedRange()
            mn, mx = box.GetMin(), box.GetMax()
            approx = prim.GetAttribute("physics:approximation").Get() if prim.GetAttribute("physics:approximation") else None
            print(f"{path}\n    approx={approx} bbox_mm min={[round(v*1000,1) for v in mn]} max={[round(v*1000,1) for v in mx]}")
        except Exception as e:
            print(f"{path}  (bbox err {e})")
