from pxr import Usd, UsdGeom, UsdPhysics
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[3]
stage = Usd.Stage.Open(str(_REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm.usd"))
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print("BODY:", prim.GetPath())
for prim in stage.Traverse():
    if "gripper" in str(prim.GetPath()).lower():
        t = prim.GetTypeName()
        apis = [s.split(":")[0] for s in (prim.GetAppliedSchemas() or [])]
        print(f"{prim.GetPath()}  type={t} apis={apis}")
