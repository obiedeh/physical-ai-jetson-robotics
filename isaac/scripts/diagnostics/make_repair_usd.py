"""Phase 2/3 repair layer: deactivate the corrupt finger MESH colliders.

Keeps the (properly parented, visually verified) pad box colliders as the
sole finger collision geometry. Original vendor layers untouched — this
is a thin override layer referencing synria_6dof_arm.usd.
"""
from pxr import Usd
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[3]

root = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2"
dst = root / "synria_6dof_arm_repaired.usd"
if dst.exists():
    dst.unlink()
new = Usd.Stage.CreateNew(str(dst))
ref = new.OverridePrim("/Alicia_D_v5_6_gripper_50mm")
ref.GetReferences().AddReference("./synria_6dof_arm.usd")
new.SetDefaultPrim(ref)
for side in ("left", "right"):
    over = new.OverridePrim(
        f"/Alicia_D_v5_6_gripper_50mm/{side}_gripper/collisions/{side}_gripper_50mm"
    )
    over.SetActive(False)
new.GetRootLayer().Save()
print("wrote", dst)
