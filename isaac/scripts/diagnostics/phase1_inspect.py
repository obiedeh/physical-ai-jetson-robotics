"""Phase 1: authoritative USD composition + transform inspection."""
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[3]
import json

root = _REPO_ROOT / "isaac/usd/robots/synria_6dof_arm_v2"
stage = Usd.Stage.Open(str(root / "synria_6dof_arm.usd"))
cache = UsdGeom.XformCache(Usd.TimeCode.Default())

report = {}
PATHS = {
    "left_link": "/Alicia_D_v5_6_gripper_50mm/left_gripper",
    "right_link": "/Alicia_D_v5_6_gripper_50mm/right_gripper",
    "left_visual": "/Alicia_D_v5_6_gripper_50mm/left_gripper/visuals",
    "right_visual": "/Alicia_D_v5_6_gripper_50mm/right_gripper/visuals",
    "left_pad_old": "/Alicia_D_v5_6_gripper_50mm/left_gripper/pad_collider",
    "right_pad_old": "/Alicia_D_v5_6_gripper_50mm/right_gripper/pad_collider",
    "left_joint": "/Alicia_D_v5_6_gripper_50mm/joints/left_finger",
    "right_joint": "/Alicia_D_v5_6_gripper_50mm/joints/right_finger",
    "link6": "/Alicia_D_v5_6_gripper_50mm/link6",
}

def mat_to_list(m):
    return [[round(m[i][j], 6) for j in range(4)] for i in range(4)]

for name, path in PATHS.items():
    prim = stage.GetPrimAtPath(path)
    if not prim:
        report[name] = "MISSING"
        continue
    entry = {"path": path, "type": prim.GetTypeName()}
    if prim.IsA(UsdGeom.Xformable):
        world = cache.GetLocalToWorldTransform(prim)
        entry["world_translate_mm"] = [round(v * 1000, 2) for v in world.ExtractTranslation()]
        entry["world_matrix"] = mat_to_list(world)
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        entry["xform_ops"] = [(o.GetOpName(), str(o.Get())) for o in ops]
    # layer authorship (which layer authors xformOp / collision)
    stack_layers = [s.layer.identifier.split("/")[-1] for s in prim.GetPrimStack()]
    entry["prim_stack_layers"] = stack_layers
    # visual mesh bbox in world
    if "visual" in name:
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default")
        r = bbox.ComputeAlignedRange()
        entry["world_bbox_mm"] = {
            "min": [round(v * 1000, 1) for v in r.GetMin()],
            "max": [round(v * 1000, 1) for v in r.GetMax()],
        }
    if "joint" in name:
        j = UsdPhysics.PrismaticJoint(prim)
        entry["axis"] = str(j.GetAxisAttr().Get())
        entry["limits_mm"] = [round((j.GetLowerLimitAttr().Get() or 0) * 1000, 2),
                              round((j.GetUpperLimitAttr().Get() or 0) * 1000, 2)]
        entry["body0"] = [str(t) for t in (prim.GetRelationship("physics:body0").GetTargets() or [])]
        entry["body1"] = [str(t) for t in (prim.GetRelationship("physics:body1").GetTargets() or [])]
        for attr in ("physics:localPos0", "physics:localPos1",
                     "physics:localRot0", "physics:localRot1"):
            a = prim.GetAttribute(attr)
            if a and a.Get() is not None:
                entry[attr] = str(a.Get())
    if "pad_old" in name:
        prim_t = prim.GetTypeName()
        entry["active"] = prim.IsActive()
        entry["attrs"] = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                         if a.GetName().startswith(("size", "xformOp", "physics"))
                         and a.Get() is not None}
    report[name] = entry

out = _REPO_ROOT / "reports/gripper_phase1_composition.json"
out.write_text(json.dumps(report, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "world_matrix"}
                  if isinstance(v, dict) else v for k, v in report.items()}, indent=1))
print("wrote", out)
