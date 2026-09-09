"""Generate a hollow dice cup USD (Milestone B2 container).

An open beaker the arm can grasp, drop the die into, shake, and invert to dump.
Built as a watertight tube+bottom mesh (trimesh) written to USD with
convexDecomposition collision so the die is CONTAINED (not a convex-hull lid).

  ~/.venv/isaacsim5/bin/python isaac/usd/props/make_dice_cup.py
"""
from __future__ import annotations

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, Gf

OUT = "isaac/usd/props/dice_cup.usda"
R_OUT, R_IN = 0.021, 0.016      # 42mm outer (graspable), 32mm inner
H = 0.040                        # 40 mm tall
BOTTOM = 0.005                   # 5 mm floor


def build_mesh() -> trimesh.Trimesh:
    # wall: open tube (annulus swept) ; bottom: a short solid disk
    wall = trimesh.creation.annulus(r_min=R_IN, r_max=R_OUT, height=H)
    wall.apply_translation([0, 0, H / 2.0])
    bottom = trimesh.creation.cylinder(radius=R_OUT, height=BOTTOM, sections=48)
    bottom.apply_translation([0, 0, BOTTOM / 2.0])
    cup = trimesh.util.concatenate([wall, bottom])
    cup.remove_duplicate_faces() if hasattr(cup, "remove_duplicate_faces") else None
    return cup


def main() -> int:
    cup = build_mesh()
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/DiceCup")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/DiceCup/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in cup.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(cup.faces))
    mesh.CreateFaceVertexIndicesAttr(cup.faces.flatten().tolist())
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.35, 0.75)])  # blue cup
    mesh.CreateExtentAttr([Gf.Vec3f(*cup.bounds[0]), Gf.Vec3f(*cup.bounds[1])])

    # rigid body on the root, collision on the mesh with concave approximation
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(0.03)                          # 30 g cup
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mc.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)

    stage.GetRootLayer().Save()
    print(f"[cup] wrote {OUT}: {len(cup.vertices)} verts, {len(cup.faces)} faces, "
          f"watertight={cup.is_watertight}, bounds={cup.bounds.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
