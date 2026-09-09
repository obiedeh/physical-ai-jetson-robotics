#!/usr/bin/env python3
"""Patch the v2 arm USD: box pad colliders + high-friction material on fingers.

Why: the URDF import gives every link a convex-hull collider. Hulls of the
curved finger meshes have ANGLED inner faces, so squeezing the 40 mm piece
converts clamp force into an ejection vector — diag_grasp_feasibility.py
measured the piece flying 201 mm out of the closing gripper. Parallel-jaw
grippers in sim standardly use flat box pad colliders instead.

This script (pure pxr, no Kit):
  1. De-instances the two finger subtrees so they can be edited.
  2. Disables the STL hull colliders on left_gripper / right_gripper only.
  3. Adds an invisible box collider matching each pad's AABB (flat, parallel
     inner faces).
  4. Binds a high-friction, zero-restitution physics material to the pads.

Usage:
    ~/.venv/isaacsim5/bin/python isaac/scripts/patch_v2_finger_pads.py
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

USD_PATH = (
    Path(__file__).resolve().parents[2]
    / "isaac" / "usd" / "robots" / "synria_6dof_arm_v2" / "synria_6dof_arm.usd"
)
ROOT = "/Alicia_D_v5_6_gripper_50mm"


def main() -> int:
    stage = Usd.Stage.Open(str(USD_PATH))

    # -- physics material (shared by both pads) ---------------------------
    mat_path = f"{ROOT}/PadPhysicsMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pmat.CreateStaticFrictionAttr(1.2)
    pmat.CreateDynamicFrictionAttr(1.0)
    pmat.CreateRestitutionAttr(0.0)

    for side in ("left", "right"):
        body_path = f"{ROOT}/{side}_gripper"
        body = stage.GetPrimAtPath(body_path)
        if not body:
            print(f"[patch] FAIL: {body_path} not found")
            return 1

        # 1. de-instance anything instanceable under the finger
        for prim in Usd.PrimRange(body):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        # 2. compute the pad AABB in the finger's local frame, then disable
        #    the hull collider
        bcache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "guide", "proxy", "render"]
        )
        hull = None
        for prim in Usd.PrimRange(body):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetName() == "node_STL_BINARY_":
                hull = prim
                break
        if hull is None:
            print(f"[patch] FAIL: no hull collider under {body_path}")
            return 1

        # bounds directly in the finger-body frame (transforming just the two
        # world AABB corners through the body rotation collapses axes)
        rel = bcache.ComputeRelativeBound(hull, body).ComputeAlignedRange()
        lo, hi = rel.GetMin(), rel.GetMax()
        center = Gf.Vec3d((lo + hi) / 2.0)
        extents = Gf.Vec3d(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
        if min(extents) < 1e-4:
            print(f"[patch] FAIL: degenerate pad extents {extents} for {side}")
            return 1

        UsdPhysics.CollisionAPI(hull).CreateCollisionEnabledAttr(False)

        # 3. flat box pad collider
        pad_path = f"{body_path}/pad_collider"
        pad = UsdGeom.Cube.Define(stage, pad_path)
        pad.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(pad.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(center)
        xf.AddScaleOp().Set(Gf.Vec3f(extents))
        UsdGeom.Imageable(pad.GetPrim()).CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())

        # 4. bind the high-friction material (physics purpose)
        binding = UsdShade.MaterialBindingAPI.Apply(pad.GetPrim())
        binding.Bind(mat, materialPurpose="physics")

        print(f"[patch] {side}: hull disabled; box pad centre={[round(v, 4) for v in center]} "
              f"extents={[round(v, 4) for v in extents]}")

    stage.GetRootLayer().Save()
    print(f"[patch] saved {USD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
