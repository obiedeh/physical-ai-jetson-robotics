#!/usr/bin/env python3
"""Convert synria_6dof_arm.urdf → USD with Isaac Sim 5.1's classic URDF importer.

Why this exists: the original conversion (import_synria_urdf.py) used the
Isaac Sim 6.x converter (omni.usd.schema.newton / URDFImporter), which is not
available in the 5.1 install this project runs on — and its output shipped
with a collision payload from a DIFFERENT robot (UR-style link names), so no
collider ever bound to the Alicia-D links and the gripper could not touch
anything (diag_grasp_feasibility.py: fingers closed to 10 mm through a 40 mm
piece).

The classic importer emits a single self-contained USD with convex-hull
colliders generated from the URDF <collision> meshes, correct joint names
(Joint1..6, left_finger, right_finger), and a proper fixed-base articulation
root.

Usage:
    $ISAAC_PYTHON isaac/scripts/import_synria_urdf_51.py \
        [--urdf PATH] [--out PATH.usd]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "isaac" / "usd" / "robots" / "synria_6dof_arm.urdf"
DEFAULT_OUT = (
    REPO_ROOT / "isaac" / "usd" / "robots" / "synria_6dof_arm_v2" / "synria_6dof_arm.usd"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    urdf_path = args.urdf.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Kit's fast shutdown can hard-exit before Python flushes stdout, so all
    # diagnostics also go to a sentinel file written & flushed before close.
    sentinel = out_path.parent / "import_result.txt"

    def note(msg: str) -> None:
        print(msg, flush=True)
        with sentinel.open("a") as fh:
            fh.write(msg + "\n")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import traceback

        try:
            import omni.kit.app
            import omni.kit.commands

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            if not ext_manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
                ext_manager.set_extension_enabled_immediate(
                    "isaacsim.asset.importer.urdf", True
                )
            for _ in range(3):  # let the extension finish loading
                app.update()

            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            if not status or import_config is None:
                note("[import] FAIL: URDFCreateImportConfig")
                return 1

            import_config.merge_fixed_joints = False  # keep tool0 as its own frame
            import_config.convex_decomp = False       # convex hull per collision mesh
            import_config.fix_base = True              # proper fixed-base articulation
            import_config.make_default_prim = True
            import_config.self_collision = False
            import_config.distance_scale = 1.0
            import_config.density = 0.0                # use URDF masses/inertia

            status, prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(urdf_path),
                import_config=import_config,
                dest_path=str(out_path),
            )
            if not status:
                note("[import] FAIL: URDFParseAndImportFile")
                return 1

            note(f"[import] URDF : {urdf_path}")
            note(f"[import] USD  : {out_path}")
            note(f"[import] prim : {prim_path}")
            note(f"[import] file exists: {out_path.exists()}")
            return 0
        except Exception:
            note("[import] EXCEPTION:\n" + traceback.format_exc())
            return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
