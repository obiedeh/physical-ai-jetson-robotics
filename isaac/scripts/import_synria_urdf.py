#!/usr/bin/env python3
"""Convert synria_6dof_arm.urdf → USD using Isaac Sim's URDF importer API.

Output: isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.usda
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "isaac" / "usd" / "robots" / "synria_6dof_arm.urdf"
DEFAULT_OUT_DIR = REPO_ROOT / "isaac" / "usd" / "robots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="URDF → USD conversion for the Synria 6DOF arm.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--collision-type",
        default="Convex Hull",
        choices=["Convex Hull", "Convex Decomposition", "Bounding Sphere", "Bounding Cube"],
    )
    parser.add_argument("--merge-mesh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urdf_path = args.urdf.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "extra_args": ["--/renderer/multiGpu/autoEnable=false"],
        }
    )

    try:
        import sys
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        for ext in (
            "omni.usd.schema.newton",
            "isaacsim.asset.importer.urdf",
            "isaacsim.asset.transformer.rules",
        ):
            if not ext_manager.is_extension_enabled(ext):
                ext_manager.set_extension_enabled_immediate(ext, True)

        # newton_usd_schemas is a sub-package of usd.schema.newton but is imported
        # as a top-level module by urdf_usd_converter.  Kit only adds the extension
        # root to sys.path; we must also add the sub-directory that contains it.
        newton_ext_path = ext_manager.get_extension_path(
            ext_manager.get_enabled_extension_id("omni.usd.schema.newton")
        )
        newton_python_path = str(Path(newton_ext_path) / "usd" / "schema" / "newton")
        if newton_python_path not in sys.path:
            sys.path.insert(0, newton_python_path)

        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        config = URDFImporterConfig(
            urdf_path=str(urdf_path),
            usd_path=str(out_dir),
            merge_mesh=args.merge_mesh,
            collision_type=args.collision_type,
            collision_from_visuals=True,
        )
        importer = URDFImporter(config)
        final_path = importer.import_urdf()

        print(f"URDF  : {urdf_path}")
        print(f"USD   : {final_path}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
