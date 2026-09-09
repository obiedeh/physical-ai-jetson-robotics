#!/usr/bin/env python3
"""Interactive Isaac Sim viewer for the Alicia-D no-robot tabletop scenes.

Three game scenes (ludo, chess, checkers) × optional Synria arm mount.
When the arm is enabled it's referenced into ``/World/SynriaArm`` at
``(0.1524, 0.0, 0.8382)`` — 6 inches from table centre on world +X —
with a 90° Z rotation so the arm's URDF forward (+X) points along world
+Y toward ``/World/Sensors/FrontCamera``. The scenes are the board-
mirrored ``*_left`` variants (board already shifted to the +X table end),
shared with the training env. The arm boots at a factory-reset home pose
and its drives actively hold torque (see ARM_STIFFNESS/ARM_DAMPING).

This sits next to ``scripts/run_synria_stack.sh``, which exposes the six
combinations through an interactive menu / CLI. The training pipeline
(``scripts/linux_rtx/train_isaaclab_synria.sh``) consumes the same
position from ``isaac/isaaclab_tasks/synria_pickplace/env_cfg.py`` so the
viewer and the trained env stay aligned.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
# Board-mirrored ("left") scene USDs — board already shifted to the +X table
# end, shared with the training env (isaac/isaaclab_tasks/synria_pickplace/
# env_cfg.py). Regenerate via isaac/usd/scenes/synria_mirrored/
# generate_mirrored_scenes.py.
MIRRORED_SCENES_DIR = REPO_ROOT / "isaac" / "usd" / "scenes" / "synria_mirrored"
SCENE_USD = {
    "ludo": MIRRORED_SCENES_DIR / "ludotable_norobot_left.usda",
    "chess": MIRRORED_SCENES_DIR / "chesstable_norobot_left.usda",
    "checkers": MIRRORED_SCENES_DIR / "checkerstable_norobot_left.usda",
}
ARM_USD = REPO_ROOT / "isaac" / "usd" / "robots" / "synria_6dof_arm" / "synria_6dof_arm.usda"
ARM_PRIM_PATH = "/World/SynriaArm"
ARM_POS = (0.1524, 0.0, 0.8382)
ARM_Z_ROT_DEG = 90.0
# Factory-reset / home pose — every joint at neutral zero. The arm boots
# here and the drives hold it (see ARM_STIFFNESS/ARM_DAMPING below).
ARM_JOINT_HOME = {
    "Joint1": 0.0, "Joint2": 0.0, "Joint3": 0.0,
    "Joint4": 0.0, "Joint5": 0.0, "Joint6": 0.0,
    "left_finger": 0.0, "right_finger": 0.0,
}
# Drive gains so the arm actively maintains torque (holds pose against
# gravity) instead of drooping. Values match env_cfg actuator cfg:
# arm joints stiff 400 / damp 40, gripper fingers stiff 2000 / damp 100.
ARM_STIFFNESS = {
    "Joint1": 400.0, "Joint2": 400.0, "Joint3": 400.0,
    "Joint4": 400.0, "Joint5": 400.0, "Joint6": 400.0,
    "left_finger": 2000.0, "right_finger": 2000.0,
}
ARM_DAMPING = {
    "Joint1": 40.0, "Joint2": 40.0, "Joint3": 40.0,
    "Joint4": 40.0, "Joint5": 40.0, "Joint6": 40.0,
    "left_finger": 100.0, "right_finger": 100.0,
}
# GUI viewport camera: match the scene's saved perspective camera so the
# lighting reads correctly (lights face this side of the table).
CAMERA_EYE = (5.0, 5.0, 5.0)
CAMERA_TARGET = (0.0, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=sorted(SCENE_USD), required=True)
    parser.add_argument("--with-robot", action="store_true", help="Reference the Synria arm USD onto the scene at the table mount.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless (smoke-check). Default is interactive GUI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_usd = SCENE_USD[args.game]
    if not scene_usd.is_file():
        raise SystemExit(f"Scene USD missing: {scene_usd}")
    if args.with_robot and not ARM_USD.is_file():
        raise SystemExit(f"Arm USD missing: {ARM_USD}")

    from isaacsim import SimulationApp

    # NOTE: the "open_usd" SimulationApp config key is silently ignored in
    # GUI mode on Isaac Sim 5.1 (the app boots an empty stage). Open the
    # scene explicitly after boot instead.
    sim_app = SimulationApp({"headless": args.headless})

    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        usd_context = omni.usd.get_context()
        if not usd_context.open_stage(str(scene_usd)):
            raise RuntimeError(f"Failed to open scene USD: {scene_usd}")
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not load the scene USD")
        print(f"[view_synria_scene] Opened stage: {scene_usd}", flush=True)

        if args.with_robot:
            arm_prim = UsdGeom.Xform.Define(stage, ARM_PRIM_PATH)
            arm_prim.AddTranslateOp().Set(Gf.Vec3d(*ARM_POS))
            arm_prim.AddRotateZOp().Set(ARM_Z_ROT_DEG)
            # Reference the asset's defaultPrim (Alicia_D_v5_6_gripper_50mm) —
            # that's the full 6-DOF arm+gripper articulation (joints Joint1-6
            # + fingers) the training env_cfg mounts. The sibling
            # /synria_6dof_arm prim pins its links to the ground via
            # !resetXformStack! and must not be used here.
            arm_prim.GetPrim().GetReferences().AddReference(str(ARM_USD))
            print(
                f"[view_synria_scene] Mounted SynriaArm at {ARM_POS} "
                f"(rot Z={ARM_Z_ROT_DEG}°) referencing {ARM_USD.name}",
                flush=True,
            )
        else:
            print("[view_synria_scene] No-robot mode — scene loaded without arm.", flush=True)

        if args.headless:
            sim_app.update()
            sim_app.update()
            print(f"[view_synria_scene] Headless OK — scene {args.game} loaded "
                  f"(with_robot={args.with_robot}). Exiting.", flush=True)
            return

        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=list(CAMERA_EYE), target=list(CAMERA_TARGET))
        print(f"[view_synria_scene] Camera set: eye={CAMERA_EYE} target={CAMERA_TARGET}", flush=True)

        # The arm's links use !resetXformStack! — their static USD poses are
        # pinned near the world origin (on the floor), so a parent Xform alone
        # cannot lift the arm onto the table. Initialising physics assembles
        # every link relative to the articulation root's world pose (the
        # /World/SynriaArm wrapper), which is how the training env mounts it.
        sim_ctx = None
        if args.with_robot:
            try:
                import numpy as np
                from isaacsim.core.api import SimulationContext
                from isaacsim.core.prims import SingleArticulation
                from isaacsim.core.utils.types import ArticulationAction

                sim_ctx = SimulationContext(stage_units_in_meters=1.0)
                sim_ctx.reset()  # play physics; assembles the articulation

                arm = SingleArticulation(prim_path=ARM_PRIM_PATH, name="synria_arm")
                arm.initialize()

                half = math.radians(ARM_Z_ROT_DEG) / 2.0
                arm.set_world_pose(
                    position=np.array(ARM_POS, dtype=float),
                    orientation=np.array([math.cos(half), 0.0, 0.0, math.sin(half)]),
                )
                dof_names = list(arm.dof_names or [])
                if dof_names:
                    home = np.array([ARM_JOINT_HOME.get(n, 0.0) for n in dof_names])
                    # Drive gains first so the joints hold torque, then boot
                    # at the factory-reset home pose and lock the drive
                    # targets there. Targets persist, so the arm keeps
                    # actively maintaining this pose every step.
                    kps = np.array([ARM_STIFFNESS.get(n, 400.0) for n in dof_names])
                    kds = np.array([ARM_DAMPING.get(n, 40.0) for n in dof_names])
                    try:
                        arm.get_articulation_controller().set_gains(kps=kps, kds=kds)
                    except Exception as gexc:
                        print(f"[view_synria_scene] WARNING: set_gains failed ({gexc!r})", flush=True)
                    arm.set_joint_positions(home)
                    # Drive position targets — this is what keeps the joints
                    # under active torque (PD hold) every step.
                    arm.apply_action(ArticulationAction(joint_positions=home))
                for _ in range(6):
                    sim_ctx.step(render=True)
                print(f"[view_synria_scene] Arm assembled at factory-reset home, "
                      f"drives holding — dofs={dof_names}", flush=True)
            except Exception as exc:  # fall back to static view rather than abort
                print(f"[view_synria_scene] WARNING: articulation setup failed ({exc!r}); "
                      f"showing static stage.", flush=True)
                sim_ctx = None

        print(f"[view_synria_scene] Viewer running — scene={args.game}, "
              f"with_robot={args.with_robot}. Close the window to exit.", flush=True)
        while sim_app.is_running():
            if sim_ctx is not None:
                sim_ctx.step(render=True)
            else:
                sim_app.update()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
