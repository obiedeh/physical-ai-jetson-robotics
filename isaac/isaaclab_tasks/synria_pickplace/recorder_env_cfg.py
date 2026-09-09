"""Recorder-only env variant: Ludo pick-place + wrist/overhead RGB cameras.

GR00T N1.7 is vision-based; the RL task is state-based. This variant exists
ONLY for demonstration recording — it adds two 224×224 RGB TiledCamera
sensors to the scene WITHOUT touching the RL task's observation manager,
so RL checkpoints load and run unchanged. Camera frames are read directly
from ``env.scene.sensors[...]`` by the recorder script, never through the
policy observation group.

Cameras (keys must match ``gr00t/modality_config.py`` and the dataset's
``meta/modality.json``):

- ``wrist``:    mounted on the ``tool0`` flange looking down the tool -Z
                axis at the fingers / grasp centre.
- ``overhead``: fixed per-env above the board centre (task_geometry board
                centre x=0.10, y=0), looking straight down.

Never used for RL training. GPU cost: 2 cameras × num_envs tiled render —
keep recorder num_envs modest (≤64) and coordinate with the RL session.
"""

from __future__ import annotations

import os

from isaac.isaaclab_tasks.synria_pickplace.env_cfg import (
    _ISAACLAB_AVAILABLE,
    SynriaLudoPickPlaceEnvCfg,
    SynriaTabletopPickPlaceEnvCfg,
)
from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z

if _ISAACLAB_AVAILABLE:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils.configclass import configclass

    # 224 = GR00T training size; override for cinematic demo renders
    _CAM_RES = int(os.environ.get("SYNRIA_CAMERA_RES", "224"))

    def add_recorder_cameras(
        cfg: SynriaTabletopPickPlaceEnvCfg,
    ) -> SynriaTabletopPickPlaceEnvCfg:
        """Attach wrist + overhead demo cameras to any synria_pickplace env cfg.

        Applied AFTER ``parse_env_cfg`` so the same RL task (and therefore
        the same trained checkpoint) runs with cameras bolted on — the
        observation manager is untouched. Returns ``cfg`` for chaining.
        """
        # Wrist camera on the tool0 flange. The grasp centre sits at
        # roughly (-0.031, 0, -0.053) m in tool0 frame (task_geometry
        # PREGRASP_CENTER_LOCAL), so look down the tool -Z axis from a
        # couple of cm behind the flange to frame fingers + cup.
        # OffsetCfg "world" convention: camera forward is +X; the
        # quaternion (0.7071, 0, 0.7071, 0) is a +90° pitch about +Y,
        # mapping forward +X → local -Z.
        cfg.scene.wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/SynriaArm/tool0/wrist_cam",
            update_period=0.0,  # every sim step; recorder samples at decimation
            width=_CAM_RES,
            height=_CAM_RES,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=6.0,
                clipping_range=(0.01, 3.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-0.031, 0.0, 0.06),
                rot=(0.7071068, 0.0, 0.7071068, 0.0),
                convention="world",
            ),
        )

        # Overhead camera fixed per-env above the board centre, looking
        # straight down. Height chosen to frame the 0.19–0.55 m board
        # annulus plus staging zones.
        cfg.scene.overhead = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/overhead_cam",
            update_period=0.0,
            width=_CAM_RES,
            height=_CAM_RES,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                clipping_range=(0.05, 5.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.10, 0.0, TABLE_SURFACE_Z + 0.85),
                rot=(0.7071068, 0.0, 0.7071068, 0.0),
                convention="world",
            ),
        )
        return cfg

    #: die footprint (m) — a small cube read by its rest orientation
    DIE_SIZE = 0.018

    def add_die(
        cfg: SynriaTabletopPickPlaceEnvCfg,
        pos: tuple[float, float, float | None] = (0.08, 0.12, None),
    ) -> SynriaTabletopPickPlaceEnvCfg:
        """Attach a pickable/rollable die (18 mm cube) adjacent to the board.
        Opt-in (demos only) — never added to the training env. The die value
        is read from its rest orientation (game_core.dice)."""
        from isaaclab.assets import RigidObjectCfg
        z = pos[2] if pos[2] is not None else TABLE_SURFACE_Z + DIE_SIZE / 2.0
        cfg.scene.die = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Die",
            spawn=sim_utils.CuboidCfg(
                size=(DIE_SIZE, DIE_SIZE, DIE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    angular_damping=0.03, linear_damping=0.05,
                    max_angular_velocity=200.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.008),  # ~8 g
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5,
                    restitution=0.35),  # a die bounces a little
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.93, 0.85)),  # ivory
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], z)),
        )
        return cfg

    def add_cup(
        cfg: SynriaTabletopPickPlaceEnvCfg,
        pos: tuple[float, float, float | None] = (0.08, -0.12, None),
    ) -> SynriaTabletopPickPlaceEnvCfg:
        """Attach the hollow dice cup (isaac/usd/props/dice_cup.usda) as a
        pickable rigid body. Opt-in (demos only). convexDecomposition collision
        (baked in the USD) makes it CONTAIN the die."""
        import pathlib

        from isaaclab.assets import RigidObjectCfg
        cup_usd = (pathlib.Path(__file__).resolve().parents[3]
                   / "isaac" / "usd" / "props" / "dice_cup.usda")
        z = pos[2] if pos[2] is not None else TABLE_SURFACE_Z + 0.001
        cfg.scene.cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(cup_usd),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    angular_damping=0.5, linear_damping=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], z)),
        )
        return cfg

    @configclass
    class SynriaLudoPickPlaceRecorderEnvCfg(SynriaLudoPickPlaceEnvCfg):
        """Ludo pick-place env + demo-recording cameras (obs space unchanged)."""

        def __post_init__(self) -> None:
            super().__post_init__()
            add_recorder_cameras(self)

    __all__ = ["SynriaLudoPickPlaceRecorderEnvCfg", "add_recorder_cameras"]
else:  # pragma: no cover - hardware-gated
    __all__ = []
