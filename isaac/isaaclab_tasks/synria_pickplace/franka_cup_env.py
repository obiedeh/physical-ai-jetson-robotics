"""Minimal ManagerBased env: Franka + Robotiq 2F-85, cup, cameras.

Surrogate honest-grasp environment (ledger R-track final): the validated
Franka+2F-85 replaces the Synria arm for sim-grasp work. Scene + actions
only — control scripts (vision pick, demo recorder) drive it directly.

Actions (8): panda_joint1..7 position targets (offsets from default) +
finger_joint target (0 open .. 0.79 closed).
"""

from __future__ import annotations

try:
    import isaaclab.envs.mdp as base_mdp
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.envs import ManagerBasedEnvCfg
    from isaaclab.envs.mdp.actions import JointPositionActionCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

    _OK = True
except ImportError:  # pragma: no cover - hardware-gated
    _OK = False

CAM_POS = (0.5, 0.0, 0.85)
RES = 224

if _OK:

    @configclass
    class FrankaCupSceneCfg(InteractiveSceneCfg):
        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
        )
        light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0),
        )
        robot: ArticulationCfg = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            # ground-mounted at the env origin (stock cfg sits on a
            # tabletop at (-0.85, 0, 0.76) — cup would be out of reach)
            init_state=FRANKA_ROBOTIQ_GRIPPER_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0)
            ),
        )
        cup: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            spawn=sim_utils.CylinderCfg(
                radius=0.02,
                height=0.04,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    angular_damping=5.0, linear_damping=0.5
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.3,
                    dynamic_friction=1.1,
                    friction_combine_mode="max",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.1, 0.1)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.021)),
        )
        overhead: TiledCameraCfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/overhead_cam",
            update_period=0.0,
            width=RES,
            height=RES,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, clipping_range=(0.05, 5.0)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=CAM_POS,
                rot=(0.7071068, 0.0, 0.7071068, 0.0),
                convention="world",
            ),
        )
        wrist: TiledCameraCfg = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            width=RES,
            height=RES,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=6.0, clipping_range=(0.01, 3.0)
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.05),
                rot=(0.0, 0.7071068, -0.7071068, 0.0),
                convention="world",
            ),
        )

    @configclass
    class ActionsCfg:
        arm: JointPositionActionCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            scale=1.0,
            use_default_offset=True,
        )
        gripper: JointPositionActionCfg = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            scale=1.0,
            use_default_offset=False,
        )

    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=base_mdp.joint_pos)

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class FrankaCupEnvCfg(ManagerBasedEnvCfg):
        scene: FrankaCupSceneCfg = FrankaCupSceneCfg(num_envs=4, env_spacing=2.5)
        actions: ActionsCfg = ActionsCfg()
        observations: ObservationsCfg = ObservationsCfg()

        def __post_init__(self) -> None:
            self.decimation = 2
            self.sim.dt = 1 / 120

    __all__ = ["FrankaCupEnvCfg"]
else:  # pragma: no cover
    __all__ = []
