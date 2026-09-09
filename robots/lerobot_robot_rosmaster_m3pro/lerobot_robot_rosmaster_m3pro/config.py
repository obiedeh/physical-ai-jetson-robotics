from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
VENDOR_HOME = (90.0, 150.0, 12.0, 20.0, 90.0, 90.0)  # servo6 gripper mid (30-180 range; 0 stalls the servo)   # servo 6 = gripper


@RobotConfig.register_subclass("rosmaster_m3pro")
@dataclass
class RosmasterM3ProConfig(RobotConfig):
    # --- ROS plumbing (vendor firmware + micro-ROS agent) ---
    ros_domain_id: int = 30
    # UDP-only FastDDS profile: required when this process runs as a different
    # Linux user than the vendor agent (shared-memory segments are per-user).
    fastdds_profile: str | None = None
    arm_topic: str = "/arm6_joints"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/odom_raw"
    battery_topic: str = "/battery"
    preflight_timeout_s: float = 8.0     # wait for the board to match our publishers

    # --- arm ---
    arm_run_time_ms: int = 150           # board-side interpolation window per command
    # max change per command, degrees (scalar or per-joint dict); None = unclamped
    max_relative_target: float | dict[str, float] | None = 10.0
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        j: (0.0, 180.0) for j in ARM_JOINTS})
    home_pose: tuple[float, float, float, float, float, float] = VENDOR_HOME
    go_home_on_connect: bool = True      # the ONLY way to have a valid open-loop state
    home_run_time_ms: int = 3000

    # --- base ---
    use_base: bool = True
    base_max_linear: float = 0.20        # m/s clamp on |vx|, |vy|
    base_max_angular: float = 0.60       # rad/s clamp on |wz|
    deadman_timeout_s: float = 0.5       # zero the base if no action arrives in time

    # --- sensors ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
