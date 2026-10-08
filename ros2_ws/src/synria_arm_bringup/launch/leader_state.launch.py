"""Start one isolated, read-only driver instance for the leader arm."""

from __future__ import annotations

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from synria_arm_bringup.validation import validate_leader_port


def _leader_node(context: object) -> list[Node]:
    try:
        get_package_share_directory("alicia_d_driver")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Required external package 'alicia_d_driver' was not found. "
            "Install and source its workspace before launching leader state capture."
        ) from exc

    port = validate_leader_port(LaunchConfiguration("leader_port").perform(context))
    return [
        Node(
            package="alicia_d_driver",
            executable="alicia_d_driver_node",
            name="alicia_d_leader_driver_node",
            output="screen",
            parameters=[
                {
                    "port": port,
                    "torque_off_on_start": False,
                    "joint_commands_enabled": False,
                    "joint_commands_dry_run": False,
                    "legacy_servo_commands": False,
                    "allow_leader_writes": False,
                }
            ],
            remappings=[
                ("/joint_states", "/leader/joint_states"),
                ("/joint_commands", "/leader/disabled_joint_commands"),
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "leader_port",
                description=(
                    "Required stable leader serial path under /dev/serial/by-id; "
                    "there is intentionally no default or auto-detection."
                ),
            ),
            OpaqueFunction(function=_leader_node),
        ]
    )
