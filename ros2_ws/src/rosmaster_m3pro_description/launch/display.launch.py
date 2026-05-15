"""Display the Yahboom ROSMASTER M3 Pro model in RViz."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("rosmaster_m3pro_description"))
    model = LaunchConfiguration("model")
    hardware_type = LaunchConfiguration("hardware_type")
    fixed_odom = LaunchConfiguration("fixed_odom")
    rviz_config = description_share / "rviz" / "display.rviz"

    robot_description = {
        "robot_description": Command(
            [
                "xacro ",
                model,
                " hardware_type:=",
                hardware_type,
                " fixed_odom:=",
                fixed_odom,
            ]
        )
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=str(description_share / "urdf" / "rosmaster_m3pro.urdf.xacro"),
                description="Path to the ROSMASTER M3 Pro Xacro file.",
            ),
            DeclareLaunchArgument(
                "hardware_type",
                default_value="fake",
                description="ros2_control hardware backend: fake or real.",
            ),
            DeclareLaunchArgument(
                "fixed_odom",
                default_value="true",
                description="Publish a fixed odom to base_footprint joint for display-only bringup.",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="screen",
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(rviz_config)],
                output="screen",
            ),
        ]
    )
