"""Display the approximate Synria arm model in RViz."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("synria_arm_description"))
    model = LaunchConfiguration("model")
    rviz_config = description_share / "rviz" / "synria_arm.rviz"

    robot_description = {"robot_description": Command(["xacro ", model])}

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=str(description_share / "urdf" / "synria_6dof_arm.urdf.xacro"),
                description="Path to the Synria arm xacro file.",
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
