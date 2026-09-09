"""Display the Yahboom ROSMASTER M3 Pro in RViz.

Launches:
  robot_state_publisher  — publishes /robot_description and TF tree
  joint_state_publisher_gui — slider GUI for arm + gripper joints
  rviz2                  — loads rosmaster_m3pro.rviz config

Usage:
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  ros2 launch rosmaster_m3pro_description display.launch.py

Optional override:
  ros2 launch rosmaster_m3pro_description display.launch.py \\
      model:=/absolute/path/to/custom.urdf.xacro
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(
        get_package_share_directory("rosmaster_m3pro_description")
    )
    default_xacro = description_share / "urdf" / "rosmaster_m3pro.urdf.xacro"
    rviz_config = description_share / "rviz" / "rosmaster_m3pro.rviz"

    model_arg = DeclareLaunchArgument(
        "model",
        default_value=str(default_xacro),
        description="Absolute path to the robot xacro/URDF file.",
    )

    robot_description = {
        "robot_description": Command(["xacro ", LaunchConfiguration("model")])
    }

    return LaunchDescription(
        [
            model_arg,
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
