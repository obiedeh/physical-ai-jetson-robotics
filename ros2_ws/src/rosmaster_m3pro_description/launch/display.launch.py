from pathlib import Path
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("rosmaster_m3pro_description")

    hardware_type = LaunchConfiguration("hardware_type")
    fixed_odom = LaunchConfiguration("fixed_odom")

    xacro_file = Path(pkg) / "urdf" / "rosmaster_m3pro.urdf.xacro"
    robot_description = xacro.process_file(
        str(xacro_file),
        mappings={
            "hardware_type": "fake",
            "fixed_odom": "true",
        },
    ).toxml()

    return LaunchDescription([
        DeclareLaunchArgument("hardware_type", default_value="fake"),
        DeclareLaunchArgument("fixed_odom", default_value="true"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", str(Path(pkg) / "rviz" / "display.rviz")],
        ),
    ])
