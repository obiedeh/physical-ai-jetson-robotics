"""Start robot_state_publisher and ros2_control for the approximate Synria arm."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("synria_arm_description"))
    gazebo_share = Path(get_package_share_directory("synria_arm_gazebo"))
    model = description_share / "urdf" / "synria_6dof_arm.urdf.xacro"
    controllers = gazebo_share / "config" / "ros2_controllers.yaml"

    robot_description = {"robot_description": Command(["xacro ", str(model)])}

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, str(controllers)],
        output="screen",
    )
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="screen",
            ),
            control_node,
            RegisterEventHandler(
                OnProcessStart(target_action=control_node, on_start=[joint_state_spawner, arm_controller_spawner])
            ),
        ]
    )
