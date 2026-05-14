"""Launch a starter MoveIt 2 demo for the approximate Synria arm."""

from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("synria_arm_description"))
    moveit_share = Path(get_package_share_directory("synria_arm_moveit_config"))

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", str(description_share / "urdf" / "synria_6dof_arm.urdf.xacro")]),
            value_type=str,
        )
    }
    robot_description_semantic = {
        "robot_description_semantic": (moveit_share / "config" / "synria_arm.srdf").read_text(encoding="utf-8")
    }
    robot_description_kinematics = {"robot_description_kinematics": load_yaml(moveit_share / "config" / "kinematics.yaml")}
    joint_limits = {"robot_description_planning": load_yaml(moveit_share / "config" / "joint_limits.yaml")}
    ompl = load_yaml(moveit_share / "config" / "ompl_planning.yaml")

    return LaunchDescription(
        [
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
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[robot_description, robot_description_semantic, robot_description_kinematics, joint_limits, ompl],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                parameters=[robot_description, robot_description_semantic, robot_description_kinematics, joint_limits, ompl],
            ),
        ]
    )
