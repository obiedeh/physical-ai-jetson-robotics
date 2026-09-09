"""Bring up the ROSMASTER M3 Pro description for Milestone 1 RViz validation."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("rosmaster_m3pro_description"))

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(description_share / "launch" / "display.launch.py")
                )
            )
        ]
    )
