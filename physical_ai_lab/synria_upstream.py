"""Synria upstream integration TODOs derived from public vendor repositories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class ActivityPhase(str, Enum):
    """When an upstream integration activity should run."""

    rtx_now = "rtx-now"
    hardware_later = "hardware-later"


@dataclass(frozen=True)
class SynriaUpstreamActivity:
    """One actionable vendor-alignment activity."""

    identifier: str
    phase: ActivityPhase
    title: str
    source_repo: str
    objective: str
    deliverable: str
    command_hint: str

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["phase"] = self.phase.value
        return data


def synria_upstream_activities() -> list[SynriaUpstreamActivity]:
    """Return Synria upstream TODOs in practical execution order."""
    return [
        SynriaUpstreamActivity(
            identifier="vendor-description-parity",
            phase=ActivityPhase.rtx_now,
            title="Vendor Robot Description Parity Audit",
            source_repo="Synria-Robotics/Synria-Robot-Descriptions and Alicia-D-ROS2",
            objective=(
                "Compare our approximate Synria URDF, SRDF, joint limits, frames, "
                "and camera link against vendor descriptions."
            ),
            deliverable="docs/reports/synria_vendor_description_parity.md",
            command_hint="ros2 launch synria_arm_description display.launch.py",
        ),
        SynriaUpstreamActivity(
            identifier="humble-to-jazzy-port",
            phase=ActivityPhase.rtx_now,
            title="ROS 2 Humble-to-Jazzy Porting Checklist",
            source_repo="Synria-Robotics/Alicia-D-ROS2",
            objective=(
                "Track package manifest, launch, controller, and MoveIt config "
                "changes needed to run Synria-style ROS 2 packages on Jazzy."
            ),
            deliverable="docs/reports/synria_ros2_jazzy_port.md",
            command_hint="bash scripts/linux_rtx/run_synria_moveit_smoke.sh",
        ),
        SynriaUpstreamActivity(
            identifier="mock-ros2-control-parity",
            phase=ActivityPhase.rtx_now,
            title="Mock ros2_control Hardware Parity",
            source_repo="Synria-Robotics/Alicia-D-ROS2",
            objective=(
                "Model the vendor control surface for joint state, joint command, "
                "gripper command, torque enable, zeroing, and speed limits."
            ),
            deliverable="ros2_ws/src/synria_arm_gazebo/config/ros2_controllers.yaml",
            command_hint="bash scripts/linux_rtx/run_ros_gazebo_factory_cell.sh",
        ),
        SynriaUpstreamActivity(
            identifier="c10-camera-sim-contract",
            phase=ActivityPhase.rtx_now,
            title="C10 Camera Simulation Contract",
            source_repo="Synria-Robotics/Synria-C10-SDK",
            objective=(
                "Define camera index, OpenCV capture, image size, calibration "
                "metadata, and ROS image topic expectations for the C10 wrist camera."
            ),
            deliverable="docs/reports/synria_c10_camera_contract.md",
            command_hint="physical-ai-lab edge-plan --compute workstation",
        ),
        SynriaUpstreamActivity(
            identifier="hand-eye-calibration-sim",
            phase=ActivityPhase.rtx_now,
            title="Simulated Hand-Eye Calibration",
            source_repo="Synria-Robotics/Alicia-D-ROS2",
            objective=(
                "Recreate the vendor hand-eye calibration workflow with simulated "
                "ArUco poses before using the real C10 camera."
            ),
            deliverable="reports/synria/hand_eye_calibration_sim.md",
            command_hint="physical-ai-lab train-synria-reach --device cuda",
        ),
        SynriaUpstreamActivity(
            identifier="cube-sort-sim-demo",
            phase=ActivityPhase.rtx_now,
            title="Cube Sorting Simulation Demo",
            source_repo="Synria-Robotics/Alicia-D-ROS2",
            objective=(
                "Build an RTX simulation of colored cube detection, MoveIt planning, "
                "and pick/place bins aligned with Synria's cube-sort package."
            ),
            deliverable="reports/demo/synria_cube_sort_sim.md",
            command_hint="bash scripts/linux_rtx/run_rtx_training_suite.sh",
        ),
        SynriaUpstreamActivity(
            identifier="lerobot-record-schema",
            phase=ActivityPhase.rtx_now,
            title="LeRobot Recording Schema Contract",
            source_repo="Synria-Robotics/lerobot",
            objective=(
                "Match our synthetic and future real demonstrations to Synria/Alicia-D "
                "record fields for camera, joint state, actions, FPS, and episode timing."
            ),
            deliverable="lerobot/configs/synria_aloha_act_notes.yaml",
            command_hint="lerobot-train --dataset.repo_id obiedeh/synria-c10-aloha-demo",
        ),
        SynriaUpstreamActivity(
            identifier="robocore-kinematics-benchmark",
            phase=ActivityPhase.rtx_now,
            title="RoboCore-Style Kinematics Benchmark",
            source_repo="Synria-Robotics/Alicia-D-SDK and RoboCore",
            objective=(
                "Benchmark FK, IK, Jacobian, and trajectory targets against our "
                "synthetic reaching trainer."
            ),
            deliverable="reports/training/synria_kinematics_benchmark.json",
            command_hint="physical-ai-lab train-synria-reach --device cuda",
        ),
        SynriaUpstreamActivity(
            identifier="real-serial-bringup",
            phase=ActivityPhase.hardware_later,
            title="Real Serial Bring-Up and Firmware Detection",
            source_repo="Synria-Robotics/Alicia-D-SDK and Alicia-D-ROS2",
            objective=(
                "Verify serial port discovery, permissions, firmware version, "
                "state reads, and debug logging on the physical arm."
            ),
            deliverable="reports/synria/real_serial_bringup.md",
            command_hint="ls /dev/ttyACM*",
        ),
        SynriaUpstreamActivity(
            identifier="safe-real-motion",
            phase=ActivityPhase.hardware_later,
            title="Safe First Motion and Zero Calibration",
            source_repo="Synria-Robotics/Alicia-D-SDK and Alicia-D-ROS2",
            objective=(
                "Run torque enable/disable, zero calibration, reduced-speed joint "
                "motion, and emergency-stop checks on hardware."
            ),
            deliverable="reports/synria/first_safe_motion.md",
            command_hint="ros2 launch alicia_d_moveit real_robot.launch.py speed_deg_s:=20",
        ),
    ]


def synria_upstream_activity_by_id(identifier: str) -> SynriaUpstreamActivity:
    """Return one upstream TODO by identifier."""
    for activity in synria_upstream_activities():
        if activity.identifier == identifier:
            return activity
    known = ", ".join(activity.identifier for activity in synria_upstream_activities())
    raise ValueError(f"Unknown Synria upstream activity '{identifier}'. Known: {known}")
