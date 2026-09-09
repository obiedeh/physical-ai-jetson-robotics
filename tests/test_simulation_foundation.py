from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ROS_SRC = REPO_ROOT / "ros2_ws" / "src"
XACRO_NS = "{http://www.ros.org/wiki/xacro}"


PACKAGE_DIRS = {
    path.name: path
    for path in ROS_SRC.iterdir()
    if path.is_dir() and (path / "package.xml").exists()
}


def parse_xml(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def test_expected_simulation_foundation_files_exist() -> None:
    expected = [
        "docs/architecture.md",
        "docs/VENDOR_INTEGRATION_MAP.md",
        "ros2_ws/src/rosmaster_m3pro_description/package.xml",
        "ros2_ws/src/rosmaster_m3pro_description/CMakeLists.txt",
        "ros2_ws/src/rosmaster_m3pro_description/urdf/rosmaster_m3pro.urdf.xacro",
        "ros2_ws/src/rosmaster_m3pro_description/launch/display.launch.py",
        "ros2_ws/src/rosmaster_m3pro_description/rviz/rosmaster_m3pro.rviz",
        "ros2_ws/src/synria_arm_description/urdf/synria_6dof_arm.urdf.xacro",
        "ros2_ws/src/synria_arm_description/launch/display.launch.py",
        "ros2_ws/src/synria_arm_gazebo/config/ros2_controllers.yaml",
        "ros2_ws/src/synria_arm_gazebo/launch/control.launch.py",
        "ros2_ws/src/physical_ai_ops_copilot/package.xml",
        "ros2_ws/src/physical_ai_ops_copilot/physical_ai_ops_copilot/ops_copilot_node.py",
        "ros2_ws/src/physical_ai_ops_copilot/launch/ops_copilot.launch.py",
        "ros2_ws/src/synria_arm_moveit_config/config/synria_arm.srdf",
        "ros2_ws/src/synria_arm_moveit_config/config/moveit_controllers.yaml",
        "ros2_ws/src/synria_arm_moveit_config/launch/demo.launch.py",
        "reports/yahboom/BRINGUP_TEMPLATE.md",
        "reports/synria/BRINGUP_TEMPLATE.md",
        "reports/simulation/BRINGUP_TEMPLATE.md",
    ]

    missing = [path for path in expected if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_ros_package_manifests_match_package_directories() -> None:
    required_packages = {
        "rosmaster_m3pro_description",
        "synria_arm_description",
        "synria_arm_gazebo",
        "synria_arm_moveit_config",
        "physical_ai_ops_copilot",
    }

    # An absent vendor package is optional; a partial package must still fail.
    required_packages |= {
        name for name in ("yahboom_M3Pro_description", "M3Pro_config")
        if (ROS_SRC / name).exists()
    }
    assert required_packages <= set(PACKAGE_DIRS)
    for package_name in required_packages:
        manifest = parse_xml(PACKAGE_DIRS[package_name] / "package.xml")
        assert manifest.findtext("name") == package_name


def test_xacro_files_are_well_formed_and_includes_resolve() -> None:
    xacro_files = sorted(ROS_SRC.glob("*/urdf/*.xacro"))
    assert xacro_files

    unresolved: list[str] = []
    for xacro_file in xacro_files:
        root = parse_xml(xacro_file)
        assert root.tag.endswith("robot")
        for include in root.findall(f".//{XACRO_NS}include"):
            filename = include.attrib["filename"]
            match = re.fullmatch(r"\$\(find ([^)]+)\)/(.+)", filename)
            if not match:
                unresolved.append(f"{xacro_file}: unsupported include {filename}")
                continue

            package_name, relative_path = match.groups()
            if (
                package_name == "yahboom_M3Pro_description"
                and not (ROS_SRC / package_name).exists()
            ):
                continue
            if (
                (package_name, relative_path)
                == ("synria_arm_description", "urdf/synria_6dof_arm.urdf")
                and not (ROS_SRC / package_name / relative_path).exists()
            ):
                continue
            package_dir = PACKAGE_DIRS.get(package_name)
            if package_dir is None or not (package_dir / relative_path).exists():
                unresolved.append(f"{xacro_file}: missing include {filename}")

    assert unresolved == []


def test_mesh_references_resolve_to_repo_files() -> None:
    unresolved: list[str] = []
    mesh_refs = 0

    mesh_pattern = re.compile(
        r"(?:package://([^/]+)/([^\s\"']+)|(?:\.{1,2}/)?([^\s\"'@]+?\.(?:stl|STL)))"
    )

    candidate_files = (
        sorted((REPO_ROOT / "ros2_ws" / "src").glob("**/*.urdf"))
        + sorted((REPO_ROOT / "ros2_ws" / "src").glob("**/*.xacro"))
        + sorted((REPO_ROOT / "isaac" / "usd").glob("**/*.urdf"))
        + sorted((REPO_ROOT / "isaac" / "usd").glob("**/*.xacro"))
        + sorted((REPO_ROOT / "isaac" / "usd").glob("**/*.usda"))
    )

    for asset_file in candidate_files:
        if not asset_file.exists():
            continue

        text = asset_file.read_text(encoding="utf-8", errors="ignore")
        for match in mesh_pattern.finditer(text):
            mesh_refs += 1
            package_name, package_relative, direct_relative = match.groups()
            if package_name:
                package_dir = PACKAGE_DIRS.get(package_name)
                if package_dir is None or not (package_dir / package_relative).exists():
                    unresolved.append(f"{asset_file}: missing mesh package://{package_name}/{package_relative}")
                continue

            candidates = [
                (asset_file.parent / direct_relative).resolve(),
                (asset_file.parent.parent / direct_relative).resolve(),
                (REPO_ROOT / "isaac" / "usd" / "robots" / direct_relative).resolve(),
            ]
            if not any(candidate.exists() for candidate in candidates):
                unresolved.append(f"{asset_file}: missing mesh {direct_relative}")

    if mesh_refs == 0:
        pytest.skip("No local vendor mesh inputs; run scripts/fetch_vendor_assets.py")
    assert unresolved == []


@pytest.mark.skipif(
    not (ROS_SRC / "yahboom_M3Pro_description").exists(),
    reason="Optional vendor assets: run scripts/fetch_vendor_assets.py yahboom",
)
def test_yahboom_description_structure_is_simulation_ready() -> None:
    yahboom = parse_xml(ROS_SRC / "yahboom_M3Pro_description" / "urdf" / "M3Pro.urdf")

    # --- Yahboom vendor URDF: 5-DOF arm + mecanum base + sensors ---
    assert yahboom.attrib["name"] == "M3Pro"
    yahboom_joints = {joint.attrib["name"] for joint in yahboom.findall("joint")}
    yahboom_links = {link.attrib["name"] for link in yahboom.findall("link")}

    # Arm joints: arm4_Joiint has the vendor double-'i' typo — preserved.
    expected_arm_joints = {
        "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joiint", "arm5_Joint",
        "arm_base_Joint",
    }
    assert expected_arm_joints <= yahboom_joints
    assert {"lwheel1", "lwheel2", "rwheel1", "rwheel2"} <= yahboom_links
    assert "base_link" in yahboom_links
    assert {"arm1", "arm2", "arm3", "arm4", "arm5"} <= yahboom_links



@pytest.mark.skipif(
    not (ROS_SRC / "synria_arm_description/urdf/synria_6dof_arm.urdf").exists(),
    reason="Optional vendor assets: run scripts/fetch_vendor_assets.py synria",
)
def test_synria_description_structure_is_simulation_ready() -> None:
    synria_urdf = parse_xml(ROS_SRC / "synria_arm_description/urdf/synria_6dof_arm.urdf")
    # --- Synria vendor URDF (real, SolidWorks export) ---
    # Robot name is the SolidWorks model name, not the ROS package name.
    assert synria_urdf.attrib["name"] == "Alicia_D_v5_6_gripper_50mm"
    urdf_joints = {j.attrib["name"] for j in synria_urdf.findall("joint")}
    urdf_links = {lnk.attrib["name"] for lnk in synria_urdf.findall("link")}

    # 6 revolute arm joints + 2 prismatic gripper fingers
    assert {"Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"} <= urdf_joints
    assert {"left_finger", "right_finger"} <= urdf_joints
    # Real link chain: base_link -> link1..link6 -> tool0, left_gripper, right_gripper
    assert {"base_link", "link1", "link2", "link3", "link4", "link5", "link6"} <= urdf_links
    assert {"tool0", "left_gripper", "right_gripper"} <= urdf_links



def test_synria_xacro_wrapper_is_simulation_ready() -> None:
    synria_xacro = parse_xml(
        ROS_SRC / "synria_arm_description/urdf/synria_6dof_arm.urdf.xacro"
    )
    # --- Synria xacro wrapper ---
    assert synria_xacro.attrib["name"] == "synria_6dof_arm"
    xacro_text = ET.tostring(synria_xacro, encoding="unicode")

    # World link and fixed joint present in the wrapper
    xacro_link_names = {lnk.attrib["name"] for lnk in synria_xacro.findall("link")}
    assert "world" in xacro_link_names

    # C10 camera macro call and vendor URDF include visible in raw text
    assert "synria_c10_camera" in xacro_text
    assert "synria_6dof_arm.urdf" in xacro_text

    # ros2_control: mock hardware, 8 joints (6 arm + 2 gripper)
    plugin_el = synria_xacro.find("ros2_control/hardware/plugin")
    assert plugin_el is not None
    assert plugin_el.text == "mock_components/GenericSystem"
    ros2_joints = {j.attrib["name"] for j in synria_xacro.findall("ros2_control/joint")}
    assert {"Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"} <= ros2_joints
    assert {"left_finger", "right_finger"} <= ros2_joints
    assert len(synria_xacro.findall("ros2_control/joint")) == 8


def test_launch_files_are_python_syntax_valid_and_define_launch_descriptions() -> None:
    launch_files = sorted(ROS_SRC.glob("*/launch/*.launch.py"))
    assert launch_files

    for launch_file in launch_files:
        tree = ast.parse(launch_file.read_text(encoding="utf-8"), filename=str(launch_file))
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert "generate_launch_description" in function_names


def test_moveit_and_controller_configs_reference_expected_synria_joints() -> None:
    # Real joint names from the vendor URDF (capital-J, SolidWorks export)
    joint_names = {"Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"}
    srdf = (ROS_SRC / "synria_arm_moveit_config" / "config" / "synria_arm.srdf").read_text(
        encoding="utf-8"
    )
    controllers = (
        ROS_SRC / "synria_arm_gazebo" / "config" / "ros2_controllers.yaml"
    ).read_text(encoding="utf-8")
    joint_limits = (
        ROS_SRC / "synria_arm_moveit_config" / "config" / "joint_limits.yaml"
    ).read_text(encoding="utf-8")
    moveit_controllers = (
        ROS_SRC / "synria_arm_moveit_config" / "config" / "moveit_controllers.yaml"
    ).read_text(encoding="utf-8")

    for joint_name in joint_names:
        assert joint_name in controllers, f"Joint {joint_name!r} missing from controllers.yaml"
        assert joint_name in joint_limits, f"Joint {joint_name!r} missing from joint_limits.yaml"
        assert joint_name in moveit_controllers, (
            f"Joint {joint_name!r} missing from moveit_controllers.yaml"
        )

    # Gripper joints also present in all three configs
    assert "left_finger" in controllers
    assert "right_finger" in controllers
    assert "left_finger" in moveit_controllers
    assert "right_finger" in moveit_controllers

    # SRDF: correct planning group, base/tip links, real adjacency pairs
    assert '<group name="synria_arm">' in srdf
    assert 'base_link="base_link"' in srdf
    assert 'tip_link="tool0"' in srdf
    assert "link1" in srdf
    assert "link6" in srdf

    # Controller manager wiring (ros2_controllers.yaml)
    assert "arm_controller" in controllers
    assert "joint_trajectory_controller/JointTrajectoryController" in controllers

    # MoveIt controller manager wiring (moveit_controllers.yaml)
    assert "moveit_simple_controller_manager" in moveit_controllers
    assert "arm_controller" in moveit_controllers
    assert "gripper_controller" in moveit_controllers
    assert "FollowJointTrajectory" in moveit_controllers
