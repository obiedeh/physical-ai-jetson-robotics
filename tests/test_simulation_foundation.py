from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path


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
        "docs/ARCHITECTURE.md",
        "docs/VENDOR_INTEGRATION_MAP.md",
        "ros2_ws/src/rosmaster_m3pro_description/urdf/rosmaster_m3pro.urdf.xacro",
        "ros2_ws/src/rosmaster_m3pro_description/launch/display.launch.py",
        "ros2_ws/src/synria_arm_description/urdf/synria_6dof_arm.urdf.xacro",
        "ros2_ws/src/synria_arm_description/launch/display.launch.py",
        "ros2_ws/src/synria_arm_gazebo/config/ros2_controllers.yaml",
        "ros2_ws/src/synria_arm_gazebo/launch/control.launch.py",
        "ros2_ws/src/synria_arm_moveit_config/config/synria_arm.srdf",
        "ros2_ws/src/synria_arm_moveit_config/launch/demo.launch.py",
        "ros2_ws/src/physical_ai_gazebo/worlds/factory_cell.sdf",
        "ros2_ws/src/physical_ai_gazebo/launch/factory_cell.launch.py",
        "isaac/usd/factory_cell_v0.usda",
        "reports/yahboom/BRINGUP_TEMPLATE.md",
        "reports/synria/BRINGUP_TEMPLATE.md",
        "reports/simulation/BRINGUP_TEMPLATE.md",
    ]

    missing = [path for path in expected if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_ros_package_manifests_match_package_directories() -> None:
    required_packages = {
        "physical_ai_description",
        "physical_ai_gazebo",
        "rosmaster_m3pro_description",
        "synria_arm_description",
        "synria_arm_gazebo",
        "synria_arm_moveit_config",
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

    assert mesh_refs >= 10
    assert unresolved == []


def test_robot_description_structure_is_simulation_ready() -> None:
    yahboom_text = (
        ROS_SRC / "rosmaster_m3pro_description" / "urdf" / "rosmaster_m3pro.urdf.xacro"
    ).read_text(encoding="utf-8")
    synria = parse_xml(
        ROS_SRC / "synria_arm_description" / "urdf" / "synria_6dof_arm.urdf.xacro"
    )

    assert 'name="rosmaster_m3pro"' in yahboom_text
    for token in [
        "rosmaster_m3pro_mecanum_base",
        "rosmaster_m3pro_arm_6dof",
        "rosmaster_m3pro_depth_camera",
        "rosmaster_m3pro_dual_tof_lidar",
        "rosmaster_m3pro_ros2_control",
        'default="fake"',
    ]:
        assert token in yahboom_text

    assert synria.attrib["name"] == "synria_6dof_arm"
    joint_names = {joint.attrib["name"] for joint in synria.findall("joint")}
    link_names = {link.attrib["name"] for link in synria.findall("link")}
    synria_text = ET.tostring(synria, encoding="unicode")

    assert {f"joint_{index}" for index in range(1, 7)} <= joint_names
    assert {"world", "base_link"} <= link_names
    assert 'name="tool0"' in synria_text
    assert "synria_c10_camera" in synria_text
    assert synria.find("ros2_control/hardware/plugin").text == "mock_components/GenericSystem"
    assert len(synria.findall("ros2_control/joint")) == 6


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
    joint_names = {f"joint_{index}" for index in range(1, 7)}
    srdf = (ROS_SRC / "synria_arm_moveit_config" / "config" / "synria_arm.srdf").read_text(
        encoding="utf-8"
    )
    controllers = (
        ROS_SRC / "synria_arm_gazebo" / "config" / "ros2_controllers.yaml"
    ).read_text(encoding="utf-8")

    for joint_name in joint_names:
        assert joint_name in controllers

    assert '<group name="synria_arm">' in srdf
    assert 'base_link="base_link"' in srdf
    assert 'tip_link="tool0"' in srdf
    assert "arm_controller" in controllers
    assert "joint_trajectory_controller/JointTrajectoryController" in controllers


def test_openusd_and_gazebo_assets_have_simulation_anchors() -> None:
    usd_text = (REPO_ROOT / "isaac" / "usd" / "factory_cell_v0.usda").read_text(
        encoding="utf-8"
    )
    sdf = parse_xml(ROS_SRC / "physical_ai_gazebo" / "worlds" / "factory_cell.sdf")

    assert usd_text.startswith("#usda 1.0")
    assert "FactoryCell" in usd_text
    assert sdf.attrib["version"]
    assert sdf.find("world") is not None
    assert len(sdf.findall(".//model")) >= 3
