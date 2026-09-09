"""Static file and parse tests for the Yahboom ROS 2 description packages.

Checks that all required files are present and valid without needing a ROS 2
installation, xacro, or any hardware. Runs in standard CI (Python-only).

Packages covered:
  - yahboom_M3Pro_description  (vendor URDF + STL meshes)
  - rosmaster_m3pro_description (our ament_cmake wrapper)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PKG = REPO_ROOT / "ros2_ws" / "src" / "yahboom_M3Pro_description"
DESC_PKG = REPO_ROOT / "ros2_ws" / "src" / "rosmaster_m3pro_description"


# ===========================================================================
# yahboom_M3Pro_description — vendor package
# ===========================================================================


@pytest.mark.skipif(
    not VENDOR_PKG.exists(),
    reason="Optional vendor assets: run scripts/fetch_vendor_assets.py yahboom",
)
class TestYahboomVendorPackage:
    """Required files exist in the vendor package."""

    def test_package_xml_exists(self) -> None:
        assert (VENDOR_PKG / "package.xml").is_file()

    def test_urdf_exists(self) -> None:
        assert (VENDOR_PKG / "urdf" / "M3Pro.urdf").is_file()

    def test_urdf_is_valid_xml(self) -> None:
        ET.parse(VENDOR_PKG / "urdf" / "M3Pro.urdf")

    def test_urdf_robot_name(self) -> None:
        root = ET.parse(VENDOR_PKG / "urdf" / "M3Pro.urdf").getroot()
        assert root.tag == "robot"
        assert root.attrib.get("name") == "M3Pro"

    def test_meshes_directory_exists(self) -> None:
        assert (VENDOR_PKG / "meshes").is_dir()

    def test_all_stl_meshes_present(self) -> None:
        expected = [
            "base_link.STL",
            "arm1.STL", "arm2.STL", "arm3.STL", "arm4.STL", "arm5.STL",
            "arm_base_Link.STL",
            "Camera.STL",
            "DCW2.STL",
            "Gripping.STL",
            "llink1.STL", "llink2.STL", "llink3.STL",
            "rlink1.STL", "rlink2.STL", "rlink3.STL",
            "lwheel1.STL", "lwheel2.STL",
            "rwheel1.STL", "rwheel2.STL",
        ]
        missing = [name for name in expected if not (VENDOR_PKG / "meshes" / name).is_file()]
        assert missing == [], f"Missing STL meshes: {missing}"

    def test_package_xml_name_matches(self) -> None:
        root = ET.parse(VENDOR_PKG / "package.xml").getroot()
        name_el = root.find("name")
        assert name_el is not None
        assert name_el.text == "yahboom_M3Pro_description"


# ===========================================================================
# Vendor URDF — structural correctness
# ===========================================================================


@pytest.mark.skipif(
    not VENDOR_PKG.exists(),
    reason="Optional vendor assets: run scripts/fetch_vendor_assets.py yahboom",
)
class TestVendorUrdfStructure:
    """Verify the M3Pro URDF has the expected links and joints."""

    def _root(self) -> ET.Element:
        return ET.parse(VENDOR_PKG / "urdf" / "M3Pro.urdf").getroot()

    def test_base_link_present(self) -> None:
        root = self._root()
        links = {lnk.attrib["name"] for lnk in root.findall("link")}
        assert "base_link" in links

    def test_five_arm_links_present(self) -> None:
        root = self._root()
        links = {lnk.attrib["name"] for lnk in root.findall("link")}
        for arm in ("arm1", "arm2", "arm3", "arm4", "arm5"):
            assert arm in links, f"Missing arm link: {arm}"

    def test_four_mecanum_wheel_links(self) -> None:
        root = self._root()
        links = {lnk.attrib["name"] for lnk in root.findall("link")}
        for wheel in ("lwheel1", "lwheel2", "rwheel1", "rwheel2"):
            assert wheel in links, f"Missing wheel link: {wheel}"

    def test_arm_joints_count(self) -> None:
        root = self._root()
        arm_joints = [
            j for j in root.findall("joint")
            if j.attrib["name"].startswith("arm") and j.attrib.get("type") == "revolute"
        ]
        assert len(arm_joints) == 5

    def test_arm4_joint_typo_preserved(self) -> None:
        """Vendor URDF has 'arm4_Joiint' (double 'i') — must stay as-is."""
        root = self._root()
        joint_names = {j.attrib["name"] for j in root.findall("joint")}
        assert "arm4_Joiint" in joint_names, (
            "arm4_Joiint typo must be preserved for M3Pro_config compatibility"
        )
        assert "arm4_Joint" not in joint_names, (
            "arm4_Joint (single 'i') must NOT exist — would break ros2_control config"
        )

    def test_arm_joint_limits(self) -> None:
        root = self._root()
        arm_joint_names = (
            "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joiint", "arm5_Joint"
        )
        for name in arm_joint_names:
            joint = next(
                (j for j in root.findall("joint") if j.attrib["name"] == name), None
            )
            assert joint is not None, f"Joint {name!r} missing"
            limit = joint.find("limit")
            assert limit is not None, f"Joint {name!r} has no <limit>"
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
            # All arm joints ±π/2
            assert abs(lower) > 1.0, f"{name} lower limit unexpectedly small: {lower}"
            assert abs(upper) > 1.0, f"{name} upper limit unexpectedly small: {upper}"

    def test_gripper_joint_present(self) -> None:
        root = self._root()
        joint_names = {j.attrib["name"] for j in root.findall("joint")}
        assert "rlink1_Joint" in joint_names

    def test_mesh_paths_use_package_prefix(self) -> None:
        """All mesh filenames reference package://yahboom_M3Pro_description/…"""
        root = self._root()
        meshes = root.findall(".//mesh")
        assert len(meshes) > 0
        for mesh in meshes:
            filename = mesh.attrib.get("filename", "")
            assert filename.startswith("package://yahboom_M3Pro_description/"), (
                f"Unexpected mesh path: {filename!r}"
            )


# ===========================================================================
# rosmaster_m3pro_description — our ament_cmake package
# ===========================================================================


class TestRosmasterDescriptionPackage:
    """Required files exist in rosmaster_m3pro_description."""

    def test_package_xml_exists(self) -> None:
        assert (DESC_PKG / "package.xml").is_file()

    def test_cmake_lists_exists(self) -> None:
        assert (DESC_PKG / "CMakeLists.txt").is_file()

    def test_urdf_xacro_exists(self) -> None:
        assert (DESC_PKG / "urdf" / "rosmaster_m3pro.urdf.xacro").is_file()

    def test_display_launch_exists(self) -> None:
        assert (DESC_PKG / "launch" / "display.launch.py").is_file()

    def test_rviz_config_exists(self) -> None:
        assert (DESC_PKG / "rviz" / "rosmaster_m3pro.rviz").is_file()

    def test_readme_exists(self) -> None:
        assert (DESC_PKG / "README.md").is_file()

    def test_package_xml_is_valid_xml(self) -> None:
        ET.parse(DESC_PKG / "package.xml")

    def test_package_xml_name(self) -> None:
        root = ET.parse(DESC_PKG / "package.xml").getroot()
        name_el = root.find("name")
        assert name_el is not None
        assert name_el.text == "rosmaster_m3pro_description"

    def test_package_xml_build_type_ament_cmake(self) -> None:
        root = ET.parse(DESC_PKG / "package.xml").getroot()
        export = root.find("export")
        assert export is not None
        build_type = export.find("build_type")
        assert build_type is not None
        assert build_type.text == "ament_cmake"

    def test_package_xml_depends_on_vendor(self) -> None:
        root = ET.parse(DESC_PKG / "package.xml").getroot()
        exec_deps = {el.text for el in root.findall("exec_depend")}
        assert "yahboom_M3Pro_description" in exec_deps

    def test_cmake_installs_urdf_launch_rviz(self) -> None:
        content = (DESC_PKG / "CMakeLists.txt").read_text()
        assert "launch" in content
        assert "rviz" in content
        assert "urdf" in content


# ===========================================================================
# xacro source — static content checks (no xacro runtime)
# ===========================================================================


class TestXacroContent:
    """Static string checks on the xacro without running xacro itself."""

    def _xacro_text(self) -> str:
        return (DESC_PKG / "urdf" / "rosmaster_m3pro.urdf.xacro").read_text()

    def test_xacro_includes_vendor_urdf(self) -> None:
        text = self._xacro_text()
        assert "yahboom_M3Pro_description" in text
        assert "M3Pro.urdf" in text

    def test_xacro_has_world_link(self) -> None:
        assert 'name="world"' in self._xacro_text()

    def test_xacro_has_world_to_base_joint(self) -> None:
        text = self._xacro_text()
        assert "world_to_base" in text
        assert 'type="fixed"' in text

    def test_xacro_has_ros2_control_block(self) -> None:
        assert "ros2_control" in self._xacro_text()

    def test_xacro_ros2_control_uses_mock_hardware(self) -> None:
        assert "mock_components/GenericSystem" in self._xacro_text()

    def test_xacro_ros2_control_covers_all_arm_joints(self) -> None:
        text = self._xacro_text()
        for joint in (
            "arm1_Joint", "arm2_Joint", "arm3_Joint", "arm4_Joiint", "arm5_Joint"
        ):
            assert joint in text, f"ros2_control block missing joint {joint!r}"

    def test_xacro_ros2_control_covers_gripper(self) -> None:
        assert "rlink1_Joint" in self._xacro_text()

    def test_xacro_arm4_typo_preserved(self) -> None:
        """arm4_Joiint (double 'i') must be in the ros2_control block."""
        assert "arm4_Joiint" in self._xacro_text()

    def test_display_launch_references_xacro(self) -> None:
        launch = (DESC_PKG / "launch" / "display.launch.py").read_text()
        assert "rosmaster_m3pro.urdf.xacro" in launch

    def test_display_launch_starts_rsp(self) -> None:
        launch = (DESC_PKG / "launch" / "display.launch.py").read_text()
        assert "robot_state_publisher" in launch

    def test_display_launch_starts_jsp_gui(self) -> None:
        launch = (DESC_PKG / "launch" / "display.launch.py").read_text()
        assert "joint_state_publisher_gui" in launch

    def test_display_launch_starts_rviz(self) -> None:
        launch = (DESC_PKG / "launch" / "display.launch.py").read_text()
        assert "rviz2" in launch

    def test_rviz_config_fixed_frame_is_base_link(self) -> None:
        rviz = (DESC_PKG / "rviz" / "rosmaster_m3pro.rviz").read_text()
        assert "Fixed Frame: base_link" in rviz
