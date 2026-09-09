#!/usr/bin/env python3
"""Obtain vendor inputs separately, then stage the required files locally.

Synria: https://github.com/Synria-Robotics/Synria-Robot-Descriptions
  git clone https://github.com/Synria-Robotics/Synria-Robot-Descriptions /tmp/synria
  Review the vendor terms and choose a revision, then pass the directory containing
  urdf/ and meshes/ (normally /tmp/synria/synriard) with --source-id COMMIT.
Yahboom: https://github.com/YahboomTechnology/ROSMASTER-M3PRO
  https://www.yahboom.net/study/ROSMASTER-M3PRO -> Download -> Code-Firmware.
  Extract the archive yourself; pass yahboomcar_ws/src with --source-id identifying
  the archive/release. If unavailable, request both yahboom_M3Pro_description and
  M3Pro_config from Yahboom product support, or obtain them from your robot.

No pinned archive URL or original upstream commit is recorded in this repo.
This script does not infer a license or promise historical byte reproduction.
It never executes downloaded code, downloads implicitly, or overwrites files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Alicia_D_v5_6_gripper_50mm"
MESH_DIR = Path("meshes/Alicia_D_v5_6/follower_standard")
DYNAMICS = {
    "Joint1": ("0.5", "0.1"),
    "Joint2": ("0.5", "0.1"),
    "Joint3": ("0.4", "0.1"),
    "Joint4": ("0.3", "0.05"),
    "Joint5": ("0.3", "0.05"),
    "Joint6": ("0.2", "0.05"),
    "left_finger": ("0.1", "0.02"),
    "right_finger": ("0.1", "0.02"),
}


def read_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular vendor file missing (or symlink): {path}")
    return path.read_bytes()


def synria_plan(source: Path) -> tuple[dict[Path, bytes], dict[str, str]]:
    urdf = source / "urdf/Alicia_D_v5_6" / f"{MODEL}.urdf"
    original = read_file(urdf)
    robot = ET.fromstring(original)
    if robot.tag != "robot" or robot.get("name") != MODEL:
        raise ValueError(f"Expected Synria {MODEL}: {urdf}")
    joints = {j.get("name"): j for j in robot.findall("joint")}
    if not set(DYNAMICS) <= joints.keys() or robot.find("link[@name='tool0']") is None:
        raise ValueError("Synria source lacks expected joints/tool0; review this revision.")
    # Preserve the existing first-party Isaac adaptation, not vendor geometry.
    for child in robot.findall("mujoco"):
        robot.remove(child)
    for name, (damping, friction) in DYNAMICS.items():
        joint = joints[name]
        for old in joint.findall("dynamics"):
            joint.remove(old)
        ET.SubElement(joint, "dynamics", damping=damping, friction=friction)
    for joint in robot.findall("joint[@type='fixed']"):
        for axis in joint.findall("axis"):
            joint.remove(axis)

    inputs = {str(urdf.relative_to(source)): hashlib.sha256(original).hexdigest()}
    meshes: dict[str, bytes] = {}
    for mesh in robot.findall(".//mesh"):
        name = Path(mesh.attrib["filename"]).name
        payload = read_file(source / MESH_DIR / name)
        meshes[name] = payload
        inputs[str(MESH_DIR / name)] = hashlib.sha256(payload).hexdigest()
    if not meshes:
        raise ValueError("Synria URDF has no mesh references.")
    plan: dict[Path, bytes] = {}
    destinations = {
        Path("isaac/usd/robots/synria_6dof_arm.urdf"): "./meshes/Alicia_D_v5_6/follower_standard",
        Path(
            "isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.urdf"
        ): "../meshes/Alicia_D_v5_6/follower_standard",
        Path(
            "ros2_ws/src/synria_arm_description/urdf/synria_6dof_arm.urdf"
        ): "./meshes/Alicia_D_v5_6/follower_standard",
    }
    for dest, prefix in destinations.items():
        model = copy.deepcopy(robot)
        for mesh in model.findall(".//mesh"):
            mesh.set("filename", f"{prefix}/{Path(mesh.attrib['filename']).name}")
        ET.indent(model)
        plan[dest] = ET.tostring(model, encoding="utf-8", xml_declaration=True) + b"\n"
    for name, payload in meshes.items():
        plan[Path("isaac/usd/robots") / MESH_DIR / name] = payload
        plan[Path("ros2_ws/src/synria_arm_description/urdf") / MESH_DIR / name] = payload
    return plan, inputs


def yahboom_plan(source: Path) -> tuple[dict[Path, bytes], dict[str, str]]:
    plan: dict[Path, bytes] = {}
    inputs: dict[str, str] = {}
    for package in ("yahboom_M3Pro_description", "M3Pro_config"):
        directory = source / package
        manifest = ET.fromstring(read_file(directory / "package.xml"))
        if manifest.findtext("name") != package:
            raise ValueError(f"Wrong package name in {directory / 'package.xml'}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Vendor package contains a symlink: {path}")
            if path.is_file() and "__pycache__" not in path.parts:
                relative = path.relative_to(source)
                payload = read_file(path)
                inputs[str(relative)] = hashlib.sha256(payload).hexdigest()
                plan[Path("ros2_ws/src") / relative] = payload
    required = (
        "yahboom_M3Pro_description/urdf/M3Pro.urdf",
        "yahboom_M3Pro_description/setup.py",
        "yahboom_M3Pro_description/setup.cfg",
        "yahboom_M3Pro_description/launch/display_launch.py",
        "M3Pro_config/CMakeLists.txt",
        "M3Pro_config/config/M3Pro.urdf.xacro",
        "M3Pro_config/config/M3Pro.srdf",
        "M3Pro_config/config/joint_limits.yaml",
        "M3Pro_config/launch/demo.launch.py",
    )
    for relative in required:
        if Path("ros2_ws/src") / relative not in plan:
            raise ValueError(f"Incomplete Yahboom source: missing {relative}")
    dest = Path("ros2_ws/src/yahboom_M3Pro_description/urdf/M3Pro.urdf")
    original = plan[dest]
    # Preserve the two package-name fixes recorded in NOTICE without rewriting XML.
    old = b"package://dofbot_M3Pro_descripton/meshes/arm_base_Link.STL"
    new = b"package://yahboom_M3Pro_description/meshes/arm_base_Link.STL"
    plan[dest] = original.replace(old, new)
    robot = ET.fromstring(plan[dest])
    if robot.get("name") != "M3Pro":
        raise ValueError("Expected Yahboom robot name M3Pro.")
    if robot.find("joint[@name='arm4_Joiint']") is None:
        raise ValueError("Yahboom joint names changed; review MoveIt compatibility.")
    meshes = robot.findall(".//mesh")
    if not meshes:
        raise ValueError("Yahboom URDF has no mesh references.")
    for mesh in meshes:
        filename = mesh.attrib["filename"]
        prefix = "package://yahboom_M3Pro_description/"
        if not filename.startswith(prefix):
            raise ValueError(f"Unresolved Yahboom mesh reference: {filename}")
        relative = Path(filename.removeprefix(prefix))
        if (
            ".." in relative.parts
            or Path("ros2_ws/src/yahboom_M3Pro_description") / relative not in plan
        ):
            raise ValueError(f"Missing Yahboom mesh: {filename}")
    return plan, inputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("asset", choices=("synria", "yahboom", "cad", "robotiq", "scenes"))
    parser.add_argument("--source", type=Path, help="User-obtained vendor directory; see above")
    parser.add_argument(
        "--source-id", help="User-supplied upstream commit or archive/release identifier"
    )
    parser.add_argument(
        "--dest-root", type=Path, default=ROOT, help="Installation root (default: this checkout)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate source and destinations without writing"
    )
    args = parser.parse_args()
    manual = {
        "cad": (
            "No direct CAD download URL is recorded. Request ROSMASTER-M3Pro.x_t from "
            "Yahboom product support. Place it in "
            "isaac/usd/robots/yahboom-rosmaster-m3-pro/ and use the retained "
            "convert_cad_to_usd.sh after configuring your Isaac path. Conversion is not "
            "validated by this script."
        ),
        "robotiq": (
            "Obtain Robotiq 2F-85 assets through NVIDIA Isaac Sim under NVIDIA/vendor "
            "terms. The assembly script uses the configured Isaac asset root at "
            "Isaac/Robots/Robotiq/2F-85/configuration/Robotiq_2F_85_config.usd. No "
            "pinned standalone URL or revision is recorded; historical flattened "
            "reproduction is unverified. See "
            "isaac/scripts/diagnostics/assemble_synria_robotiq.py."
        ),
        "scenes": (
            "Mirrored scene layers reference an external Alicia-D-ROS2 checkout "
            "(scripts/isaac_sim/scenes/*table_norobot.usda). Request those game-table "
            "scene files and their source revision from Synria Robotics if absent. No "
            "pinned direct download is recorded. Configure and run "
            "isaac/usd/scenes/synria_mirrored/generate_mirrored_scenes.py locally."
        ),
    }
    try:
        if args.asset in manual:
            raise ValueError(manual[args.asset])
        if args.source is None or not args.source_id:
            raise ValueError(
                f"{args.asset}: supply --source and --source-id after obtaining the vendor files. "
                "Run --help for official sources and request instructions. Nothing installed."
            )
        source = args.source.expanduser().resolve()
        plan, inputs = (synria_plan if args.asset == "synria" else yahboom_plan)(source)
        root = args.dest_root.expanduser().resolve()
        receipt = Path(".vendor-assets") / f"{args.asset}.json"
        plan[receipt] = (
            json.dumps(
                {
                    "asset": args.asset,
                    "source_id": args.source_id,
                    "source_id_basis": "user supplied; not independently verified",
                    "input_sha256": inputs,
                    "installed_sha256": {
                        str(p): hashlib.sha256(b).hexdigest() for p, b in plan.items()
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
        # Validate the complete plan before creating any files; refuse replacement.
        for path in plan:
            target = root / path
            if not target.resolve().is_relative_to(root) or target.exists() or target.is_symlink():
                raise ValueError(
                    f"Destination exists or escapes root: {target}; nothing installed."
                )
        if args.check:
            print(f"Validated {args.asset}: {len(plan) - 1} asset files; no writes.")
            return 0
        created: list[Path] = []
        try:
            for path, payload in plan.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    created.append(target)
                    handle.write(payload)
        except BaseException:
            for path in reversed(created):
                path.unlink()
            raise
        print(f"Installed {args.asset} locally; receipt: {root / receipt}")
        if args.asset == "synria":
            print(
                'Next: "$ISAAC_PYTHON" isaac/scripts/import_synria_urdf_51.py --out '
                '.vendor-assets/synria-import/synria_6dof_arm.usd'
            )
            print(
                "Then: cp -R .vendor-assets/synria-import/configuration "
                "isaac/usd/robots/synria_6dof_arm_v2/"
            )
            print(
                "The retained first-party v2 root layer references the generated "
                "configuration layers. Revalidate simulation; historical conversion parity "
                "is not established."
            )
        return 0
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"Vendor setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
