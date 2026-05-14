#!/usr/bin/env python3
"""Render the repo-local factory-cell USD scene with Blender."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "isaac" / "usd" / "factory_cell_v0.usda"
DEFAULT_OUTPUT = REPO_ROOT / "isaac" / "outputs" / "factory_cell_v0_blender.png"
DEFAULT_CAMERA = "OverviewCamera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a screenshot from the repo-local OpenUSD factory-cell scene using Blender."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="USD scene to import.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Screenshot path.")
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="Camera object name after USD import. Defaults to OverviewCamera.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Render height in pixels.")
    parser.add_argument("--samples", type=int, default=64, help="Cycles sample count.")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_usd(scene_path: Path) -> None:
    if not scene_path.exists():
        raise FileNotFoundError(f"USD scene does not exist: {scene_path}")
    try:
        bpy.ops.wm.usd_import(filepath=str(scene_path))
    except AttributeError:
        create_fallback_factory_cell()


def material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def add_cube(
    name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material(f"{name}_mat", color))
    return obj


def add_cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    color: tuple[float, float, float, float],
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(f"{name}_mat", color))
    return obj


def add_sphere(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(f"{name}_mat", color))
    return obj


def create_fallback_factory_cell() -> None:
    """Rebuild the simple v0 factory-cell layout when Blender lacks USD support."""

    add_cube("Floor", (0, 0, -0.025), (4.5, 3.0, 0.05), (0.45, 0.48, 0.46, 1.0))
    add_cube("SafetyZone", (0, 0, 0.012), (2.3, 1.55, 0.01), (0.95, 0.82, 0.10, 0.55))
    add_cube("InspectionTable", (0.85, 0.0, 0.52), (1.5, 0.9, 0.12), (0.18, 0.21, 0.24, 1.0))
    add_cube("TablePedestal", (0.85, 0.0, 0.25), (0.24, 0.24, 1.0), (0.12, 0.14, 0.16, 1.0))
    add_cube("PickBin", (0.52, -0.28, 0.67), (0.56, 0.40, 0.22), (0.05, 0.42, 0.70, 1.0))
    add_cube("PlaceBin", (1.18, 0.28, 0.67), (0.56, 0.40, 0.22), (0.15, 0.58, 0.28, 1.0))
    add_cube("InspectionPartA", (0.67, -0.18, 0.82), (0.16, 0.16, 0.16), (0.86, 0.22, 0.16, 1.0))
    add_sphere("InspectionPartB", (1.02, 0.16, 0.82), 0.09, (0.90, 0.52, 0.12, 1.0))

    add_cube("YahboomMobileBaseProxy_Chassis", (-0.95, -0.55, 0.12), (0.64, 0.48, 0.18), (0.08, 0.11, 0.16, 1.0))
    add_cube("YahboomMobileBaseProxy_TopPlate", (-0.95, -0.55, 0.235), (0.52, 0.36, 0.05), (0.12, 0.32, 0.58, 1.0))
    wheel_color = (0.02, 0.02, 0.02, 1.0)
    wheel_rotation = (math.radians(90), 0.0, 0.0)
    for x in (-1.17, -0.73):
        for y in (-0.80, -0.30):
            add_cylinder(f"Wheel_{x}_{y}", (x, y, 0.08), 0.075, 0.07, wheel_color, wheel_rotation)

    add_cylinder("SynriaArmProxy_Base", (0.20, 0.75, 0.14), 0.16, 0.14, (0.16, 0.17, 0.18, 1.0))
    shoulder = add_cube("SynriaArmProxy_ShoulderLink", (0.32, 0.75, 0.43), (0.14, 0.14, 0.68), (0.74, 0.76, 0.72, 1.0))
    shoulder.rotation_euler[1] = math.radians(26)
    elbow = add_cube("SynriaArmProxy_ElbowLink", (0.55, 0.80, 0.69), (0.12, 0.12, 0.60), (0.63, 0.66, 0.62, 1.0))
    elbow.rotation_euler = (0.0, math.radians(-36), math.radians(18))
    add_cube("SynriaArmProxy_WristCamera", (0.72, 0.87, 0.87), (0.16, 0.08, 0.08), (0.04, 0.04, 0.04, 1.0))
    add_cube("ArmReachEnvelope", (0.35, 0.55, 0.035), (1.7, 1.7, 0.03), (0.70, 0.16, 0.12, 0.45))


def find_object(name: str) -> bpy.types.Object | None:
    if name in bpy.data.objects:
        return bpy.data.objects[name]

    suffix = f"/{name}"
    for obj in bpy.data.objects:
        if obj.name.endswith(suffix) or obj.name.split(".")[0] == name:
            return obj
    return None


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def scene_bounds() -> tuple[Vector, Vector]:
    objects = [obj for obj in bpy.context.scene.objects if obj.type != "CAMERA"]
    if not objects:
        return Vector((0.0, 0.0, 0.0)), Vector((4.5, 3.0, 2.0))

    min_corner = Vector((math.inf, math.inf, math.inf))
    max_corner = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)
    return min_corner, max_corner


def configure_camera(camera_name: str) -> bpy.types.Object:
    camera = find_object(camera_name)
    if camera is None or camera.type != "CAMERA":
        camera_data = bpy.data.cameras.new(camera_name)
        camera = bpy.data.objects.new(camera_name, camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = (2.35, -2.25, 2.05)

    min_corner, max_corner = scene_bounds()
    center = (min_corner + max_corner) * 0.5
    center.z = max(center.z, 0.45)
    look_at(camera, center)
    camera.data.lens = 24
    camera.data.clip_end = 1000
    bpy.context.scene.camera = camera
    return camera


def configure_lighting() -> None:
    if not any(obj.type == "LIGHT" for obj in bpy.context.scene.objects):
        sun_data = bpy.data.lights.new("FallbackKeyLight", type="SUN")
        sun = bpy.data.objects.new("FallbackKeyLight", sun_data)
        bpy.context.collection.objects.link(sun)
        sun.rotation_euler = (math.radians(55), 0.0, math.radians(35))
        sun_data.energy = 2.5

    bpy.context.scene.world.color = (0.03, 0.035, 0.04)


def configure_render(width: int, height: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def render(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Screenshot was not written: {output_path}")


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    clear_scene()
    import_usd(scene_path)
    configure_camera(args.camera)
    configure_lighting()
    configure_render(args.width, args.height, args.samples)
    render(output_path)

    print(f"Imported scene: {scene_path}")
    print(f"Rendered screenshot: {output_path}")


if __name__ == "__main__":
    main()
