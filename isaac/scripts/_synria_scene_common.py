"""Shared helpers for Synria tabletop training-scene builders.

Each scene builder (Ludo, chess, checkers) shares the same floor / table /
arm-mount / lighting / camera layout; only the board and the pieces change.
This module owns the shared parts so they cannot drift across scenes.

Import this module from a scene builder script. Importing it boots Isaac
Sim's ``SimulationApp`` headlessly — the import side-effect is intentional
and must happen before any ``omni`` or ``pxr`` import. Re-export ``sim_app``
to your script so you can ``sim_app.close()`` in a ``finally`` block.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Boot Isaac Sim's SimulationApp first — required before any omni / pxr imports.
try:
    from isaacsim import SimulationApp  # noqa: F401  (Isaac Sim 5.1 entry)

    sim_app = SimulationApp({"headless": True})
except ImportError as exc:  # pragma: no cover - hardware-gated import
    sys.stderr.write(
        "This module requires Isaac Sim 5.1's Python environment. "
        "Activate the Isaac Sim venv before running, e.g. "
        "`source ~/.venv/isaacsim5/bin/activate`.\n"
    )
    raise SystemExit(1) from exc

import omni.usd as _omni_usd  # noqa: E402
from omni.isaac.core.utils.stage import create_new_stage as _create_new_stage  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402


def create_new_stage():
    """Create a new USD stage and return the Usd.Stage object.

    In Isaac Sim 5.1, omni.isaac.core.utils.stage.create_new_stage() returns
    True instead of the stage — fetch it from the context afterward.
    """
    _create_new_stage()
    return _omni_usd.get_context().get_stage()


def save_stage(stage, path: str) -> None:
    """Export the composed stage to a USDA file."""
    stage.Export(path)

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNRIA_ARM_USD = REPO_ROOT / "isaac" / "usd" / "robots" / "synria_6dof_arm_v2" / "configuration" / "synria_6dof_arm_physics.usd"

# ---------------------------------------------------------------------------
# Shared layout constants (meters)
# ---------------------------------------------------------------------------

TABLE_HEIGHT = 0.75
TABLE_THICKNESS = 0.04
TABLE_SIZE_X = 1.20
TABLE_SIZE_Y = 0.80
TABLE_TOP_Z = TABLE_HEIGHT + TABLE_THICKNESS / 2.0
TABLE_SURFACE_Z = TABLE_HEIGHT + TABLE_THICKNESS

# Synria arm reach is 0.65 m; mount the base near one edge of the table.
ARM_BASE_X = -0.50
ARM_BASE_Y = 0.0
ARM_BASE_Z = TABLE_SURFACE_Z

# Center every game board at the same target so all three scenes share the
# arm reach budget.
BOARD_CENTER_X = 0.10
BOARD_CENTER_Y = 0.0

# ---------------------------------------------------------------------------
# Primitive makers
# ---------------------------------------------------------------------------


def set_display_color(prim: UsdGeom.Gprim, rgb: tuple[float, float, float]) -> None:
    prim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def make_cube(stage, path: str, size: float, position: tuple[float, float, float],
              rgb: tuple[float, float, float], rigid: bool = True) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(*position))
    set_display_color(cube, rgb)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    return cube


def make_box(stage, path: str, size: tuple[float, float, float],
             position: tuple[float, float, float],
             rgb: tuple[float, float, float], rigid: bool = False) -> UsdGeom.Cube:
    """Non-cube rectangular box, implemented as a unit Cube scaled per-axis."""
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(box)
    api.SetTranslate(Gf.Vec3d(*position))
    api.SetScale(Gf.Vec3f(*size))
    set_display_color(box, rgb)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    return box


def make_cylinder(stage, path: str, radius: float, height: float,
                  position: tuple[float, float, float],
                  rgb: tuple[float, float, float], rigid: bool = True) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(*position))
    set_display_color(cyl, rgb)
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cyl.GetPrim())
    return cyl


def make_sphere(stage, path: str, radius: float,
                position: tuple[float, float, float],
                rgb: tuple[float, float, float], rigid: bool = True) -> UsdGeom.Sphere:
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(radius)
    UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*position))
    set_display_color(sph, rgb)
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(sph.GetPrim())
    return sph


def make_cone(stage, path: str, radius: float, height: float,
              position: tuple[float, float, float],
              rgb: tuple[float, float, float], rigid: bool = True) -> UsdGeom.Cone:
    cone = UsdGeom.Cone.Define(stage, path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(height)
    cone.CreateAxisAttr("Z")
    UsdGeom.XformCommonAPI(cone).SetTranslate(Gf.Vec3d(*position))
    set_display_color(cone, rgb)
    UsdPhysics.CollisionAPI.Apply(cone.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(cone.GetPrim())
    return cone


# ---------------------------------------------------------------------------
# Shared scene parts
# ---------------------------------------------------------------------------


def set_stage_metadata(stage) -> None:
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    UsdPhysics.Scene.Define(stage, "/World/Physics/PhysicsScene")


def add_lighting(stage) -> None:
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/SunLight")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-45.0, 35.0, 0.0))

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/EnvLight")
    dome.CreateIntensityAttr(600.0)


def add_cameras(stage, board_top_z: float) -> None:
    """Overhead board camera + operator perspective camera.

    Args:
        board_top_z: Z height of the board's top surface, in meters. The
            overhead camera is placed 0.55 m above it.
    """
    overhead = UsdGeom.Camera.Define(stage, "/World/Cameras/OverheadBoardCamera")
    overhead.CreateFocalLengthAttr(18.0)
    UsdGeom.XformCommonAPI(overhead).SetTranslate(
        Gf.Vec3d(BOARD_CENTER_X, BOARD_CENTER_Y, board_top_z + 0.55)
    )
    UsdGeom.XformCommonAPI(overhead).SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0))

    operator = UsdGeom.Camera.Define(stage, "/World/Cameras/OperatorCamera")
    operator.CreateFocalLengthAttr(24.0)
    UsdGeom.XformCommonAPI(operator).SetTranslate(Gf.Vec3d(0.85, -0.85, 1.20))
    UsdGeom.XformCommonAPI(operator).SetRotate(Gf.Vec3f(-25.0, 0.0, 45.0))


def add_floor(stage) -> None:
    make_box(
        stage,
        "/World/Floor",
        size=(5.0, 5.0, 0.02),
        position=(0.0, 0.0, -0.01),
        rgb=(0.55, 0.55, 0.55),
        rigid=False,
    )


def add_table(stage) -> None:
    make_box(
        stage,
        "/World/Table/Top",
        size=(TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_THICKNESS),
        position=(0.0, 0.0, TABLE_TOP_Z),
        rgb=(0.78, 0.65, 0.50),
        rigid=False,
    )
    leg_inset = 0.05
    leg_size = (0.04, 0.04, TABLE_HEIGHT)
    leg_z = TABLE_HEIGHT / 2.0
    for sign_x in (-1.0, 1.0):
        for sign_y in (-1.0, 1.0):
            x = sign_x * (TABLE_SIZE_X / 2.0 - leg_inset)
            y = sign_y * (TABLE_SIZE_Y / 2.0 - leg_inset)
            tag = f"{'p' if sign_x > 0 else 'n'}{'p' if sign_y > 0 else 'n'}"
            make_box(
                stage,
                f"/World/Table/Leg_{tag}",
                size=leg_size,
                position=(x, y, leg_z),
                rgb=(0.55, 0.40, 0.28),
                rigid=False,
            )


def add_synria_arm(stage) -> None:
    """Reference the Synria 6DOF arm onto the table top."""
    arm_path = "/World/SynriaArm"
    xform = UsdGeom.Xform.Define(stage, arm_path)
    UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(ARM_BASE_X, ARM_BASE_Y, ARM_BASE_Z))
    # Reference the arm USD by relative path.
    xform.GetPrim().GetReferences().AddReference("../../robots/synria_6dof_arm_v2/configuration/synria_6dof_arm_physics.usd")


# ---------------------------------------------------------------------------
# Staging zones for pick-and-place tasks
# ---------------------------------------------------------------------------
#
# Four rectangular pads flank the board on all four sides (left, right,
# top, bottom — orientation from the operator camera's point of view).
# They are the targets for the primary pick-and-place training task: pick
# a piece from the board, place it on a staging zone, return it to the
# board.
#
# Zone dimensions are sized to hold one row of pieces from any of the three
# games (Ludo tokens, chess pieces, checker discs). Color-coded so the
# overhead camera can identify them in the observation image.
#
# Orientation:
#   - `top`    sits at -X of the board (the side closest to the arm base)
#   - `bottom` sits at +X of the board (the side closest to the operator)
#   - `left`   sits at -Y of the board (operator camera's left)
#   - `right`  sits at +Y of the board (operator camera's right)

STAGING_ZONE_THICKNESS = 0.0012  # 1.2 mm pad
STAGING_ZONE_OFFSET = 0.06       # pad inner edge sits this far from the board edge

STAGING_ZONE_COLORS: dict[str, tuple[float, float, float]] = {
    "left": (0.15, 0.55, 0.85),    # blue
    "right": (0.85, 0.55, 0.15),   # orange
    "bottom": (0.30, 0.75, 0.30),  # green
    "top": (0.75, 0.30, 0.75),     # purple
}


def add_staging_zones(stage, board_size: float) -> dict[str, tuple[float, float]]:
    """Add left / right / top / bottom staging pads around a square board.

    Args:
        stage: the USD stage to write into.
        board_size: edge length of the (square) board, meters. Pads are
            sized as ``(board_size, 0.10)`` for left/right and
            ``(0.10, board_size)`` for top/bottom — large enough to hold
            one row of game pieces.

    Returns:
        Dict ``{zone_name: (x, y)}`` mapping each zone name to its center
        XY on the table surface. Useful as a reference for task configs.
    """
    UsdGeom.Xform.Define(stage, "/World/StagingZones")

    long_pad = board_size  # match the board edge
    short_pad = 0.10       # 10 cm depth perpendicular to the board edge

    half_board = board_size / 2.0
    edge_clearance = half_board + STAGING_ZONE_OFFSET + short_pad / 2.0

    pad_z = TABLE_SURFACE_Z + STAGING_ZONE_THICKNESS / 2.0

    zones: dict[str, tuple[float, float]] = {
        # Left: -Y of the board (operator's left from the operator camera)
        "left": (BOARD_CENTER_X, BOARD_CENTER_Y - edge_clearance),
        # Right: +Y of the board
        "right": (BOARD_CENTER_X, BOARD_CENTER_Y + edge_clearance),
        # Bottom: +X of the board (away from the arm, toward the operator)
        "bottom": (BOARD_CENTER_X + edge_clearance, BOARD_CENTER_Y),
        # Top: -X of the board (toward the arm base)
        "top": (BOARD_CENTER_X - edge_clearance, BOARD_CENTER_Y),
    }

    for name, (cx, cy) in zones.items():
        if name in ("bottom", "top"):
            # Pads aligned along Y, depth along X
            size = (short_pad, long_pad, STAGING_ZONE_THICKNESS)
        else:
            # Pads aligned along X, depth along Y
            size = (long_pad, short_pad, STAGING_ZONE_THICKNESS)
        make_box(
            stage,
            f"/World/StagingZones/{name}",
            size=size,
            position=(cx, cy, pad_z),
            rgb=STAGING_ZONE_COLORS[name],
            rigid=False,
        )

    return zones


def build_common_scene(stage, board_top_z: float) -> None:
    """Set up the shared parts of every Synria tabletop scene.

    Defines /World, /World/Lighting, /World/Cameras, /World/Table, and adds
    the floor, table, arm reference, lighting, cameras, and physics scene.
    Game-specific board + pieces are layered on top by the caller.

    Args:
        board_top_z: Z height of the board top, used to place the overhead
            camera. The board itself is the caller's responsibility.
    """
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Lighting")
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    UsdGeom.Xform.Define(stage, "/World/Table")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    set_stage_metadata(stage)
    add_lighting(stage)
    add_cameras(stage, board_top_z=board_top_z)
    add_floor(stage)
    add_table(stage)
    add_synria_arm(stage)


def ensure_arm_usd_exists() -> None:
    if not SYNRIA_ARM_USD.exists():
        raise FileNotFoundError(
            f"Synria arm USD not found at {SYNRIA_ARM_USD}. "
            "Verify isaac/usd/robots/synria_6dof_arm_v2/ is present."
        )


__all__ = [
    "sim_app",
    "create_new_stage",
    "save_stage",
    "Gf",
    "Sdf",
    "UsdGeom",
    "UsdLux",
    "UsdPhysics",
    "REPO_ROOT",
    "TABLE_HEIGHT",
    "TABLE_THICKNESS",
    "TABLE_TOP_Z",
    "TABLE_SURFACE_Z",
    "ARM_BASE_X",
    "ARM_BASE_Y",
    "ARM_BASE_Z",
    "BOARD_CENTER_X",
    "BOARD_CENTER_Y",
    "STAGING_ZONE_COLORS",
    "make_cube",
    "make_box",
    "make_cylinder",
    "make_sphere",
    "make_cone",
    "set_display_color",
    "build_common_scene",
    "ensure_arm_usd_exists",
    "add_staging_zones",
]
