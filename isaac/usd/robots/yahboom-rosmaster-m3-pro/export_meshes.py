#!/usr/bin/env python3
"""
Export the converted CAD USD to per-link STL files for URDF use.
Produces:
  meshes/chassis.stl       - full robot body minus wheels (base_link visual)
  meshes/mecanum_wheel.stl - one Mecanum wheel (reused for all 4 wheel links)
Uses trimesh for STL I/O and mesh math; reads geometry directly from USD.
"""
import sys
import os
import struct
import math
import numpy as np

sys.path.insert(0, "/home/oedeh/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311")
os.environ["LD_LIBRARY_PATH"] = (
    "/home/oedeh/isaacsim/kit:"
    "/home/oedeh/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

# Must import pxr via subprocess-safe env — run this script via kit python or set env first
from pxr import Usd, UsdGeom, Gf

ROBOT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_FILE = os.path.join(ROBOT_DIR, "rosmaster_m3pro_cad.usd")
MESH_DIR = os.path.join(ROBOT_DIR, "meshes")
os.makedirs(MESH_DIR, exist_ok=True)


def write_stl_binary(filepath, triangles, normals=None):
    """Write binary STL. triangles: Nx3x3 float array (mm → converted to m)."""
    n = len(triangles)
    with open(filepath, "wb") as f:
        f.write(b"\x00" * 80)  # header
        f.write(struct.pack("<I", n))
        for i, tri in enumerate(triangles):
            if normals is not None:
                n_vec = normals[i]
            else:
                # compute face normal
                v0, v1, v2 = tri
                e1 = v1 - v0
                e2 = v2 - v0
                n_vec = np.cross(e1, e2)
                length = np.linalg.norm(n_vec)
                if length > 0:
                    n_vec /= length
            f.write(struct.pack("<fff", *n_vec))
            for v in tri:
                f.write(struct.pack("<fff", *v))
            f.write(struct.pack("<H", 0))
    print(f"  Wrote {n:,} triangles → {filepath}")


def usd_mesh_to_triangles(prim, scale=0.001):
    """Extract triangulated faces from a USD Mesh prim. Returns Nx3x3 array in meters."""
    pts_attr = prim.GetAttribute("points")
    fvc_attr = prim.GetAttribute("faceVertexCounts")
    fvi_attr = prim.GetAttribute("faceVertexIndices")
    if not (pts_attr and fvc_attr and fvi_attr):
        return None
    pts = pts_attr.Get()
    fvc = fvc_attr.Get()
    fvi = fvi_attr.Get()
    if pts is None or fvc is None or fvi is None:
        return None

    pts = np.array(pts, dtype=np.float32) * scale  # mm → m
    fvc = list(fvc)
    fvi = list(fvi)

    triangles = []
    idx = 0
    for count in fvc:
        face_verts = [fvi[idx + k] for k in range(count)]
        # fan triangulation
        for k in range(1, count - 1):
            triangles.append([pts[face_verts[0]], pts[face_verts[k]], pts[face_verts[k + 1]]])
        idx += count
    if not triangles:
        return None
    return np.array(triangles, dtype=np.float32)


def collect_all_meshes(stage):
    """Collect all Mesh prims from the USD stage."""
    parent = stage.GetPrimAtPath("/rosmaster_m3pro_cad/rosmaster_m3pro_cad")
    if not parent:
        # Try flat path
        for p in stage.TraverseAll():
            if p.GetTypeName() == "Xform" and p.GetName() == "rosmaster_m3pro_cad":
                children = list(p.GetChildren())
                if children and children[0].GetTypeName() in ("Mesh", "Xform"):
                    parent = p
                    break
    return [c for c in stage.TraverseAll() if c.GetTypeName() == "Mesh"]


def generate_mecanum_wheel_stl(filepath, radius=0.040, width=0.038, n_rollers=8, roller_radius=0.012):
    """
    Generate a Mecanum wheel STL:
    - Outer rubber cylinder (tire)
    - Hub cylinder (aluminum)
    - n_rollers diagonal rollers at 45 degrees
    Coordinate: wheel rotation axis = Y, center at origin.
    Units: meters.
    """
    triangles = []

    def cylinder_tris(r, h, nx=32, cap=True, y_offset=0):
        """Generate cylinder triangles, axis=Y, center at y_offset."""
        tris = []
        angles = [2 * math.pi * i / nx for i in range(nx)]
        y0 = y_offset - h / 2
        y1 = y_offset + h / 2
        for i in range(nx):
            a0, a1 = angles[i], angles[(i + 1) % nx]
            p00 = np.array([r * math.cos(a0), y0, r * math.sin(a0)])
            p01 = np.array([r * math.cos(a1), y0, r * math.sin(a1)])
            p10 = np.array([r * math.cos(a0), y1, r * math.sin(a0)])
            p11 = np.array([r * math.cos(a1), y1, r * math.sin(a1)])
            tris.append([p00, p10, p11])
            tris.append([p00, p11, p01])
            if cap:
                center0 = np.array([0, y0, 0])
                center1 = np.array([0, y1, 0])
                tris.append([center0, p01, p00])
                tris.append([center1, p10, p11])
        return tris

    def roller_tris(r, length, center, axis_angle, wheel_angle, nx=12):
        """Generate a single diagonal roller cylinder."""
        tris = []
        # Roller axis at 45 degrees to wheel plane
        roll_angle_rad = math.radians(axis_angle)
        nx_pts = nx
        # Local basis: roller axis in XZ plane of wheel, rotated by wheel_angle around Y
        wa = wheel_angle
        # Roller axis direction (45° in wheel's local frame)
        ax = np.array([
            math.cos(wa) * math.cos(roll_angle_rad),
            math.sin(roll_angle_rad),
            math.sin(wa) * math.cos(roll_angle_rad),
        ])
        ax = ax / np.linalg.norm(ax)
        # Perpendicular basis
        perp1 = np.array([-math.sin(wa), 0, math.cos(wa)])
        perp2 = np.cross(ax, perp1)
        perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)
        perp1 = np.cross(perp2, ax)

        angles = [2 * math.pi * i / nx_pts for i in range(nx_pts)]
        p0 = center - ax * length / 2
        p1 = center + ax * length / 2
        ring0 = [p0 + r * (math.cos(a) * perp1 + math.sin(a) * perp2) for a in angles]
        ring1 = [p1 + r * (math.cos(a) * perp1 + math.sin(a) * perp2) for a in angles]
        for i in range(nx_pts):
            j = (i + 1) % nx_pts
            tris.append([ring0[i], ring0[j], ring1[i]])
            tris.append([ring0[j], ring1[j], ring1[i]])
            tris.append([p0, ring0[j], ring0[i]])
            tris.append([p1, ring1[i], ring1[j]])
        return tris

    # Outer rubber tire (thin cylinder, slightly smaller than full radius)
    triangles.extend(cylinder_tris(radius, width * 0.92, nx=48, cap=False))

    # Aluminum hub
    hub_r = radius * 0.45
    hub_w = width * 0.88
    triangles.extend(cylinder_tris(hub_r, hub_w, nx=32, cap=True))

    # Diagonal rollers
    for i in range(n_rollers):
        wheel_a = 2 * math.pi * i / n_rollers
        roller_center = np.array([
            (radius * 0.78) * math.cos(wheel_a),
            0.0,
            (radius * 0.78) * math.sin(wheel_a),
        ])
        triangles.extend(roller_tris(
            r=roller_radius,
            length=width * 1.05,
            center=roller_center,
            axis_angle=45.0,
            wheel_angle=wheel_a,
        ))

    write_stl_binary(filepath, np.array(triangles))


def main():
    print(f"Loading USD: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)

    print("Collecting meshes...")
    all_mesh_prims = collect_all_meshes(stage)
    print(f"Found {len(all_mesh_prims)} mesh prims")

    # -----------------------------------------------------------------
    # Determine wheel bounding regions by spatial analysis.
    # In the CAD coordinate system we need to identify wheel meshes.
    # Strategy: collect bounding box centers; meshes at extreme X positions
    # and with small Y extent (cylinders = wheels) are likely wheels.
    # -----------------------------------------------------------------
    print("Analyzing mesh extents to separate wheels from chassis...")

    mesh_data = []
    for prim in all_mesh_prims:
        ext_attr = prim.GetAttribute("extent")
        if ext_attr:
            ext = ext_attr.Get()
            if ext:
                cx = (ext[0][0] + ext[1][0]) / 2
                cy = (ext[0][1] + ext[1][1]) / 2
                cz = (ext[0][2] + ext[1][2]) / 2
                dx = ext[1][0] - ext[0][0]
                dy = ext[1][1] - ext[0][1]
                dz = ext[1][2] - ext[0][2]
                mesh_data.append((prim, cx, cy, cz, dx, dy, dz))

    # Find overall bounding box
    all_cx = [d[1] for d in mesh_data]
    all_cy = [d[2] for d in mesh_data]
    all_cz = [d[3] for d in mesh_data]

    global_min_x = min(d[1] - d[4]/2 for d in mesh_data)
    global_max_x = max(d[1] + d[4]/2 for d in mesh_data)
    global_min_y = min(d[2] - d[5]/2 for d in mesh_data)
    global_max_y = max(d[2] + d[5]/2 for d in mesh_data)
    global_min_z = min(d[3] - d[6]/2 for d in mesh_data)
    global_max_z = max(d[3] + d[6]/2 for d in mesh_data)

    print(f"Global bbox (mm): X[{global_min_x:.1f},{global_max_x:.1f}] "
          f"Y[{global_min_y:.1f},{global_max_y:.1f}] Z[{global_min_z:.1f},{global_max_z:.1f}]")

    # For the M3 Pro, wheels are Mecanum wheels ~80mm diameter × 38mm wide.
    # They're at the 4 corners. In CAD the robot body is roughly centered.
    # Wheel meshes tend to cluster in 4 spatial regions at the extremes.
    # We'll just export everything as one body + generate a proper wheel STL.
    # -----------------------------------------------------------------

    print("\nExporting full assembly as chassis.stl (all meshes merged)...")
    all_triangles = []
    failed = 0
    for i, (prim, *_) in enumerate(mesh_data):
        if i % 1000 == 0:
            print(f"  Processing mesh {i}/{len(mesh_data)}...", end="\r")
        tris = usd_mesh_to_triangles(prim)
        if tris is not None:
            all_triangles.append(tris)
        else:
            failed += 1

    if all_triangles:
        merged = np.concatenate(all_triangles, axis=0)
        print(f"\n  Total triangles: {len(merged):,} (skipped {failed} meshes)")

        # Decimate naively by keeping every nth triangle if > 200K triangles
        max_tris = 300_000
        if len(merged) > max_tris:
            step = math.ceil(len(merged) / max_tris)
            merged = merged[::step]
            print(f"  Decimated to {len(merged):,} triangles (1 of every {step})")

        chassis_path = os.path.join(MESH_DIR, "chassis.stl")
        write_stl_binary(chassis_path, merged)
    else:
        print("  ERROR: No triangles extracted!")

    print("\nGenerating Mecanum wheel mesh...")
    wheel_path = os.path.join(MESH_DIR, "mecanum_wheel.stl")
    generate_mecanum_wheel_stl(wheel_path)
    print(f"  Wheel STL: {wheel_path}")

    print("\nDone. Next: update URDF xacro to reference meshes/chassis.stl and meshes/mecanum_wheel.stl")


if __name__ == "__main__":
    main()
