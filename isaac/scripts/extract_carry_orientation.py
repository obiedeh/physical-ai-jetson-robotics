#!/usr/bin/env python3
"""Extract the demonstrated carry-orientation reference (pitch_ref) from
the expert corpus — the operator's wrist-pitch requirement, grounded in
recorded data rather than an arbitrary tolerance.

Pure-python FK from the URDF (no Isaac needed): joint states (T,8) ->
tool rotation -> tilt of the gripper approach axis from vertical.
Phases from the recorded gripper action channel:
    grasp  = first sustained finger-close command
    carry  = grasp .. first reopen (the placement release)
Reports per-episode: tilt at grasp, at lift(+15 steps), carry mean/max;
aggregates pitch_ref (median carry tilt) and tolerance (p95 of max
carry deviation from each episode's own mean).

    python3 isaac/scripts/extract_carry_orientation.py \
        --corpus reports/training/synria_e46_lerobot_raw --sample 200
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load_chain(urdf_path: Path, tip: str = "tool0"):
    """Return ordered [(xyz, rpy, axis)] for the revolute chain to tip."""
    root = ET.parse(urdf_path).getroot()
    joints = {}
    child_of = {}
    for j in root.findall("joint"):
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = [float(x) for x in (origin.get("xyz", "0 0 0")).split()]
        rpy = [float(x) for x in (origin.get("rpy", "0 0 0")).split()]
        axis_el = j.find("axis")
        axis = [float(x) for x in axis_el.get("xyz").split()] if axis_el is not None else [0, 0, 1]
        jtype = j.get("type")
        joints[child] = (name, parent, xyz, rpy, axis, jtype)
        child_of[parent] = child
    # walk back from tip to base
    chain = []
    link = tip
    while link in joints:
        chain.append(joints[link])
        link = joints[link][1]
    chain.reverse()
    return chain


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def axis_angle_R(axis, th):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def fk_R(chain, q):
    R = np.eye(3)
    qi = 0
    for (_, _, _, rpy, axis, jtype) in chain:
        R = R @ rpy_to_R(*rpy)
        if jtype in ("revolute", "continuous"):
            R = R @ axis_angle_R(axis, q[qi])
            qi += 1
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=REPO / "reports/training/synria_e46_lerobot_raw")
    ap.add_argument("--urdf", type=Path,
                    default=REPO / "isaac/usd/robots/synria_6dof_arm.urdf")
    ap.add_argument("--tip", default="tool0")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports/carry_orientation_ref.json")
    args = ap.parse_args()

    chain = load_chain(args.urdf, args.tip)
    nrev = sum(1 for c in chain if c[5] in ("revolute", "continuous"))
    print(f"[ori] chain to {args.tip}: {len(chain)} joints ({nrev} revolute)")

    eps = sorted(args.corpus.glob("episode_*"))[: args.sample]
    rows = []
    for ep in eps:
        try:
            states = np.load(ep / "states.npy")   # (T,8)
            actions = np.load(ep / "actions.npy")
        except Exception:
            continue
        # gripper close command: finger-target offset (dim 6) strongly neg
        closed_cmd = actions[:, 6] < -0.006
        if not closed_cmd.any():
            continue
        g = int(np.argmax(closed_cmd))            # grasp command moment
        reopen = np.where(~closed_cmd[g:])[0]
        r = g + int(reopen[0]) if len(reopen) else len(states) - 1
        if r - g < 30:                             # carry too short: reject
            continue
        tilts = []
        for t in range(g, r, 3):
            R = fk_R(chain, states[t, :6])
            approach = R[:, 2]                    # tool z-axis, world frame
            tilt = math.degrees(math.acos(
                min(1.0, max(-1.0, abs(approach[2])))))
            tilts.append(tilt)
        tilts = np.array(tilts)
        rows.append({
            "ep": ep.name,
            "tilt_grasp_deg": float(tilts[0]),
            "tilt_lift_deg": float(tilts[min(5, len(tilts) - 1)]),
            "carry_mean_deg": float(tilts.mean()),
            "carry_max_dev_deg": float(np.abs(tilts - tilts.mean()).max()),
            "carry_len_steps": int(r - g),
        })
    if not rows:
        print("[ori] no usable episodes")
        return 1
    mean_tilts = np.array([r["carry_mean_deg"] for r in rows])
    max_devs = np.array([r["carry_max_dev_deg"] for r in rows])
    ref = {
        "episodes_analyzed": len(rows),
        "pitch_ref_deg": float(np.median(mean_tilts)),
        "pitch_ref_p5_p95_deg": [float(np.percentile(mean_tilts, 5)),
                                 float(np.percentile(mean_tilts, 95))],
        "carry_max_dev_p95_deg": float(np.percentile(max_devs, 95)),
        "tolerance_recommendation_deg": float(np.percentile(max_devs, 95)),
        "convention": "tilt = angle of tool z-axis from world vertical; "
                      "0 deg = gripper axis vertical (cup level)",
    }
    args.out.write_text(json.dumps({"reference": ref, "episodes": rows[:20]},
                                   indent=2))
    print(f"[ori] pitch_ref = {ref['pitch_ref_deg']:.1f} deg "
          f"(p5-p95 {ref['pitch_ref_p5_p95_deg'][0]:.1f}-"
          f"{ref['pitch_ref_p5_p95_deg'][1]:.1f}), "
          f"carry max-dev p95 = {ref['carry_max_dev_p95_deg']:.1f} deg "
          f"over {len(rows)} episodes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
