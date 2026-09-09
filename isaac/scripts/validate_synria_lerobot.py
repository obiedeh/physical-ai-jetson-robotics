"""Validate every episode of a converted Synria LeRobot dataset (gate 3).

Checks per episode (GR00T ledger staged-validation gate 3 / mission
objective 4):

    1.  images/states/actions synchronized: parquet row count == frame
        count of BOTH videos == episodes.jsonl length
    2.  timestamps: start at 0, strictly monotonic, uniform 1/fps spacing
    3.  joint order/units: state[0:6] within URDF limits (radians —
        degrees would blow past them), fingers within physical stroke
    4.  action dims == 8; reconstructed targets (default + action) within
        URDF joint limits + finger stroke (unsafe/out-of-range detection)
    5.  gripper commands: reconstructed finger targets within
        [0, 0.025] / [-0.025, 0] (small tolerance for clipping)
    6.  missing/duplicated frames: frame_index == arange(T); global
        index contiguous across episodes
    7.  episode boundaries: episode_index constant per file and matches
        the file name
    8.  language instructions: task_index maps into tasks.jsonl and
        matches episodes.jsonl tasks

Exit 0 = all episodes pass. Runs in the gr00t venv:

    ~/.venv/gr00t/bin/python isaac/scripts/validate_synria_lerobot.py \
        --dataset reports/training/synria_cup_lerobot_demos
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-4


def video_frame_count(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return int(out.stdout.strip())
    except ValueError:
        return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    ds = args.dataset

    info = json.loads((ds / "meta" / "info.json").read_text())
    fps = info["fps"]
    episodes = [
        json.loads(line)
        for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    tasks = {
        t["task_index"]: t["task"]
        for t in (
            json.loads(line)
            for line in (ds / "meta" / "tasks.jsonl").read_text().splitlines()
            if line
        )
    }
    defaults = json.loads((ds / "meta" / "synria_defaults.json").read_text())
    arm_default = np.asarray(defaults["arm_default_rad"], dtype=np.float64)
    finger_default = np.asarray(
        [defaults["left_finger_default_m"], defaults["right_finger_default_m"]]
    )
    jlim = np.asarray(list(defaults["joint_limits_rad"].values()))  # (6, 2)
    flim = np.asarray(
        [defaults["finger_limits_m"]["left"], defaults["finger_limits_m"]["right"]]
    )  # (2, 2)

    failures: list[str] = []
    expected_next_index = 0
    total_frames = 0

    for ep in episodes:
        idx = ep["episode_index"]
        tag = f"ep{idx:06d}"
        pq = ds / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        df = pd.read_parquet(pq)
        T = len(df)

        if T != ep["length"]:
            failures.append(f"{tag}: parquet rows {T} != meta length {ep['length']}")

        # 1. sync with videos
        for cam in ("wrist", "overhead"):
            vid = (
                ds / "videos" / "chunk-000"
                / f"observation.images.{cam}" / f"episode_{idx:06d}.mp4"
            )
            if not vid.exists():
                failures.append(f"{tag}: missing video {vid.name} ({cam})")
                continue
            nf = video_frame_count(vid)
            if nf != T:
                failures.append(f"{tag}: {cam} video frames {nf} != rows {T}")

        # 2. timestamps
        ts = df["timestamp"].to_numpy()
        expect = np.arange(T) / fps
        if abs(ts[0]) > TOL or np.abs(ts - expect).max() > 1.0 / fps * 0.01:
            failures.append(f"{tag}: timestamps not uniform 1/{fps}s from 0")

        # 3-5. state/action ranges
        states = np.stack(df["observation.state"].to_numpy())
        actions = np.stack(df["action"].to_numpy())
        if states.shape != (T, 8) or actions.shape != (T, 8):
            failures.append(
                f"{tag}: bad dims states{states.shape} actions{actions.shape}"
            )
            continue
        if np.isnan(states).any() or np.isnan(actions).any():
            failures.append(f"{tag}: NaNs present")
        arm_pos = states[:, :6]
        if (arm_pos < jlim[:, 0] - 0.02).any() or (arm_pos > jlim[:, 1] + 0.02).any():
            failures.append(f"{tag}: arm joint STATE outside URDF limits (units?)")
        fing_pos = states[:, 6:8]
        if (fing_pos < flim[:, 0] - 0.002).any() or (fing_pos > flim[:, 1] + 0.002).any():
            failures.append(f"{tag}: finger STATE outside stroke")
        arm_tgt = arm_default[None, :] + actions[:, :6]
        if (arm_tgt < jlim[:, 0] - 0.02).any() or (arm_tgt > jlim[:, 1] + 0.02).any():
            failures.append(f"{tag}: reconstructed arm TARGET outside URDF limits")
        fing_tgt = finger_default[None, :] + actions[:, 6:8]
        if (fing_tgt < flim[:, 0] - 0.002).any() or (fing_tgt > flim[:, 1] + 0.002).any():
            failures.append(f"{tag}: reconstructed finger TARGET outside stroke")

        # 6. frame indices
        if not np.array_equal(df["frame_index"].to_numpy(), np.arange(T)):
            failures.append(f"{tag}: frame_index not contiguous (missing/dup frames)")
        gi = df["index"].to_numpy()
        if gi[0] != expected_next_index or not np.array_equal(
            gi, np.arange(gi[0], gi[0] + T)
        ):
            failures.append(f"{tag}: global index not contiguous")
        expected_next_index = int(gi[-1]) + 1

        # 7. episode boundary
        if not (df["episode_index"] == idx).all():
            failures.append(f"{tag}: episode_index mismatch inside file")

        # 8. instructions
        ti = df["task_index"].to_numpy()
        if len(set(ti.tolist())) != 1 or int(ti[0]) not in tasks:
            failures.append(f"{tag}: task_index invalid/inconsistent")
        elif tasks[int(ti[0])] not in ep["tasks"]:
            failures.append(f"{tag}: instruction mismatch vs episodes.jsonl")

        total_frames += T

    if total_frames != info["total_frames"]:
        failures.append(
            f"info.json total_frames {info['total_frames']} != sum {total_frames}"
        )

    print(f"[validate] {len(episodes)} episodes, {total_frames} frames checked")
    if failures:
        print(f"[validate] FAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[validate] PASS — all episodes clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
