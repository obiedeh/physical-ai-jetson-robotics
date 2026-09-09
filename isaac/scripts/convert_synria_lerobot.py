"""Convert raw recorded Synria episodes → GR00T-ready LeRobot v2.1 dataset.

Input:  the ``--raw`` directory produced by ``record_synria_lerobot_demos.py``
        (episode_XXXXXX/{states.npy,actions.npy,wrist.mp4,overhead.mp4} +
        raw_manifest.jsonl).
Output: a LeRobot v2.1 dataset at ``--out`` in the exact layout of
        Isaac-GR00T's ``demo_data/cube_to_bowl_5`` reference:

    data/chunk-000/episode_000000.parquet   (action, observation.state,
                                             timestamp, frame_index,
                                             episode_index, index, task_index)
    videos/chunk-000/observation.images.{wrist,overhead}/episode_XXXXXX.mp4
    meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}

stats.json / relative_stats.json are generated automatically by GR00T's
DatasetFactory on first training run — not written here.

Runs in the gr00t venv (needs pandas/pyarrow, NOT Isaac):

    ~/.venv/gr00t/bin/python isaac/scripts/convert_synria_lerobot.py \
        --raw <raw_root> --out reports/training/synria_cup_lerobot_demos
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaac.isaaclab_tasks.synria_pickplace.gr00t.dataset import (  # noqa: E402
    modality_json_spec,
)

FPS = 30
CHUNKS_SIZE = 1000
STATE_NAMES = [
    "joint_1.pos", "joint_2.pos", "joint_3.pos",
    "joint_4.pos", "joint_5.pos", "joint_6.pos",
    "left_finger.pos", "right_finger.pos",
]
# surrogate embodiment (S2): Franka + Robotiq 2F-85, 7 arm + 1 gripper
FRANKA_STATE_NAMES = [
    "panda_joint1.pos", "panda_joint2.pos", "panda_joint3.pos",
    "panda_joint4.pos", "panda_joint5.pos", "panda_joint6.pos",
    "panda_joint7.pos", "finger_joint.pos",
]


def franka_modality_json_spec() -> dict:
    return {
        "state": {
            "arm": {"start": 0, "end": 7},
            "gripper": {"start": 7, "end": 8},
        },
        "action": {
            "arm": {"start": 0, "end": 7},
            "gripper": {"start": 7, "end": 8},
        },
        "video": {
            "wrist": {"original_key": "observation.images.wrist"},
            "overhead": {"original_key": "observation.images.overhead"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embodiment", choices=("synria", "franka"),
                        default="synria")
    args = parser.parse_args()
    state_names = FRANKA_STATE_NAMES if args.embodiment == "franka" else STATE_NAMES
    mod_spec = (
        franka_modality_json_spec() if args.embodiment == "franka"
        else modality_json_spec()
    )
    robot_type = (
        "franka_panda_robotiq_2f85" if args.embodiment == "franka"
        else "synria_6dof_alicia_d"
    )

    manifest_path = args.raw / "raw_manifest.jsonl"
    if not manifest_path.exists():
        sys.stderr.write(f"raw manifest not found: {manifest_path}\n")
        return 2
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if not entries:
        sys.stderr.write("raw manifest is empty — nothing to convert\n")
        return 2

    tasks: dict[str, int] = {}
    for ent in entries:
        tasks.setdefault(ent["instruction"], len(tasks))

    data_dir = args.out / "data" / "chunk-000"
    meta_dir = args.out / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for cam in ("wrist", "overhead"):
        (args.out / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(
            parents=True, exist_ok=True
        )

    episodes_meta = []
    global_index = 0
    total_frames = 0
    for new_idx, ent in enumerate(entries):
        src = args.raw / f"episode_{ent['episode_index']:06d}"
        states = np.load(src / "states.npy")
        actions = np.load(src / "actions.npy")
        T = len(states)
        assert len(actions) == T, f"{src}: states {T} != actions {len(actions)}"
        assert states.shape[1] == 8 and actions.shape[1] == 8, f"{src}: bad dims"

        df = pd.DataFrame(
            {
                "action": [row.astype(np.float32) for row in actions],
                "observation.state": [row.astype(np.float32) for row in states],
                "timestamp": (np.arange(T) / FPS).astype(np.float32),
                "frame_index": np.arange(T, dtype=np.int64),
                "episode_index": np.full(T, new_idx, dtype=np.int64),
                "index": np.arange(global_index, global_index + T, dtype=np.int64),
                "task_index": np.full(T, tasks[ent["instruction"]], dtype=np.int64),
            }
        )
        df.to_parquet(data_dir / f"episode_{new_idx:06d}.parquet")
        for cam in ("wrist", "overhead"):
            shutil.copy2(
                src / f"{cam}.mp4",
                args.out / "videos" / "chunk-000"
                / f"observation.images.{cam}" / f"episode_{new_idx:06d}.mp4",
            )
        episodes_meta.append(
            {"episode_index": new_idx, "tasks": [ent["instruction"]], "length": T}
        )
        global_index += T
        total_frames += T

    with (meta_dir / "episodes.jsonl").open("w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    with (meta_dir / "tasks.jsonl").open("w") as f:
        for task, idx in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    features = {
        "action": {"dtype": "float32", "names": state_names, "shape": [8]},
        "observation.state": {"dtype": "float32", "names": state_names, "shape": [8]},
        "timestamp": {"dtype": "float32", "names": None, "shape": [1]},
        "frame_index": {"dtype": "int64", "names": None, "shape": [1]},
        "episode_index": {"dtype": "int64", "names": None, "shape": [1]},
        "index": {"dtype": "int64", "names": None, "shape": [1]},
        "task_index": {"dtype": "int64", "names": None, "shape": [1]},
    }
    for cam in ("wrist", "overhead"):
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [224, 224, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": 224,
                "video.width": 224,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(entries),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(entries)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_chunks": 1,
        "total_videos": 2 * len(entries),
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (meta_dir / "modality.json").write_text(
        json.dumps(mod_spec, indent=4) + "\n"
    )
    defaults = args.raw / "defaults.json"
    if defaults.exists():
        shutil.copy2(defaults, meta_dir / "synria_defaults.json")

    print(f"[convert] {len(entries)} episodes, {total_frames} frames → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
