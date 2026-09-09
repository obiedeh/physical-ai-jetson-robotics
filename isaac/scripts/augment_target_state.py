#!/usr/bin/env python3
"""Append target XY to a target-labelled Synria raw corpus.

The source corpus must contain ``raw_manifest.jsonl`` entries with
``place: ["track", index]`` (or another BoardGeometry position) and
``episode_XXXXXX/states.npy`` / ``actions.npy``. The output keeps actions at
8D and changes states from 8D to 10D: six arm joints, two fingers, target x,
and target y. Media and metadata are hard-linked when possible, so this does
not duplicate large videos on the same filesystem.

This is data preparation only. It does not claim that target coordinates are
available from a physical camera or arm state; the field is a simulator/task
label and must be treated as privileged information for sim experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ludo_engine.board import BoardGeometry  # noqa: E402


def _copy_or_link(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _target_xy(entry: dict, geometry: BoardGeometry) -> tuple[float, float]:
    place = entry.get("place")
    if not isinstance(place, list) or len(place) != 2:
        raise ValueError(f"episode {entry.get('episode_index')}: missing place [kind, index]")
    kind, index = place
    if kind not in {"track", "home", "yard", "done"} or not isinstance(index, int):
        raise ValueError(f"episode {entry.get('episode_index')}: invalid place {place!r}")
    # Track positions are color-independent; non-track positions need the
    # manifest's color to resolve the board quadrant/home column.
    color = entry.get("color", "red")
    if color not in {"red", "blue", "green", "yellow"}:
        raise ValueError(f"episode {entry.get('episode_index')}: invalid color {color!r}")
    return geometry.world_xy(color, (kind, index))


def augment(raw: Path, out: Path, geometry: BoardGeometry) -> int:
    manifest_path = raw / "raw_manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if not entries:
        raise ValueError("raw manifest is empty")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    augmented_entries: list[dict] = []
    for entry in entries:
        idx = entry.get("episode_index")
        if not isinstance(idx, int):
            raise ValueError(f"invalid episode_index: {idx!r}")
        source_ep = raw / f"episode_{idx:06d}"
        target_ep = out / f"episode_{idx:06d}"
        states_path = source_ep / "states.npy"
        actions_path = source_ep / "actions.npy"
        states = np.load(states_path)
        actions = np.load(actions_path)
        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError(f"episode {idx}: states must have shape (T, 8), got {states.shape}")
        if actions.ndim != 2 or actions.shape != (len(states), 8):
            raise ValueError(
                f"episode {idx}: actions must have shape {(len(states), 8)}, got {actions.shape}"
            )
        target = np.asarray(_target_xy(entry, geometry), dtype=np.float32)
        augmented = np.concatenate(
            [states.astype(np.float32), np.broadcast_to(target, (len(states), 2))], axis=1
        )
        target_ep.mkdir(parents=True, exist_ok=True)
        np.save(target_ep / "states.npy", augmented)
        _copy_or_link(actions_path, target_ep / "actions.npy")
        for source_file in sorted(source_ep.iterdir()):
            if source_file.name in {"states.npy", "actions.npy"}:
                continue
            if source_file.is_file():
                _copy_or_link(source_file, target_ep / source_file.name)
        updated = dict(entry)
        updated["target_xy"] = [float(target[0]), float(target[1])]
        updated["state_dim"] = 10
        updated["target_state_fields"] = ["target_x", "target_y"]
        augmented_entries.append(updated)

    for source_file in sorted(raw.iterdir()):
        if source_file.name == "raw_manifest.jsonl" or source_file.is_dir():
            continue
        _copy_or_link(source_file, out / source_file.name)
    (out / "raw_manifest.jsonl").write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in augmented_entries)
    )
    print(f"[augment] {len(augmented_entries)} episodes -> {out} (state 8D -> 10D, action 8D)")
    return len(augmented_entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board-size", type=float, default=0.34)
    parser.add_argument("--center-x", type=float, default=0.15)
    parser.add_argument("--center-y", type=float, default=0.0)
    args = parser.parse_args()
    try:
        augment(args.raw, args.out, BoardGeometry(args.board_size, args.center_x, args.center_y))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
