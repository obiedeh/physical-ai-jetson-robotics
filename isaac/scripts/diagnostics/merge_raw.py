"""Merge base raw demos + dwell raw demos into one renumbered raw dir."""
import json
import os
import sys
from pathlib import Path

# repo root derived from this file, so the script works from any
# checkout of the repository (worktree or clone)
_REPO_ROOT = Path(__file__).resolve().parents[3]

repo = _REPO_ROOT
base = repo / "reports/training/franka_cup_lerobot_raw"
dwell = repo / "reports/training/franka_cup_dwell_raw"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

idx = 0
manifest_lines = []


def take(src_root: Path) -> None:
    global idx
    man = {}
    for line in (src_root / "raw_manifest.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            man[rec["episode_index"]] = rec
    for ep_dir in sorted(src_root.glob("episode_*")):
        src_idx = int(ep_dir.name.split("_")[1])
        dst = out / f"episode_{idx:06d}"
        if not dst.exists():
            os.symlink(ep_dir.resolve(), dst)
        rec = dict(man.get(src_idx, {"length": 0}))
        rec["episode_index"] = idx
        rec["source"] = str(src_root.name)
        manifest_lines.append(json.dumps(rec))
        idx += 1


take(base)
take(dwell)
(out / "raw_manifest.jsonl").write_text("\n".join(manifest_lines) + "\n")
print(f"[merge] {idx} episodes -> {out}")
