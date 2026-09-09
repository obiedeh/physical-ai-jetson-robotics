"""Generate meta/episodes_stats.jsonl for a GR00T-flavor LeRobot v2.1 dataset.

Our converter (convert_synria_lerobot.py) targets GR00T's v2.1 dialect, which
omits episodes_stats.jsonl because GR00T computes its own normalization.
lerobot's v2.1 -> v3.0 migrator requires that file, and lerobot 0.6.1 ships
the exact stats math (`lerobot.datasets.compute_stats`) — so this uses THEIR
functions rather than reimplementing the quantile bookkeeping, and only adds
the video-frame sampling they normally do at record time from PNG paths
(here: decoded from the episode mp4s, 12 frames spread across the episode).

Run in the lerobot venv, from OUTSIDE the repo root (the repo shadows
package names):

    cd ~ && ~/.venv/lerobot/bin/python \
        ~/github/physical-ai-jetson-robotics/isaac/scripts/gen_episodes_stats_v21.py \
        --root ~/github/physical-ai-jetson-robotics/reports/training/synria_e33_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import get_feature_stats


def _serialize(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def sample_video_frames(path: Path, n: int = 12) -> np.ndarray:
    """(N, C, H, W) uint8 frames spread across the video."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        frames = []
        if total > 0:
            want = set(np.linspace(0, total - 1, n, dtype=int).tolist())
            for i, frame in enumerate(container.decode(stream)):
                if i in want:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= n:
                    break
        else:
            for i, frame in enumerate(container.decode(stream)):
                if i % 30 == 0:
                    frames.append(frame.to_ndarray(format="rgb24"))
    arr = np.stack(frames)                     # (N, H, W, C)
    return arr.transpose(0, 3, 1, 2)           # (N, C, H, W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.expanduser()

    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    numeric_keys = [k for k, v in features.items()
                    if v["dtype"] not in ("video", "image", "string", "language")]

    out_path = root / "meta/episodes_stats.jsonl"
    rows = []
    for ep in range(info["total_episodes"]):
        chunk = ep // info["chunks_size"]
        pq = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
        df = pd.read_parquet(pq)
        ep_stats: dict = {}
        for key in numeric_keys:
            col = df[key].to_numpy()
            if col.dtype == object:            # list-valued column -> (T, D)
                arr = np.stack(col.tolist()).astype(np.float32)
            else:
                arr = col.astype(np.float32)
            ep_stats[key] = get_feature_stats(
                arr, axis=0, keepdims=arr.ndim == 1)
        for key in video_keys:
            vid = root / info["video_path"].format(
                episode_chunk=chunk, video_key=key, episode_index=ep)
            frames = sample_video_frames(vid)
            st = get_feature_stats(frames, axis=(0, 2, 3), keepdims=True)
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                for k, v in st.items()
            }
        rows.append({"episode_index": ep,
                     "stats": {k: _serialize(v) for k, v in ep_stats.items()}})
        if ep % 20 == 0:
            print(f"[stats] episode {ep}/{info['total_episodes']}", flush=True)

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[stats] wrote {out_path} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
