#!/usr/bin/env python3
"""Ludo demo composer: session frames + per-turn narration -> MP4.

Overlays each overhead frame with the current turn's description (the
game brain's plain-language plan, e.g. "red rolls 4: token 0 ...") and
a running turn scoreboard as results land. Pure post-processing — no
Isaac needed; runs in any venv with PIL + ffmpeg on PATH.

    python3 isaac/scripts/ludo_compose.py --session reports/ludo_demo_02
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()
    meta = [json.loads(l) for l in
            (args.session / "meta.jsonl").read_text().splitlines() if l.strip()]
    turns_fp = args.session / "turns.jsonl"
    turns = ([json.loads(l) for l in turns_fp.read_text().splitlines()
              if l.strip()] if turns_fp.exists() else [])
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()

    # map each turn description to the frame index where it first appears,
    # so the scoreboard only shows turns already completed on screen
    first_frame = {}
    for m in meta:
        first_frame.setdefault(m["desc"], m["f"])
    done_after = []      # (frame_after_which_done, label)
    for i, t in enumerate(turns):
        nxt = turns[i + 1]["desc"] if i + 1 < len(turns) else None
        end_f = first_frame.get(nxt, meta[-1]["f"]) if meta else 0
        mark = "OK" if all(t["ok"]) else "MISS"
        done_after.append((end_f, f"T{t['turn']} {mark}"))

    anno = args.session / "anno"
    anno.mkdir(exist_ok=True)
    for m in meta:
        fp = args.session / "frames" / f"f_{m['f']:05d}.png"
        if not fp.exists():
            continue
        im = Image.open(fp).convert("RGB")
        im = im.resize((im.width * args.scale, im.height * args.scale),
                       Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 40], fill=(0, 0, 0))
        d.text((10, 8), m["desc"][:110], fill=(255, 210, 0), font=font)
        board = "  ".join(lbl for f0, lbl in done_after if m["f"] >= f0)
        d.rectangle([0, im.height - 30, im.width, im.height], fill=(0, 0, 0))
        d.text((10, im.height - 26),
               f"SYNRIA plays LUDO  |  GR00T approach + scripted grasp "
               f"@30Hz  |  {board}", fill=(120, 220, 255), font=small)
        im.save(anno / f"a_{m['f']:05d}.png")

    out = args.session / "ludo_demo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-pattern_type", "glob", "-i", str(anno / "a_*.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)],
        check=True,
    )
    # media checksum manifest (append-only): ties the clip to the session
    # and its turns so a video can be cited against a specific failure
    import hashlib
    import time
    mm = args.session / "media_manifest.json"
    entries = json.loads(mm.read_text()) if mm.exists() else []
    entries.append({
        "file": out.name,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frames": len(meta), "turns": [t["turn"] for t in turns],
    })
    mm.write_text(json.dumps(entries, indent=2))
    print(f"[compose] {out} ({len(meta)} frames, {len(turns)} turns) "
          f"sha256 -> {mm.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
