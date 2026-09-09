#!/usr/bin/env python3
"""G3 composer: merge episode frames + sidecar answers into the demo MP4.

For each frame (by wall-clock), overlay the most recent Gemini answer that
had ARRIVED by that moment — so the crosshair lags reality by the true
round-trip, and the age readout shows honest staleness. Yellow crosshair =
Gemini's cup fix; text = query frame, latency, age.

    python3 isaac/scripts/g3_compose.py --session reports/g3_session_01
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
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    meta = [json.loads(l) for l in
            (args.session / "frames_meta.jsonl").read_text().splitlines() if l.strip()]
    answers = [json.loads(l) for l in
               (args.session / "sidecar.jsonl").read_text().splitlines() if l.strip()]
    answers = [a for a in answers if a.get("point")]
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    anno = args.session / "anno"
    anno.mkdir(exist_ok=True)
    phases = {0: "approach", 1: "grasped", 2: "carrying", 3: "placing"}
    for m in meta:
        fp = args.session / "frames" / f"f_{m['f']:05d}.png"
        if not fp.exists():
            continue
        im = Image.open(fp).convert("RGB")
        im = im.resize((im.width * 3, im.height * 3), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        avail = [a for a in answers if a["t_answer"] <= m["t"]]
        if avail:
            a = avail[-1]
            y, x = a["point"]
            W, H = im.size
            u, v = x / 1000 * W, y / 1000 * H
            L = 30
            d.line([u - L, v, u + L, v], fill=(255, 210, 0), width=5)
            d.line([u, v - L, u, v + L], fill=(255, 210, 0), width=5)
            d.ellipse([u - 14, v - 14, u + 14, v + 14],
                      outline=(255, 210, 0), width=3)
            age = m["t"] - a["t_answer"]
            d.text((10, 10),
                   f"GEMINI ER-2 cup fix  |  rtt {a['lat']:.1f}s  |  "
                   f"age {age:+.1f}s", fill=(255, 210, 0), font=font)
        else:
            d.text((10, 10), "GEMINI ER-2: awaiting first answer...",
                   fill=(180, 180, 180), font=font)
        d.text((10, im.height - 34),
               f"ACT policy driving @30Hz  |  phase: "
               f"{phases.get(m.get('phase', 0), m.get('phase'))}",
               fill=(120, 220, 255), font=font)
        im.save(anno / f"a_{m['f']:05d}.png")

    out = args.session / "g3_demo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-pattern_type", "glob", "-i", str(anno / "a_*.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out)],
        check=True,
    )
    print(f"[compose] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
