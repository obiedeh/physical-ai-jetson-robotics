"""Score a Gemini model's pointing on the overhead perception benchmark.

Reads a benchmark folder written by ``isaac/scripts/gen_perception_bench.py``
(``overhead_XXX.png`` plus ``truth.jsonl``), asks the model for one point on
the named object in each frame, and reports the error in millimetres at the
table using the per-frame ``mm_per_px`` scale from the truth file.

    python -m physical_ai_lab.gemini_robotics.point_bench \
        --model gemini-4-hydrogen --bench reports/perception_bench \
        --out reports/gemini_robotics/g1_gemini-4-hydrogen.json

The API key is read from ``GEMINI_API_KEY`` or ``~/.config/gemini/api_key``.
Frames leave the machine: use rendered or cleared images only.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PROMPT = (
    "Point to the centre of the {target} in this overhead image of a robot "
    'workspace. Answer JSON only: [{{"point": [y, x], "label": "{target}"}}] '
    "with y and x normalized to 0-1000."
)


def load_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    return (Path.home() / ".config/gemini/api_key").read_text().strip()


def parse_point(text: str) -> tuple[float, float] | None:
    """Return (y, x) in 0-1000 from the model text, or None."""
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        y, x = data[0]["point"]
        return float(y), float(x)
    except Exception:
        m = re.search(r'"point"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
        return (float(m.group(1)), float(m.group(2))) if m else None


def to_pixels(point_yx: tuple[float, float], image_wh: tuple[int, int]) -> tuple[float, float]:
    """Convert a (y, x) 0-1000 point to (x, y) pixels."""
    y, x = point_yx
    return x / 1000.0 * image_wh[0], y / 1000.0 * image_wh[1]


def error_mm(
    pred_px: tuple[float, float], truth_px: tuple[float, float], mm_per_px: float
) -> float:
    return math.hypot(pred_px[0] - truth_px[0], pred_px[1] - truth_px[1]) * mm_per_px


def ask(
    model: str,
    png: bytes,
    prompt: str,
    key: str,
    temperature: float,
    timeout: float,
    retries: int = 4,
) -> tuple[str | None, float, str | None]:
    """Return (text, seconds of the successful call, last error). Waits between retries."""
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(png).decode(),
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"temperature": temperature},
        }
    ).encode()
    error = None
    for attempt in range(retries):
        start = time.monotonic()
        try:
            req = urllib.request.Request(
                API.format(model=model),
                data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            parts = resp["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts), time.monotonic() - start, error
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}"
        except Exception as exc:  # timeout, connection, malformed response
            error = type(exc).__name__
        time.sleep(min(60.0, 5.0 * 2**attempt))
    return None, 0.0, error


def summarize(model: str, rows: list[dict], total: int) -> dict:
    errs = sorted(r["err_mm"] for r in rows if r["ok"])
    lat = sorted(r["seconds"] for r in rows if r["ok"])
    out = {"model": model, "n_total": total, "n_answered": len(errs)}
    if errs:
        out.update(
            {
                "median_mm": round(statistics.median(errs), 2),
                "p90_mm": round(errs[min(len(errs) - 1, int(0.9 * len(errs)))], 2),
                "max_mm": round(errs[-1], 2),
                "within_20mm": sum(e <= 20 for e in errs),
                "median_seconds": round(statistics.median(lat), 2),
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--bench", type=Path, default=Path("reports/perception_bench"))
    p.add_argument("--target", default="red cup")
    p.add_argument("--truth-key", default="cup_px")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    key = load_key()
    truth = [
        json.loads(line)
        for line in (args.bench / "truth.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        truth = truth[: args.limit]
    prompt = PROMPT.format(target=args.target)
    rows = []
    for t in truth:
        png = (args.bench / f"overhead_{t['i']:03d}.png").read_bytes()
        text, seconds, error = ask(args.model, png, prompt, key, args.temperature, args.timeout)
        point = parse_point(text) if text else None
        row = {"i": t["i"], "ok": point is not None, "seconds": round(seconds, 2)}
        if error:
            row["error"] = error
        if text and point is None:
            row["unparsed"] = text[:200]
        if point is not None:
            px = to_pixels(point, tuple(t["image_wh"]))
            row["pred_px"] = [round(px[0], 1), round(px[1], 1)]
            row["err_mm"] = round(error_mm(px, tuple(t[args.truth_key]), t["mm_per_px"]), 1)
        rows.append(row)
    report = summarize(args.model, rows, len(truth))
    report.update(
        {
            "bench": str(args.bench),
            "target": args.target,
            "truth_key": args.truth_key,
            "temperature": args.temperature,
            "prompt": prompt,
            "results": rows,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "prompt")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
