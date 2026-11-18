"""Score a Gemini model on reading Ludo board states from overhead frames.

The model points at every token and names its color. Each point is mapped to
a grid cell with the known camera geometry and compared with the engine truth
written by `isaac/scripts/render_ludo_board_bench.py`.

    python -m physical_ai_lab.gemini_robotics.board_bench --model gemini-4-hydrogen \
        --out reports/gemini_robotics/ludo_board_gemini-4-hydrogen_672.json

`--size 224` downsamples each frame first, to the resolution the arm's
overhead camera records today. Perception only: nothing here commands a robot.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
from pathlib import Path

from .ludo_board import COLORS, OverheadCamera, occupied, score_state
from .point_bench import ask, load_key

PROMPT = (
    "This is an overhead image of a Ludo board. The playing pieces are round "
    "tokens colored red, blue, green or yellow. Point to the centre of every "
    'token. Answer JSON only: [{"point": [y, x], "label": "<color>"}, ...] '
    "with y and x normalized to 0-1000 and one entry per visible token."
)


def parse_tokens(text: str) -> list[tuple[str, float, float]] | None:
    """Return [(color, y, x)] in 0-1000 from the model text, or None if it is not a list."""
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        try:
            y, x = item["point"]
            label = str(item.get("label", "")).lower()
            color = next((c for c in COLORS if c in label), "unknown")
            out.append((color, float(y), float(x)))
        except Exception:
            continue
    return out


def to_cells(tokens: list[tuple[str, float, float]], cam: OverheadCamera) -> tuple[set, int]:
    """Map pointed tokens to (color, col, row). Returns the set and the count pointed off the board."""
    cells, off = set(), 0
    w, h = cam.image_wh
    for color, y, x in tokens:
        cell = cam.px_to_cell(x / 1000.0 * w, y / 1000.0 * h)
        if cell is None:
            off += 1
        else:
            cells.add((color, cell[0], cell[1]))
    return cells, off


def summarize(model: str, rows: list[dict], total: int) -> dict:
    ok = [r for r in rows if r["ok"]]
    out = {"model": model, "n_total": total, "n_answered": len(ok)}
    if ok:
        truth = sum(r["n_truth"] for r in ok)
        pred = sum(r["n_pred"] for r in ok)
        hit = sum(r["n_hit"] for r in ok)
        out.update({
            "states_exact": sum(r["exact"] for r in ok),
            "cells_truth": truth, "cells_pred": pred, "cells_hit": hit,
            "recall": round(hit / truth, 3), "precision": round(hit / pred, 3) if pred else 0.0,
            "wrong_color": sum(r["wrong_color"] for r in ok),
            "off_board": sum(r["off_board"] for r in ok),
            "median_seconds": round(statistics.median(r["seconds"] for r in ok), 2),
        })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--bench", type=Path, default=Path("reports/ludo_board_bench"))
    p.add_argument("--size", type=int, default=0, help="downsample frames to this width first")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    key = load_key()
    states = [json.loads(line) for line in (args.bench / "truth.jsonl").read_text().splitlines() if line.strip()]
    if args.limit:
        states = states[: args.limit]
    rows = []
    for s in states:
        png = (args.bench / s["image"]).read_bytes()
        if args.size:
            from PIL import Image
            buf = io.BytesIO()
            Image.open(io.BytesIO(png)).resize((args.size, args.size), Image.LANCZOS).save(buf, "PNG")
            png = buf.getvalue()
        cam = OverheadCamera(tuple(s["image_wh"]), s["camera"]["focal_mm"], s["camera"]["aperture_mm"],
                             s["camera"]["distance_m"])
        text, seconds, error = ask(args.model, png, PROMPT, key, args.temperature, args.timeout)
        tokens = parse_tokens(text) if text else None
        row = {"i": s["i"], "ok": tokens is not None, "seconds": round(seconds, 2)}
        if error:
            row["error"] = error
        if text and tokens is None:
            row["unparsed"] = text[:200]
        if tokens is not None:
            cells, off = to_cells(tokens, cam)
            row.update(score_state(occupied(s["tokens"]), cells))
            row["off_board"] = off
            row["n_points"] = len(tokens)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("i", "ok", "n_truth", "n_hit", "n_pred", "exact", "error")}), flush=True)
    report = summarize(args.model, rows, len(states))
    report.update({"bench": str(args.bench), "input_width": args.size or states[0]["image_wh"][0],
                   "temperature": args.temperature, "prompt": PROMPT, "results": rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "prompt")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
