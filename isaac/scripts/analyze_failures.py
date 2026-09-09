"""Aggregate stage-eval ledgers into failure distributions (Goal 2/10).

    python isaac/scripts/analyze_failures.py reports/closed_loop_v6d.jsonl \
        [more.jsonl ...] [--csv reports/stage_metrics.csv]

Groups by (takeover_stage, instruction, fixed_spawn) and prints success
rate with a 95% Wilson interval plus the failure-label distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson(k: int, nn: int, z: float = 1.96) -> tuple[float, float]:
    if nn == 0:
        return (0.0, 0.0)
    p = k / nn
    d = 1 + z * z / nn
    c = (p + z * z / (2 * nn)) / d
    h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledgers", nargs="+")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    groups: dict = defaultdict(list)
    for path in args.ledgers:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (
                rec.get("takeover_stage", "none"),
                rec.get("instruction", "")[:40],
                rec.get("fixed_spawn", False),
            )
            groups[key].append(rec)

    rows = []
    for key, recs in sorted(groups.items()):
        stage, instr, fixed = key
        nn = len(recs)
        k = sum(1 for r in recs if r["outcome"] == "success")
        lo, hi = wilson(k, nn)
        print(f"\n=== takeover={stage} fixed_spawn={fixed} instr='{instr}' ===")
        print(f"success: {k}/{nn} = {k/nn:.1%}  (95% CI {lo:.1%}-{hi:.1%})")
        fails = Counter(
            r["first_failed_stage"] for r in recs if r["outcome"] != "success"
        )
        for label, cnt in fails.most_common():
            print(f"  {label:26s} {cnt:4d}  ({cnt/nn:.1%})")
        med_min_dxy = sorted(r["min_hand_cup_xy"] for r in recs)[nn // 2]
        med_max_z = sorted(r["max_cup_z"] for r in recs)[nn // 2]
        print(f"  median min hand-cup xy: {med_min_dxy:.3f}  "
              f"median max cup z: {med_max_z:.3f}")
        rows.append(
            {
                "takeover_stage": stage,
                "fixed_spawn": fixed,
                "instruction": instr,
                "episodes": nn,
                "successes": k,
                "rate": round(k / nn, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "top_failure": fails.most_common(1)[0][0] if fails else "",
                "median_min_hand_cup_xy": round(med_min_dxy, 4),
                "median_max_cup_z": round(med_max_z, 4),
            }
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[analyze] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
