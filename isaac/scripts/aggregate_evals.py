"""Aggregate several eval reports into one cross-seed table.

Why this exists
---------------
Every keep/revise/revert decision in the experiment ledger from E1 to E23 was
made on a single seed. E25 then measured what a seed is worth: at an identical
budget, on identical code, three seeds spread 2.8x on grasp and ~18x on
completed cycles. Deltas smaller than that were never distinguishable from
luck, and one wrong conclusion was drawn and nearly acted on after two of three
seeds had reported.

A point estimate cannot show that. This prints the spread, so a decision either
clears seed noise visibly or does not get made.

    python isaac/scripts/aggregate_evals.py reports/eval/*.json
    python isaac/scripts/aggregate_evals.py a.json b.json --baseline c.json

Pure Python — no torch, no Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stats import spread  # noqa: E402

STAGES = (
    "reach", "align", "grasp", "lift",
    "transport", "at_plate", "place", "upright", "full",
)

#: Below this many seeds, the spread column is not measuring seed variance and
#: says so rather than implying a tight result.
MIN_SEEDS_FOR_A_DECISION = 3

#: Deltas inside +/-10% are not worth flagging regardless of spread — on a
#: saturated stage the spread floor approaches 1.0x and everything clears it.
MIN_MATERIAL_DELTA = 1.10


def load(paths: list[Path]) -> list[dict]:
    reports = []
    for p in paths:
        try:
            reports.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[aggregate] skipping {p}: {exc}", file=sys.stderr)
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aggregate_evals",
        description="Cross-seed spread over eval reports written by "
        "eval_synria_sequence.py.",
    )
    ap.add_argument("reports", type=Path, nargs="+", help="eval JSON reports")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional single report to compare the group against.",
    )
    args = ap.parse_args(argv)

    reports = load(args.reports)
    if not reports:
        print("[aggregate] no readable reports", file=sys.stderr)
        return 1

    baseline = None
    if args.baseline:
        loaded = load([args.baseline])
        baseline = loaded[0] if loaded else None

    n = len(reports)
    print(f"[aggregate] {n} run(s):")
    for r in reports:
        print(
            f"  {Path(r.get('checkpoint', '?')).parent.name}  "
            f"{r.get('completed_episodes')} episodes  "
            f"(eval seed {r.get('eval_seed', r.get('seed'))})"
        )
    if baseline:
        print(
            f"[aggregate] baseline: "
            f"{Path(baseline.get('checkpoint', '?')).parent.name}"
        )

    head = f"\n{'stage':10} {'mean':>7} {'min':>7} {'max':>7} {'spread':>8}"
    if baseline:
        head += f" {'baseline':>9} {'vs base':>8}"
    print(head)
    print("-" * (len(head) - 1))

    for st in STAGES:
        vals = [r["stages"][st]["rate"] for r in reports if st in r.get("stages", {})]
        if not vals:
            continue
        sp = spread(vals)
        ratio = "inf" if sp.ratio == float("inf") else f"{sp.ratio:.1f}x"
        line = (
            f"{st:10} {sp.mean:7.3f} {sp.lo:7.3f} {sp.hi:7.3f} {ratio:>8}"
        )
        if baseline and st in baseline.get("stages", {}):
            b = baseline["stages"][st]["rate"]
            if b > 0 and sp.mean > 0:
                delta = sp.mean / b
                # Flag only when the group mean moves further than the group's
                # own seeds move among themselves AND the move is worth
                # noticing. Without the second clause a saturated stage like
                # reach (spread 1.02x because every seed is near 1.0) flags on
                # a 2% difference, which is noise dressed as a finding.
                material = not (MIN_MATERIAL_DELTA > delta > 1 / MIN_MATERIAL_DELTA)
                beats_noise = sp.ratio == float("inf") or delta > sp.ratio
                mark = " *" if (material and beats_noise) else ""
                line += f" {b:9.3f} {delta:7.2f}x{mark}"
            else:
                line += f" {b:9.3f} {'--':>8}"
        print(line)

    if n < MIN_SEEDS_FOR_A_DECISION:
        print(
            f"\n[aggregate] WARNING: {n} seed(s). The spread column is not "
            f"measuring seed variance below {MIN_SEEDS_FOR_A_DECISION} runs, "
            "and a single run reports a spread of 1.0x while telling you "
            "nothing. Do not make a keep/revert call on this."
        )
    else:
        print(
            f"\n[aggregate] {n} seeds. A difference smaller than the spread "
            "column is not evidence -- that column is what the same code and "
            "budget produce from RNG alone. '*' marks a vs-baseline delta that "
            "exceeds it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
