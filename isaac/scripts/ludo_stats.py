#!/usr/bin/env python3
"""Aggregate Ludo executor sessions into cross-session statistics and a
failure taxonomy. One command regenerates everything from the raw
append-only artifacts (turns.jsonl + provenance.json per session dir);
never mutates inputs.

    python3 isaac/scripts/ludo_stats.py            # scan reports/ludo_*
    python3 isaac/scripts/ludo_stats.py --glob 'reports/ludo_soak_*'

Outputs (overwritten — they are derived views, the raw data is the
record): reports/ludo_stats.csv (one row per attempt) and
reports/ludo_stats_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _percentile(sorted_vals: list, p: float):
    """Index-based percentile, ported from physical-ai-safety-observability
    telemetry/runtime.py — the portfolio audit (2026-08-19) found three
    distinct off-by-one percentile implementations and exactly one correct
    one; this is it. round((p/100)*(n-1)) on the sorted values."""
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    return sorted_vals[round((p / 100.0) * (n - 1))]


def classify_failure(d: dict) -> str:
    """Symptom-level taxonomy from the attempt record alone."""
    if not d.get("grasped"):
        return "never-grasped (approach failed)"
    if d.get("final_tilt_deg", 0) > 45:
        return "cup toppled"
    if d.get("hold_armed"):
        return "hold armed but scored out (hold-window edge)"
    if d.get("err_mm", 1e9) < 60:
        return "near-miss placement (35-60mm)"
    return "carry/place lost the cup (>60mm)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="reports/ludo_*")
    args = ap.parse_args()

    rows = []
    for sdir in sorted(REPO.glob(args.glob)):
        tj = sdir / "turns.jsonl"
        if not tj.exists():
            continue
        prov = {}
        pj = sdir / "provenance.json"
        if pj.exists():
            prov = json.loads(pj.read_text())
        for line in tj.read_text().splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            for d in t.get("attempts", []):
                rows.append({
                    "session": sdir.name,
                    "git_sha": prov.get("git_sha", "")[:9],
                    "seed": prov.get("seed"),
                    "turn": t["turn"],
                    "attempt": d["attempt"],
                    "ok": d["ok"],
                    "err_mm": d["err_mm"],
                    "final_tilt_deg": d["final_tilt_deg"],
                    "steps": d["steps"],
                    "lane": d["lane"],
                    "far_pick": d.get("far_pick"),
                    "grasped": d.get("grasped"),
                    "hold_armed": d.get("hold_armed"),
                    "closest_approach_mm": d.get("closest_approach_mm"),
                    "failure_class": (None if d["ok"]
                                      else classify_failure(d)),
                })
    if not rows:
        print("[stats] no sessions with attempt-level data found "
              "(pre-instrumentation sessions lack 'attempts')")
        return 1

    # a NON-default glob is a per-campaign view: write it beside its

    # frozen copy only — never over the tracked campaign-wide files

    # (the groot17 eval chain clobbered them once, 2026-08-20)

    _default_glob = "reports/ludo_*"

    _slug = args.glob.replace("/", "_").replace("*", "STAR")

    _view_dir = REPO / "reports/ludo_stats_frozen"

    _view_dir.mkdir(exist_ok=True)

    out_csv = (REPO / "reports/ludo_stats.csv" if args.glob == _default_glob

               else _view_dir / f"view_{_slug}.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    firsts = [r for r in rows if r["attempt"] == 1]
    n = len(firsts)
    k = sum(1 for r in firsts if r["ok"])
    p = k / n
    sd = math.sqrt(p * (1 - p) / n) if n else 0.0
    turn_keys = {(r["session"], r["turn"]) for r in rows}
    turn_ok = sum(1 for s, t in turn_keys if any(
        r["ok"] for r in rows if r["session"] == s and r["turn"] == t))
    errs = sorted(r["err_mm"] for r in firsts if r["ok"])
    approaches = sorted(r["closest_approach_mm"] for r in rows
                        if r.get("closest_approach_mm") is not None)
    tax: dict = {}
    for r in rows:
        if r["failure_class"]:
            tax[r["failure_class"]] = tax.get(r["failure_class"], 0) + 1
    summary = {
        "sessions": len({r["session"] for r in rows}),
        "attempts": len(rows),
        "first_try": {"n": n, "ok": k, "rate": round(p, 3),
                      "binomial_sd": round(sd, 3),
                      "binomial_sd_note": "assumes independent Bernoulli "
                      "trials; the ledger disputes this (the deterministic "
                      "rails are per-session-history, not per-turn), so "
                      "treat the SD as a lower bound on uncertainty"},
        "turn_level": {"n": len(turn_keys), "ok": turn_ok,
                       "rate": round(turn_ok / max(1, len(turn_keys)), 3)},
        "ok_err_mm": {"p50": _percentile(errs, 50),
                      "p95": _percentile(errs, 95),
                      "n": len(errs)},
        "closest_approach_mm": {"p50": _percentile(approaches, 50),
                                "p95": _percentile(approaches, 95),
                                "n": len(approaches)},
        "failure_taxonomy_counts": tax,
        "data_kind": "simulated",
        "note": "first_try = attempt 1 only; turn_level counts the "
                "built-in single retry; success != reward != normalized "
                "score (no reward exists in this pipeline)",
        "glob": args.glob,
        "generated_utc": __import__("time").strftime(
            "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
    }
    out_json = (REPO / "reports/ludo_stats_summary.json"
                if args.glob == _default_glob
                else _view_dir / f"view_{_slug}_summary.json")
    out_json.write_text(json.dumps(summary, indent=2))
    # frozen, campaign-stamped copy — never overwritten (audit A3: the
    # 6-seed block's numbers previously survived only in ledger prose)
    frozen_dir = REPO / "reports/ludo_stats_frozen"
    frozen_dir.mkdir(exist_ok=True)
    slug = args.glob.replace("/", "_").replace("*", "STAR")
    fp = frozen_dir / f"stats_{summary['generated_utc'].replace(':', '')}_{slug}.json"
    i = 1
    while fp.exists():
        fp = frozen_dir / f"{fp.stem}.{i}.json"
        i += 1
    fp.write_text(json.dumps(summary, indent=2))
    print(f"[stats] frozen -> {fp}")
    print(f"[stats] {len(rows)} attempts, {len(turn_keys)} turns -> "
          f"{out_csv.name}, {out_json.name}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
