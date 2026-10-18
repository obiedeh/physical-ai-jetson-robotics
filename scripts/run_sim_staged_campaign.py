"""Serve the staged executor and run one seeded trial campaign against it.

Usage::

    python scripts/run_sim_staged_campaign.py --task-id pick_cup --trials 50 \\
        --placement-seed 2026 --dataset-root DATASET --output-dir reports/sim_policy/staged

starts ``scripts/serve_staged_executor.py`` on ``--port``, waits for its metadata,
runs ``scripts/run_sim_policy_trials.py`` with the same seed, scale and timing as the
checkpoint campaigns, stops the server, and writes beside the trial report a
``stages_<utc>.json`` that counts the final phase of every trial and the
infeasible plans, so the report says which stage failed and not only that a
trial did. The scene must already be running in policy mode. Simulation only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.run_sim_checkpoint_campaigns import (  # noqa: E402
    SERVER_START_TIMEOUT_S,
    wait_for_server,
)

DEFAULT_REGISTRY = REPOSITORY / "config" / "synria_tasks.json"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; trial timing is explicit, as for the checkpoint campaigns."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--command-period-s", type=float, default=0.2)
    parser.add_argument("--response-timeout-s", type=float, default=0.5)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--randomize-scale", type=float, default=1.0)
    parser.add_argument("--placement-seed", type=int, default=None)
    parser.add_argument("--task-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--policy-config", type=Path, default=None,
                        help="Safety file for the trial client and the executor's step size.")
    parser.add_argument("--speed-fraction", type=float, default=0.8)
    parser.add_argument("--margin-deg", type=float, default=3.0)
    parser.add_argument("--port", type=int, default=8091)
    return parser


def summarize_stages(stage_log: Path) -> dict[str, object]:
    """Count final phases, infeasible plans, stalls and which waypoint failed."""
    trials = [json.loads(line) for line in stage_log.read_text(encoding="utf-8").splitlines()
              if line.strip() and json.loads(line).get("kind") == "trial"]
    final = Counter(trial["final_phase"] for trial in trials)
    failed_waypoints: Counter[str] = Counter()
    for trial in trials:
        if trial.get("feasible") is False:
            for name, entry in (trial.get("plan") or {}).items():
                if isinstance(entry, dict) and entry.get("feasible") is False:
                    failed_waypoints[name] += 1
    stalls = Counter(stall["phase"] for trial in trials for stall in trial.get("stalls", []))
    return {
        "trials": len(trials),
        "final_phase": dict(sorted(final.items())),
        "infeasible": sum(1 for trial in trials if trial.get("feasible") is False),
        "infeasible_waypoints": dict(sorted(failed_waypoints.items())),
        "stalls_by_phase": dict(sorted(stalls.items())),
        "stage_log": str(stage_log),
    }


def main(argv: list[str] | None = None) -> int:
    """Serve, run, stop, summarize."""
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = args.output_dir / f"trials_staged_expert_{args.task_id}_{stamp}.json"
    stage_log = args.output_dir / f"stages_{args.task_id}_{stamp}.jsonl"
    endpoint = f"http://127.0.0.1:{args.port}"
    policy_config = ["--policy-config", str(args.policy_config)] if args.policy_config else []
    server = subprocess.Popen([
        sys.executable, str(REPOSITORY / "scripts" / "serve_staged_executor.py"),
        "--task-id", args.task_id, "--task-registry", str(args.task_registry),
        "--command-period-s", str(args.command_period_s), "--port", str(args.port),
        "--speed-fraction", str(args.speed_fraction), "--margin-deg", str(args.margin_deg),
        "--stage-log", str(stage_log), *policy_config,
    ])
    try:
        metadata = wait_for_server(endpoint, SERVER_START_TIMEOUT_S)
        if metadata.get("policy_id") != "staged_expert_v1":
            raise RuntimeError("the endpoint is not the staged executor")
        subprocess.run([
            sys.executable, str(REPOSITORY / "scripts" / "run_sim_policy_trials.py"),
            "--endpoint", endpoint, "--trials", str(args.trials),
            "--command-period-s", str(args.command_period_s),
            "--response-timeout-s", str(args.response_timeout_s),
            "--dataset-root", str(args.dataset_root), "--output", str(report),
            "--randomize-scale", str(args.randomize_scale),
            *(["--placement-seed", str(args.placement_seed)]
              if args.placement_seed is not None else []),
            *policy_config,
        ], check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    summary = summarize_stages(stage_log)
    result = json.loads(report.read_text(encoding="utf-8"))
    summary.update(report=str(report), successes=result["successes"],
                   success_rate=result["success_rate"])
    stages_path = args.output_dir / f"stages_{args.task_id}_{stamp}.json"
    stages_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
