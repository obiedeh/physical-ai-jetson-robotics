"""Serve several saved checkpoints in turn and run one scored campaign against each.

Usage::

    python scripts/run_sim_checkpoint_campaigns.py \\
        --checkpoint "$RUN/checkpoints/checkpoint-step-000020000" \\
                     "$EVIDENCE/checkpoint-step-000020000.json" \\
        --checkpoint "$RUN/checkpoints/averaged-last4-step-000020000" \\
                     "$EVIDENCE/checkpoint-averaged-last4.json" \\
        --trials 50 --command-period-s 0.2 --response-timeout-s 0.5 --placement-seed 2026 \\
        --dataset-root "$DATASET_ROOT" --output-dir reports/sim_policy/<study>/campaigns

Every ``--checkpoint`` names a checkpoint directory and its completion receipt, as the
policy server takes them. For each one the script starts a simulation-lane server on
``--port``, waits for it, runs ``scripts/run_sim_policy_trials.py`` with the same seed
and scale, stops the server and moves on; one report per checkpoint lands in
``--output-dir`` as ``trials_<policy_id>_<checkpoint name>_<utc>.json``. With a
placement seed every campaign faces the same layouts, so the reports compare trial by
trial. The Isaac scene must already be in its policy launch mode. Nothing trains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from run_sim_data_scaling_study import SERVER_START_TIMEOUT_S, wait_for_server  # noqa: E402

DEFAULT_REGISTRY = REPOSITORY / "config" / "synria_tasks.json"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; the trial timing values are explicit."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", nargs=2, action="append", required=True,
                        metavar=("DIRECTORY", "RECORD"),
                        help="A checkpoint directory and its completion receipt; repeatable.")
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--command-period-s", type=float, required=True)
    parser.add_argument("--response-timeout-s", type=float, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--randomize-scale", type=float, default=1.0)
    parser.add_argument("--placement-seed", type=int, default=None)
    parser.add_argument("--task-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--port", type=int, default=8090)
    return parser


def run_campaign(args: argparse.Namespace, checkpoint: Path, record: Path) -> Path:
    """Serve one checkpoint, run its campaign, stop the server; returns the report path."""
    endpoint = f"http://127.0.0.1:{args.port}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = json.loads(record.read_text(encoding="utf-8"))
    report = args.output_dir / f"trials_{receipt['policy_id']}_{checkpoint.name}_{stamp}.json"
    server = subprocess.Popen([
        sys.executable, str(REPOSITORY / "scripts" / "serve_synria_policy.py"),
        "--checkpoint", str(checkpoint), "--checkpoint-record", str(record),
        "--task-registry", str(args.task_registry),
        "--device", args.device, "--port", str(args.port), "--lane", "simulation",
    ])
    try:
        metadata = wait_for_server(endpoint, SERVER_START_TIMEOUT_S)
        if metadata.get("checkpoint_content_sha256") != receipt["checkpoint_content_sha256"]:
            raise RuntimeError(f"served checkpoint differs from the receipt {record}")
        subprocess.run([
            sys.executable, str(REPOSITORY / "scripts" / "run_sim_policy_trials.py"),
            "--endpoint", endpoint, "--trials", str(args.trials),
            "--command-period-s", str(args.command_period_s),
            "--response-timeout-s", str(args.response_timeout_s),
            "--dataset-root", str(args.dataset_root), "--output", str(report),
            "--randomize-scale", str(args.randomize_scale),
            *(["--placement-seed", str(args.placement_seed)]
              if args.placement_seed is not None else []),
        ], check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
    return report


def main(argv: list[str] | None = None) -> int:
    """Run every campaign in order and print one summary line per report."""
    args = build_parser().parse_args(argv)
    pairs = [(Path(directory), Path(record)) for directory, record in args.checkpoint]
    for directory, record in pairs:
        if not directory.is_dir() or not record.is_file():
            raise SystemExit(f"missing checkpoint directory or receipt: {directory} {record}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for directory, record in pairs:
        report = run_campaign(args, directory, record)
        result = json.loads(report.read_text(encoding="utf-8"))
        print(json.dumps({"checkpoint": str(directory), "successes": result["successes"],
                          "trials": len(result["trials"]), "report": str(report)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
