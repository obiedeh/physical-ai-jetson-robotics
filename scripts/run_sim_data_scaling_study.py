"""Run the simulation data-scaling study: nested-subset training, optional scene trials, curve.

The study specification is a JSON file like ``config/synria_sim_data_scaling.json`` with
the frozen dataset, its contract, a simulation-lane training configuration whose probe
set is registered and committed, the subset sizes, seed and steps, a raw output root
outside git and an evidence root inside the repository. Usage::

    python scripts/run_sim_data_scaling_study.py --spec reports/sim_policy/<study>/spec.json
    python scripts/run_sim_data_scaling_study.py --spec ... --trials 20 \\
        --command-period-s 0.2 --response-timeout-s 0.5 --device cuda

Without ``--trials`` only training and the held-out error run. With it, after each
subset finishes, the final checkpoint is served on a local port with
``--lane simulation`` and ``scripts/run_sim_policy_trials.py`` runs the campaign against
the Isaac scene, which must already be in policy mode. ``--trials-only`` adds
the campaigns to a completed study's saved checkpoints without training anything.
The study record and the Markdown table land in the evidence root; both say they
are simulation evidence only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.policy_client import HttpPolicyTransport  # noqa: E402
from synria_lerobot.sim_data_scaling import add_trials, run_study  # noqa: E402

SERVER_START_TIMEOUT_S = 120.0


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--trials", type=int, help="Run a scored campaign per subset.")
    parser.add_argument("--trials-only", action="store_true",
                        help="Add campaigns to a completed study's saved checkpoints; no training.")
    parser.add_argument("--command-period-s", type=float)
    parser.add_argument("--response-timeout-s", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--task-registry", type=Path,
                        default=REPOSITORY / "config" / "synria_tasks.json")
    return parser


def wait_for_server(endpoint: str, timeout_s: float) -> dict[str, Any]:
    """Poll the served metadata until it answers or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    transport = HttpPolicyTransport(endpoint)
    last = ""
    while time.monotonic() < deadline:
        try:
            return transport.metadata(5.0)
        except Exception as error:  # noqa: BLE001 - the server is still starting
            last = str(error)
            time.sleep(1.0)
    raise RuntimeError(f"policy server did not answer within {timeout_s:.0f} s: {last}")


def campaign_for(args: argparse.Namespace, spec: dict[str, Any]) -> Any:
    """A trial function that serves each subset's checkpoint and runs the campaign on it."""
    if args.command_period_s is None or args.response_timeout_s is None:
        raise SystemExit("--trials needs --command-period-s and --response-timeout-s")
    endpoint = f"http://127.0.0.1:{args.port}"

    def run(row: dict[str, Any]) -> dict[str, Any]:
        """Serve, wait, run the campaign, stop; the row's report path is returned."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = Path(spec["evidence_root"]) / f"trials_{row['policy_id']}_{stamp}.json"
        server = subprocess.Popen([
            sys.executable, str(REPOSITORY / "scripts" / "serve_synria_policy.py"),
            "--checkpoint", row["checkpoint_path"],
            "--checkpoint-record", row["checkpoint_record"],
            "--task-registry", str(args.task_registry),
            "--device", args.device, "--port", str(args.port), "--lane", "simulation",
        ])
        try:
            metadata = wait_for_server(endpoint, SERVER_START_TIMEOUT_S)
            if metadata.get("checkpoint_content_sha256") != row["checkpoint_content_sha256"]:
                raise RuntimeError("served checkpoint differs from the study row")
            subprocess.run([
                sys.executable, str(REPOSITORY / "scripts" / "run_sim_policy_trials.py"),
                "--endpoint", endpoint, "--trials", str(args.trials),
                "--command-period-s", str(args.command_period_s),
                "--response-timeout-s", str(args.response_timeout_s),
                "--dataset-root", spec["dataset_root"], "--output", str(report),
            ], check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
        result = json.loads(report.read_text(encoding="utf-8"))
        return {
            "report": str(report), "trials": len(result["trials"]),
            "successes": result["successes"], "success_rate": result["success_rate"],
        }

    return run


def main(argv: list[str] | None = None) -> int:
    """Run the study and write its record and table."""
    args = build_parser().parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    trial_fn = campaign_for(args, spec) if args.trials else None
    evidence = Path(spec["evidence_root"])
    if args.trials_only:
        if trial_fn is None:
            raise SystemExit("--trials-only needs --trials")
        study = json.loads((evidence / "study.json").read_text(encoding="utf-8"))
        study = add_trials(study, trial_fn, write_to=evidence)
    else:
        study = run_study(spec, trial_fn=trial_fn, write_to=evidence)
    print(json.dumps({"status": study["status"], "rows": len(study["rows"]),
                      "record": str(evidence / "study.json"), "table": str(evidence / "STUDY.md")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
