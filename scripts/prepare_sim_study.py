"""Prepare a simulation data-scaling study for a dataset: probes, training configuration, spec.

Usage::

    python scripts/prepare_sim_study.py --dataset-root DATASET --study-id staged_place_01 \\
        --policy-prefix simplace01 --held-out 8 --subset-sizes 50 100 150 --steps 20000 \\
        --registered-by "staged_expert"

writes ``reports/sim_policy/<study-id>/probes.json`` (registered, with its hash),
``train_config.json`` (the committed defaults with the dataset's contract, the
simulation lane, the probe set and its hash) and ``spec.json`` for
``scripts/run_sim_data_scaling_study.py``. Held-out takes are chosen evenly across
the dataset's non-excluded takes so the probe set spans the recording; excluded
takes are never held out. The probe file must then be committed before training,
which the trainer enforces. Nothing here trains or runs a trial.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.checkpoint_eval import (  # noqa: E402
    DIAGNOSTIC_PURPOSE,
    PROBE_VERSION,
    read_episode_metadata,
    register_probes,
    strict_content_hash,
)
from synria_lerobot.exclusions import read_exclusions  # noqa: E402
from synria_lerobot.physical_contract import SIMULATION_LANE, PhysicalDatasetContract  # noqa: E402

DEFAULT_TRAINING = REPOSITORY / "config" / "synria_act_training.json"
SIM_LEDGER = "docs/ludo_flagship/SIM_POLICY_LEDGER.md"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; the step count and seed are explicit study identity."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--policy-prefix", required=True)
    parser.add_argument("--held-out", type=int, required=True,
                        help="How many takes to hold out, spread evenly over the dataset.")
    parser.add_argument("--subset-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--command-period-s", type=float, default=0.2)
    parser.add_argument("--response-timeout-s", type=float, default=0.5)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--registered-by", required=True)
    parser.add_argument("--base-training-config", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--evidence-root", type=Path, default=None,
                        help="Defaults to reports/sim_policy/<study-id> in the repository.")
    parser.add_argument("--raw-output-root", type=Path, default=None,
                        help="Defaults to ~/synria-sim-training/<study-id>.")
    return parser


def choose_held_out(episodes: list[int], count: int) -> list[int]:
    """``count`` takes spread evenly over the ordered takes, never the first or last."""
    if count < 1 or count >= len(episodes):
        raise ValueError("hold out at least one take and fewer than all of them")
    chosen = []
    for index in range(count):
        position = round((index + 1) * len(episodes) / (count + 1))
        chosen.append(episodes[min(len(episodes) - 1, max(0, position))])
    return sorted(set(chosen))


def prepare(args: argparse.Namespace) -> dict[str, Path]:
    """Write the three files; returns their paths."""
    root = args.dataset_root.resolve()
    contract_payload = json.loads((root / "physical_contract.json").read_text(encoding="utf-8"))
    contract = PhysicalDatasetContract.from_dict(contract_payload)
    contract.require_lane(SIMULATION_LANE)
    lengths = read_episode_metadata(root)
    excluded = set(read_exclusions(root).excluded_ids)
    usable = [index for index in sorted(lengths) if index not in excluded]
    held_out = choose_held_out(usable, args.held_out)
    evidence = (args.evidence_root or (REPOSITORY / "reports" / "sim_policy" / args.study_id))
    evidence = evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    probes_path = evidence / "probes.json"
    if probes_path.exists():
        raise SystemExit(f"{probes_path} exists; a probe set is frozen once")
    probes = {
        "version": PROBE_VERSION,
        "purpose": DIAGNOSTIC_PURPOSE,
        "probe_set_id": f"{args.study_id}_sim",
        "registered_by": args.registered_by,
        "registered_on": date.today().isoformat(),
        "dataset_content_sha256": strict_content_hash(root),
        "task_id": contract.task_id,
        "physical_contract": contract_payload,
        "held_out_episodes": held_out,
        "held_out_frame_counts": {str(index): int(lengths[index]) for index in held_out},
        "physical_trials": [],
        "capture": {"fps": contract_payload["requested_rate_hz"], "max_duration_s": 120,
                    "shutdown_timeout_s": 5},
        "lane": SIMULATION_LANE,
    }
    probes_path.write_text(json.dumps(probes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    probe_hash = register_probes(probes_path)
    base = json.loads(args.base_training_config.read_text(encoding="utf-8"))
    base.update(
        lane=SIMULATION_LANE, expected_contract=contract_payload,
        dataset_repo_id=f"local/{args.study_id}_sim", repository=str(REPOSITORY),
        probe_set=(str(probes_path.relative_to(REPOSITORY))
                   if probes_path.is_relative_to(REPOSITORY) else str(probes_path)),
        probe_sha256=probe_hash,
        policy_ledger=SIM_LEDGER, command_period_s=args.command_period_s,
        response_timeout_s=args.response_timeout_s, chunk_size=args.chunk_size,
        device="cuda", save_freq=args.save_freq, dataloader_workers=args.dataloader_workers,
    )
    config_path = evidence / "train_config.json"
    config_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    spec = {
        "version": "synria_sim_data_scaling_v1",
        "study_id": args.study_id,
        "dataset_root": str(root),
        "contract": str(root / "physical_contract.json"),
        "evidence_root": str(evidence),
        "raw_output_root": str((args.raw_output_root or (
            Path.home() / "synria-sim-training" / args.study_id)).resolve()),
        "policy_prefix": args.policy_prefix,
        "training_config": str(config_path),
        "subset_sizes": sorted(set(args.subset_sizes)),
        "seed": args.seed,
        "steps": args.steps,
    }
    spec_path = evidence / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"probes": probes_path, "train_config": config_path, "spec": spec_path}


def main(argv: list[str] | None = None) -> int:
    """Prepare and print the paths; remind that the probe set must be committed."""
    paths = prepare(build_parser().parse_args(argv))
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    print("commit probes.json before training; the trainer refuses an uncommitted probe set",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
