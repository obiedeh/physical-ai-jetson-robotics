"""Average a run's last checkpoints into one servable checkpoint with its own receipt.

Usage::

    python scripts/average_synria_checkpoints.py \\
        --evidence reports/sim_policy/pick_cup_17oct3/train_n045 --last 4 \\
        --dataset-root "$DATASET_ROOT" --device cuda

The averaged checkpoint lands beside its sources as ``averaged-last<k>-step-<step>``
and its receipt as ``checkpoint-averaged-last<k>.json`` in the evidence directory. It is
evaluated on the frozen held-out set first, so the receipt says ``evaluated`` and
``scripts/serve_synria_policy.py`` accepts it like any saved step. Nothing trains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.checkpoint_average import average_checkpoints  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evidence", type=Path, required=True,
                        help="The run's evidence directory holding training_run.json.")
    parser.add_argument("--last", type=int, default=4, help="How many final checkpoints.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Average, evaluate, receipt, and print the summary."""
    args = build_parser().parse_args(argv)
    summary = average_checkpoints(
        args.evidence, args.last, repository=args.repository, dataset_root=args.dataset_root,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
