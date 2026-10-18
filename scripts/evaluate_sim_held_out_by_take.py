"""Split a simulation study's held-out error by take and write the table beside the study.

Usage::

    python scripts/evaluate_sim_held_out_by_take.py \\
        --study reports/sim_policy/pick_cup_17oct3/study.json --device cuda \\
        --label 36=randomized --label 44=randomized

Each row's final checkpoint is reloaded and scored on every held-out take on its own,
with the same metric the trainer used. ``--label`` names a take in the table header,
for example which takes were recorded with randomized object placement. The outputs are
``held_out_by_take.json`` and ``HELD_OUT_BY_TAKE.md`` in the study directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.sim_held_out import held_out_by_take, write_result  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--label", action="append", default=[], metavar="TAKE=TEXT")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Read the study, its probe set and training configuration, then score and write."""
    args = build_parser().parse_args(argv)
    study = json.loads(args.study.read_text(encoding="utf-8"))
    directory = args.study.parent
    probes = json.loads((directory / "probes.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "train_config.json").read_text(encoding="utf-8"))
    lengths = {int(index): int(count) for index, count in probes["held_out_frame_counts"].items()}
    labels = dict(item.split("=", 1) for item in args.label)
    result = held_out_by_take(
        study, device=args.device, chunk_size=int(config["chunk_size"]),
        repo_id=config["dataset_repo_id"],
        fps=float(config["expected_contract"]["requested_rate_hz"]),
        held_out_lengths=lengths,
    )
    record, table = write_result(result, directory, labels)
    print(json.dumps({"rows": len(result["rows"]), "record": str(record), "table": str(table)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
