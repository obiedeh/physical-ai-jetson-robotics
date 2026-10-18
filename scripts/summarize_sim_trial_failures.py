"""Sort every trial of a study's campaign reports into a failure mode and write a table.

Usage::

    python scripts/summarize_sim_trial_failures.py --study reports/sim_policy/pick_cup_17oct3

reads every ``trials_*.json`` in the study directory, marks the campaigns the study's
``study.json`` lists as superseded, and writes ``FAILURE_MODES.md`` and
``failure_modes.json`` beside them. Give ``--reports`` instead to name the reports
yourself. Nothing here trains or runs a trial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.sim_trial_analysis import (  # noqa: E402
    render_markdown,
    summarize_reports,
    superseded_reports,
    write_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", type=Path, help="Study directory holding trials_*.json.")
    parser.add_argument("--reports", type=Path, nargs="*", help="Trial reports, in order.")
    parser.add_argument("--output", type=Path,
                        help="Directory for the table; defaults to the study directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Summarize, write and print the table."""
    args = build_parser().parse_args(argv)
    if (args.study is None) == (not args.reports):
        raise SystemExit("give exactly one of --study and --reports")
    paths = list(args.reports or sorted(args.study.glob("trials_*.json")))
    output = args.output or args.study or paths[0].parent
    summary = summarize_reports(
        paths, superseded=superseded_reports(args.study) if args.study else set(),
    )
    write_summary(summary, output)
    print(render_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
