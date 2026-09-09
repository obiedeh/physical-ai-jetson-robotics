"""Rebuild ``training_results.json`` for a run from its TensorBoard events.

Why this exists
---------------
``train_synria_pickplace.py`` used to read its final reward statistics off
``runner.logger``, an attribute RSL-RL does not have. The read raised, a bare
``except Exception: pass`` swallowed it, and every run since has written
``mean_reward_last_iter: NaN``. The summaries were unusable as evidence, so
the real numbers were being read off the eval reports instead.

The statistics are not recoverable from the runner after the fact:
``OnPolicyRunner.learn`` keeps ``rewbuffer``/``lenbuffer`` as function locals
and never stores them on ``self``. But it does log them through
``self.writer``, so the TensorBoard event file the run already wrote is the
authoritative record — and it stays readable long after the process exits.

That is what this module reads, which makes it work in two directions:
``train_synria_pickplace.py`` imports :func:`final_scalars` to write a correct
summary at the end of a run, and the CLI here repairs the summary of any run
that already finished (including one still in progress — the event file is
flushed per iteration).

    python isaac/scripts/summarize_training_run.py reports/training/<run>
    python isaac/scripts/summarize_training_run.py reports/training/<run> --write

Note this reads TensorBoard only — it does not need Isaac Sim, so it runs in
a plain Python environment as well as inside the Isaac venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Scalars RSL-RL logs once per iteration.
_REWARD_TAG = "Train/mean_reward"
_EP_LEN_TAG = "Train/mean_episode_length"

# Per-term breakdowns. These are the numbers that make the summary usable as
# evidence — grasp_confirmed, piece_lift and friends are the task-level rates
# the experiment ledger is actually judged on.
_TERM_PREFIXES = ("Episode_Reward/", "Episode_Termination/")


def _latest_event_file(run_dir: Path) -> Path | None:
    """Newest ``events.out.tfevents.*`` in ``run_dir``, or None if there is none.

    A resumed run leaves several event files behind (one per process). The
    newest is the one this run wrote, which is why this sorts by mtime rather
    than by name — the filename timestamp is the writer's start time, and a
    resumed run that outlives a later smoke test would otherwise lose.
    """
    events = sorted(
        run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime
    )
    return events[-1] if events else None


def final_scalars(run_dir: Path) -> dict[str, Any]:
    """Return the last logged value of every scalar tag in ``run_dir``.

    Returns keys ``mean_reward_last_iter``, ``mean_episode_length_last_iter``,
    ``last_iteration`` and ``reward_terms`` (a per-term dict). Missing values
    come back as ``None`` rather than NaN, so a gap is visible in JSON instead
    of serialising to the bare token ``NaN`` — which is what the old summaries
    emitted, and which is not valid JSON for strict parsers.

    Raises ``FileNotFoundError`` if the run has no event file, and lets
    TensorBoard's own import error propagate. Both are conditions the caller
    should see, not swallow — swallowing them is the original bug.
    """
    from tensorboard.backend.event_processing.event_accumulator import (  # noqa: PLC0415
        EventAccumulator,
    )

    event_file = _latest_event_file(run_dir)
    if event_file is None:
        raise FileNotFoundError(f"no TensorBoard event file under {run_dir}")

    # size_guidance 0 = keep every scalar; the default downsamples and would
    # drop the final point we care about.
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])

    def last(tag: str) -> tuple[float, int] | None:
        if tag not in tags:
            return None
        events = acc.Scalars(tag)
        if not events:
            return None
        return float(events[-1].value), int(events[-1].step)

    reward = last(_REWARD_TAG)
    ep_len = last(_EP_LEN_TAG)

    terms: dict[str, float] = {}
    for tag in sorted(tags):
        if tag.startswith(_TERM_PREFIXES) and not tag.endswith("/time"):
            point = last(tag)
            if point is not None:
                terms[tag] = round(point[0], 6)

    return {
        "mean_reward_last_iter": None if reward is None else round(reward[0], 4),
        "mean_episode_length_last_iter": None if ep_len is None else round(ep_len[0], 2),
        "last_iteration": None if reward is None else reward[1],
        "reward_terms": terms,
        "source_event_file": event_file.name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="summarize_training_run",
        description=(
            "Rebuild training_results.json for a run from its TensorBoard events."
        ),
    )
    parser.add_argument("run_dir", type=Path, help="reports/training/<experiment_name>")
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Merge the recovered scalars into the run's training_results.json. "
            "Without this the summary is printed and nothing is modified."
        ),
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 2

    try:
        recovered = final_scalars(run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results_path = run_dir / "training_results.json"
    existing: dict[str, Any] = {}
    if results_path.exists():
        # A summary written before this fix contains the bare token NaN, which
        # json.loads accepts but strict parsers reject; read it leniently and
        # rewrite it clean.
        existing = json.loads(results_path.read_text())

    merged = {**existing, **recovered}

    if not args.write:
        print(json.dumps(merged, indent=2))
        print(f"\n(dry run — pass --write to update {results_path})", file=sys.stderr)
        return 0

    results_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"[summarize] wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
