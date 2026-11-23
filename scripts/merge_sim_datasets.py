"""Merge two simulation datasets with the same contract into one, for one training run.

    python scripts/merge_sim_datasets.py --base "$HUMAN_DATASET" --extra "$GENERATED_DATASET" \\
        --output ~/synria-sim-datasets/<name>/dataset --repo-id local/<name>

The base dataset's episodes keep their indices (0 to N-1), so a probe set's held-out
episodes stay the same takes; the extra dataset's episodes follow as N, N+1, ...
LeRobot's own aggregation joins the frames, videos and statistics. The three per-episode
sidecar logs (capture provenance, episode metadata, quality records) are joined with the
extra episodes renumbered, and each final still is copied under its new episode number
with the logs pointing at the copy. The exclusion logs beside the two datasets are joined
the same way, so a take excluded in either source stays excluded. The two contracts must
be identical and the output must not exist; nothing in either source is changed.
Simulation lane only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.checkpoint_eval import read_episode_metadata  # noqa: E402
from synria_lerobot.exclusions import exclusion_path, read_exclusions  # noqa: E402

SIDECARS = ("physical_capture_provenance", "physical_episode_metadata",
            "physical_quality_records")
STILL_NAME = re.compile(r"^episode_(\d{6})_(.+)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every row of a JSON-lines file; a missing file has none."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def renumber_still(name: str, offset: int) -> str:
    """A final still's file name with its episode number moved by ``offset``."""
    match = STILL_NAME.match(name)
    if match is None:
        raise ValueError(f"unexpected final still name {name!r}")
    return f"episode_{int(match.group(1)) + offset:06d}_{match.group(2)}"


def merge_rows(base: list[dict[str, Any]], extra: list[dict[str, Any]], offset: int,
               still_dir: Path | None) -> list[dict[str, Any]]:
    """Base rows as they are, then extra rows renumbered by ``offset``.

    When ``still_dir`` is given, every ``final_still`` path is rewritten to that
    directory under the row's (new) episode number.
    """
    merged: list[dict[str, Any]] = []
    for rows, shift in ((base, 0), (extra, offset)):
        for row in rows:
            row = dict(row)
            if type(row.get("episode_index")) is not int:
                raise ValueError("every sidecar row needs an integer episode_index")
            row["episode_index"] += shift
            if still_dir is not None and isinstance(row.get("final_still"), str):
                row["final_still"] = str(
                    still_dir / renumber_still(Path(row["final_still"]).name, shift))
            merged.append(row)
    return merged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows once; an existing file is never overwritten."""
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Check the contracts, aggregate, join the sidecars and exclusions, verify."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args(argv)
    base, extra, output = args.base.resolve(), args.extra.resolve(), args.output.resolve()
    if output.exists() or exclusion_path(output).exists():
        raise SystemExit("the output dataset and its exclusion log must not exist")
    if output.is_relative_to(REPOSITORY):
        raise SystemExit("datasets stay outside the repository")
    contracts = [json.loads((root / "physical_contract.json").read_text(encoding="utf-8"))
                 for root in (base, extra)]
    if contracts[0] != contracts[1]:
        raise SystemExit("the two datasets' contracts differ; refusing to merge")
    offset = len(read_episode_metadata(base))
    extra_count = len(read_episode_metadata(extra))

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(repo_ids=["local/merge_base", "local/merge_extra"],
                       aggr_repo_id=args.repo_id, roots=[base, extra], aggr_root=output)
    shutil.copyfile(base / "physical_contract.json", output / "physical_contract.json")
    still_dir = output / "final_stills"
    still_dir.mkdir(exist_ok=True)
    for root, shift in ((base, 0), (extra, offset)):
        for still in sorted((root / "final_stills").glob("*")):
            shutil.copyfile(still, still_dir / renumber_still(still.name, shift))
    for name in SIDECARS:
        rows = merge_rows(read_jsonl(base / f"{name}.jsonl"),
                          read_jsonl(extra / f"{name}.jsonl"), offset, still_dir)
        write_jsonl(output / f"{name}.jsonl", rows)
    exclusions = merge_rows(read_jsonl(exclusion_path(base)),
                            read_jsonl(exclusion_path(extra)), offset, None)
    if exclusions:
        write_jsonl(exclusion_path(output), exclusions)

    episodes = read_episode_metadata(output)
    if len(episodes) != offset + extra_count:
        raise SystemExit(f"merged dataset has {len(episodes)} episodes, expected "
                         f"{offset + extra_count}")
    state = read_exclusions(output)
    print(json.dumps({"episodes": len(episodes), "base_episodes": offset,
                      "extra_episodes": extra_count,
                      "excluded": sorted(state.excluded_ids),
                      "output": str(output)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
