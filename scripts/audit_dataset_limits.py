"""Count the frames of a recorded dataset that press, leave or exceed the operator limits.

Usage::

    python scripts/audit_dataset_limits.py --dataset-root DATASET --output reports/sim_policy/STUDY

reads the dataset's parquet files, takes the gripper type from its physical contract
unless ``--gripper`` says otherwise, compares every commanded and observed value with
``config/synria_limits.yaml`` and writes ``LIMITS_AUDIT.md`` and ``limits_audit.json``
into the output directory. Nothing here changes the dataset or trains anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.dataset_limits import (  # noqa: E402
    DEFAULT_PRESS_DEG,
    audit_dataset,
    render_markdown,
    write_audit,
)
from synria_lerobot.quality_gates import load_limits  # noqa: E402

DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="LeRobot dataset directory holding data/ and physical_contract.json.")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--gripper", help="Gripper type; defaults to the dataset's contract.")
    parser.add_argument("--press-deg", type=float, default=DEFAULT_PRESS_DEG,
                        help="A joint this close to a bound counts as pressing it.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Directory for LIMITS_AUDIT.md and limits_audit.json.")
    return parser


def gripper_type_from_contract(root: Path) -> str:
    """The gripper type the dataset's physical contract declares."""
    contract = root / "physical_contract.json"
    if not contract.exists():
        raise SystemExit(f"{contract} is missing; give --gripper")
    payload = json.loads(contract.read_text(encoding="utf-8"))
    gripper = payload.get("gripper_type")
    if not isinstance(gripper, str) or not gripper:
        raise SystemExit("the physical contract names no gripper type; give --gripper")
    return gripper


def main(argv: list[str] | None = None) -> int:
    """Audit, write and print the table."""
    args = build_parser().parse_args(argv)
    gripper = args.gripper or gripper_type_from_contract(args.dataset_root)
    audit = audit_dataset(args.dataset_root, load_limits(args.limits), gripper,
                          press_deg=args.press_deg)
    write_audit(audit, args.output)
    print(render_markdown(audit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
