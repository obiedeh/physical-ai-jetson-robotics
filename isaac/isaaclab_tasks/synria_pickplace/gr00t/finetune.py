"""Fine-tune GR00T N1.7 on Synria pick-and-place demonstrations.

Wraps NVIDIA's official ``gr00t/experiment/launch_finetune.py`` CLI with
Synria-specific defaults (embodiment tag, modality config, dataset paths,
output paths). The CLI is the upstream entry point — we subprocess-invoke
it rather than re-implement training in Python, which is the pattern
documented in the official Isaac-GR00T README.

Reference:
    https://github.com/NVIDIA/Isaac-GR00T (Apache-2.0)

Runs on the Linux RTX 5090 only — Windows imports succeed but invoking
``main()`` raises a clear ``SystemExit`` with the install hint.

Usage on RTX:

    uv run python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune \\
        --game chess \\
        --amplified \\
        --output-dir runs/gr00t_synria_chess_v0 \\
        --base-model nvidia/GR00T-N1.7-3B \\
        --max-steps 10000 \\
        --global-batch-size 16

The recipe mirrors NVIDIA's published LeRobot SO-101 post-training recipe.
The Synria is a structurally similar single 6DOF + 2-finger gripper arm.

Python 3.10 is required (gr00t pyproject pins ``==3.10.*``). The Synria
fine-tune likely needs a separate ``uv``-managed venv from this repo's
main Python 3.11+ venv. See ``docs/SYNRIA_GR00T_FINETUNE.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GROOT_LAUNCH_SCRIPT_REL = "gr00t/experiment/launch_finetune.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune",
        description=(
            "Wrap NVIDIA's gr00t/experiment/launch_finetune.py with "
            "Synria pick-and-place defaults."
        ),
    )
    parser.add_argument(
        "--game",
        choices=("ludo", "chess", "checkers"),
        required=True,
        help="Which game's demonstrations to fine-tune on.",
    )
    parser.add_argument(
        "--amplified",
        action="store_true",
        help="Use the GR00T-Mimic amplified dataset instead of seed demos.",
    )
    parser.add_argument(
        "--base-model",
        default="nvidia/GR00T-N1.7-3B",
        help=(
            "HuggingFace model ID or local path to the base GR00T checkpoint. "
            "Default is N1.7 3B per the official README."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "gr00t_finetune",
        help="Where to write LoRA adapters / checkpoints / logs.",
    )
    # Names match gr00t.configs.finetune_config.FinetuneConfig (tyro CLI):
    # there is no --num-epochs/--batch-size — training is bounded by
    # --max-steps and batched by --global-batch-size.
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gr00t-repo",
        type=Path,
        default=None,
        help=(
            "Path to a local clone of NVIDIA/Isaac-GR00T. If omitted, the "
            "wrapper assumes `gr00t` is on PYTHONPATH and launches via "
            "`python -m gr00t.experiment.launch_finetune`."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command without invoking it.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help=(
            "Any remaining arguments are forwarded verbatim to "
            "launch_finetune.py (e.g. --max-steps 50000, --lora-rank 16). "
            "Use `--` to separate them from wrapper args."
        ),
    )
    return parser


def _build_launch_cmd(args: argparse.Namespace) -> list[str]:
    """Construct the subprocess command for gr00t/experiment/launch_finetune.py."""
    from .dataset import SYNRIA_DEMO_ROOTS, SYNRIA_MIMIC_AMPLIFIED_ROOTS
    from .embodiment import SYNRIA_EMBODIMENT_TAG

    dataset_root = SYNRIA_MIMIC_AMPLIFIED_ROOTS if args.amplified else SYNRIA_DEMO_ROOTS
    dataset_path = dataset_root[args.game]

    if args.gr00t_repo is not None:
        # Invoke as a script from a local clone — matches the README pattern:
        #   uv run python gr00t/experiment/launch_finetune.py ...
        launch_script = args.gr00t_repo / _GROOT_LAUNCH_SCRIPT_REL
        cmd = [sys.executable, str(launch_script)]
    else:
        # Invoke as a module — assumes gr00t is pip-installed and on PYTHONPATH.
        cmd = [sys.executable, "-m", "gr00t.experiment.launch_finetune"]

    # The modality config is a Python file that registers the Synria layout
    # under EmbodimentTag.NEW_EMBODIMENT when launch_finetune imports it.
    modality_config_path = Path(__file__).resolve().parent / "modality_config.py"

    cmd.extend(
        [
            "--base-model-path", str(args.base_model),
            "--dataset-path", str(dataset_path),
            "--output-dir", str(args.output_dir),
            "--embodiment-tag", SYNRIA_EMBODIMENT_TAG,
            "--modality-config-path", str(modality_config_path),
            "--max-steps", str(args.max_steps),
            "--global-batch-size", str(args.global_batch_size),
            "--learning-rate", str(args.learning_rate),
        ]
    )

    if args.extra:
        # argparse leaves a leading "--" in REMAINDER args; drop it.
        extra = list(args.extra)
        if extra and extra[0] == "--":
            extra = extra[1:]
        cmd.extend(extra)

    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Validate dataset path early so the user gets a useful error before
    # invoking the heavy GR00T import + training loop.
    from .dataset import SYNRIA_DEMO_ROOTS, SYNRIA_MIMIC_AMPLIFIED_ROOTS

    dataset_root = SYNRIA_MIMIC_AMPLIFIED_ROOTS if args.amplified else SYNRIA_DEMO_ROOTS
    dataset_path = dataset_root[args.game]
    if not args.dry_run and not dataset_path.exists():
        which = "amplified" if args.amplified else "seed"
        sys.stderr.write(
            f"Synria {which} demo dataset not found at {dataset_path}. "
            f"Collect demonstrations first — see docs/SYNRIA_GR00T_FINETUNE.md.\n"
        )
        raise SystemExit(2)

    # Light gating: verify gr00t is importable when not dry-run + no local clone.
    if not args.dry_run and args.gr00t_repo is None and shutil.which("python") is not None:
        try:
            import gr00t  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:  # pragma: no cover - hardware-gated
            sys.stderr.write(
                "GR00T is not importable in this Python environment. "
                "Either install it (`uv pip install gr00t` from a clone, "
                "Python 3.10 strict), or pass --gr00t-repo /path/to/Isaac-GR00T.\n"
            )
            raise SystemExit(1) from exc

    cmd = _build_launch_cmd(args)
    print("[gr00t-finetune] launch command:")
    print(" ", " ".join(cmd))
    print("[gr00t-finetune] cwd:", _REPO_ROOT)

    if args.dry_run:
        print("[gr00t-finetune] dry run — not invoking.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "wrapper": "isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune",
        "game": args.game,
        "amplified": args.amplified,
        "base_model": args.base_model,
        "command": cmd,
    }
    (args.output_dir / "wrapper_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Forward exit code so caller sees the GR00T launch_finetune's result.
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


if __name__ == "__main__":  # pragma: no cover - hardware-gated
    raise SystemExit(main())
