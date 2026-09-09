"""Inference wrapper: load a fine-tuned GR00T checkpoint and roll out
inside the Isaac Lab Synria pick-and-place env.

Two patterns supported:

    Pattern A — direct in-process: load ``Gr00tPolicy`` from
        ``gr00t.policy.gr00t_policy`` and call it inside an Isaac Lab
        rollout loop. Lowest latency, requires GR00T + Isaac Lab in the
        same Python venv (and Python 3.10).

    Pattern B — policy server: wrap the GR00T policy in
        ``gr00t.policy.server_client.PolicyClient`` running in a separate
        process. Pattern NVIDIA uses for hardware deployment. Useful if
        the Isaac Lab venv and the GR00T venv don't co-exist.

This module is a SCAFFOLD — runs on the Linux RTX 5090 only. Windows
imports succeed, ``main()`` raises ``SystemExit`` with a clear hint.

Used for:
    - V1 success-rate evaluation after fine-tuning
    - Generating amplified trajectories to feed back into GR00T-Mimic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m isaac.isaaclab_tasks.synria_pickplace.gr00t.inference",
        description="Roll out a fine-tuned GR00T policy inside the Isaac Lab Synria env.",
    )
    parser.add_argument(
        "--task",
        choices=(
            "Synria-Ludo-PickPlace-v0",
            "Synria-Chess-PickPlace-v0",
            "Synria-Checkers-PickPlace-v0",
        ),
        required=True,
        help="Which registered Isaac Lab task ID to roll out against.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the fine-tuned GR00T checkpoint (output_dir from finetune.py).",
    )
    parser.add_argument(
        "--base-model",
        default="nvidia/GR00T-N1.7-3B",
        help="Base GR00T model the fine-tune was started from.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes to roll out.",
    )
    parser.add_argument(
        "--instruction",
        default=(
            "pick the target piece and move it to the target staging zone, "
            "then return it"
        ),
        help="Language instruction passed to the GR00T policy each episode.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_REPO_ROOT / "reports" / "training" / "gr00t_eval.json",
        help="Where to write per-episode success metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        import gymnasium as gym  # type: ignore[import-not-found]

        # Correct import path per the official Isaac-GR00T repo:
        # gr00t/policy/gr00t_policy.py exports Gr00tPolicy.
        from gr00t.policy.gr00t_policy import Gr00tPolicy  # type: ignore[import-not-found]

        # Importing the parent package triggers gym.register() for the
        # three Synria task IDs.
        import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    except ImportError as exc:  # pragma: no cover - hardware-gated
        sys.stderr.write(
            "GR00T or Isaac Lab is not available in this Python environment. "
            "Inference runs on the Linux RTX 5090 only (Python 3.10 strict).\n"
        )
        raise SystemExit(1) from exc

    from .embodiment import SYNRIA_EMBODIMENT_TAG, synria_modality_config

    policy = Gr00tPolicy.from_pretrained(
        # TODO(RTX): verify Gr00tPolicy.from_pretrained's actual signature
        # against your installed gr00t version — args may be
        # (model_path, embodiment_tag, modality_config, device) or differ.
        model_path=args.base_model,
        adapter_path=str(args.checkpoint),
        embodiment_tag=SYNRIA_EMBODIMENT_TAG,
        modality_config=synria_modality_config(),
    )

    env = gym.make(args.task)

    # TODO(RTX): iterate over args.num_episodes, step the env with
    # policy(obs, instruction=args.instruction), and accumulate the V1
    # success metrics (piece returned to origin within tolerance) into
    # args.report.
    #
    # The policy call convention depends on your installed gr00t version.
    # The official deployment script is the authoritative reference:
    #   scripts/deployment/standalone_inference_script.py
    _ = policy  # claimed-by-design; reused once the rollout loop is filled in
    _ = env

    return 0


if __name__ == "__main__":  # pragma: no cover - hardware-gated
    raise SystemExit(main())
