"""Synria board ↔ staging pick-and-place tasks for Isaac Lab.

Registers three gym task IDs — one per game:

    Synria-Ludo-PickPlace-v0
    Synria-Chess-PickPlace-v0
    Synria-Checkers-PickPlace-v0

Each task uses the same env structure (Synria arm + tabletop + four staging
zones); only the scene USD and the per-game initial-state seeding differ.

This package must run inside Isaac Lab's Python on the Linux RTX 5090 — the
import of ``isaaclab`` and ``gymnasium`` happens at registration time. On
Windows the package imports as a no-op so static parsing, ruff, and mypy
still pass.
"""

from __future__ import annotations

# env_cfg is itself import-safe (wraps its isaaclab imports in try/except),
# so we always pull in the config-class names. The gymnasium.register calls
# below are gated behind a try/except so the package import succeeds on
# environments without gymnasium or isaaclab installed.
from .env_cfg import (
    SynriaCheckersPickPlaceEnvCfg,
    SynriaChessPickPlaceEnvCfg,
    SynriaLudoPickPlaceEnvCfg,
)

try:
    import gymnasium as gym  # noqa: F401

    gym.register(
        id="Synria-Ludo-PickPlace-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": SynriaLudoPickPlaceEnvCfg},
        disable_env_checker=True,
    )
    gym.register(
        id="Synria-Chess-PickPlace-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": SynriaChessPickPlaceEnvCfg},
        disable_env_checker=True,
    )
    gym.register(
        id="Synria-Checkers-PickPlace-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": SynriaCheckersPickPlaceEnvCfg},
        disable_env_checker=True,
    )

except ImportError:  # pragma: no cover - hardware-gated registration
    # Isaac Lab / gymnasium not installed in this environment (e.g. on Windows
    # dev machine). The package is still import-safe; tests that exercise the
    # registration path will skip.
    pass

__all__ = [
    "SynriaLudoPickPlaceEnvCfg",
    "SynriaChessPickPlaceEnvCfg",
    "SynriaCheckersPickPlaceEnvCfg",
]
