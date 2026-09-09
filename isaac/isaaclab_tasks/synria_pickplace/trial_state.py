"""Per-environment trial state manager for the Synria pick-and-place task.

Tracks the four-phase pick-and-place cycle for every parallel environment
instance.  The manager lives on the CPU side and is stored in
``env.extras["trial_state"]`` so that MDP term functions can query the
current phase and target coordinates for each env.

No Isaac Lab or torch dependency — fully testable without a GPU.

Phase progression (V1 task):

    PICK_FROM_BOARD  →  PLACE_ON_ZONE  →  PICK_FROM_ZONE  →  RETURN_TO_BOARD  →  SUCCESS
         ↓                   ↓                   ↓                   ↓
       FAILED              FAILED              FAILED              FAILED

Usage::

    mgr = TrialStateManager(num_envs=4, game="chess")
    target_xy = mgr.current_target_xy(env_id=0)
    mgr.advance_phase(env_id=0)   # call after reward confirms phase complete
    mgr.reset(env_ids=[0, 1])     # call on episode reset
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import IntEnum

from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    PIECE_COUNTS,
    ZONE_NAMES,
    piece_start_xy,
    zone_center_xy,
)

# ---------------------------------------------------------------------------
# Task phase enum
# ---------------------------------------------------------------------------


class TaskPhase(IntEnum):
    """The four sequential phases of the V1 pick-and-place task.

    The enum values are monotonically increasing so that ``phase >= X``
    can express "at or past phase X".
    """

    PICK_FROM_BOARD = 0
    """Step 1: Move EE to the target piece and pick it off the board."""

    PLACE_ON_ZONE = 1
    """Step 2: Carry the piece to the target staging zone and release it."""

    PICK_FROM_ZONE = 2
    """Step 3: Move EE back to the staging zone and pick the piece up again."""

    RETURN_TO_BOARD = 3
    """Step 4: Carry the piece back and release it at the original square."""

    SUCCESS = 4
    """Episode completed successfully."""

    FAILED = 5
    """Episode terminated with a failure (drop, collision, or timeout)."""


# ---------------------------------------------------------------------------
# Per-env trial record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialInfo:
    """Immutable snapshot of one env's trial state.

    Attributes:
        phase: Current task phase.
        target_piece_idx: Which piece is the target (0-based game index).
        target_zone_idx: Which staging zone is the target (0=left, 1=right,
            2=top, 3=bottom — matches the one-hot encoding in the obs space).
        origin_xy: World (x, y) of the piece's starting board square.
        zone_xy: World (x, y) of the target staging zone centre.
        consec_grasp_steps: Consecutive steps where grasp criteria were met.
        phase_step_count: Steps elapsed in the current phase.
        episode_step: Total steps elapsed in the episode.
    """

    phase: TaskPhase
    target_piece_idx: int
    target_zone_idx: int
    origin_xy: tuple[float, float]
    zone_xy: tuple[float, float]
    consec_grasp_steps: int = 0
    phase_step_count: int = 0
    episode_step: int = 0


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class TrialStateManager:
    """CPU-side manager for per-env trial state.

    Instantiate once in the env's ``__post_init__`` and store in
    ``env.extras["trial_state"]``.  Isaac Lab's vectorised env calls
    :meth:`reset` on episode boundaries and the MDP term functions call
    :meth:`current_target_xy`, :meth:`get_phase`, etc. every step.

    Args:
        num_envs: Number of parallel environments.
        game: Board game — ``"chess"``, ``"checkers"``, or ``"ludo"``.
        seed: Random seed for target piece + zone sampling.
    """

    def __init__(self, num_envs: int, game: str, seed: int = 0) -> None:
        if game not in PIECE_COUNTS:
            raise ValueError(f"Unknown game={game!r}. Valid: {list(PIECE_COUNTS)}")
        if num_envs < 1:
            raise ValueError(f"num_envs={num_envs} must be ≥ 1.")
        self._num_envs = num_envs
        self._game = game
        self._rng = random.Random(seed)
        self._states: list[TrialInfo] = [self._new_trial() for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Resample target piece and zone for the given env IDs.

        Args:
            env_ids: Env indices to reset.  Defaults to all envs.
        """
        ids = env_ids if env_ids is not None else list(range(self._num_envs))
        for eid in ids:
            self._states[eid] = self._new_trial()

    def _new_trial(self) -> TrialInfo:
        piece_count = PIECE_COUNTS[self._game]
        piece_idx = self._rng.randrange(piece_count)
        zone_idx = self._rng.randrange(len(ZONE_NAMES))
        zone_name = ZONE_NAMES[zone_idx]
        origin = piece_start_xy(self._game, piece_idx)
        zone = zone_center_xy(zone_name, self._game)
        return TrialInfo(
            phase=TaskPhase.PICK_FROM_BOARD,
            target_piece_idx=piece_idx,
            target_zone_idx=zone_idx,
            origin_xy=origin,
            zone_xy=zone,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, env_id: int) -> None:
        """Increment step counters for one env.  Call once per sim step."""
        s = self._states[env_id]
        self._states[env_id] = replace(
            s,
            phase_step_count=s.phase_step_count + 1,
            episode_step=s.episode_step + 1,
        )

    def increment_grasp(self, env_id: int) -> None:
        """Increment the consecutive-grasp counter for one env."""
        s = self._states[env_id]
        self._states[env_id] = replace(s, consec_grasp_steps=s.consec_grasp_steps + 1)

    def reset_grasp(self, env_id: int) -> None:
        """Reset the consecutive-grasp counter (gripper opened or lost piece)."""
        s = self._states[env_id]
        self._states[env_id] = replace(s, consec_grasp_steps=0)

    def advance_phase(self, env_id: int) -> None:
        """Move one env to the next phase and reset its phase step counter.

        Clamped at SUCCESS so callers do not need range checks.
        """
        s = self._states[env_id]
        next_phase_val = min(int(s.phase) + 1, int(TaskPhase.SUCCESS))
        self._states[env_id] = replace(
            s,
            phase=TaskPhase(next_phase_val),
            phase_step_count=0,
            consec_grasp_steps=0,
        )

    def mark_failed(self, env_id: int) -> None:
        """Mark one env as failed (drop / collision)."""
        s = self._states[env_id]
        self._states[env_id] = replace(s, phase=TaskPhase.FAILED)

    # ------------------------------------------------------------------
    # Queries — called every step by MDP term functions
    # ------------------------------------------------------------------

    def get_state(self, env_id: int) -> TrialInfo:
        """Return the current :class:`TrialInfo` for one env."""
        return self._states[env_id]

    def get_phase(self, env_id: int) -> TaskPhase:
        return self._states[env_id].phase

    def get_phases(self) -> list[TaskPhase]:
        """Return the current phase for every env."""
        return [s.phase for s in self._states]

    def current_target_xy(self, env_id: int) -> tuple[float, float]:
        """Phase-dependent EE target (x, y) in world frame.

        * Phase 0 (PICK_FROM_BOARD) and 3 (RETURN_TO_BOARD): piece origin on the board.
        * Phase 1 (PLACE_ON_ZONE) and 2 (PICK_FROM_ZONE): staging zone centre.
        """
        s = self._states[env_id]
        if s.phase in (TaskPhase.PICK_FROM_BOARD, TaskPhase.RETURN_TO_BOARD):
            return s.origin_xy
        return s.zone_xy

    def all_target_xys(self) -> list[tuple[float, float]]:
        """Return the phase-dependent target XY for every env (vectorised query)."""
        return [self.current_target_xy(i) for i in range(self._num_envs)]

    def is_success(self, env_id: int) -> bool:
        return self._states[env_id].phase == TaskPhase.SUCCESS

    def is_failed(self, env_id: int) -> bool:
        return self._states[env_id].phase == TaskPhase.FAILED

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def game(self) -> str:
        return self._game
