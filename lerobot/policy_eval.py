"""Policy evaluation helpers for the Synria LeRobot track.

Provides a ``DeterministicArmPolicy`` that replays the planned trajectory
deltas as its predicted actions — this is the ``oracle`` baseline policy
that should score perfectly on synthetic episodes.

It is intentionally simple: no neural network, no ACT, no hardware. It
exists so the evaluation pipeline can be unit-tested end-to-end before
a real ACT policy is trained.

When a real policy is available, replace ``DeterministicArmPolicy`` with
a wrapper around the LeRobot policy checkpoint:

    from lerobot.policy_eval import evaluate_policy_on_episode

    class MyACTPolicy:
        def predict(self, obs: ObservationFrame) -> ActionFrame:
            ...  # call ACT model

    results = [evaluate_policy_on_episode(MyACTPolicy(), ep) for ep in dataset]

Usage::

    from lerobot.policy_eval import DeterministicArmPolicy, evaluate_policy_on_dataset
    from lerobot.dataset import generate_synthetic_episode, SynriaEpisodeDataset

    dataset = SynriaEpisodeDataset([generate_synthetic_episode(zone="left")])
    policy = DeterministicArmPolicy()
    results = evaluate_policy_on_dataset(policy, dataset)
    assert all(r.passed for r in results)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lerobot.schema import (
    ActionFrame,
    Episode,
    EpisodeStep,
    ObservationFrame,
)

if TYPE_CHECKING:
    from lerobot.dataset import SynriaEpisodeDataset

# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyEvalResult:
    """Aggregate evaluation metrics for one episode.

    Attributes:
        episode_id: Episode identifier.
        n_steps: Number of steps evaluated.
        mean_joint_tracking_error_rad: Mean absolute joint-position tracking
            error across all steps and joints (radians).
        max_joint_tracking_error_rad: Worst-case joint tracking error (radians).
        completion_rate: Fraction of steps where the policy action was within
            the ``pass_threshold_rad`` of the ground-truth action.
        passed: ``True`` when both mean and max errors are within policy
            acceptance criteria.
        notes: Free-text evaluation context.
    """

    episode_id: str
    n_steps: int
    mean_joint_tracking_error_rad: float
    max_joint_tracking_error_rad: float
    completion_rate: float
    passed: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "n_steps": self.n_steps,
            "mean_joint_tracking_error_rad": self.mean_joint_tracking_error_rad,
            "max_joint_tracking_error_rad": self.max_joint_tracking_error_rad,
            "completion_rate": self.completion_rate,
            "passed": self.passed,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


class DeterministicArmPolicy:
    """Oracle baseline policy that echoes the ground-truth trajectory deltas.

    Looks up the planned action from the current step index. When the index
    exceeds the episode length, returns a zero action (hold in place).

    This policy achieves zero tracking error on synthetic episodes, providing
    a deterministic upper bound for the evaluation pipeline.

    Args:
        pass_threshold_rad: Joint-position tolerance for marking a step as
            "completed" (default 0.05 rad ≈ 2.9°).
    """

    def __init__(self, pass_threshold_rad: float = 0.05) -> None:
        self.pass_threshold_rad = pass_threshold_rad
        self._episode_steps: list[EpisodeStep] = []

    def load_episode(self, episode: Episode) -> None:
        """Load ground-truth steps for replay."""
        self._episode_steps = list(episode.steps)

    def predict(self, obs: ObservationFrame) -> ActionFrame:
        """Return the ground-truth delta for the current frame index.

        Falls back to a zero action when the frame index is out of range.
        """
        idx = obs.frame_index
        if 0 <= idx < len(self._episode_steps):
            return self._episode_steps[idx].action
        # Hold in place: zero deltas, no gripper change.
        return ActionFrame(
            joint_deltas_rad=(0.0,) * 6,
            gripper_command=0.0,
            timestamp_s=obs.timestamp_s,
        )


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def evaluate_policy_on_episode(
    policy: DeterministicArmPolicy,
    episode: Episode,
    pass_threshold_rad: float = 0.05,
) -> PolicyEvalResult:
    """Run a policy against one episode and compute tracking metrics.

    For each step, the policy predicts an action given the ground-truth
    observation. The predicted action is compared to the ground-truth action
    from the episode.

    Tracking error = mean absolute difference between predicted and
    ground-truth joint deltas across all 6 joints.

    Args:
        policy: A ``DeterministicArmPolicy`` (or any object with
            ``load_episode`` + ``predict`` methods).
        episode: Ground-truth demonstration episode.
        pass_threshold_rad: Per-step pass tolerance in radians.

    Returns:
        :class:`PolicyEvalResult` with aggregate metrics.
    """
    policy.load_episode(episode)

    if not episode.steps:
        return PolicyEvalResult(
            episode_id=episode.metadata.episode_id,
            n_steps=0,
            mean_joint_tracking_error_rad=0.0,
            max_joint_tracking_error_rad=0.0,
            completion_rate=1.0,
            passed=True,
            notes="Empty episode — nothing to evaluate.",
        )

    all_errors: list[float] = []
    steps_passed = 0

    for step in episode.steps:
        pred_action = policy.predict(step.observation)
        gt_action = step.action

        # Per-joint absolute delta error.
        joint_errors = [
            abs(pred - gt)
            for pred, gt in zip(
                pred_action.joint_deltas_rad,
                gt_action.joint_deltas_rad,
                strict=False,
            )
        ]
        step_mean_err = sum(joint_errors) / len(joint_errors) if joint_errors else 0.0
        all_errors.extend(joint_errors)

        if step_mean_err <= pass_threshold_rad:
            steps_passed += 1

    total_joints_evaluated = len(all_errors)
    mean_err = sum(all_errors) / total_joints_evaluated if total_joints_evaluated else 0.0
    max_err = max(all_errors) if all_errors else 0.0
    completion = steps_passed / len(episode.steps)

    passed = (
        mean_err <= pass_threshold_rad
        and max_err <= pass_threshold_rad * 5.0  # max can be up to 5× mean threshold
    )

    return PolicyEvalResult(
        episode_id=episode.metadata.episode_id,
        n_steps=len(episode.steps),
        mean_joint_tracking_error_rad=round(mean_err, 6),
        max_joint_tracking_error_rad=round(max_err, 6),
        completion_rate=round(completion, 4),
        passed=passed,
        notes=(
            f"Evaluated with {type(policy).__name__} on "
            f"zone={episode.metadata.staging_zone!r}, "
            f"game={episode.metadata.game!r}."
        ),
    )


def evaluate_policy_on_dataset(
    policy: DeterministicArmPolicy,
    dataset: SynriaEpisodeDataset,
    pass_threshold_rad: float = 0.05,
) -> list[PolicyEvalResult]:
    """Evaluate a policy across all episodes in a dataset.

    Args:
        policy: Policy to evaluate.
        dataset: Dataset of demonstration episodes.
        pass_threshold_rad: Per-step pass tolerance passed to
            :func:`evaluate_policy_on_episode`.

    Returns:
        List of :class:`PolicyEvalResult`, one per episode, in dataset order.
    """
    return [
        evaluate_policy_on_episode(policy, ep, pass_threshold_rad=pass_threshold_rad)
        for ep in dataset
    ]
