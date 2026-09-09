"""Tests for lerobot.policy_eval — DeterministicArmPolicy and evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.dataset import SynriaEpisodeDataset, generate_synthetic_episode
from lerobot.policy_eval import (
    DeterministicArmPolicy,
    PolicyEvalResult,
    evaluate_policy_on_dataset,
    evaluate_policy_on_episode,
)
from lerobot.schema import (
    N_JOINTS,
    ActionFrame,
    Episode,
    EpisodeMetadata,
    ObservationFrame,
    now_iso,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ep(zone: str = "left", n_steps: int = 20) -> Episode:
    return generate_synthetic_episode(zone=zone, n_steps=n_steps)


def _empty_ep() -> Episode:
    meta = EpisodeMetadata(
        episode_id="empty",
        task_variant="V1",
        game="chess",
        staging_zone="left",
        robot_id="test",
        fps=30.0,
        n_steps=0,
        captured_at=now_iso(),
        synthetic=True,
    )
    return Episode(metadata=meta, steps=[])


# ---------------------------------------------------------------------------
# DeterministicArmPolicy
# ---------------------------------------------------------------------------


def test_policy_predict_returns_action_frame() -> None:
    ep = _make_ep(n_steps=5)
    policy = DeterministicArmPolicy()
    policy.load_episode(ep)
    action = policy.predict(ep.steps[0].observation)
    assert isinstance(action, ActionFrame)


def test_policy_predict_exact_match() -> None:
    """Deterministic policy must echo the ground-truth action exactly."""
    ep = _make_ep(n_steps=5)
    policy = DeterministicArmPolicy()
    policy.load_episode(ep)
    for step in ep.steps:
        pred = policy.predict(step.observation)
        assert pred.joint_deltas_rad == step.action.joint_deltas_rad
        assert pred.gripper_command == step.action.gripper_command


def test_policy_predict_out_of_bounds_returns_zero() -> None:
    ep = _make_ep(n_steps=3)
    policy = DeterministicArmPolicy()
    policy.load_episode(ep)
    # Create an observation with frame_index beyond episode length.
    obs = ep.steps[-1].observation
    # Manually build an obs with large frame_index.
    far_obs = ObservationFrame(
        joint_state=obs.joint_state,
        ee_pose=obs.ee_pose,
        frame_index=9999,
        timestamp_s=obs.timestamp_s,
    )
    action = policy.predict(far_obs)
    assert all(d == 0.0 for d in action.joint_deltas_rad)
    assert action.gripper_command == 0.0


def test_policy_action_vector_length() -> None:
    ep = _make_ep(n_steps=5)
    policy = DeterministicArmPolicy()
    policy.load_episode(ep)
    for step in ep.steps:
        vec = policy.predict(step.observation).as_vector()
        assert len(vec) == N_JOINTS + 1


# ---------------------------------------------------------------------------
# evaluate_policy_on_episode
# ---------------------------------------------------------------------------


def test_deterministic_policy_zero_error() -> None:
    """Oracle policy should produce zero tracking error on synthetic data."""
    ep = _make_ep(n_steps=30)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert result.mean_joint_tracking_error_rad == pytest.approx(0.0, abs=1e-9)
    assert result.max_joint_tracking_error_rad == pytest.approx(0.0, abs=1e-9)


def test_deterministic_policy_passes() -> None:
    ep = _make_ep(n_steps=30)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert result.passed is True


def test_deterministic_policy_completion_rate_one() -> None:
    ep = _make_ep(n_steps=30)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert result.completion_rate == pytest.approx(1.0)


def test_eval_result_n_steps() -> None:
    ep = _make_ep(n_steps=20)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert result.n_steps == 20


def test_eval_result_episode_id() -> None:
    ep = _make_ep(n_steps=10)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert result.episode_id == ep.metadata.episode_id


def test_eval_result_is_policy_eval_result() -> None:
    ep = _make_ep(n_steps=10)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert isinstance(result, PolicyEvalResult)


def test_eval_empty_episode() -> None:
    ep = _empty_ep()
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    assert result.n_steps == 0
    assert result.passed is True
    assert "Empty episode" in result.notes


def test_eval_result_to_dict() -> None:
    ep = _make_ep(n_steps=10)
    policy = DeterministicArmPolicy()
    result = evaluate_policy_on_episode(policy, ep)
    d = result.to_dict()
    assert "mean_joint_tracking_error_rad" in d
    assert "passed" in d
    assert "completion_rate" in d


# ---------------------------------------------------------------------------
# evaluate_policy_on_dataset
# ---------------------------------------------------------------------------


def test_eval_dataset_length() -> None:
    eps = [generate_synthetic_episode(zone=z, n_steps=10) for z in ("left", "right", "top")]
    ds = SynriaEpisodeDataset(eps)
    policy = DeterministicArmPolicy()
    results = evaluate_policy_on_dataset(policy, ds)
    assert len(results) == 3


def test_eval_dataset_all_pass() -> None:
    from lerobot.schema import VALID_STAGING_ZONES

    eps = [generate_synthetic_episode(zone=z, n_steps=20) for z in VALID_STAGING_ZONES]
    ds = SynriaEpisodeDataset(eps)
    policy = DeterministicArmPolicy()
    results = evaluate_policy_on_dataset(policy, ds)
    for r in results:
        assert r.passed, (
            f"Zone {r.episode_id!r} did not pass: "
            f"mean_err={r.mean_joint_tracking_error_rad:.6f}"
        )


def test_eval_dataset_returns_list_of_results() -> None:
    ds = SynriaEpisodeDataset([generate_synthetic_episode(zone="left", n_steps=5)])
    policy = DeterministicArmPolicy()
    results = evaluate_policy_on_dataset(policy, ds)
    assert isinstance(results, list)
    assert isinstance(results[0], PolicyEvalResult)
