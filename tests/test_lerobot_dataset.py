"""Tests for lerobot.dataset — synthetic episode generation and dataset wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.dataset import (
    SynriaEpisodeDataset,
    generate_synthetic_episode,
    write_dataset_summary,
)
from lerobot.schema import (
    N_JOINTS,
    VALID_GAMES,
    VALID_STAGING_ZONES,
    VALID_TASK_VARIANTS,
    Episode,
)

# Joint position limits from arm_control.safety (used for constraint checks).
_JOINT_LIMITS = {
    "joint_1": (-3.1416, 3.1416),
    "joint_2": (-1.5708, 1.5708),
    "joint_3": (-2.3562, 2.3562),
    "joint_4": (-3.1416, 3.1416),
    "joint_5": (-1.5708, 1.5708),
    "joint_6": (-3.1416, 3.1416),
}
_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")


# ---------------------------------------------------------------------------
# generate_synthetic_episode — basic contracts
# ---------------------------------------------------------------------------


def test_episode_is_episode_instance() -> None:
    ep = generate_synthetic_episode(zone="left")
    assert isinstance(ep, Episode)


def test_episode_has_steps() -> None:
    ep = generate_synthetic_episode(zone="left")
    assert len(ep) > 0


def test_episode_metadata_zone() -> None:
    ep = generate_synthetic_episode(zone="right")
    assert ep.metadata.staging_zone == "right"


def test_episode_metadata_game() -> None:
    ep = generate_synthetic_episode(game="ludo")
    assert ep.metadata.game == "ludo"


def test_episode_metadata_synthetic_flag() -> None:
    ep = generate_synthetic_episode()
    assert ep.metadata.synthetic is True


def test_episode_n_steps_cap() -> None:
    ep = generate_synthetic_episode(zone="left", n_steps=10)
    assert len(ep) == 10


def test_episode_n_steps_cap_zero_gives_full() -> None:
    ep_full = generate_synthetic_episode(zone="left", n_steps=0)
    ep_cap = generate_synthetic_episode(zone="left", n_steps=5)
    assert len(ep_full) > len(ep_cap)


def test_episode_step_indices_are_sequential() -> None:
    ep = generate_synthetic_episode(zone="left", n_steps=20)
    for i, step in enumerate(ep):
        assert step.step_index == i


def test_episode_step_observation_frame_index_sequential() -> None:
    ep = generate_synthetic_episode(zone="left", n_steps=15)
    for i, step in enumerate(ep):
        assert step.observation.frame_index == i


def test_episode_timestamps_non_decreasing() -> None:
    ep = generate_synthetic_episode(zone="left", n_steps=30)
    ts = [s.observation.timestamp_s for s in ep]
    for a, b in zip(ts, ts[1:], strict=False):
        assert b >= a


def test_episode_action_vector_length() -> None:
    ep = generate_synthetic_episode(zone="left", n_steps=5)
    for step in ep:
        vec = step.action.as_vector()
        assert len(vec) == N_JOINTS + 1  # 6 deltas + gripper


def test_episode_observation_joint_count() -> None:
    ep = generate_synthetic_episode(zone="left", n_steps=5)
    for step in ep:
        assert len(step.observation.joint_state.positions_rad) == N_JOINTS


# ---------------------------------------------------------------------------
# All zones and games produce valid episodes
# ---------------------------------------------------------------------------


def test_all_zones_produce_episodes() -> None:
    for zone in VALID_STAGING_ZONES:
        ep = generate_synthetic_episode(zone=zone, n_steps=10)
        assert len(ep) == 10, f"Zone {zone!r} produced fewer steps than expected"
        assert ep.metadata.staging_zone == zone


def test_all_games_produce_episodes() -> None:
    for game in VALID_GAMES:
        ep = generate_synthetic_episode(game=game, n_steps=10)
        assert ep.metadata.game == game


def test_all_task_variants_produce_episodes() -> None:
    for variant in VALID_TASK_VARIANTS:
        ep = generate_synthetic_episode(task_variant=variant, n_steps=10)
        assert ep.metadata.task_variant == variant


# ---------------------------------------------------------------------------
# Safety / physics constraints
# ---------------------------------------------------------------------------


def test_joint_positions_within_urdf_limits() -> None:
    """All generated joint positions must respect URDF joint limits."""
    for zone in VALID_STAGING_ZONES:
        ep = generate_synthetic_episode(zone=zone, n_steps=0)
        for step in ep:
            pos = step.observation.joint_state.positions_rad
            for name, val in zip(_JOINT_NAMES, pos, strict=False):
                lo, hi = _JOINT_LIMITS[name]
                assert lo <= val <= hi, (
                    f"Zone {zone!r}: {name}={val:.4f} out of limits [{lo}, {hi}]"
                )


def test_gripper_values_in_range() -> None:
    ep = generate_synthetic_episode(zone="top", n_steps=0)
    for step in ep:
        gm = step.observation.joint_state.gripper_open_m
        assert 0.0 <= gm <= 0.085, f"gripper_open_m={gm} out of range"


def test_ee_pose_height_above_table() -> None:
    """EE z should be above zero (arm doesn't go below the base)."""
    ep = generate_synthetic_episode(zone="left", n_steps=0)
    for step in ep:
        assert step.observation.ee_pose.z_m >= 0.0, (
            f"EE z={step.observation.ee_pose.z_m:.4f} — below table surface"
        )


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_invalid_zone_raises() -> None:
    with pytest.raises(ValueError, match="Unknown zone"):
        generate_synthetic_episode(zone="diagonal")


def test_invalid_game_raises() -> None:
    with pytest.raises(ValueError, match="Unknown game"):
        generate_synthetic_episode(game="monopoly")


def test_invalid_task_variant_raises() -> None:
    with pytest.raises(ValueError, match="Unknown task_variant"):
        generate_synthetic_episode(task_variant="V9")


# ---------------------------------------------------------------------------
# SynriaEpisodeDataset
# ---------------------------------------------------------------------------


def test_dataset_len() -> None:
    eps = [generate_synthetic_episode(zone=z, n_steps=5) for z in ("left", "right")]
    ds = SynriaEpisodeDataset(eps)
    assert len(ds) == 2


def test_dataset_getitem() -> None:
    eps = [generate_synthetic_episode(zone=z, n_steps=5) for z in ("left", "right")]
    ds = SynriaEpisodeDataset(eps)
    assert ds[0].metadata.staging_zone == "left"
    assert ds[1].metadata.staging_zone == "right"


def test_dataset_total_steps() -> None:
    eps = [generate_synthetic_episode(zone=z, n_steps=10) for z in ("left", "right")]
    ds = SynriaEpisodeDataset(eps)
    assert ds.total_steps == 20


def test_dataset_to_dict() -> None:
    eps = [generate_synthetic_episode(zone="left", n_steps=5)]
    ds = SynriaEpisodeDataset(eps)
    d = ds.to_dict()
    assert d["n_episodes"] == 1
    assert d["total_steps"] == 5
    assert len(d["episodes"]) == 1


def test_dataset_iteration() -> None:
    zones = list(VALID_STAGING_ZONES)
    eps = [generate_synthetic_episode(zone=z, n_steps=5) for z in zones]
    ds = SynriaEpisodeDataset(eps)
    for ep, zone in zip(ds, zones, strict=False):
        assert ep.metadata.staging_zone == zone


def test_dataset_write_json(tmp_path: Path) -> None:
    ds = SynriaEpisodeDataset([generate_synthetic_episode(zone="left", n_steps=5)])
    out = ds.write_json(tmp_path / "test_dataset.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert "n_episodes" in data
    assert "episodes" in data


# ---------------------------------------------------------------------------
# write_dataset_summary
# ---------------------------------------------------------------------------


def test_write_dataset_summary(tmp_path: Path) -> None:
    eps = [generate_synthetic_episode(zone=z, n_steps=10) for z in VALID_STAGING_ZONES]
    ds = SynriaEpisodeDataset(eps)
    out = write_dataset_summary(ds, tmp_path / "summary.json")
    assert out.exists()
    summary = json.loads(out.read_text())
    assert summary["n_episodes"] == 4
    assert set(summary["staging_zones"]) == set(VALID_STAGING_ZONES)
    assert summary["total_steps"] == 40
