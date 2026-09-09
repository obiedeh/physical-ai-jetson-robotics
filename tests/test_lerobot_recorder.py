"""Tests for lerobot.recorder — EpisodeRecorder and RecordingSession."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.dataset import SynriaEpisodeDataset
from lerobot.recorder import (
    EpisodeRecorder,
    RecordingSession,
    benchmark_recording_fps,
)
from lerobot.schema import VALID_STAGING_ZONES, Episode

# ---------------------------------------------------------------------------
# EpisodeRecorder
# ---------------------------------------------------------------------------


def test_recorder_returns_episode() -> None:
    rec = EpisodeRecorder(zone="left", mock=True)
    ep = rec.record()
    assert isinstance(ep, Episode)


def test_recorder_episode_zone() -> None:
    rec = EpisodeRecorder(zone="right", mock=True)
    ep = rec.record()
    assert ep.metadata.staging_zone == "right"


def test_recorder_episode_game() -> None:
    rec = EpisodeRecorder(zone="left", game="ludo", mock=True)
    ep = rec.record()
    assert ep.metadata.game == "ludo"


def test_recorder_episode_fps_stored() -> None:
    rec = EpisodeRecorder(zone="left", fps=15.0, mock=True)
    ep = rec.record()
    assert ep.metadata.fps == pytest.approx(15.0)


def test_recorder_n_steps_cap() -> None:
    rec = EpisodeRecorder(zone="left", mock=True)
    ep = rec.record(n_steps=20)
    assert len(ep) == 20


def test_recorder_full_trajectory() -> None:
    rec = EpisodeRecorder(zone="left", mock=True)
    ep_full = rec.record(n_steps=0)
    ep_cap = rec.record(n_steps=5)
    assert len(ep_full) > len(ep_cap)


def test_recorder_synthetic_flag() -> None:
    rec = EpisodeRecorder(zone="left", mock=True)
    ep = rec.record()
    assert ep.metadata.synthetic is True


def test_recorder_invalid_zone_raises() -> None:
    with pytest.raises(ValueError, match="Unknown zone"):
        EpisodeRecorder(zone="diagonal")


def test_recorder_invalid_game_raises() -> None:
    with pytest.raises(ValueError, match="Unknown game"):
        EpisodeRecorder(zone="left", game="monopoly")


def test_recorder_invalid_fps_raises() -> None:
    with pytest.raises(ValueError, match="fps"):
        EpisodeRecorder(zone="left", fps=0.0)


def test_recorder_camera_loop_stats() -> None:
    rec = EpisodeRecorder(zone="left", mock=True)
    stats = rec.camera_loop_stats(n_frames=10)
    assert "n_frames" in stats
    assert stats["n_frames"] == 10
    assert stats["mean_fps"] > 0.0


# ---------------------------------------------------------------------------
# RecordingSession
# ---------------------------------------------------------------------------


def test_session_returns_dataset() -> None:
    session = RecordingSession(n_episodes=2, zones=("left", "right"), mock=True)
    ds = session.record_all(n_steps=10)
    assert isinstance(ds, SynriaEpisodeDataset)


def test_session_n_episodes() -> None:
    session = RecordingSession(n_episodes=4, mock=True)
    ds = session.record_all(n_steps=5)
    assert len(ds) == 4


def test_session_zones_cycled() -> None:
    """Episodes should cycle through zones."""
    session = RecordingSession(
        n_episodes=4,
        zones=("left", "right"),
        games=("chess",),
        mock=True,
    )
    ds = session.record_all(n_steps=5)
    zones = [ep.metadata.staging_zone for ep in ds]
    # With 4 episodes and 2 zones, should cycle: left, right, left, right.
    assert zones[0] == "left"
    assert zones[1] == "right"
    assert zones[2] == "left"
    assert zones[3] == "right"


def test_session_all_zones_covered() -> None:
    session = RecordingSession(
        n_episodes=4,
        zones=tuple(VALID_STAGING_ZONES),
        mock=True,
    )
    ds = session.record_all(n_steps=5)
    covered = {ep.metadata.staging_zone for ep in ds}
    assert covered == set(VALID_STAGING_ZONES)


def test_session_n_recorded_property() -> None:
    session = RecordingSession(n_episodes=3, mock=True)
    assert session.n_recorded == 0
    session.record_all(n_steps=5)
    assert session.n_recorded == 3


def test_session_invalid_n_episodes_raises() -> None:
    with pytest.raises(ValueError, match="n_episodes"):
        RecordingSession(n_episodes=0)


def test_session_invalid_zone_raises() -> None:
    with pytest.raises(ValueError, match="Unknown zone"):
        RecordingSession(n_episodes=1, zones=("diagonal",))


# ---------------------------------------------------------------------------
# benchmark_recording_fps
# ---------------------------------------------------------------------------


def test_benchmark_fps_returns_dict() -> None:
    stats = benchmark_recording_fps(n_frames=10)
    assert "mock_fps" in stats
    assert "n_frames" in stats
    assert stats["n_frames"] == 10


def test_benchmark_fps_target_stored() -> None:
    stats = benchmark_recording_fps(n_frames=5, fps=30.0)
    assert stats["target_fps"] == pytest.approx(30.0)


def test_benchmark_fps_positive() -> None:
    stats = benchmark_recording_fps(n_frames=20)
    assert stats["mock_fps"] > 0.0
