"""Tests for lerobot.cube_sort — CubeSortSimulation, MockColorDetector, CubeSortPlanner."""

from __future__ import annotations

import pytest

from lerobot.cube_sort import (
    COLOR_TO_ZONE,
    CUBE_COLORS,
    CubeDetection,
    CubeSortSimulation,
    MockColorDetector,
)

# ---------------------------------------------------------------------------
# MockColorDetector
# ---------------------------------------------------------------------------


def test_detector_returns_n_cubes() -> None:
    det = MockColorDetector(seed=0, n_cubes=4)
    assert len(det.detect()) == 4


def test_detector_colors_are_valid() -> None:
    det = MockColorDetector(seed=1, n_cubes=8)
    for d in det.detect():
        assert d.color in CUBE_COLORS


def test_detector_confidence_in_range() -> None:
    for d in MockColorDetector(seed=7, n_cubes=6).detect():
        assert 0.0 <= d.confidence <= 1.0


def test_detector_cube_idx_sequential() -> None:
    detections = MockColorDetector(seed=0, n_cubes=5).detect()
    assert [d.cube_idx for d in detections] == list(range(5))


def test_detector_is_deterministic() -> None:
    a = MockColorDetector(seed=42, n_cubes=6).detect()
    b = MockColorDetector(seed=42, n_cubes=6).detect()
    assert [(d.color, d.position_x_m, d.position_y_m) for d in a] == [
        (d.color, d.position_x_m, d.position_y_m) for d in b
    ]


def test_detector_different_seeds_differ() -> None:
    # Positions will differ even if colors cycle the same.
    a_pos = [d.position_x_m for d in MockColorDetector(seed=1, n_cubes=8).detect()]
    b_pos = [d.position_x_m for d in MockColorDetector(seed=99, n_cubes=8).detect()]
    assert a_pos != b_pos


# ---------------------------------------------------------------------------
# CubeDetection
# ---------------------------------------------------------------------------


def test_cube_detection_target_zone_red() -> None:
    d = CubeDetection(0, "red", 0.3, 0.0, 0.02, 0.9)
    assert d.target_zone() == "right"


def test_cube_detection_target_zone_green() -> None:
    d = CubeDetection(1, "green", 0.3, 0.0, 0.02, 0.9)
    assert d.target_zone() == "left"


def test_cube_detection_target_zone_blue() -> None:
    d = CubeDetection(2, "blue", 0.3, 0.0, 0.02, 0.9)
    assert d.target_zone() == "top"


def test_cube_detection_target_zone_yellow() -> None:
    d = CubeDetection(3, "yellow", 0.3, 0.0, 0.02, 0.9)
    assert d.target_zone() == "bottom"


def test_cube_detection_to_dict_keys() -> None:
    d = CubeDetection(0, "red", 0.3, 0.0, 0.02, 0.95)
    dd = d.to_dict()
    for key in ("cube_idx", "color", "position_m", "confidence", "target_zone"):
        assert key in dd


# ---------------------------------------------------------------------------
# CubeSortSimulation
# ---------------------------------------------------------------------------


def test_simulation_default_n_cubes() -> None:
    sim = CubeSortSimulation()
    results = sim.run()
    assert len(results) == 6


def test_simulation_custom_n_cubes() -> None:
    sim = CubeSortSimulation(n_cubes=3)
    assert len(sim.run()) == 3


def test_simulation_n_cubes_invalid_raises() -> None:
    with pytest.raises(ValueError, match="n_cubes"):
        CubeSortSimulation(n_cubes=0)


def test_simulation_all_results_pass() -> None:
    sim = CubeSortSimulation(n_cubes=6, seed=42)
    results = sim.run()
    for r in results:
        assert r.passed, f"Cube {r.cube_idx} ({r.color}) failed: {r.safety_errors}"


def test_simulation_is_deterministic() -> None:
    r1 = CubeSortSimulation(n_cubes=4, seed=7).run()
    r2 = CubeSortSimulation(n_cubes=4, seed=7).run()
    assert [(r.color, r.target_zone) for r in r1] == [(r.color, r.target_zone) for r in r2]


def test_simulation_result_colors_valid() -> None:
    for r in CubeSortSimulation(n_cubes=8, seed=0).run():
        assert r.color in CUBE_COLORS


def test_simulation_result_zones_valid() -> None:
    valid_zones = set(COLOR_TO_ZONE.values())
    for r in CubeSortSimulation(n_cubes=8, seed=0).run():
        assert r.target_zone in valid_zones


def test_simulation_trajectory_duration_positive() -> None:
    for r in CubeSortSimulation(n_cubes=4, seed=0).run():
        assert r.trajectory_duration_s > 0


def test_simulation_summary_pass_rate_one() -> None:
    sim = CubeSortSimulation(n_cubes=4, seed=42)
    results = sim.run()
    summary = sim.summary(results)
    assert summary["pass_rate"] == pytest.approx(1.0)


def test_simulation_summary_color_counts_sum_to_n_cubes() -> None:
    n = 8
    sim = CubeSortSimulation(n_cubes=n, seed=0)
    results = sim.run()
    summary = sim.summary(results)
    assert sum(summary["color_counts"].values()) == n


def test_simulation_summary_json_serialisable() -> None:
    import json as _json

    sim = CubeSortSimulation(n_cubes=4, seed=0)
    summary = sim.summary(sim.run())
    s = _json.dumps(summary)
    assert len(s) > 0
