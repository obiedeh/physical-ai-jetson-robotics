"""Tests for the slam map_store and nav_plan modules."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slam.map_store import MapMetadata, MapRecord, MapStore
from slam.nav_plan import (
    Waypoint,
    WaypointType,
    build_nav_plan,
    validate_nav_plan,
)

# ---------------------------------------------------------------------------
# MapMetadata tests
# ---------------------------------------------------------------------------


def _sample_metadata(name: str = "test_map") -> MapMetadata:
    return MapMetadata(
        map_name=name,
        captured_at="2026-05-27T10:00:00+00:00",
        robot_id="yahboom-orin-01",
        resolution_m_per_px=0.05,
        origin_x_m=-5.0,
        origin_y_m=-5.0,
        width_px=200,
        height_px=200,
        notes="Unit test map.",
    )


def test_map_metadata_area_m2() -> None:
    meta = _sample_metadata()
    # 200 × 0.05 = 10 m each side → 100 m²
    assert meta.area_m2() == pytest.approx(100.0, abs=0.01)


def test_map_metadata_width_height_m() -> None:
    meta = _sample_metadata()
    assert meta.width_m() == pytest.approx(10.0, abs=0.001)
    assert meta.height_m() == pytest.approx(10.0, abs=0.001)


def test_map_metadata_to_dict_has_required_fields() -> None:
    meta = _sample_metadata()
    d = meta.to_dict()
    for key in ("map_name", "captured_at", "robot_id", "resolution_m_per_px",
                "width_px", "height_px"):
        assert key in d


# ---------------------------------------------------------------------------
# MapRecord tests
# ---------------------------------------------------------------------------


def test_map_record_is_complete_false_when_no_paths() -> None:
    record = MapRecord(metadata=_sample_metadata())
    assert record.is_complete() is False


def test_map_record_is_complete_false_when_files_missing() -> None:
    record = MapRecord(
        metadata=_sample_metadata(),
        pgm_path=Path("/nonexistent/map.pgm"),
        yaml_path=Path("/nonexistent/map.yaml"),
    )
    assert record.is_complete() is False


def test_map_record_is_complete_true_when_files_exist(tmp_path: Path) -> None:
    pgm = tmp_path / "map.pgm"
    yaml = tmp_path / "map.yaml"
    pgm.write_bytes(b"P5 1 1 255 0")
    yaml.write_text("resolution: 0.05\n", encoding="utf-8")

    record = MapRecord(metadata=_sample_metadata(), pgm_path=pgm, yaml_path=yaml)
    assert record.is_complete() is True


def test_map_record_to_dict_complete_key(tmp_path: Path) -> None:
    record = MapRecord(metadata=_sample_metadata())
    d = record.to_dict()
    assert "complete" in d
    assert d["complete"] is False


# ---------------------------------------------------------------------------
# MapStore tests
# ---------------------------------------------------------------------------


def test_map_store_add_and_get(tmp_path: Path) -> None:
    store = MapStore(tmp_path)
    record = MapRecord(metadata=_sample_metadata("my_map"))
    store.add(record)

    retrieved = store.get("my_map")
    assert retrieved is not None
    assert retrieved.metadata.map_name == "my_map"


def test_map_store_list_maps_sorted(tmp_path: Path) -> None:
    store = MapStore(tmp_path)
    store.add(MapRecord(metadata=_sample_metadata("b_map")))
    store.add(MapRecord(metadata=_sample_metadata("a_map")))

    names = store.list_maps()
    assert names == ["a_map", "b_map"]


def test_map_store_persists_to_json(tmp_path: Path) -> None:
    store = MapStore(tmp_path)
    store.add(MapRecord(metadata=_sample_metadata("persist_map")))

    # Reload from disk
    store2 = MapStore(tmp_path)
    assert "persist_map" in store2.list_maps()


def test_map_store_complete_maps_filters_correctly(tmp_path: Path) -> None:
    pgm = tmp_path / "real.pgm"
    yaml = tmp_path / "real.yaml"
    pgm.write_bytes(b"P5 1 1 255 0")
    yaml.write_text("resolution: 0.05\n", encoding="utf-8")

    store = MapStore(tmp_path)
    store.add(MapRecord(metadata=_sample_metadata("incomplete")))
    store.add(MapRecord(metadata=_sample_metadata("complete"), pgm_path=pgm, yaml_path=yaml))

    complete = store.complete_maps()
    names = {r.metadata.map_name for r in complete}
    assert "complete" in names
    assert "incomplete" not in names


def test_map_store_remove(tmp_path: Path) -> None:
    store = MapStore(tmp_path)
    store.add(MapRecord(metadata=_sample_metadata("to_remove")))
    removed = store.remove("to_remove")
    assert removed is True
    assert store.get("to_remove") is None


def test_map_store_remove_nonexistent_returns_false(tmp_path: Path) -> None:
    store = MapStore(tmp_path)
    assert store.remove("ghost_map") is False


# ---------------------------------------------------------------------------
# nav_plan tests
# ---------------------------------------------------------------------------


def _sample_waypoints() -> list[Waypoint]:
    return [
        Waypoint("home",     x_m=0.0, y_m=0.0),
        Waypoint("workcell", x_m=3.0, y_m=0.0),
        Waypoint("inspect",  x_m=3.0, y_m=2.0, waypoint_type=WaypointType.inspect),
        Waypoint("dock",     x_m=0.0, y_m=0.0, waypoint_type=WaypointType.dock),
    ]


def test_build_nav_plan_computes_distances() -> None:
    plan = build_nav_plan("lab_run", _sample_waypoints())
    assert plan.name == "lab_run"
    assert len(plan.waypoints) == 4
    assert len(plan.segments) == 3  # 4 waypoints → 3 segments

    # home → workcell: 3.0 m
    assert plan.segments[0].distance_m == pytest.approx(3.0, abs=1e-4)
    # workcell → inspect: 2.0 m
    assert plan.segments[1].distance_m == pytest.approx(2.0, abs=1e-4)


def test_build_nav_plan_total_distance() -> None:
    plan = build_nav_plan("total_dist", _sample_waypoints())
    # 3.0 + 2.0 + sqrt(9+4) ≈ 5.0 + 3.6056
    expected = 3.0 + 2.0 + math.sqrt(3.0**2 + 2.0**2)
    assert plan.total_distance_m == pytest.approx(expected, abs=1e-3)


def test_build_nav_plan_headings_are_correct() -> None:
    wps = [
        Waypoint("a", x_m=0.0, y_m=0.0),
        Waypoint("b", x_m=1.0, y_m=0.0),  # heading should be 0 (east)
        Waypoint("c", x_m=1.0, y_m=1.0),  # heading should be pi/2 (north)
    ]
    plan = build_nav_plan("heading_test", wps)
    assert plan.segments[0].heading_rad == pytest.approx(0.0, abs=1e-4)
    assert plan.segments[1].heading_rad == pytest.approx(math.pi / 2, abs=1e-4)


def test_build_nav_plan_single_waypoint() -> None:
    plan = build_nav_plan("solo", [Waypoint("only", x_m=1.0, y_m=2.0)])
    assert len(plan.waypoints) == 1
    assert len(plan.segments) == 0
    assert plan.total_distance_m == 0.0


def test_build_nav_plan_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="at least one waypoint"):
        build_nav_plan("empty", [])


def test_validate_nav_plan_passes_valid_plan() -> None:
    plan = build_nav_plan("valid", _sample_waypoints())
    issues = validate_nav_plan(plan)
    assert issues == []


def test_validate_nav_plan_flags_duplicate_labels() -> None:
    wps = [
        Waypoint("home", x_m=0.0, y_m=0.0),
        Waypoint("home", x_m=1.0, y_m=0.0),  # duplicate
    ]
    plan = build_nav_plan("dup", wps)
    issues = validate_nav_plan(plan)
    assert any("Duplicate" in issue for issue in issues)


def test_validate_nav_plan_flags_long_segment() -> None:
    wps = [
        Waypoint("a", x_m=0.0, y_m=0.0),
        Waypoint("b", x_m=20.0, y_m=0.0),  # 20 m > default max 5 m
    ]
    plan = build_nav_plan("long", wps)
    issues = validate_nav_plan(plan, max_segment_m=5.0)
    assert any("exceeds max" in issue for issue in issues)


def test_nav_plan_to_dict_structure() -> None:
    plan = build_nav_plan("dict_test", _sample_waypoints())
    d = plan.to_dict()
    assert d["name"] == "dict_test"
    assert d["n_waypoints"] == 4
    assert len(d["segments"]) == 3
    assert "total_distance_m" in d
