"""Offline tests for scripts/jetson/slam_session.py: mark, summarize and the /proc summary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jetson" / "slam_session.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("slam_session", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def test_mark_appends_rows(mod, tmp_path: Path) -> None:
    assert mod.main(["mark", "lap_start", "--out", str(tmp_path)]) == 0
    assert mod.main(["mark", "lap_end", "--out", str(tmp_path), "--note", "lap 1"]) == 0
    rows = [json.loads(line) for line in (tmp_path / "marks.jsonl").read_text().splitlines()]
    assert [r["label"] for r in rows] == ["lap_start", "lap_end"]
    assert rows[1]["note"] == "lap 1"
    assert rows[0]["t"] <= rows[1]["t"]


def test_summarize_writes_metrics_with_session(mod, tmp_path: Path) -> None:
    (tmp_path / "session.json").write_text(json.dumps({"schema": "slam-session-v1", "host": "x"}))
    with (tmp_path / "poses.jsonl").open("w") as fh:
        for k in range(20):
            row = {"t": 10.0 + k, "frame": "map", "x": 0.1 * k, "y": 0.0, "yaw": 0.0}
            fh.write(json.dumps(row) + "\n")
    mod.main(["mark", "lap_start", "--out", str(tmp_path)])
    marks = [json.loads(line) for line in (tmp_path / "marks.jsonl").read_text().splitlines()]
    marks[0]["t"] = 10.0
    marks.append({"t": 29.0, "label": "lap_end"})
    (tmp_path / "marks.jsonl").write_text("".join(json.dumps(m) + "\n" for m in marks))
    assert mod.main(["summarize", "--out", str(tmp_path)]) == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["session"]["host"] == "x"
    assert metrics["loop_closure"]["map"]["laps_scored"] == 1
    assert metrics["loop_closure"]["odom"]["laps_scored"] == 0


def test_proc_summary_aggregates_by_pattern(mod) -> None:
    samples = [
        {"t": 1.0, "procs": {"11": {"name": "slam_toolbox", "cpu_percent": None,
                                    "rss_mb": 100.0}}},
        {"t": 3.0, "procs": {"11": {"name": "slam_toolbox", "cpu_percent": 80.0, "rss_mb": 120.0},
                             "12": {"name": "micro_ros_agent", "cpu_percent": 5.0,
                                    "rss_mb": 20.0}}},
        {"t": 5.0, "procs": {"11": {"name": "slam_toolbox", "cpu_percent": 120.0,
                                    "rss_mb": 110.0}}},
    ]
    s = mod.summarize_proc_samples(samples)
    assert s["samples"] == 3
    st = s["processes"]["slam_toolbox"]
    assert st["cpu_percent_peak"] == 120.0 and st["cpu_percent_p50"] == 80.0
    assert st["rss_mb_peak"] == 120.0 and st["n"] == 3
    assert s["processes"]["micro_ros_agent"]["n"] == 1
