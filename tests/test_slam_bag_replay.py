"""Pure-Python checks for the offline SLAM replay harness (scripts/jetson/slam_bag_replay.py).

The ROS 2 parts need the rover's board; these tests cover the map writer, the
occupied-cell comparison and the tegrastats parser, and check the committed
replay record against its own map file.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "jetson" / "slam_bag_replay.py"
RECORD = (
    REPO / "reports" / "slam" / "sessions" / "2026-09-16_stationary" / "offline_replay_2026-10-30"
)


def _mod():
    spec = importlib.util.spec_from_file_location("slam_bag_replay", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_map_trinary_and_counts(tmp_path: Path) -> None:
    m = _mod()
    data = np.array([[-1, 0, 100], [30, 70, 25]], dtype=np.int16)
    meta = {
        "resolution_m": 0.05,
        "width": 3,
        "height": 2,
        "origin": [1.0, -2.0, 0.0],
        "frame_id": "map",
        "stamp_s": 0.0,
    }
    rec = m.write_map(data, meta, tmp_path, "map")
    assert (rec["occupied"], rec["free"], rec["unknown"], rec["uncertain"]) == (2, 2, 1, 1)
    assert rec["known_area_m2"] == pytest.approx(4 * 0.05 * 0.05)
    pgm = (tmp_path / "map.pgm").read_bytes()
    header, _, body = pgm.partition(b"255\n")
    assert header.startswith(b"P5\n3 2\n")
    # rows are flipped for the image: first image row is the grid's last row
    assert list(body) == [205, 0, 254, 205, 254, 0]
    assert "origin: [1.000000, -2.000000, 0.0]" in (tmp_path / "map.yaml").read_text()


def test_compare_aligns_by_world_coordinates() -> None:
    m = _mod()
    a = np.zeros((4, 4), dtype=np.int16)
    a[1, 1] = 100
    am = {"resolution_m": 0.05, "origin": [0.0, 0.0, 0.0]}
    # same occupied cell seen from a map whose origin is one cell further left and down
    b = np.zeros((4, 4), dtype=np.int16)
    b[2, 2] = 100
    bm = {"resolution_m": 0.05, "origin": [-0.05, -0.05, 0.0]}
    c = m.compare(a, am, b, bm)
    assert c["comparable"] and c["occupied_iou"] == 1.0
    bm_other = {"resolution_m": 0.1, "origin": [0.0, 0.0, 0.0]}
    assert m.compare(a, am, b, bm_other)["comparable"] is False


def test_parse_tegrastats(tmp_path: Path) -> None:
    m = _mod()
    log = tmp_path / "t.log"
    log.write_text(
        "x VDD_IN 5000mW/5000mW tj@50.5C\nx VDD_IN 7000mW/7000mW tj@52C\n"
        "x VDD_IN 6000mW/6000mW tj@51C\n"
    )
    p = m.parse_tegrastats(log)
    assert (
        p["samples"] == 3
        and p["vdd_in_w_p50"] == 6.0
        and p["vdd_in_w_max"] == 7.0
        and p["tj_c_max"] == 52.0
    )
    assert m.parse_tegrastats(tmp_path / "missing.log") == {"available": False}


@pytest.mark.skipif(not RECORD.exists(), reason="replay record not present")
def test_committed_replay_record_matches_its_map() -> None:
    rec = json.loads((RECORD / "metrics.json").read_text())
    assert rec["schema"] == "slam-offline-replay-v1"
    assert rec["playback"]["exit_code"] == 0
    rm = rec["replay_map"]
    pgm = (RECORD / "map.pgm").read_bytes()
    header, _, body = pgm.partition(b"255\n")
    assert header == f"P5\n{rm['width']} {rm['height']}\n".encode()
    counts = {0: 0, 205: 0, 254: 0}
    for v in body:
        counts[v] += 1
    assert (
        counts[0] == rm["occupied"]
        and counts[254] == rm["free"]
        and counts[205] == rm["unknown"] + rm["uncertain"]
    )
    assert rec["notes"][0].startswith("stationary session")
    assert "jetson" not in json.dumps(rec).lower().replace("jetson-orin-nx", "").replace(
        "nvidia jetson orin nx", ""
    )
