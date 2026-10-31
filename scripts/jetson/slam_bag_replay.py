#!/usr/bin/env python3
"""Rebuild a map offline from a recorded rover session with SLAM Toolbox.

The session recorder (slam_session.py) keeps the raw bag of a SLAM run next to
its metrics. This script replays that bag through a fresh SLAM Toolbox mapping
node on the robot's own board and saves what comes out, so the mapping stack
can be exercised and measured again without driving the rover:

1. reads the bag's metadata and hashes its storage file;
2. extracts the last /map the live session itself published (the "session
   map"), straight from the bag, as the reference;
3. starts ``async_slam_toolbox_node`` with the committed replay parameters and
   simulated time;
4. plays the bag's /scan, /odom, /tf and /tf_static with the bag clock. The
   recorded /tf also carries the live session's own map->odom transform, so it
   is remapped to /tf_bag and republished to /tf with every map-parented
   transform removed; only the replayed node may say where the map is;
5. keeps the latest /map from the node, writes it as map.pgm and map.yaml
   (the map_server format), and scores it against the session map;
6. records the node's peak resident memory and CPU time, board power from
   tegrastats when available, and every count that describes what happened.

Nothing is installed and nothing on the board is changed. ROS 2 (Humble on the
rover's Orin NX), slam_toolbox and the bag are expected to be present. The
output directory gets map.pgm, map.yaml, session_map.pgm, session_map.yaml,
metrics.json, slam_toolbox.log, bag_play.log and tegrastats.log.

A stationary session gives a stationary map: with minimum_travel_distance at
0.5 m, SLAM Toolbox adds one node and the map is the footprint of one laser
sweep. That is still a real run of the mapping stack on the recorded data, and
the metrics say so; loop closure and drift need a driven session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "slam-offline-replay-v1"
PLAY_TOPICS = ["/scan", "/odom", "/tf", "/tf_static"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip("\x00\n ")
    except OSError:
        return None


def grid_to_arrays(msg: Any) -> tuple[np.ndarray, dict[str, Any]]:
    info = msg.info
    data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
    meta = {
        "resolution_m": float(info.resolution),
        "width": int(info.width),
        "height": int(info.height),
        "origin": [float(info.origin.position.x), float(info.origin.position.y), 0.0],
        "frame_id": msg.header.frame_id,
        "stamp_s": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
    }
    return data, meta


def write_map(data: np.ndarray, meta: dict[str, Any], out: Path, stem: str) -> dict[str, Any]:
    """map_server trinary PGM: occupied black, free white, unknown grey."""
    img = np.full(data.shape, 205, dtype=np.uint8)
    img[(data >= 0) & (data <= 25)] = 254
    img[data >= 65] = 0
    pgm = out / f"{stem}.pgm"
    with pgm.open("wb") as f:
        f.write(f"P5\n{data.shape[1]} {data.shape[0]}\n255\n".encode())
        f.write(np.flipud(img).tobytes())  # PGM rows run top down; grid rows run bottom up
    yaml_text = (
        f"image: {stem}.pgm\nmode: trinary\nresolution: {meta['resolution_m']}\n"
        f"origin: [{meta['origin'][0]:.6f}, {meta['origin'][1]:.6f}, 0.0]\nnegate: 0\n"
        f"occupied_thresh: 0.65\nfree_thresh: 0.25\n"
    )
    (out / f"{stem}.yaml").write_text(yaml_text)
    cells = int(data.size)
    counts = {
        "cells": cells,
        "occupied": int((data >= 65).sum()),
        "free": int(((data >= 0) & (data <= 25)).sum()),
        "unknown": int((data < 0).sum()),
        "uncertain": int(((data > 25) & (data < 65)).sum()),
    }
    counts["known_area_m2"] = round(
        (counts["occupied"] + counts["free"]) * meta["resolution_m"] ** 2, 3
    )
    return {**meta, **counts, "pgm_sha256": sha256(pgm), "yaml": f"{stem}.yaml"}


def occupied_world_cells(
    data: np.ndarray, meta: dict[str, Any], threshold: int = 65
) -> set[tuple[int, int]]:
    """Occupied cells as integer world-grid indices (floor of the cell centre
    over the resolution, with a small epsilon so a centre that lands on a
    boundary within float error is binned the same way from either map)."""
    res = meta["resolution_m"]
    ys, xs = np.nonzero(data >= threshold)
    ox, oy = meta["origin"][0], meta["origin"][1]
    eps = 1e-6
    return {
        (
            int(math.floor((ox + (x + 0.5) * res) / res + eps)),
            int(math.floor((oy + (y + 0.5) * res) / res + eps)),
        )
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    }


def compare(a: np.ndarray, am: dict[str, Any], b: np.ndarray, bm: dict[str, Any]) -> dict[str, Any]:
    if abs(am["resolution_m"] - bm["resolution_m"]) > 1e-9:
        return {"comparable": False, "reason": "different resolutions"}
    sa, sb = occupied_world_cells(a, am), occupied_world_cells(b, bm)
    inter = len(sa & sb)
    union = len(sa | sb)
    return {
        "comparable": True,
        "occupied_cells_replay": len(sa),
        "occupied_cells_session": len(sb),
        "occupied_intersection": inter,
        "occupied_union": union,
        "occupied_iou": round(inter / union, 4) if union else None,
        "note": (
            "occupied cells compared in world coordinates at the shared resolution; "
            "the two maps may have different origins and extents"
        ),
    }


def bag_session_map(bag_dir: Path) -> tuple[np.ndarray, dict[str, Any], int] | None:
    """The last /map message the live session recorded, read straight from the bag."""
    import rosbag2_py
    from nav_msgs.msg import OccupancyGrid
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/map"]))
    last = None
    count = 0
    while reader.has_next():
        _topic, raw, _t = reader.read_next()
        last = raw
        count += 1
    if last is None:
        return None
    msg = deserialize_message(last, OccupancyGrid)
    data, meta = grid_to_arrays(msg)
    return data, meta, count


def node_pid(launcher_pid: int, name: str, wait_s: float = 10.0) -> int:
    """``ros2 run`` execs the node through a Python launcher; the process to
    measure is its child whose command names the node executable."""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().split()
                if int(fields[3]) == launcher_pid and name in Path(
                    f"/proc/{fields[0]}/cmdline"
                ).read_bytes().decode(errors="replace"):
                    return int(fields[0])
            except (OSError, ValueError, IndexError):
                continue
        time.sleep(0.2)
    return launcher_pid


class ProcSampler(threading.Thread):
    """Peak RSS and CPU time of one pid from /proc, sampled every second."""

    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.rss_peak_kb = 0
        self.samples = 0
        self.stop = threading.Event()
        self.cpu_ticks = 0

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        self.rss_peak_kb = max(self.rss_peak_kb, int(line.split()[1]))
                fields = Path(f"/proc/{self.pid}/stat").read_text().split()
                self.cpu_ticks = int(fields[13]) + int(fields[14])
                self.samples += 1
            except (OSError, IndexError, ValueError):
                break
            self.stop.wait(1.0)


def parse_tegrastats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    vin: list[float] = []
    temps: list[float] = []
    for line in path.read_text().splitlines():
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            vin.append(int(m.group(1)) / 1000.0)
        m = re.search(r"tj@([\d.]+)C", line)
        if m:
            temps.append(float(m.group(1)))
    if not vin:
        return {"available": False, "samples": 0}
    return {
        "available": True,
        "samples": len(vin),
        "vdd_in_w_p50": round(statistics.median(vin), 3),
        "vdd_in_w_max": round(max(vin), 3),
        "tj_c_max": max(temps) if temps else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--bag", type=Path, required=True, help="rosbag2 directory (metadata.yaml and *.db3)"
    )
    ap.add_argument(
        "--params",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "slam_toolbox_replay_params.yaml",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=float, default=1.0, help="bag playback rate")
    ap.add_argument(
        "--settle-s",
        type=float,
        default=4.0,
        help="seconds to let slam_toolbox start before playback",
    )
    ap.add_argument("--no-tegrastats", action="store_true")
    args = ap.parse_args()

    import rclpy
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from tf2_msgs.msg import TFMessage

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    bag_dir = args.bag.resolve()
    meta_path = bag_dir / "metadata.yaml"
    db_files = sorted(bag_dir.glob("*.db3"))
    if not meta_path.exists() or not db_files:
        print(f"no rosbag2 directory at {bag_dir}", file=sys.stderr)
        return 2
    started = time.time()
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    session = bag_session_map(bag_dir)
    session_map = None
    session_data = session_meta = None
    if session is not None:
        session_data, session_meta, n_maps = session
        session_map = write_map(session_data, session_meta, out, "session_map")
        session_map["messages_in_bag"] = n_maps

    env = dict(os.environ)
    slam_log = (out / "slam_toolbox.log").open("w")
    slam = subprocess.Popen(
        [
            "ros2",
            "run",
            "slam_toolbox",
            "async_slam_toolbox_node",
            "--ros-args",
            "--params-file",
            str(args.params),
            "-p",
            "use_sim_time:=true",
        ],
        stdout=slam_log,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    slam_node_pid = node_pid(slam.pid, "async_slam_toolbox_node")
    sampler = ProcSampler(slam_node_pid)
    sampler.start()
    tegra = None
    tegra_path = out / "tegrastats.log"
    if not args.no_tegrastats and shutil.which("tegrastats"):
        tegra = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=tegra_path.open("w"),
            stderr=subprocess.DEVNULL,
        )

    rclpy.init()

    class Harness(Node):
        def __init__(self) -> None:
            super().__init__("slam_bag_replay_harness")
            self.tf_pub = self.create_publisher(TFMessage, "/tf", 200)
            self.create_subscription(TFMessage, "/tf_bag", self.on_tf_bag, 200)
            self.create_subscription(TFMessage, "/tf", self.on_tf, 200)
            map_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
            self.tf_in = self.tf_out = self.tf_dropped_map_parent = 0
            self.live_map_odom = 0
            self.maps = 0
            self.last_map: OccupancyGrid | None = None
            self.first_map_wall: float | None = None
            self.lock = threading.Lock()

        def on_tf_bag(self, msg: TFMessage) -> None:
            keep = [t for t in msg.transforms if t.header.frame_id.lstrip("/") != "map"]
            self.tf_in += len(msg.transforms)
            self.tf_dropped_map_parent += len(msg.transforms) - len(keep)
            if keep:
                self.tf_out += len(keep)
                self.tf_pub.publish(TFMessage(transforms=keep))

        def on_tf(self, msg: TFMessage) -> None:
            # the bag's map->odom never reaches /tf, so a map-parented transform
            # here comes from the live node
            self.live_map_odom += sum(
                1 for t in msg.transforms if t.header.frame_id.lstrip("/") == "map"
            )

        def on_map(self, msg: OccupancyGrid) -> None:
            with self.lock:
                self.maps += 1
                self.last_map = msg
                if self.first_map_wall is None:
                    self.first_map_wall = time.time()

    node = Harness()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    time.sleep(args.settle_s)

    play_log = (out / "bag_play.log").open("w")
    play_cmd = [
        "ros2",
        "bag",
        "play",
        str(bag_dir),
        "--clock",
        "100",
        "--rate",
        str(args.rate),
        "--topics",
        *PLAY_TOPICS,
        "--remap",
        "/tf:=/tf_bag",
    ]
    play_started = time.time()
    play = subprocess.Popen(
        play_cmd, stdout=play_log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid
    )
    play_rc = play.wait()
    play_wall = time.time() - play_started
    time.sleep(3.0)  # the last map the node published before the clock stopped

    with node.lock:
        last_map = node.last_map
    replay_map = None
    comparison: dict[str, Any] | None = None
    if last_map is not None:
        data, meta = grid_to_arrays(last_map)
        replay_map = write_map(data, meta, out, "map")
        if session_data is not None and session_meta is not None:
            comparison = compare(data, meta, session_data, session_meta)

    sampler.stop.set()
    os.killpg(os.getpgid(slam.pid), signal.SIGINT)
    try:
        slam.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(slam.pid), signal.SIGKILL)
    if tegra is not None:
        tegra.terminate()
        tegra.wait(timeout=5)
    rclpy.shutdown()
    slam_log.close()
    play_log.close()

    clk_tck = os.sysconf("SC_CLK_TCK")
    metadata_text = meta_path.read_text()
    duration_ns = re.search(r"duration:\s*\n\s*nanoseconds:\s*(\d+)", metadata_text)
    slam_version = run(["dpkg-query", "-W", "-f=${Version}", "ros-humble-slam-toolbox"]) or run(
        ["dpkg-query", "-W", "-f=${Version}", f"ros-{env.get('ROS_DISTRO', 'humble')}-slam-toolbox"]
    )
    metrics: dict[str, Any] = {
        "schema": SCHEMA,
        "recorded_at": recorded_at,
        "wall_s": round(time.time() - started, 1),
        "device_model": read_text("/proc/device-tree/model"),
        "board": "jetson-orin-nx",
        "ros_distro": env.get("ROS_DISTRO"),
        "slam_toolbox_version": slam_version or None,
        "bag": {
            "dir": str(bag_dir.name),
            "storage_files": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in db_files
            ],
            "metadata_sha256": sha256(meta_path),
            "duration_s": round(int(duration_ns.group(1)) / 1e9, 3) if duration_ns else None,
            "topics_played": PLAY_TOPICS,
            "tf_remapped_to": "/tf_bag",
        },
        "params": {"file": str(Path(args.params).name), "sha256": sha256(args.params)},
        "playback": {
            "command": play_cmd[:5] + ["..."],
            "rate": args.rate,
            "exit_code": play_rc,
            "wall_s": round(play_wall, 1),
        },
        "tf_filter": {
            "transforms_in": node.tf_in,
            "transforms_republished": node.tf_out,
            "map_parented_transforms_dropped_from_bag": node.tf_dropped_map_parent,
            "map_parented_transforms_published_by_replayed_node": node.live_map_odom,
        },
        "map_messages_from_replayed_node": node.maps,
        "first_map_after_playback_start_s": round(node.first_map_wall - play_started, 1)
        if node.first_map_wall
        else None,
        "replay_map": replay_map,
        "session_map": session_map,
        "comparison": comparison,
        "slam_toolbox_process": {
            "pid_sampled": slam_node_pid,
            "launcher_pid": slam.pid,
            "rss_peak_mb": round(sampler.rss_peak_kb / 1024, 1),
            "cpu_s": round(sampler.cpu_ticks / clk_tck, 1),
            "samples": sampler.samples,
            "exit_code": slam.returncode,
        },
        "power": parse_tegrastats(tegra_path),
        "not_measured": [
            "loop closure and drift: the session has no laps",
            "localisation: no targets",
        ],
        "notes": [
            "stationary session: one scan node expected (minimum_travel_distance 0.5 m); "
            "the map is one sweep's footprint",
            "host name withheld",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "map_messages_from_replayed_node",
                    "replay_map",
                    "comparison",
                    "slam_toolbox_process",
                    "power",
                )
            },
            indent=1,
        )
    )
    return 0 if replay_map is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
