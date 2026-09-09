#!/usr/bin/env python3
"""Rover SLAM session recorder for the ROSMASTER M3 Pro on Jetson Orin NX.

Records what a SLAM measurement needs and nothing that decides the result:

  start      provenance first, then a still baseline, then ``ros2 bag record``
             of the named topics while sampling INA3221 rails, thermal zones,
             RAM and per-process CPU of the SLAM stack; writes session.json
  mark       append an operator mark (lap_start, lap_end, kidnap_lift,
             kidnap_place, or free text) with a timestamp to marks.jsonl
  summarize  compute loop-closure and re-localisation metrics from
             poses.jsonl (see slam_pose_log.py) and marks.jsonl into metrics.json

The SLAM stack itself (vendor slam_toolbox / cartographer / gmapping launch on
the Orin) is started by the operator before ``start``; this script observes it.
It never publishes /cmd_vel. Run from the repo root on the Orin after
``source scripts/jetson/orin_ros_env.sh``.

Protocol: docs/rosmaster_m3pro/SLAM_FIRST_RUN_PROTOCOL.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physical_ai_lab.jetson_provenance import (  # noqa: E402
    PowerWatch,
    ThermalWatch,
    write_provenance,
)
from slam.metrics import compute_metrics, percentile  # noqa: E402

DEFAULT_TOPICS = [
    "/scan0", "/scan1", "/imu/data_raw", "/odom_raw", "/tf", "/tf_static", "/map", "/battery",
]
DEFAULT_PROC_PATTERNS = [
    "slam_toolbox", "cartographer", "slam_gmapping", "rtabmap", "micro_ros_agent",
    "laserscan_multi_merger", "robot_state_publisher", "ekf_node", "ros2 bag", "slam_pose_log",
]


class ProcWatch:
    """Per-process CPU percent and RSS from /proc for processes matching patterns."""

    def __init__(self, patterns: list[str], interval_s: float = 2.0) -> None:
        self.patterns = [re.compile(p) for p in patterns]
        self.interval = interval_s
        self.samples: list[dict] = []
        self._prev: dict[int, tuple[float, int]] = {}
        self._clk = os.sysconf("SC_CLK_TCK")
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def _matching(self) -> dict[int, str]:
        found: dict[int, str] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                cmd = open(f"/proc/{entry}/cmdline", "rb").read().replace(b"\0", b" ").decode(
                    errors="replace").strip()
            except OSError:
                continue
            for pat in self.patterns:
                if pat.search(cmd):
                    found[int(entry)] = pat.pattern
                    break
        return found

    def _sample(self) -> dict:
        now = time.time()
        out: dict = {"t": now, "procs": {}}
        for pid, name in self._matching().items():
            try:
                stat = open(f"/proc/{pid}/stat").read()
                fields = stat.rsplit(")", 1)[1].split()
                ticks = int(fields[11]) + int(fields[12])  # utime + stime
                rss_kb = 0
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
            except (OSError, IndexError, ValueError):
                continue
            cpu = None
            if pid in self._prev:
                t0, ticks0 = self._prev[pid]
                dt = now - t0
                if dt > 0:
                    cpu = round((ticks - ticks0) / self._clk / dt * 100.0, 1)
            self._prev[pid] = (now, ticks)
            out["procs"][str(pid)] = {"name": name, "cpu_percent": cpu,
                                      "rss_mb": round(rss_kb / 1024, 1)}
        return out

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval)

    def __enter__(self) -> ProcWatch:
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)

    def summary(self) -> dict:
        return summarize_proc_samples(self.samples)


def summarize_proc_samples(samples: list[dict]) -> dict:
    by_name: dict[str, dict[str, list[float]]] = {}
    for s in samples:
        for rec in s.get("procs", {}).values():
            slot = by_name.setdefault(rec["name"], {"cpu": [], "rss": []})
            if rec.get("cpu_percent") is not None:
                slot["cpu"].append(rec["cpu_percent"])
            slot["rss"].append(rec["rss_mb"])
    out: dict = {"samples": len(samples), "interval_note": "cpu_percent is per process, "
                 "over the sample interval, 100 = one core", "processes": {}}
    for name, v in sorted(by_name.items()):
        out["processes"][name] = {
            "cpu_percent_p50": percentile(v["cpu"], 50),
            "cpu_percent_peak": max(v["cpu"]) if v["cpu"] else None,
            "rss_mb_peak": max(v["rss"]) if v["rss"] else None,
            "n": len(v["rss"]),
        }
    return out


def _mem_available_kb() -> int | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def _power_mode_nosudo() -> str:
    try:
        status = open("/var/lib/nvpmodel/status").read().strip()
        pid = int(status.split(":")[1])
        for line in open("/etc/nvpmodel.conf"):
            if line.startswith(f"< POWER_MODEL ID={pid} "):
                return f"{line.strip()} ({status})"
        return status
    except Exception:
        return "unavailable"


def _device_model() -> str:
    try:
        return open("/proc/device-tree/model").read().replace("\0", "").strip()
    except OSError:
        return "unavailable"


def _du_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _rail_window(pw: PowerWatch, start_idx: int, end_idx: int) -> dict:
    out: dict = {}
    for rail in pw.rails:
        vals = [s[rail + "_mW"] for s in pw.samples[start_idx:end_idx] if rail + "_mW" in s]
        if vals:
            out[rail] = {"mW_p50": percentile(vals, 50), "mW_peak": max(vals), "n": len(vals)}
    return out


def cmd_start(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topics = args.topics.split(",") if args.topics else DEFAULT_TOPICS
    model = _device_model()
    prov = write_provenance(
        out,
        data_kind=(f"REAL hardware measurement ({model}; rover SLAM session: ros2 bag of "
                   f"{len(topics)} topics, INA3221 power, thermal zones, per-process CPU; "
                   "NOT simulated)"),
        seed_record={},
        precision="n/a",
        batch_size=1,
        extra={
            "power_mode_nosudo": _power_mode_nosudo(),
            "slam_stack": args.stack,
            "operator_note": args.note,
            "topics": topics,
            "still_baseline_s": args.still_seconds,
            "mem_available_kb_at_start": _mem_available_kb(),
        },
    )
    print(f"[slam-session] provenance -> {prov}", flush=True)
    session: dict = {
        "schema": "slam-session-v1",
        "host": platform.node(),
        "device_model": model,
        "provenance": prov.name,
        "slam_stack": args.stack,
        "topics": topics,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "incomplete",
    }
    bag_dir = out / "bag"
    bag_proc: subprocess.Popen | None = None
    t0 = time.time()
    t_rec = t0
    still_n = record_n = 0
    with ThermalWatch(interval_s=1.0) as tw, PowerWatch(0.5) as pw, \
            ProcWatch(args.proc_patterns.split(","), 2.0) as procs:
        try:
            time.sleep(args.still_seconds)
            still_n = len(pw.samples)
            session["still_baseline"] = {
                "window_s": args.still_seconds,
                "note": "SLAM stack running, robot stationary, before bag record started",
            }
            if not args.no_bag:
                bag_cmd = ["ros2", "bag", "record", "-o", str(bag_dir), *topics]
                bag_proc = subprocess.Popen(
                    bag_cmd, stdin=subprocess.DEVNULL,
                    stdout=open(out / "bag_record.log", "w"), stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                session["bag_command"] = bag_cmd
                print(f"[slam-session] bag record pid {bag_proc.pid} -> {bag_dir}", flush=True)
            t_rec = time.time()
            session["record_started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            last_beat = t_rec
            while True:
                if args.duration and time.time() - t_rec >= args.duration:
                    break
                if bag_proc is not None and bag_proc.poll() is not None:
                    session["status"] = "bag_exited_early"
                    session["bag_returncode"] = bag_proc.returncode
                    break
                time.sleep(0.5)
                if time.time() - last_beat >= 30:
                    last_beat = time.time()
                    vin = [s.get("VDD_IN_mW") for s in pw.samples[-60:] if s.get("VDD_IN_mW")]
                    print(f"[slam-session] {time.time() - t_rec:6.0f}s  mem_avail "
                          f"{(_mem_available_kb() or 0) // 1024} MB  VDD_IN p50 "
                          f"{percentile(vin, 50) if vin else 'n/a'} mW", flush=True)
        except KeyboardInterrupt:
            print("[slam-session] interrupted by operator", flush=True)
        finally:
            if bag_proc is not None and bag_proc.poll() is None:
                os.killpg(bag_proc.pid, signal.SIGINT)
                try:
                    bag_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(bag_proc.pid, signal.SIGTERM)
                    bag_proc.wait(timeout=10)
            record_n = len(pw.samples)
    if session["status"] == "incomplete":
        session["status"] = "complete"
    session["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session["wall_clock_s"] = round(time.time() - t0, 1)
    session["record_s"] = round(time.time() - t_rec, 1)
    session["thermal"] = tw.summary()
    session["power"] = pw.summary()
    session["power"]["still_baseline"] = _rail_window(pw, 0, still_n)
    session["power"]["recording"] = _rail_window(pw, still_n, record_n)
    session["processes"] = procs.summary()
    mem = [s["mem_available_kb"] for s in tw.samples if "mem_available_kb" in s]
    session["mem_available_mb"] = (
        {"min": min(mem) // 1024, "p50": (percentile(mem, 50) or 0) // 1024, "n": len(mem)}
        if mem else None
    )
    if bag_dir.exists():
        meta = bag_dir / "metadata.yaml"
        session["bag"] = {
            "dir": bag_dir.name,
            "bytes": _du_bytes(bag_dir),
            "metadata_sha256": (hashlib.sha256(meta.read_bytes()).hexdigest()
                                if meta.exists() else None),
            "note": "bag data files are not committed; metadata.yaml is",
        }
    with (out / "samples.jsonl").open("w") as fh:
        for s in pw.samples:
            fh.write(json.dumps({"kind": "power", **s}) + "\n")
        for s in tw.samples:
            fh.write(json.dumps({"kind": "thermal", **s}) + "\n")
        for s in procs.samples:
            fh.write(json.dumps({"kind": "proc", **s}) + "\n")
    (out / "session.json").write_text(json.dumps(session, indent=2) + "\n")
    print(f"[slam-session] {session['status']} -> {out / 'session.json'}", flush=True)
    return 0 if session["status"] == "complete" else 1


def cmd_mark(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "label": args.label, "note": args.note}
    with (out / "marks.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[slam-session] mark {args.label} at {row['utc']}", flush=True)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    targets = json.loads(Path(args.targets).read_text()) if args.targets else None
    metrics = compute_metrics(
        out, targets, converge_m=args.converge_m, converge_deg=args.converge_deg,
        hold_s=args.hold_s, timeout_s=args.timeout_s,
    )
    session_path = out / "session.json"
    metrics["session"] = json.loads(session_path.read_text()) if session_path.exists() else None
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for frame, lc in metrics["loop_closure"].items():
        print(f"[slam-session] {frame}: laps scored {lc['laps_scored']}/{lc['laps_marked']} "
              f"translation error {lc['translation_error_m']}")
    if metrics["relocalisation"]:
        r = metrics["relocalisation"]
        print(f"[slam-session] relocalisation: {r['converged']}/{r['trials_scored']} converged, "
              f"time {r['time_to_converge_s']}")
    for item in metrics["not_measured"]:
        print(f"[slam-session] not measured: {item}")
    print(f"[slam-session] -> {out / 'metrics.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="record a session")
    s.add_argument("--out", required=True,
                   help="session directory, e.g. reports/slam/sessions/2026-09-10_run_a")
    s.add_argument("--stack", required=True,
                   help="SLAM stack and launch in use, e.g. 'vendor slam_toolbox "
                        "(M3Pro_ws yahboom_mapping)'")
    s.add_argument("--note", default="", help="operator note, e.g. room, route, who is present")
    s.add_argument("--topics", default="", help="comma list; default " + ",".join(DEFAULT_TOPICS))
    s.add_argument("--duration", type=float, default=0.0,
                   help="seconds to record; 0 = until Ctrl-C")
    s.add_argument("--still-seconds", type=float, default=15.0)
    s.add_argument("--proc-patterns", default=",".join(DEFAULT_PROC_PATTERNS))
    s.add_argument("--no-bag", action="store_true", help="sample resources only")
    s.set_defaults(fn=cmd_start)

    m = sub.add_parser("mark", help="append an operator mark")
    m.add_argument("label")
    m.add_argument("--out", required=True)
    m.add_argument("--note", default="")
    m.set_defaults(fn=cmd_mark)

    z = sub.add_parser("summarize", help="compute metrics from poses.jsonl and marks.jsonl")
    z.add_argument("--out", required=True)
    z.add_argument("--targets", default="",
                   help="JSON list of {x,y,yaw} kidnap targets in the map frame")
    z.add_argument("--converge-m", type=float, default=0.15)
    z.add_argument("--converge-deg", type=float, default=10.0)
    z.add_argument("--hold-s", type=float, default=2.0)
    z.add_argument("--timeout-s", type=float, default=60.0)
    z.set_defaults(fn=cmd_summarize)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
