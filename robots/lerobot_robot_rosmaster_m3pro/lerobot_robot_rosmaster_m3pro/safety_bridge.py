"""Leg C safety bridge: stream M3 Pro telemetry -> SafetyEvents -> the
physical-ai-safety-observability API (POST /events), with a local JSONL
fallback when the API is unreachable. Runs on the 5090 against the Orin host.

  cd /tmp && ~/.venv/lerobot17/bin/python -m \
      lerobot_robot_rosmaster_m3pro.safety_bridge \
      --remote-ip 192.168.1.243 --api http://localhost:8000 --hz 5

Telemetry sources (from the host obs stream): battery.v, base odom velocity,
min_obstacle_m (LiDAR). Optional --person-source wires a detector later; e-stop
and deadman can be injected by the controller. Events are de-duplicated per
(rule_id, severity) with a cooldown so a standing condition is not spammed.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from math import hypot
from pathlib import Path

from .safety import SafetyConfig, evaluate


def _post(api: str, event: dict, timeout: float = 3.0) -> bool:
    req = urllib.request.Request(
        api.rstrip("/") + "/events",
        data=json.dumps(event).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-ip", default="192.168.1.243")
    ap.add_argument("--api", default="http://localhost:8000",
                    help="physical-ai-safety-observability base URL")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--cooldown-s", type=float, default=5.0,
                    help="min seconds between repeats of the same (rule,severity)")
    ap.add_argument("--fallback", default="reports/safety/m3pro_events.jsonl")
    ap.add_argument("--camera-id", default="m3pro_front")
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = run until Ctrl-C")
    args = ap.parse_args()

    from .client import RosmasterM3ProClient, RosmasterM3ProClientConfig
    cfg = SafetyConfig(camera_id=args.camera_id)
    robot = RosmasterM3ProClient(RosmasterM3ProClientConfig(
        id="safety", remote_ip=args.remote_ip, use_base=True,
        camera_shapes={"front": (480, 640)}))
    robot.connect()
    fb = Path(args.fallback); fb.parent.mkdir(parents=True, exist_ok=True)
    fbf = fb.open("a")
    print(f"[safety] bridge up: host {args.remote_ip} -> API {args.api} "
          f"(fallback {fb})", flush=True)

    last_sent: dict[tuple, float] = {}
    period = 1.0 / args.hz
    t_end = time.time() + args.duration_s if args.duration_s > 0 else None
    posted = dropped = 0
    try:
        while t_end is None or time.time() < t_end:
            t0 = time.time()
            obs = robot.get_observation()
            snap = {
                "battery_v": float(obs.get("battery.v", float("nan"))),
                "min_obstacle_m": obs.get("min_obstacle_m"),
                "base_speed_mps": hypot(float(obs.get("base.vx", 0.0)),
                                        float(obs.get("base.vy", 0.0))),
                "captured_at": None,
            }
            for ev in evaluate(snap, cfg):
                key = (ev["rule_id"], ev["severity"])
                now = time.time()
                if now - last_sent.get(key, 0.0) < args.cooldown_s:
                    continue
                last_sent[key] = now
                ok = _post(args.api, ev)
                fbf.write(json.dumps({"posted": ok, **ev}) + "\n"); fbf.flush()
                posted += ok; dropped += (not ok)
                print(f"[safety] {'POST' if ok else 'LOCAL'} {ev['rule_id']} "
                      f"{ev['severity']} :: {ev['summary']}", flush=True)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        fbf.close(); robot.disconnect()
        print(f"[safety] done: {posted} posted, {dropped} local-only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
