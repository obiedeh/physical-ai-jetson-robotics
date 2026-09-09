"""m3pro-smoke: connect (board preflight), go home, read an observation, nudge
joint1 +5/-5 deg, optional 1 s base pulse, disconnect (base zeroed). Robot on
a stand, operator present. Prints a provenance-style summary."""
from __future__ import annotations

import argparse
import json
import os
import time

from .config import RosmasterM3ProConfig
from .robot import RosmasterM3Pro


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-base", action="store_true", help="skip the base pulse")
    ap.add_argument("--camera", type=int, default=None, help="V4L2 index for one OpenCV camera")
    ap.add_argument("--profile", default=os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE"))
    args = ap.parse_args()
    cams = {}
    if args.camera is not None:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        cams["front"] = OpenCVCameraConfig(index_or_path=args.camera, width=640, height=480, fps=15)
    cfg = RosmasterM3ProConfig(id="m3pro", fastdds_profile=args.profile, cameras=cams,
                               use_base=not args.no_base)
    robot = RosmasterM3Pro(cfg)
    t0 = time.time()
    robot.connect()
    print(f"[smoke] connected + homed in {time.time()-t0:.1f}s")
    obs = robot.get_observation()
    print("[smoke] obs:", json.dumps({k: (v if not hasattr(v, "shape") else list(v.shape))
                                      for k, v in obs.items()}, default=float))
    try:
        home = dict(zip([f"{j}.pos" for j in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")],
                        cfg.home_pose))
        up = dict(home); up["joint1.pos"] = home["joint1.pos"] + 5
        sent = robot.send_action(up); time.sleep(1.0)
        print("[smoke] joint1 +5 ->", sent["joint1.pos"])
        sent = robot.send_action(home); time.sleep(1.0)
        print("[smoke] joint1 home ->", sent["joint1.pos"])
        if not args.no_base:
            t1 = time.time()
            while time.time() - t1 < 1.0:
                robot.send_action({**home, "base.vx": 0.10}); time.sleep(0.05)
            robot.send_action({**home, "base.vx": 0.0})
            time.sleep(0.5)
            o = robot.get_observation()
            print(f"[smoke] base pulse done; odom x={o['base.odom_x']:.3f} vx={o['base.vx']:.3f}")
    finally:
        robot.disconnect()
        print("[smoke] disconnected (base zeroed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
