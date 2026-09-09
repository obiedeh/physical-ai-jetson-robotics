"""Record a LeRobot dataset by driving the M3 Pro from the 5090 KEYBOARD
(active mode) — no joystick needed. The Orin host runs in ACTIVE mode
(m3pro-host.service); this drives the arm via the client over ZMQ and records
observation.images.front / observation.state / action.

  cd /tmp && DISPLAY=:1 ~/.venv/lerobot17/bin/python -m \
      lerobot_robot_rosmaster_m3pro.record_kbd --repo-id oedeh/m3pro_kbd_v1 \
      --remote-ip 192.168.1.251 --episodes 8 --episode-time-s 25 --use-base

Keys (focus the terminal running this): q/a joint1  w/s joint2  e/d joint3
  r/f joint4  t/g joint5  y/h gripper(6)  i/k base vx  j/l base vy  u/o wz
  space=stop base  0=home  ESC=stop teleop. Frames record continuously; a
  short reset window separates episodes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .client import RosmasterM3ProClient, RosmasterM3ProClientConfig
from .teleop import RosmasterM3ProKeyboard, RosmasterM3ProKeyboardConfig
from .config import ARM_JOINTS

BASE = ("base.vx", "base.vy", "base.wz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--remote-ip", default="192.168.1.251")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--episode-time-s", type=float, default=25.0)
    ap.add_argument("--reset-time-s", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--task", default="pick up the cube and place it on the plate")
    ap.add_argument("--use-base", action="store_true")
    ap.add_argument("--cam-h", type=int, default=480)
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = list(ARM_JOINTS) + (["base.vx", "base.vy", "base.wz"] if args.use_base else [])
    features = {
        "observation.images.front": {"dtype": "video", "shape": (args.cam_h, args.cam_w, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features,
                               robot_type="rosmaster_m3pro", use_videos=True)

    robot = RosmasterM3ProClient(RosmasterM3ProClientConfig(
        id="kbd", remote_ip=args.remote_ip, use_base=args.use_base, passive=False,
        camera_shapes={"front": (args.cam_h, args.cam_w)}))
    teleop = RosmasterM3ProKeyboard(RosmasterM3ProKeyboardConfig(
        id="kbd", use_base=args.use_base))
    robot.connect(); teleop.connect()
    print(f"[kbd] connected to host {args.remote_ip}; FOCUS THIS TERMINAL and drive "
          f"with the keys. First key press establishes the arm state.", flush=True)

    def _vec(d: dict) -> np.ndarray:
        v = [float(d.get(f"{j}.pos", d.get(j, 0.0))) for j in ARM_JOINTS]
        if args.use_base:
            v += [float(d.get(k, 0.0)) for k in BASE]
        return np.asarray(v, dtype=np.float32)

    period = 1.0 / args.fps
    try:
        for ep in range(args.episodes):
            print(f"[kbd] EPISODE {ep+1}/{args.episodes} — {args.episode_time_s:.0f}s "
                  f"(Ctrl-C to stop)", flush=True)
            t_end = time.time() + args.episode_time_s; n = 0
            while time.time() < t_end:
                t0 = time.time()
                action = teleop.get_action()          # keyboard -> joint/base targets
                robot.send_action(action)             # -> ZMQ -> host -> board
                obs = robot.get_observation()
                if any(obs[f"{j}.pos"] != obs[f"{j}.pos"] for j in ARM_JOINTS):
                    time.sleep(period); continue      # NaN state (no cmd yet) -> skip
                ds.add_frame({
                    "observation.images.front": obs["front"],
                    "observation.state": _vec(obs),
                    "action": _vec(action),
                    "task": args.task,
                }); n += 1
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
            ds.save_episode()
            print(f"[kbd] episode {ep+1} saved ({n} frames)", flush=True)
            if ep < args.episodes - 1:
                print(f"[kbd] RESET — reposition; press 0 to home; next in "
                      f"{args.reset_time_s:.0f}s", flush=True)
                t1 = time.time()
                while time.time() - t1 < args.reset_time_s:
                    robot.send_action(teleop.get_action()); time.sleep(period)
    except KeyboardInterrupt:
        print("\n[kbd] stopped; saving current episode", flush=True)
        try:
            ds.save_episode()
        except Exception:
            pass
    finally:
        teleop.disconnect(); robot.disconnect()
    if args.push_to_hub:
        ds.push_to_hub()
    print(f"[kbd] done: {ds.num_episodes} eps, {ds.num_frames} frames at {ds.root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
