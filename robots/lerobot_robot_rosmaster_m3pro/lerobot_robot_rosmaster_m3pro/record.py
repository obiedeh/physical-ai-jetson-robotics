"""Record a LeRobot dataset from the M3 Pro while the operator drives with the
vendor joystick. Runs on the 5090 (lerobot >= 0.6.2); the Orin host must run in
--passive mode so the joystick owns the board and the host captures its
/arm6_joints + /cmd_vel commands as the action.

  # on the Orin:
  bash scripts/jetson/m3pro_host.sh --passive --cameras front:0:640:480:15
  # on the 5090 (from OUTSIDE the repo; repo-local lerobot/ shadows the lib):
  cd /tmp && ~/.venv/lerobot17/bin/python -m lerobot_robot_rosmaster_m3pro.record \
      --repo-id oedeh/m3pro_teleop_v1 --episodes 10 --episode-time-s 20 \
      --remote-ip 192.168.1.251 --task "pick up the cube and place it on the plate"

Dataset schema (vision-based imitation; the board publishes no joint feedback,
so observation.state is the joystick's last arm command and the CAMERA is the
real state; action is the joystick command this frame):
  observation.images.front  (H,W,3 uint8)
  observation.state         (6,)  joint1..6 deg  [+ base odom if --use-base]
  action                    (6,)  joint1..6 deg  [+ base vx,vy,wz if --use-base]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .client import RosmasterM3ProClient, RosmasterM3ProClientConfig
from .config import ARM_JOINTS

BASE = ("base.vx", "base.vy", "base.wz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF dataset id, e.g. oedeh/m3pro_teleop_v1")
    ap.add_argument("--remote-ip", default="192.168.1.251")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-time-s", type=float, default=20.0)
    ap.add_argument("--reset-time-s", type=float, default=8.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--task", default="teleop demonstration")
    ap.add_argument("--use-base", action="store_true", help="also record base vx,vy,wz")
    ap.add_argument("--cam-h", type=int, default=480)
    ap.add_argument("--cam-w", type=int, default=640)
    ap.add_argument("--push-to-hub", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    state_names = list(ARM_JOINTS) + (["base.vx", "base.vy", "base.wz"] if args.use_base else [])
    features = {
        "observation.images.front": {"dtype": "video", "shape": (args.cam_h, args.cam_w, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(state_names),),
                              "names": state_names},
        "action": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=features,
                               robot_type="rosmaster_m3pro", use_videos=True)

    cfg = RosmasterM3ProClientConfig(id="rec", remote_ip=args.remote_ip, passive=True,
                                     use_base=args.use_base,
                                     camera_shapes={"front": (args.cam_h, args.cam_w)})
    robot = RosmasterM3ProClient(cfg)
    robot.connect()
    print(f"[record] connected to host {args.remote_ip}; drive with the joystick.", flush=True)

    def _vec(d: dict) -> np.ndarray:
        v = [float(d.get(f"{j}.pos", d.get(j, 0.0))) for j in ARM_JOINTS]
        if args.use_base:
            v += [float(d.get(k, 0.0)) for k in BASE]
        return np.asarray(v, dtype=np.float32)

    period = 1.0 / args.fps
    try:
        for ep in range(args.episodes):
            print(f"[record] EPISODE {ep+1}/{args.episodes} — recording {args.episode_time_s:.0f}s "
                  f"(Ctrl-C to stop early)", flush=True)
            t_end = time.time() + args.episode_time_s
            n = 0
            while time.time() < t_end:
                t0 = time.time()
                obs = robot.get_observation()
                action = robot.captured_action()
                if action is None:      # joystick hasn't published yet this episode
                    time.sleep(period); continue
                frame = {
                    "observation.images.front": obs["front"],
                    "observation.state": _vec(obs),
                    "action": _vec(action),
                    "task": args.task,
                }
                ds.add_frame(frame)
                n += 1
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
            ds.save_episode()
            print(f"[record] episode {ep+1} saved ({n} frames)", flush=True)
            if ep < args.episodes - 1:
                print(f"[record] RESET — reposition; next episode in {args.reset_time_s:.0f}s", flush=True)
                time.sleep(args.reset_time_s)
    except KeyboardInterrupt:
        print("\n[record] stopped by operator; saving current episode", flush=True)
        try:
            ds.save_episode()
        except Exception:
            pass
    finally:
        robot.disconnect()
    if args.push_to_hub:
        ds.push_to_hub()
        print("[record] pushed to hub", flush=True)
    print(f"[record] done: {ds.num_episodes} episodes, {ds.num_frames} frames at "
          f"{ds.root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
