"""ROSMASTER M3 Pro ZMQ host — runs ON THE ORIN (Python 3.10, no lerobot).

Owns the board via M3ProHardware (safety floor: clamps, deadman, zero-on-
exit) and the USB cameras via cv2, and serves the LeKiwi-style wire so a
LeRobot client on the 5090 (Python 3.12) can record/eval:

  cmd  socket  PULL  + CONFLATE  on :5555   <- JSON action dict from client
  obs  socket  PUSH  + SNDHWM 2  on :5556   -> [JSON header, jpeg0, jpeg1...]

The host applies its OWN 500 ms watchdog (zeros the base if the client goes
quiet) on top of the hardware deadman — defence in depth for a base with no
firmware watchdog.

Run (on the robot):
  source /opt/ros/humble/setup.bash
  source /home/jetson/yahboomcar_ws/install/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=.../config/fastdds_no_shm.xml   # non-vendor user
  python3 -m lerobot_robot_rosmaster_m3pro.m3pro_host --cameras 0
"""
from __future__ import annotations

import argparse
import json
import time

from .hardware import ARM_JOINTS, M3ProHardware


def _open_cameras(specs: list[tuple[str, int, int, int, int]]):
    """Open each camera; a camera that fails to open is WARNED and skipped
    (never fatal) so the arm/state loop still serves. A bounded open timeout
    avoids the indefinite V4L2 hang seen when a prior host was hard-killed
    and left /dev/videoN locked."""
    import cv2
    cams = {}
    for name, index, w, h, fps in specs:
        print(f"[host] opening camera '{name}' (/dev/video{index})...", flush=True)
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        try:  # OpenCV >= 4.5 backend timeout props (best-effort)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok = cap.isOpened() and cap.read()[0]
        if not ok:
            print(f"[host] WARNING: camera '{name}' (/dev/video{index}) failed to "
                  "open/read — serving WITHOUT it (fuser /dev/video{index}?)", flush=True)
            cap.release()
            continue
        cams[name] = cap
    return cams


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd-port", type=int, default=5555)
    ap.add_argument("--obs-port", type=int, default=5556)
    ap.add_argument("--domain", type=int, default=30)
    ap.add_argument("--profile", default=None, help="FastDDS XML for a non-vendor user")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--no-home", action="store_true", help="do not move to home on connect")
    ap.add_argument("--passive", action="store_true",
                    help="joystick-observe mode: the vendor joystick drives the "
                         "board; the host only captures its /arm6_joints + /cmd_vel "
                         "commands as the action (for recording) and never publishes")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--watchdog-ms", type=int, default=500)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--cameras", default="", help="comma list, e.g. 'front:0:640:480:15,wrist:2:640:480:15' or bare '0'")
    args = ap.parse_args()

    specs = []
    for tok in filter(None, (t.strip() for t in args.cameras.split(","))):
        parts = tok.split(":")
        if len(parts) == 1:
            specs.append(("front", int(parts[0]), 640, 480, 15))
        else:
            name, idx = parts[0], int(parts[1])
            w = int(parts[2]) if len(parts) > 2 else 640
            h = int(parts[3]) if len(parts) > 3 else 480
            fps = int(parts[4]) if len(parts) > 4 else 15
            specs.append((name, idx, w, h, fps))

    import cv2
    import zmq

    hw = M3ProHardware(ros_domain_id=args.domain, fastdds_profile=args.profile,
                       use_base=not args.no_base, passive=args.passive)
    # Board FIRST (so a camera problem never blocks state serving), cameras after.
    mode = "PASSIVE (joystick drives; we observe)" if args.passive else "active"
    print(f"[host] connecting board (domain {args.domain}, base={not args.no_base}, {mode})...", flush=True)
    hw.connect(go_home=not args.no_home and not args.passive)
    print(f"[host] board up; battery {hw.read_state()['battery.v']:.2f} V", flush=True)
    cams = _open_cameras(specs)
    print(f"[host] cameras serving: {list(cams)}", flush=True)

    ctx = zmq.Context()
    cmd = ctx.socket(zmq.PULL); cmd.setsockopt(zmq.CONFLATE, 1); cmd.bind(f"tcp://*:{args.cmd_port}")
    obs = ctx.socket(zmq.PUSH); obs.setsockopt(zmq.SNDHWM, 2); obs.bind(f"tcp://*:{args.obs_port}")
    print(f"[host] serving cmd :{args.cmd_port}  obs :{args.obs_port}", flush=True)

    period = 1.0 / args.hz
    last_cmd_t = time.time()
    watchdog_active = False
    try:
        while True:
            t0 = time.time()
            try:
                msg = cmd.recv_string(zmq.NOBLOCK)
                action = json.loads(msg)
                hw.apply_action(action)
                last_cmd_t = time.time(); watchdog_active = False
            except zmq.Again:
                if not watchdog_active and (time.time() - last_cmd_t) > args.watchdog_ms / 1000.0:
                    if not args.no_base:
                        hw.zero_base(n=2)
                    watchdog_active = True
            state = hw.read_state()
            header = {"state": state, "cams": [], "t": t0}
            if args.passive:
                header["action"] = hw.read_action()   # joystick command, or None
            frames = []
            for name, cap in cams.items():
                ok, fr = cap.read()
                if not ok:
                    continue
                ok2, jpg = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                if not ok2:
                    continue
                header["cams"].append({"name": name, "h": fr.shape[0], "w": fr.shape[1]})
                frames.append(jpg.tobytes())
            try:
                obs.send_multipart([json.dumps(header).encode()] + frames, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[host] shutting down", flush=True)
    finally:
        try:
            hw.close()   # zeros the base
        finally:
            for cap in cams.values():
                cap.release()
            cmd.close(); obs.close(); ctx.term()
            print("[host] closed (base zeroed)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
