#!/usr/bin/env python3
"""Log the SLAM pose and raw odometry to poses.jsonl for slam/metrics.py.

Two sources, one file, one row per sample:
  frame "map":  TF lookup map -> base frame at --hz (the SLAM estimate)
  frame "odom": every /odom_raw message from the board (raw wheel odometry)

Run on the Orin next to ``slam_session.py start`` (same --out). Read-only:
subscribes and looks up transforms, publishes nothing. If the map frame is
not being published yet, the lookup failures are counted and reported, and
the log still carries odometry.

  python3 scripts/jetson/slam_pose_log.py --out reports/slam/sessions/<name> \
      --map-frame map --base-frame base_footprint
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_footprint")
    ap.add_argument("--odom-topic", default="/odom_raw")
    args = ap.parse_args()

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = (out / "poses.jsonl").open("a", buffering=1)
    counts = {"map": 0, "odom": 0, "tf_fail": 0}

    rclpy.init()
    node = Node("slam_pose_log")
    buf = Buffer()
    TransformListener(buf, node)

    def on_odom(msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        row = {"t": time.time(), "frame": "odom", "x": round(p.x, 4), "y": round(p.y, 4),
               "yaw": round(yaw_from_quaternion(q.x, q.y, q.z, q.w), 5)}
        fh.write(json.dumps(row) + "\n")
        counts["odom"] += 1

    node.create_subscription(Odometry, args.odom_topic, on_odom, 20)

    def on_timer() -> None:
        try:
            tf = buf.lookup_transform(args.map_frame, args.base_frame, Time())
        except Exception:
            counts["tf_fail"] += 1
            return
        tr, q = tf.transform.translation, tf.transform.rotation
        row = {"t": time.time(), "frame": "map", "x": round(tr.x, 4), "y": round(tr.y, 4),
               "yaw": round(yaw_from_quaternion(q.x, q.y, q.z, q.w), 5)}
        fh.write(json.dumps(row) + "\n")
        counts["map"] += 1

    node.create_timer(1.0 / args.hz, on_timer)
    last = time.time()
    print(f"[pose-log] {args.map_frame}->{args.base_frame} at {args.hz} Hz, {args.odom_topic} "
          f"-> {out / 'poses.jsonl'}", flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - last >= 10:
                last = time.time()
                print(f"[pose-log] map {counts['map']}  odom {counts['odom']}  "
                      f"tf_fail {counts['tf_fail']}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        print(f"[pose-log] final: {counts}", flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
