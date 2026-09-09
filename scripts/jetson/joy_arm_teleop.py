#!/usr/bin/env python3
"""Smooth stick-based arm+base teleop for the ROSMASTER M3 Pro, publishing the
FULL 6-joint target to /arm6_joints (ArmJoints) and /cmd_vel (Twist) — the
topics the passive recorder already captures. Bypasses the vendor button-layout
mapper (which assumes an Xbox layout that this DirectInput pad doesn't match).

Consumes /joy (from js0_joy). Each stick axis integrates one joint target at
`arm_speed` deg/s; a deadman gate (any configured enable input, or --no-deadman)
guards motion. Joint targets are clamped to the vendor limits.

  ros2 run ... OR: python3 joy_arm_teleop.py [--no-deadman] [--arm-speed 40]
Axis/joint map is configurable; defaults follow the common gamepad layout and
can be remapped live via --map "joint:axis:scale,...".
"""
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from arm_msgs.msg import ArmJoints

HOME = [90.0, 150.0, 12.0, 20.0, 90.0, 0.0]
# per-joint (min,max) degrees (vendor: j5 0..270, j6 gripper 30..180)
LIMITS = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270), (30, 180)]
# default: joint index -> (joy axis index, scale). Common layout:
#  LX=0 LY=1 RX=3 RY=4 LT=2 RT=5 ; signs make "up/right" increase the joint.
DEFAULT_MAP = {0: (0, 1.0), 1: (1, -1.0), 2: (4, -1.0), 3: (3, 1.0),
               4: (6, 1.0), 5: (7, 1.0)}


class JoyArmTeleop(Node):
    def __init__(self, args):
        super().__init__("joy_arm_teleop")
        self.args = args
        self.amap = args.amap
        self.target = list(HOME)
        self.axes = []
        self.buttons = []
        self.deadman = not args.no_deadman
        self.enable_btn = args.enable_btn
        self.pub_arm = self.create_publisher(ArmJoints, "/arm6_joints", 10)
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Joy, "/joy", self._on_joy, 10)
        self.dt = 1.0 / args.rate
        self.create_timer(self.dt, self._tick)
        self._last = time.time()
        self.get_logger().info(
            f"joy_arm_teleop: sticks->/arm6_joints @ {args.rate}Hz, "
            f"arm_speed {args.arm_speed} deg/s, deadman={self.deadman} "
            f"(btn {self.enable_btn}); map {self.amap}")

    def _on_joy(self, m):
        self.axes = list(m.axes)
        self.buttons = list(m.buttons)

    def _ax(self, i):
        return self.axes[i] if 0 <= i < len(self.axes) else 0.0

    def _enabled(self):
        if not self.deadman:
            return True
        return (0 <= self.enable_btn < len(self.buttons)
                and self.buttons[self.enable_btn] == 1)

    def _tick(self):
        now = time.time(); dt = now - self._last; self._last = now
        moved = False
        if self._enabled():
            for j, (ax, sc) in self.amap.items():
                v = self._ax(ax) * sc
                if abs(v) < self.args.deadzone:
                    continue
                self.target[j] += v * self.args.arm_speed * dt
                lo, hi = LIMITS[j]
                self.target[j] = min(max(self.target[j], lo), hi)
                moved = True
            # base: right stick or dpad optionally -> cmd_vel (off by default)
            if self.args.base_lx is not None:
                t = Twist()
                t.linear.x = self._dz(self._ax(self.args.base_lx)) * self.args.base_speed
                t.angular.z = self._dz(self._ax(self.args.base_az)) * self.args.base_turn
                self.pub_vel.publish(t)
        # always publish the current target so the board holds it (and the
        # recorder captures a continuous action stream)
        m = ArmJoints()
        for i in range(6):
            setattr(m, f"joint{i+1}", int(round(self.target[i])))
        m.time = int(self.dt * 1000) + 50
        self.pub_arm.publish(m)

    def _dz(self, v):
        return 0.0 if abs(v) < self.args.deadzone else v


def parse_map(s):
    if not s:
        return dict(DEFAULT_MAP)
    out = {}
    for tok in s.split(","):
        j, ax, sc = tok.split(":")
        out[int(j)] = (int(ax), float(sc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--arm-speed", type=float, default=40.0, help="deg/s per full stick")
    ap.add_argument("--deadzone", type=float, default=0.15)
    ap.add_argument("--no-deadman", action="store_true", help="move without holding enable")
    ap.add_argument("--enable-btn", type=int, default=0, help="deadman button index")
    ap.add_argument("--map", dest="map_s", default="", help="joint:axis:scale,...")
    ap.add_argument("--base-lx", type=int, default=None, help="axis for base vx (opt)")
    ap.add_argument("--base-az", type=int, default=None, help="axis for base wz (opt)")
    ap.add_argument("--base-speed", type=float, default=0.15)
    ap.add_argument("--base-turn", type=float, default=0.4)
    a = ap.parse_args()
    a.amap = parse_map(a.map_s)
    rclpy.init()
    n = JoyArmTeleop(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
