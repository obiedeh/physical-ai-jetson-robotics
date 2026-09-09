#!/usr/bin/env python3
"""ONE-process joystick teleop for the ROSMASTER M3 Pro: reads /dev/input/js0
directly (legacy joystick API, no SDL) AND integrates the sticks into the full
6-joint target, publishing /arm6_joints (ArmJoints) — no inter-process /joy
hop (which FastDDS fails to deliver between two systemd services). Also
publishes /joy for anything else that wants it. The micro-ROS agent (a
separate, non-systemd process) delivers /arm6_joints to the board fine.

  python3 js0_to_arm.py --dev /dev/input/js0 --arm-speed 25 \
      --map 0:4:1,1:5:-1,2:0:1,3:1:-1,4:6:1,5:7:1 [--no-deadman] [--enable-btn N]
"""
from __future__ import annotations

import argparse
import os
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from arm_msgs.msg import ArmJoints

_EV = struct.Struct("IhBB")
JS_BUTTON, JS_AXIS, JS_INIT = 0x01, 0x02, 0x80
HOME = [90.0, 150.0, 12.0, 20.0, 90.0, 0.0]
LIMITS = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270), (30, 180)]
DEFAULT_MAP = {0: (4, 1.0), 1: (5, -1.0), 2: (0, 1.0), 3: (1, -1.0), 4: (6, 1.0), 5: (7, 1.0)}


class Js0ToArm(Node):
    def __init__(self, a):
        super().__init__("js0_to_arm")
        self.a = a
        self.amap = a.amap
        self.deadman = not a.no_deadman
        self.enable_btn = a.enable_btn
        self.target = list(HOME)
        self.axes: list[float] = []
        self.buttons: list[int] = []
        self._lock = threading.Lock()
        self._stop = False
        self._fd = os.open(a.dev, os.O_RDONLY)
        self.pub_joy = self.create_publisher(Joy, "/joy", 10)
        self.pub_arm = self.create_publisher(ArmJoints, "/arm6_joints", 10)
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.dt = 1.0 / a.rate
        self._last = time.time()
        threading.Thread(target=self._reader, daemon=True).start()
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"js0_to_arm: {a.dev} -> /arm6_joints @ {a.rate}Hz speed {a.arm_speed} "
            f"deadman={self.deadman}(btn {self.enable_btn}) map {self.amap}")

    def _grow(self, lst, i, fill):
        while len(lst) <= i:
            lst.append(fill)

    def _reader(self):
        while not self._stop:
            try:
                buf = os.read(self._fd, _EV.size)
            except OSError:
                time.sleep(0.1); continue
            if len(buf) != _EV.size:
                continue
            _t, value, etype, number = _EV.unpack(buf)
            base = etype & ~JS_INIT
            with self._lock:
                if base == JS_AXIS:
                    self._grow(self.axes, number, 0.0)
                    self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
                elif base == JS_BUTTON:
                    self._grow(self.buttons, number, 0)
                    self.buttons[number] = int(value)

    def _ax(self, i):
        return self.axes[i] if 0 <= i < len(self.axes) else 0.0

    def _enabled(self):
        if not self.deadman:
            return True
        return 0 <= self.enable_btn < len(self.buttons) and self.buttons[self.enable_btn] == 1

    def _tick(self):
        now = time.time(); dt = now - self._last; self._last = now
        with self._lock:
            axes = list(self.axes); buttons = list(self.buttons)
        # /joy passthrough
        jm = Joy(); jm.header.stamp = self.get_clock().now().to_msg()
        jm.axes = [float(x) for x in axes]; jm.buttons = [int(b) for b in buttons]
        self.pub_joy.publish(jm)
        # integrate
        if self._enabled():
            for j, (ax, sc) in self.amap.items():
                v = (axes[ax] if ax < len(axes) else 0.0) * sc
                if abs(v) < self.a.deadzone:
                    continue
                self.target[j] += v * self.a.arm_speed * dt
                lo, hi = LIMITS[j]
                self.target[j] = min(max(self.target[j], lo), hi)
            if self.a.base_vx_ax is not None:
                t = Twist()
                bx = axes[self.a.base_vx_ax] if self.a.base_vx_ax < len(axes) else 0.0
                bz = axes[self.a.base_wz_ax] if self.a.base_wz_ax < len(axes) else 0.0
                t.linear.x = (0.0 if abs(bx) < self.a.deadzone else bx) * self.a.base_speed
                t.angular.z = (0.0 if abs(bz) < self.a.deadzone else bz) * self.a.base_turn
                self.pub_vel.publish(t)
        m = ArmJoints()
        for i in range(6):
            setattr(m, f"joint{i+1}", int(round(self.target[i])))
        m.time = int(self.dt * 1000) + 50
        self.pub_arm.publish(m)

    def destroy_node(self):
        self._stop = True
        try:
            os.close(self._fd)
        except OSError:
            pass
        super().destroy_node()


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
    ap.add_argument("--dev", default="/dev/input/js0")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--arm-speed", type=float, default=25.0)
    ap.add_argument("--deadzone", type=float, default=0.15)
    ap.add_argument("--no-deadman", action="store_true")
    ap.add_argument("--enable-btn", type=int, default=0)
    ap.add_argument("--map", dest="map_s", default="")
    ap.add_argument("--base-vx-ax", type=int, default=None)
    ap.add_argument("--base-wz-ax", type=int, default=None)
    ap.add_argument("--base-speed", type=float, default=0.15)
    ap.add_argument("--base-turn", type=float, default=0.4)
    a = ap.parse_args(); a.amap = parse_map(a.map_s)
    rclpy.init(); n = Js0ToArm(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
