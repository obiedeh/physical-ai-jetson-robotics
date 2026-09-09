#!/usr/bin/env python3
"""Publish sensor_msgs/Joy from /dev/input/js0 using the legacy Linux joystick
API directly — no SDL, no display, no extra ROS package. Written because the
ROS `joy` (SDL) driver enumerates no device headless on this Orin, while raw
js0 streams fine. The vendor `yahboom_joy_M3Pro` node maps /joy -> /arm6_joints.

  source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=30
  python3 scripts/jetson/js0_joy.py --dev /dev/input/js0 --rate 20
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

_EV = struct.Struct("IhBB")   # time(u32), value(i16), type(u8), number(u8)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80


class Js0Joy(Node):
    def __init__(self, dev: str, rate: float):
        super().__init__("js0_joy")
        self.dev = dev
        self.pub = self.create_publisher(Joy, "/joy", 10)
        self.axes: list[float] = []
        self.buttons: list[int] = []
        self._lock = threading.Lock()
        self._stop = False
        self._fd = os.open(dev, os.O_RDONLY)
        threading.Thread(target=self._reader, daemon=True).start()
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f"js0_joy reading {dev}, publishing /joy at {rate} Hz")

    def _grow(self, lst, idx, fill):
        while len(lst) <= idx:
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
            base = etype & ~JS_EVENT_INIT
            with self._lock:
                if base == JS_EVENT_AXIS:
                    self._grow(self.axes, number, 0.0)
                    self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
                elif base == JS_EVENT_BUTTON:
                    self._grow(self.buttons, number, 0)
                    self.buttons[number] = int(value)

    def _publish(self):
        m = Joy()
        m.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            m.axes = [float(a) for a in self.axes]
            m.buttons = [int(b) for b in self.buttons]
        self.pub.publish(m)

    def destroy_node(self):
        self._stop = True
        try:
            os.close(self._fd)
        except OSError:
            pass
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/input/js0")
    ap.add_argument("--rate", type=float, default=20.0)
    args = ap.parse_args()
    rclpy.init()
    node = Js0Joy(args.dev, args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
