#!/usr/bin/env python3
"""Deterministic one-shot publisher for the ROSMASTER M3 Pro board topics
(the ros2 CLI's `topic pub --once` hangs on shutdown against micro-ROS
subscribers). Publishes N messages at a rate and exits. ROS_DOMAIN_ID=30.

  m3pro_pub.py beep 100            # buzzer value (vendor UInt16), then 0
  m3pro_pub.py rgb 0 0 255         # ColorRGBA r g b (0-255), then off
  m3pro_pub.py base 0.10 0 0 1.0   # cmd_vel vx vy wz for SECONDS, then zero
  m3pro_pub.py arm 90 150 12 20 90 0 3000   # ArmJoints 6 deg + run_time ms
"""
import sys, time
import rclpy
from rclpy.node import Node


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); return 2
    kind = args[0]
    rclpy.init()
    node = Node("m3pro_pub")
    try:
        if kind == "beep":
            from std_msgs.msg import UInt16
            pub = node.create_publisher(UInt16, "/beep", 10)
            _settle(node, pub)
            pub.publish(UInt16(data=int(args[1]))); _spin(node, 0.6)
            pub.publish(UInt16(data=0)); _spin(node, 0.3)
        elif kind == "rgb":
            from std_msgs.msg import ColorRGBA
            pub = node.create_publisher(ColorRGBA, "/rgb", 10)
            _settle(node, pub)
            r, g, b = (float(x) for x in args[1:4])
            pub.publish(ColorRGBA(r=r, g=g, b=b, a=0.0)); _spin(node, 1.5)
            pub.publish(ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.0)); _spin(node, 0.3)
        elif kind == "base":
            from geometry_msgs.msg import Twist
            pub = node.create_publisher(Twist, "/cmd_vel", 10)
            _settle(node, pub)
            vx, vy, wz, secs = (float(x) for x in args[1:5])
            t = Twist(); t.linear.x = vx; t.linear.y = vy; t.angular.z = wz
            t0 = time.time()
            while time.time() - t0 < secs:
                pub.publish(t); _spin(node, 0.05)
            for _ in range(5):
                pub.publish(Twist()); _spin(node, 0.05)
        elif kind == "arm":
            from arm_msgs.msg import ArmJoints
            pub = node.create_publisher(ArmJoints, "/arm6_joints", 10)
            _settle(node, pub)
            j = [int(x) for x in args[1:7]]; rt = int(args[7]) if len(args) > 7 else 3000
            m = ArmJoints(joint1=j[0], joint2=j[1], joint3=j[2], joint4=j[3],
                          joint5=j[4], joint6=j[5], time=rt)
            pub.publish(m); _spin(node, 0.5)
        else:
            print("unknown kind", kind); return 2
        print(f"[m3pro_pub] {kind} sent; subscribers matched: "
              f"{pub.get_subscription_count()}")
        return 0
    finally:
        node.destroy_node(); rclpy.try_shutdown()


def _spin(node, secs):
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def _settle(node, pub, wait=4.0):
    end = time.time() + wait
    while time.time() < end and pub.get_subscription_count() == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print("[m3pro_pub] WARNING: no subscriber matched (board not connected?)")


if __name__ == "__main__":
    sys.exit(main())
