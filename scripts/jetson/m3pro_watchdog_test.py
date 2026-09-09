#!/usr/bin/env python3
"""Does the M3 Pro base stop by itself when /cmd_vel goes silent?
Sends vx for 1 s, then stays SILENT and samples /odom_raw velocity every
0.5 s for 4 s, then sends explicit zeros (always). Robot must be on a stand.
"""
import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def main():
    vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    rclpy.init(); n = Node("m3pro_watchdog_test")
    pub = n.create_publisher(Twist, "/cmd_vel", 10)
    vel = {"v": None, "t": None}
    def cb(m):
        vel["v"] = (round(m.twist.twist.linear.x, 3), round(m.twist.twist.linear.y, 3)); vel["t"] = time.time()
    n.create_subscription(Odometry, "/odom_raw", cb, 10)
    def spin(s):
        e = time.time() + s
        while time.time() < e: rclpy.spin_once(n, timeout_sec=0.05)
    try:
        spin(3.0)
        print(f"subscribers on /cmd_vel: {pub.get_subscription_count()}; odom seen: {vel['v'] is not None}")
        t = Twist(); t.linear.x = vx
        t0 = time.time()
        while time.time() - t0 < 1.0:
            pub.publish(t); spin(0.05)
        print(f"t+0.0s (last cmd) odom vel = {vel['v']}  -> now SILENT")
        for k in range(8):
            spin(0.5); print(f"t+{(k+1)*0.5:.1f}s odom vel = {vel['v']}")
    finally:
        for _ in range(10): pub.publish(Twist()); spin(0.05)
        print("explicit zero sent")
        n.destroy_node(); rclpy.try_shutdown()

if __name__ == "__main__":
    main()
