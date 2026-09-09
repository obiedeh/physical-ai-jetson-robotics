"""Lerobot-free hardware layer for the Yahboom ROSMASTER M3 Pro board.

This is the single source of truth for the board contract and the safety
floor. It depends only on ROS 2 (rclpy + the vendor arm_msgs) and numpy —
NOT on lerobot — so it runs on the Orin's Python 3.10 (where lerobot 0.6.2,
py>=3.12, cannot be installed) inside the ZMQ host, and is also reused by the
direct lerobot Robot class where a py3.12+ROS environment exists.

Board contract (measured 2026-08-20, see docs/notes/yahboom_strategy):
  publish  arm_msgs/ArmJoints /arm6_joints  (joint1..6 int16 deg + time ms)
  publish  geometry_msgs/Twist /cmd_vel      (NO firmware watchdog)
  subscribe nav_msgs/Odometry /odom_raw, std_msgs/Float32 /battery
  the board publishes no joint feedback -> arm state is the last command.
"""
from __future__ import annotations

import math
import os
import threading
import time

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
VENDOR_HOME = (90.0, 150.0, 12.0, 20.0, 90.0, 90.0)  # servo6 gripper mid (30-180 range; 0 stalls the servo)   # servo 6 = gripper


class M3ProHardware:
    """Board I/O + safety. No lerobot, no cameras (the caller owns cameras).

    Safety guarantees, unconditional:
      * every arm target is clamped to joint_limits and (vs the last command)
        to max_relative_target;
      * base velocity is clamped to base_max_linear / base_max_angular;
      * a deadman thread zeros the base when no fresh command arrives within
        deadman_timeout_s (the base firmware has no watchdog);
      * zeros are sent on close() and on any send exception.
    """

    def __init__(self, *, ros_domain_id: int = 30, fastdds_profile: str | None = None,
                 arm_topic: str = "/arm6_joints", cmd_vel_topic: str = "/cmd_vel",
                 odom_topic: str = "/odom_raw", battery_topic: str = "/battery",
                 lidar_topics: tuple = ("/scan0", "/scan1"),
                 arm_run_time_ms: int = 150, home_run_time_ms: int = 3000,
                 max_relative_target: float | dict | None = 10.0,
                 joint_limits: dict | None = None,
                 home_pose: tuple = VENDOR_HOME, use_base: bool = True,
                 base_max_linear: float = 0.20, base_max_angular: float = 0.60,
                 deadman_timeout_s: float = 0.5, preflight_timeout_s: float = 8.0,
                 passive: bool = False):
        # passive: the vendor joystick owns the board; we only SUBSCRIBE to
        # /arm6_joints + /cmd_vel to capture the commanded action for recording.
        # No publishing, no homing, no deadman. read_action() returns the last
        # joystick command.
        self.passive = passive
        self.ros_domain_id = ros_domain_id
        self.fastdds_profile = fastdds_profile
        self.arm_topic, self.cmd_vel_topic = arm_topic, cmd_vel_topic
        self.arm_joint_topic = "/arm_joint"   # vendor single-joint teleop topic
        self.odom_topic, self.battery_topic = odom_topic, battery_topic
        self.lidar_topics = tuple(lidar_topics)
        self.arm_run_time_ms, self.home_run_time_ms = arm_run_time_ms, home_run_time_ms
        self.max_relative_target = max_relative_target
        self.joint_limits = joint_limits or {j: (0.0, 180.0) for j in ARM_JOINTS}
        self.home_pose = tuple(home_pose)
        self.use_base = use_base
        self.base_max_linear, self.base_max_angular = base_max_linear, base_max_angular
        self.deadman_timeout_s = deadman_timeout_s
        self.preflight_timeout_s = preflight_timeout_s

        self._node = None
        self._executor = None
        self._spin_thread = None
        self._deadman_thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._commanded: dict | None = None
        self._base_cmd = (0.0, 0.0, 0.0)
        self._last_cmd_t = 0.0
        self._odom = {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._battery = float("nan")
        self._lidar_min = {}   # per-topic nearest valid range (m)
        self._connected = False
        self._joy_arm: dict | None = (dict(zip(ARM_JOINTS, self.home_pose))
                                      if passive else None)  # passive: running joint target
        self._joy_base = (0.0, 0.0, 0.0)     # passive: last /cmd_vel command
        self._joy_seen = False

    # ---------------------------------------------------------------- lifecycle
    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, *, go_home: bool = True) -> None:
        os.environ["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        if self.fastdds_profile:
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = self.fastdds_profile
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Float32
            from arm_msgs.msg import ArmJoints
        except ImportError as e:
            raise ImportError(
                "ROS 2 Humble + vendor arm_msgs required: source /opt/ros/humble/"
                "setup.bash and /home/jetson/yahboomcar_ws/install/setup.bash "
                f"first ({e})") from e
        self._Twist, self._ArmJoints = Twist, ArmJoints
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node(f"m3pro_hw_{os.getpid()}")
        self._pub_arm = self._node.create_publisher(ArmJoints, self.arm_topic, 10)
        self._pub_vel = self._node.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._node.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        self._node.create_subscription(Float32, self.battery_topic, self._on_battery, 10)
        try:
            from sensor_msgs.msg import LaserScan
            for i, topic in enumerate(self.lidar_topics):
                self._node.create_subscription(
                    LaserScan, topic,
                    (lambda m, t=topic: self._on_scan(t, m)), 10)
        except ImportError:
            pass  # sensor_msgs absent: min_obstacle_m stays None
        if self.passive:
            # capture the vendor joystick's commands to the board as the action:
            # /arm6_joints (full 6) AND /arm_joint (single-joint increments, the
            # topic the vendor yahboom_joy_M3Pro mapper actually publishes).
            self._node.create_subscription(ArmJoints, self.arm_topic, self._on_joy_arm, 10)
            self._node.create_subscription(Twist, self.cmd_vel_topic, self._on_joy_base, 10)
            try:
                from arm_msgs.msg import ArmJoint as _AJ
            except ImportError:
                from arm_interface.msg import ArmJoint as _AJ  # type: ignore
            self._node.create_subscription(_AJ, self.arm_joint_topic, self._on_arm_joint, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop.clear()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        t0 = time.time()
        while time.time() - t0 < self.preflight_timeout_s and \
                self._pub_arm.get_subscription_count() == 0:
            time.sleep(0.1)
        if self._pub_arm.get_subscription_count() == 0:
            self.close()
            raise ConnectionError(
                f"board not visible on {self.arm_topic} within "
                f"{self.preflight_timeout_s}s (agent down? wrong domain? "
                "non-vendor user needs fastdds_profile?)")
        self._connected = True
        if self.passive:
            # joystick owns the board: no deadman, no homing, no publishing
            return
        if self.use_base:
            self._deadman_thread = threading.Thread(target=self._deadman, daemon=True)
            self._deadman_thread.start()
        if go_home:
            self.send_arm(dict(zip(ARM_JOINTS, self.home_pose)),
                          run_time_ms=self.home_run_time_ms, clamp=False)
            time.sleep(self.home_run_time_ms / 1000.0 + 0.3)

    def close(self) -> None:
        self._stop.set()
        try:
            if not self.passive and self.use_base and getattr(self, "_pub_vel", None) is not None:
                self.zero_base(n=5)
        finally:
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=1.0)
            try:
                if self._executor is not None and self._node is not None:
                    self._executor.remove_node(self._node)
                if self._node is not None:
                    self._node.destroy_node()
            except Exception:
                pass
            self._node = self._executor = None
            self._connected = False

    # ---------------------------------------------------------------- state / cmd
    def read_state(self) -> dict:
        with self._lock:
            # passive: the joystick's last arm command is the best state proxy
            # (the board publishes no joint feedback); active: our last command.
            cmd = dict(self._joy_arm) if self.passive else (
                dict(self._commanded) if self._commanded else None)
            odom = dict(self._odom)
            batt = self._battery
        state = {f"{j}.pos": (float(cmd[j]) if cmd else float("nan")) for j in ARM_JOINTS}
        if self.use_base:
            state.update({"base.odom_x": odom["x"], "base.odom_y": odom["y"],
                          "base.odom_yaw": odom["yaw"], "base.vx": odom["vx"],
                          "base.vy": odom["vy"], "base.wz": odom["wz"]})
        state["battery.v"] = batt
        with self._lock:
            mins = [v for v in self._lidar_min.values() if v is not None]
        state["min_obstacle_m"] = (min(mins) if mins else None)
        return state

    def read_action(self) -> dict | None:
        """Passive mode: the last vendor-joystick command (arm degrees + base
        velocities), as a flat action dict. None until the first command seen."""
        with self._lock:
            if not self._joy_seen or self._joy_arm is None:
                return None
            arm = dict(self._joy_arm)
            vx, vy, wz = self._joy_base
        act = {f"{j}.pos": float(arm[j]) for j in ARM_JOINTS}
        if self.use_base:
            act.update({"base.vx": float(vx), "base.vy": float(vy), "base.wz": float(wz)})
        return act

    def apply_action(self, action: dict) -> dict:
        """Clamp + publish arm and/or base from a flat action dict; return the
        actually-sent values. Any exception zeros the base first. In passive
        mode this is a no-op (the joystick owns the board)."""
        if self.passive:
            return {}
        applied: dict = {}
        try:
            goal = {k[:-4]: float(v) for k, v in action.items()
                    if k.endswith(".pos") and k[:-4] in ARM_JOINTS}
            if goal:
                with self._lock:
                    prev = dict(self._commanded) if self._commanded else None
                full = dict(prev) if prev else dict(zip(ARM_JOINTS, self.home_pose))
                for j, v in goal.items():
                    lo, hi = self.joint_limits.get(j, (0.0, 180.0))
                    v = min(max(v, lo), hi)
                    if prev is not None and self.max_relative_target is not None:
                        lim = (self.max_relative_target
                               if isinstance(self.max_relative_target, (int, float))
                               else self.max_relative_target.get(j, math.inf))
                        v = min(max(v, prev[j] - lim), prev[j] + lim)
                    full[j] = v
                self.send_arm(full, run_time_ms=self.arm_run_time_ms, clamp=False)
                applied.update({f"{j}.pos": full[j] for j in ARM_JOINTS})
            if self.use_base:
                vx = max(-self.base_max_linear, min(self.base_max_linear, float(action.get("base.vx", 0.0))))
                vy = max(-self.base_max_linear, min(self.base_max_linear, float(action.get("base.vy", 0.0))))
                wz = max(-self.base_max_angular, min(self.base_max_angular, float(action.get("base.wz", 0.0))))
                self.send_base(vx, vy, wz)
                applied.update({"base.vx": vx, "base.vy": vy, "base.wz": wz})
            self._last_cmd_t = time.time()
            return applied
        except Exception:
            if self.use_base:
                self.zero_base(n=3)
            raise

    def send_arm(self, joints_deg: dict, *, run_time_ms: int, clamp: bool = True) -> None:
        if clamp:
            clamped = {}
            for j in ARM_JOINTS:
                lo, hi = self.joint_limits.get(j, (0.0, 180.0))
                clamped[j] = min(max(float(joints_deg[j]), lo), hi)
            joints_deg = clamped
        m = self._ArmJoints()
        for i, j in enumerate(ARM_JOINTS, start=1):
            setattr(m, f"joint{i}", int(round(joints_deg[j])))
        m.time = int(run_time_ms)
        self._pub_arm.publish(m)
        with self._lock:
            self._commanded = {j: float(int(round(joints_deg[j]))) for j in ARM_JOINTS}

    def send_base(self, vx: float, vy: float, wz: float) -> None:
        t = self._Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self._pub_vel.publish(t)
        self._base_cmd = (vx, vy, wz)
        self._last_cmd_t = time.time()

    def zero_base(self, n: int = 3) -> None:
        if getattr(self, "_pub_vel", None) is None:
            return
        for _ in range(n):
            try:
                t = self._Twist(); self._pub_vel.publish(t)
                self._base_cmd = (0.0, 0.0, 0.0)
            except Exception:
                return
            time.sleep(0.02)

    # ---------------------------------------------------------------- internals
    def _deadman(self) -> None:
        period = min(0.1, self.deadman_timeout_s / 3)
        while not self._stop.is_set():
            time.sleep(period)
            if self._base_cmd != (0.0, 0.0, 0.0) and \
                    time.time() - self._last_cmd_t > self.deadman_timeout_s:
                self.zero_base(n=3)

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                if self._stop.is_set():
                    break
                raise

    def _on_odom(self, msg) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._odom = {"x": msg.pose.pose.position.x, "y": msg.pose.pose.position.y,
                          "yaw": yaw, "vx": msg.twist.twist.linear.x,
                          "vy": msg.twist.twist.linear.y, "wz": msg.twist.twist.angular.z}

    def _on_battery(self, msg) -> None:
        with self._lock:
            self._battery = float(msg.data)

    def _on_scan(self, topic, msg) -> None:
        import math as _m
        best = None
        rmin, rmax = msg.range_min, msg.range_max
        for r in msg.ranges:
            if r is None or _m.isinf(r) or _m.isnan(r):
                continue
            if rmin <= r <= rmax and (best is None or r < best):
                best = r
        with self._lock:
            self._lidar_min[topic] = best

    def _on_joy_arm(self, msg) -> None:  # passive: /arm6_joints from the joystick
        with self._lock:
            self._joy_arm = {f"joint{i}": float(getattr(msg, f"joint{i}"))
                             for i in range(1, 7)}
            self._joy_seen = True

    def _on_arm_joint(self, msg) -> None:  # passive: /arm_joint single-joint cmd
        # msg.id in 1..6, msg.angle in degrees; maintain the running 6-vector
        jid = int(getattr(msg, "id", 0))
        ang = float(getattr(msg, "angle", getattr(msg, "joint", 0.0)))
        if 1 <= jid <= 6:
            with self._lock:
                if self._joy_arm is None:
                    self._joy_arm = dict(zip(ARM_JOINTS, self.home_pose))
                self._joy_arm[f"joint{jid}"] = ang
                self._joy_seen = True

    def _on_joy_base(self, msg) -> None:  # passive: /cmd_vel from the joystick
        with self._lock:
            self._joy_base = (msg.linear.x, msg.linear.y, msg.angular.z)
            self._joy_seen = True
