"""Yahboom ROSMASTER M3 Pro as a LeRobot Robot (LeRobot >= 0.6.2).

Contract (measured on the robot, 2026-08-20):
  publish  arm_msgs/ArmJoints on /arm6_joints  (joint1..6 int16 degrees + time ms)
  publish  geometry_msgs/Twist on /cmd_vel      (NO firmware watchdog -> deadman here)
  subscribe nav_msgs/Odometry /odom_raw, std_msgs/Float32 /battery
  the board (/YB_Node) publishes no joint angles -> arm state is the last command.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import ARM_JOINTS, RosmasterM3ProConfig

logger = logging.getLogger(__name__)

BASE_ACTIONS = ("base.vx", "base.vy", "base.wz")


class RosmasterM3Pro(Robot):
    config_class = RosmasterM3ProConfig
    name = "rosmaster_m3pro"

    def __init__(self, config: RosmasterM3ProConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._deadman_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._commanded: dict[str, float] | None = None     # open-loop arm state (deg)
        self._base_cmd = (0.0, 0.0, 0.0)
        self._last_action_t = 0.0
        self._odom = {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._battery = float("nan")
        self._connected = False

    # ------------------------------------------------------------------ features
    @property
    def _arm_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in ARM_JOINTS}

    @property
    def _base_obs_ft(self) -> dict[str, type]:
        if not self.config.use_base:
            return {}
        return {"base.odom_x": float, "base.odom_y": float, "base.odom_yaw": float,
                "base.vx": float, "base.vy": float, "base.wz": float}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
                for cam in self.config.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._arm_ft, **self._base_obs_ft, "battery.v": float, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        ft = dict(self._arm_ft)
        if self.config.use_base:
            ft.update({k: float for k in BASE_ACTIONS})
        return ft

    # ------------------------------------------------------------------ lifecycle
    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True          # bus servos are factory-calibrated via the vendor tool

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        cfg = self.config
        os.environ["ROS_DOMAIN_ID"] = str(cfg.ros_domain_id)
        if cfg.fastdds_profile:
            os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = cfg.fastdds_profile
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Float32
            from arm_msgs.msg import ArmJoints
        except ImportError as e:  # pragma: no cover - environment error
            raise ImportError(
                "ROS 2 Humble + the vendor arm_msgs are required: `source "
                "/opt/ros/humble/setup.bash && source /home/jetson/yahboomcar_ws/"
                f"install/setup.bash` before running ({e})") from e
        self._Twist, self._ArmJoints = Twist, ArmJoints
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node(f"lerobot_{self.name}_{os.getpid()}")
        self._pub_arm = self._node.create_publisher(ArmJoints, cfg.arm_topic, 10)
        self._pub_vel = self._node.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self._node.create_subscription(Odometry, cfg.odom_topic, self._on_odom, 10)
        self._node.create_subscription(Float32, cfg.battery_topic, self._on_battery, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop.clear()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # preflight: the board (/YB_Node) must have matched our arm publisher
        t0 = time.time()
        while time.time() - t0 < cfg.preflight_timeout_s and self._pub_arm.get_subscription_count() == 0:
            time.sleep(0.1)
        if self._pub_arm.get_subscription_count() == 0:
            self._teardown_ros()
            raise ConnectionError(
                f"board not visible on {cfg.arm_topic} within {cfg.preflight_timeout_s}s "
                "(micro-ROS agent down? wrong ROS_DOMAIN_ID? need fastdds_profile for "
                "a non-vendor user?)")
        logger.info("%s: board matched on %s; battery %.2f V", self, cfg.arm_topic, self._battery)

        for cam in self.cameras.values():
            cam.connect()
        self._connected = True
        if cfg.use_base:
            self._deadman_thread = threading.Thread(target=self._deadman, daemon=True)
            self._deadman_thread.start()
        if cfg.go_home_on_connect:
            self._send_arm(dict(zip(ARM_JOINTS, cfg.home_pose)), cfg.home_run_time_ms)
            time.sleep(cfg.home_run_time_ms / 1000.0 + 0.3)
        else:
            logger.warning("%s: go_home_on_connect=False — arm state is UNKNOWN until the "
                           "first action (observation will carry NaN)", self)

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop.set()
        try:
            if self.config.use_base:
                self._zero_base(n=5)
        finally:
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            self._teardown_ros()
            self._connected = False
            logger.info("%s disconnected (base zeroed)", self)

    # ------------------------------------------------------------------ I/O
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict[str, Any] = {}
        with self._lock:
            cmd = self._commanded
            odom = dict(self._odom)
            batt = self._battery
        for j in ARM_JOINTS:
            obs[f"{j}.pos"] = float(cmd[j]) if cmd else float("nan")
        if self.config.use_base:
            obs.update({"base.odom_x": odom["x"], "base.odom_y": odom["y"],
                        "base.odom_yaw": odom["yaw"], "base.vx": odom["vx"],
                        "base.vy": odom["vy"], "base.wz": odom["wz"]})
        obs["battery.v"] = batt
        for name, cam in self.cameras.items():
            obs[name] = cam.async_read()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Clamp to joint limits and max_relative_target, publish, and return
        what was actually sent (so the dataset records the real command)."""
        cfg = self.config
        applied: dict[str, float] = {}
        try:
            goal = {k.removesuffix(".pos"): float(v) for k, v in action.items()
                    if k.endswith(".pos") and k.removesuffix(".pos") in ARM_JOINTS}
            if goal:
                with self._lock:
                    prev = dict(self._commanded) if self._commanded else None
                full = dict(prev) if prev else dict(zip(ARM_JOINTS, cfg.home_pose))
                for j, v in goal.items():
                    lo, hi = cfg.joint_limits.get(j, (0.0, 180.0))
                    v = min(max(v, lo), hi)
                    if prev is not None and cfg.max_relative_target is not None:
                        lim = (cfg.max_relative_target if isinstance(cfg.max_relative_target, (int, float))
                               else cfg.max_relative_target.get(j, math.inf))
                        v = min(max(v, prev[j] - lim), prev[j] + lim)
                    full[j] = v
                self._send_arm(full, cfg.arm_run_time_ms)
                applied.update({f"{j}.pos": full[j] for j in ARM_JOINTS})
            if cfg.use_base:
                vx = float(action.get("base.vx", 0.0)); vy = float(action.get("base.vy", 0.0))
                wz = float(action.get("base.wz", 0.0))
                vx = max(-cfg.base_max_linear, min(cfg.base_max_linear, vx))
                vy = max(-cfg.base_max_linear, min(cfg.base_max_linear, vy))
                wz = max(-cfg.base_max_angular, min(cfg.base_max_angular, wz))
                self._send_base(vx, vy, wz)
                applied.update({"base.vx": vx, "base.vy": vy, "base.wz": wz})
            self._last_action_t = time.time()
            return applied
        except Exception:
            if cfg.use_base:
                self._zero_base(n=3)
            raise

    # ------------------------------------------------------------------ internals
    def _send_arm(self, joints_deg: dict[str, float], run_time_ms: int) -> None:
        m = self._ArmJoints()
        for i, j in enumerate(ARM_JOINTS, start=1):
            setattr(m, f"joint{i}", int(round(joints_deg[j])))
        m.time = int(run_time_ms)
        self._pub_arm.publish(m)
        with self._lock:
            self._commanded = {j: float(int(round(joints_deg[j]))) for j in ARM_JOINTS}

    def _send_base(self, vx: float, vy: float, wz: float) -> None:
        t = self._Twist()
        t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        self._pub_vel.publish(t)
        self._base_cmd = (vx, vy, wz)

    def _zero_base(self, n: int = 3) -> None:
        if getattr(self, "_pub_vel", None) is None:
            return
        for _ in range(n):
            try:
                self._send_base(0.0, 0.0, 0.0)
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.02)

    def _deadman(self) -> None:
        """No firmware watchdog on the base: if no action arrived within
        deadman_timeout_s and the last base command was non-zero, send zeros."""
        period = min(0.1, self.config.deadman_timeout_s / 3)
        while not self._stop.is_set():
            time.sleep(period)
            if self._base_cmd != (0.0, 0.0, 0.0) and \
               time.time() - self._last_action_t > self.config.deadman_timeout_s:
                logger.warning("%s: deadman — no action for %.2fs, zeroing base",
                               self, time.time() - self._last_action_t)
                self._zero_base(n=3)

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    break
                raise

    def _teardown_ros(self) -> None:
        self._stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        try:
            if self._executor is not None and self._node is not None:
                self._executor.remove_node(self._node)
            if self._node is not None:
                self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        self._node = None
        self._executor = None

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
