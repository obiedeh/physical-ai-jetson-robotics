"""ROSMASTER M3 Pro LeRobot client — runs on the 5090 (LeRobot 0.6.2, py3.12).

A LeRobot Robot that talks to the Orin-side m3pro_host over ZMQ (LeKiwi wire):
records LeRobotDatasets, runs policies, evaluates — while the robot itself is
served on the Orin. Registered as robot type `rosmaster_m3pro_client`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import ARM_JOINTS

logger = logging.getLogger(__name__)
BASE_ACTIONS = ("base.vx", "base.vy", "base.wz")


@RobotConfig.register_subclass("rosmaster_m3pro_client")
@dataclass
class RosmasterM3ProClientConfig(RobotConfig):
    remote_ip: str = "192.168.1.243"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connect_timeout_s: float = 10.0
    polling_timeout_ms: int = 200
    use_base: bool = True
    # passive/joystick-observe: the host embeds the joystick command as the
    # action in each obs frame; the client surfaces it and never commands.
    passive: bool = False
    # camera name -> (h, w); must match what the host actually streams
    camera_shapes: dict[str, tuple] = field(default_factory=lambda: {"front": (480, 640)})
    cameras: dict[str, CameraConfig] = field(default_factory=dict)  # unused; host owns cameras


class RosmasterM3ProClient(Robot):
    config_class = RosmasterM3ProClientConfig
    name = "rosmaster_m3pro_client"

    def __init__(self, config: RosmasterM3ProClientConfig):
        super().__init__(config)
        self.config = config
        self._zmq = None
        self._ctx = None
        self._cmd = None
        self._obs = None
        self._connected = False
        self._last_state: dict[str, float] = {}
        self._captured_action: dict[str, float] | None = None  # passive: joystick cmd
        self._passive_warned = False

    @property
    def _arm_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in ARM_JOINTS}

    @property
    def _base_obs_ft(self) -> dict[str, type]:
        if not self.config.use_base:
            return {}
        return {k: float for k in ("base.odom_x", "base.odom_y", "base.odom_yaw",
                                   "base.vx", "base.vy", "base.wz")}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cams = {name: (h, w, 3) for name, (h, w) in self.config.camera_shapes.items()}
        return {**self._arm_ft, **self._base_obs_ft, "battery.v": float, **cams}

    @cached_property
    def action_features(self) -> dict[str, type]:
        ft = dict(self._arm_ft)
        if self.config.use_base:
            ft.update({k: float for k in BASE_ACTIONS})
        return ft

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import zmq
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._cmd = self._ctx.socket(zmq.PUSH)
        self._cmd.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_cmd}")
        self._cmd.setsockopt(zmq.CONFLATE, 1)
        self._obs = self._ctx.socket(zmq.PULL)
        self._obs.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_observations}")
        self._obs.setsockopt(zmq.RCVHWM, 2)
        poller = zmq.Poller(); poller.register(self._obs, zmq.POLLIN)
        socks = dict(poller.poll(int(self.config.connect_timeout_s * 1000)))
        if self._obs not in socks:
            self._teardown()
            raise ConnectionError(
                f"no observation from m3pro_host at {self.config.remote_ip}:"
                f"{self.config.port_zmq_observations} within "
                f"{self.config.connect_timeout_s}s (host running? firewall?)")
        self._connected = True
        logger.info("%s connected to host %s", self, self.config.remote_ip)

    @check_if_not_connected
    def disconnect(self) -> None:
        # tell the host to stop the base, then drop the sockets
        try:
            zero = {f"{j}.pos": self._last_state.get(f"{j}.pos", 90.0) for j in ARM_JOINTS}
            if self.config.use_base:
                zero.update({k: 0.0 for k in BASE_ACTIONS})
            self._cmd.send_string(json.dumps(zero))
        except Exception:
            pass
        self._teardown()
        self._connected = False

    def _teardown(self) -> None:
        for s in (self._cmd, self._obs):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        try:
            if self._ctx is not None:
                self._ctx.term()
        except Exception:
            pass
        self._cmd = self._obs = self._ctx = None

    def _latest(self) -> list[bytes] | None:
        zmq = self._zmq
        poller = zmq.Poller(); poller.register(self._obs, zmq.POLLIN)
        last = None
        # drain to the newest frame available within the polling window
        socks = dict(poller.poll(self.config.polling_timeout_ms))
        while self._obs in socks:
            last = self._obs.recv_multipart()
            socks = dict(poller.poll(0))
        return last

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        import cv2
        frames = self._latest()
        if frames is None:
            raise TimeoutError("no observation frame from m3pro_host within polling window")
        header = json.loads(frames[0].decode())
        obs: dict[str, Any] = dict(header["state"])
        self._last_state = {k: v for k, v in header["state"].items() if k.endswith(".pos")}
        if self.config.passive:
            self._captured_action = header.get("action")  # joystick cmd this frame, or None
        for i, cam in enumerate(header.get("cams", [])):
            buf = np.frombuffer(frames[1 + i], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)          # BGR
            obs[cam["name"]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return obs

    def captured_action(self) -> dict[str, float] | None:
        """Passive mode: the joystick command captured in the most recent
        observation (call get_observation() first). None until one is seen."""
        return self._captured_action

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.passive:
            if not self._passive_warned:
                logger.warning("%s is passive (joystick-observe): send_action is a "
                               "no-op; the joystick commands the board", self)
                self._passive_warned = True
            return action
        payload = {k: float(v) for k, v in action.items()
                   if k.endswith(".pos") or k in BASE_ACTIONS}
        self._cmd.send_string(json.dumps(payload))
        return action
