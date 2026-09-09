"""Keyboard teleoperator for the ROSMASTER M3 Pro (pynput, X11 session).

Arm (absolute targets, stepped per tick):   q/a joint1  w/s joint2  e/d joint3
                                             r/f joint4  t/g joint5  y/h gripper(joint6)
Base (held keys -> velocity, released -> 0): i/k vx  j/l vy  u/o wz     space = stop base
                                             0 = arm to home pose
Emits a full RobotAction for RosmasterM3Pro every call, so the robot's
clamps (limits, max_relative_target) and deadman remain the safety floor.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config import ARM_JOINTS, VENDOR_HOME

logger = logging.getLogger(__name__)

try:
    from pynput import keyboard
    PYNPUT = True
except Exception:  # noqa: BLE001
    keyboard = None
    PYNPUT = False

ARM_KEYS = {"q": ("joint1", +1), "a": ("joint1", -1), "w": ("joint2", +1), "s": ("joint2", -1),
            "e": ("joint3", +1), "d": ("joint3", -1), "r": ("joint4", +1), "f": ("joint4", -1),
            "t": ("joint5", +1), "g": ("joint5", -1), "y": ("joint6", +1), "h": ("joint6", -1)}
BASE_KEYS = {"i": ("vx", +1), "k": ("vx", -1), "j": ("vy", +1), "l": ("vy", -1),
             "u": ("wz", +1), "o": ("wz", -1)}


@TeleoperatorConfig.register_subclass("rosmaster_m3pro_keyboard")
@dataclass
class RosmasterM3ProKeyboardConfig(TeleoperatorConfig):
    step_deg: float = 2.0            # degrees per tick while a key is held
    linear_speed: float = 0.10       # m/s while i/k/j/l held
    angular_speed: float = 0.40      # rad/s while u/o held
    start_pose: tuple[float, float, float, float, float, float] = VENDOR_HOME
    use_base: bool = True


class RosmasterM3ProKeyboard(Teleoperator):
    config_class = RosmasterM3ProKeyboardConfig
    name = "rosmaster_m3pro_keyboard"

    def __init__(self, config: RosmasterM3ProKeyboardConfig):
        super().__init__(config)
        self.config = config
        self._queue: Queue = Queue()
        self._pressed: dict[Any, bool] = {}
        self._listener = None
        self._target = dict(zip(ARM_JOINTS, config.start_pose))

    @property
    def action_features(self) -> dict:
        ft = {f"{j}.pos": float for j in ARM_JOINTS}
        if self.config.use_base:
            ft.update({"base.vx": float, "base.vy": float, "base.wz": float})
        return ft

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT and self._listener is not None and self._listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not PYNPUT:
            raise ImportError("pynput is required for keyboard teleop (pip install pynput); "
                              "it needs an X11 session, not headless/Wayland")
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        logger.info("%s: keyboard listener started (see module docstring for keys)", self)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key):
        self._queue.put((getattr(key, "char", key), True))

    def _on_release(self, key):
        self._queue.put((getattr(key, "char", key), False))

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        while not self._queue.empty():
            k, down = self._queue.get_nowait()
            self._pressed[k] = down
        held = {k for k, v in self._pressed.items() if v}
        if "0" in held:
            self._target = dict(zip(ARM_JOINTS, self.config.start_pose))
        for k in held:
            if k in ARM_KEYS:
                j, sgn = ARM_KEYS[k]
                self._target[j] = min(180.0, max(0.0, self._target[j] + sgn * self.config.step_deg))
        action: dict[str, float] = {f"{j}.pos": self._target[j] for j in ARM_JOINTS}
        if self.config.use_base:
            v = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
            if keyboard.Key.space not in held:
                for k in held:
                    if k in BASE_KEYS:
                        axis, sgn = BASE_KEYS[k]
                        v[axis] = sgn * (self.config.angular_speed if axis == "wz" else self.config.linear_speed)
            action.update({"base.vx": v["vx"], "base.vy": v["vy"], "base.wz": v["wz"]})
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def teleop_events(self) -> dict[str, Any]:
        return {"t": time.time()}
