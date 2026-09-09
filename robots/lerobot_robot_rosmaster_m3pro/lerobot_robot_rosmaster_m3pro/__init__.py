"""LeRobot plugin for the Yahboom ROSMASTER M3 Pro.

Two runtime contexts share this package:
  * the Orin HOST (Python 3.10, ROS 2, NO lerobot) imports only `hardware`
    and runs `m3pro_host`; the lerobot-dependent classes below are skipped;
  * the 5090 CLIENT (Python 3.12, lerobot >= 0.6.2) gets the registered
    `rosmaster_m3pro_client` (and the direct `rosmaster_m3pro` / keyboard
    teleop) via the standard `lerobot_robot_*` plugin discovery.
"""
from .hardware import ARM_JOINTS, VENDOR_HOME, M3ProHardware  # noqa: F401 (lerobot-free)

__all__ = ["ARM_JOINTS", "VENDOR_HOME", "M3ProHardware"]

try:  # lerobot present (client/5090 side) -> register the LeRobot types
    from .config import RosmasterM3ProConfig  # noqa: F401
    from .robot import RosmasterM3Pro  # noqa: F401
    from .teleop import RosmasterM3ProKeyboardConfig, RosmasterM3ProKeyboard  # noqa: F401
    from .client import RosmasterM3ProClient, RosmasterM3ProClientConfig  # noqa: F401

    __all__ += ["RosmasterM3ProConfig", "RosmasterM3Pro",
                "RosmasterM3ProKeyboardConfig", "RosmasterM3ProKeyboard",
                "RosmasterM3ProClient", "RosmasterM3ProClientConfig"]
except ImportError:  # Orin host: lerobot not installed — hardware + host only
    pass
