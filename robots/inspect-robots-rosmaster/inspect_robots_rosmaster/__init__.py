"""Inspect Robots embodiment for the Yahboom ROSMASTER M3 Pro arm.

Registered via the ``inspect_robots.embodiments`` entry point as
``rosmaster_m3pro``; ``make_embodiment(**kwargs)`` is the factory Inspect
Robots calls (kwargs come from ``-E key=value`` on the CLI).
"""
from __future__ import annotations

import time

import numpy as np

from inspect_robots.embodiment import (
    RENDERABLE,
    RESETTABLE,
    EmbodimentInfo,
)
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics, Box, CameraSpec, ObservationSpace, StateField, StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult

ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


class RosmasterM3ProEmbodiment:
    """Real M3 Pro 6-DOF arm over the m3pro_host ZMQ bridge (active mode).

    Action: 6 absolute joint angles in degrees (``joint_pos``). Observation:
    the ``front`` camera + the 6 joint angles (the board has no joint feedback,
    so state is the last commanded pose and the camera is the real state).
    Connection is lazy (in ``reset``) so ``doctor`` can inspect the spaces
    without any robot present.
    """

    def __init__(self, *, remote_ip: str = "192.168.1.251", control_hz: float = 10.0,
                 cam_h: int = 480, cam_w: int = 640,
                 joint_low: float = 0.0, joint_high: float = 180.0,
                 settle_s: float = 0.0):
        self.remote_ip = remote_ip
        self.control_hz = float(control_hz)
        self.cam_h, self.cam_w = int(cam_h), int(cam_w)
        self.settle_s = float(settle_s)
        self._robot = None
        self.info = EmbodimentInfo(
            name="rosmaster_m3pro",
            action_space=Box(
                shape=(6,),
                low=np.full((6,), float(joint_low)),
                high=np.full((6,), float(joint_high)),
                semantics=ActionSemantics(
                    control_mode="joint_pos", frame="joint",
                    dim_labels=ARM),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(name="front", height=cam_h, width=cam_w, channels=3),),
                state_keys=frozenset({"joint_pos"}),
                # absolute joint_pos control needs one (6,) proprioceptive field
                state=StateSpec(fields=(
                    StateField(key="joint_pos", shape=(6,), unit="deg"),)),
            ),
            control_hz=self.control_hz,
            is_simulated=False,
            # real robot: human reset, NO privileged success oracle, NO seedable world
            capabilities=frozenset({RESETTABLE, RENDERABLE}),
            docs=("Yahboom ROSMASTER M3 Pro 6-DOF arm. Action = 6 absolute joint "
                  "angles in degrees [0..180], servo 6 is the gripper. The 'front' "
                  "camera (Orbbec RGB) is the real scene; joint feedback is not "
                  "published by the board, so 'joint_pos' state is the last "
                  "commanded pose. Vendor home is [90,150,12,20,90,0]."),
        )

    # -- lazy client -------------------------------------------------------
    def _ensure_robot(self):
        if self._robot is None:
            from lerobot_robot_rosmaster_m3pro.client import (
                RosmasterM3ProClient, RosmasterM3ProClientConfig,
            )
            cfg = RosmasterM3ProClientConfig(
                id="ir", remote_ip=self.remote_ip, use_base=False, passive=False,
                camera_shapes={"front": (self.cam_h, self.cam_w)})
            self._robot = RosmasterM3ProClient(cfg)
            self._robot.connect()
        return self._robot

    def _observe(self, instruction):
        obs = self._ensure_robot().get_observation()
        jp = np.array([float(obs[f"{j}.pos"]) for j in ARM], dtype=np.float64)
        return Observation(
            images={"front": np.asarray(obs["front"])},
            state={"joint_pos": jp},
            instruction=instruction,
        )

    # -- Inspect Robots Embodiment API ------------------------------------
    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Human-in-the-loop reset: ask the operator to set up the physical
        scene, then return the first observation. No privileged state."""
        self._ensure_robot()
        from inspect_robots.console import poll  # attended reset prompt
        setup = scene.setup or "reset the workspace for this trial"
        try:
            poll((f"[rosmaster_m3pro] RESET — {setup}",
                  f"instruction: {scene.instruction}",
                  "place the object(s), clear the arm path, then press Enter"))
        except Exception:
            # non-interactive run: fall back to a fixed settle
            time.sleep(max(self.settle_s, 1.0))
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Send 6 absolute joint targets; return the next observation. Success
        is decided by the scorer/grader (no oracle on real hardware)."""
        robot = self._ensure_robot()
        data = np.asarray(action.data, dtype=np.float64).reshape(-1)[:6]
        cmd = {f"{j}.pos": float(v) for j, v in zip(ARM, data)}
        robot.send_action(cmd)
        if self.control_hz > 0:
            time.sleep(1.0 / self.control_hz)
        return StepResult(
            observation=self._observe(None),
            reward=0.0,
            terminated=False,
            termination_reason=None,
            truncated=False,
            info={},
        )

    def close(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            finally:
                self._robot = None


def make_embodiment(**kwargs) -> RosmasterM3ProEmbodiment:
    """Entry-point factory; CLI ``-E key=value`` args arrive as kwargs."""
    for k in ("control_hz", "cam_h", "cam_w", "joint_low", "joint_high", "settle_s"):
        if k in kwargs and isinstance(kwargs[k], str):
            kwargs[k] = float(kwargs[k]) if "." in kwargs[k] or k in ("control_hz", "settle_s") else int(kwargs[k])
    return RosmasterM3ProEmbodiment(**kwargs)
