"""Bounded policy requests and gated, clamped command offers for Synria."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

from .embodiment import SynriaEmbodiment, SynriaObservation
from .physical_contract import DRIVER_JOINT_NAMES
from .quality_gates import PhysicalLimits


class PolicyTransport(Protocol):
    def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any] | None: ...


class CommandSink(Protocol):
    """Deployment adapter must implement a controller-supported hold, never torque-off."""

    def offer(self, action: tuple[float, ...]) -> None: ...
    def hold(self) -> None: ...


@dataclass(frozen=True)
class PolicySafetyConfig:
    max_joint_speed_rad_s: tuple[float, ...]
    max_gripper_speed_m_s: float
    command_period_s: float
    response_timeout_s: float
    max_observation_age_s: float
    verified_by: str = ""
    verified_on: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.max_joint_speed_rad_s, (tuple, list)) or len(
            self.max_joint_speed_rad_s
        ) != 6:
            raise ValueError("six joint speeds required")
        for name, values in (
            ("joint speeds", self.max_joint_speed_rad_s),
            ("gripper speed", (self.max_gripper_speed_m_s,)),
            ("command period", (self.command_period_s,)),
            ("per-policy response timeout", (self.response_timeout_s,)),
            ("observation age", (self.max_observation_age_s,)),
        ):
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in values):
                raise ValueError(f"{name} must be positive finite numbers, not booleans")
        if any(not isinstance(value, str) for value in (self.verified_by, self.verified_on)):
            raise ValueError("policy verification fields must be strings")
        object.__setattr__(self, "max_joint_speed_rad_s", tuple(self.max_joint_speed_rad_s))
        if any(not math.isfinite(value) or value <= 0 for value in self.max_delta):
            raise ValueError("speed times command period must be positive and finite")

    @property
    def max_delta(self) -> tuple[float, ...]:
        """Derived rad/m bounds; never an independently configured step limit."""
        return tuple(
            speed * self.command_period_s
            for speed in (*self.max_joint_speed_rad_s, self.max_gripper_speed_m_s)
        )

    @property
    def verified(self) -> bool:
        return bool(self.verified_by.strip() and self.verified_on.strip())

    def require_command_period(self, period_s: float) -> None:
        if type(period_s) not in (int, float) or period_s != self.command_period_s:
            raise ValueError("motion-loop period must match the configured command period")

    @classmethod
    def load(
        cls, path: Path, *, command_period_s: float, response_timeout_s: float
    ) -> PolicySafetyConfig:
        data = json.loads(path.read_text(encoding="utf-8"))
        if {"max_delta", "response_timeout_s", "command_period_s"} & data.keys():
            raise ValueError("shared config cannot set step deltas, period or policy timeout")
        return cls(
            data["max_joint_speed_rad_s"], data["max_gripper_speed_m_s"],
            command_period_s, response_timeout_s, data["max_observation_age_s"],
            data["verified_by"], data["verified_on"],
        )


@dataclass(frozen=True)
class PolicyDecision:
    action: tuple[float, ...] | None
    hold: bool
    reason: str
    inference_s: float | None
    end_to_end_s: float
    request_id: str


class HttpPolicyTransport:
    """JSON protocol with a bounded socket wait; no executable deserialization."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
        def encode(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(f"unsupported observation field: {type(value)}")

        body = json.dumps(payload, default=encode, allow_nan=False).encode()
        request = Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("oversized policy response")
        result: dict[str, Any] = json.loads(raw)
        return result


class FakePolicy:
    def __init__(self, action: tuple[float, ...]) -> None:
        self.action = action

    def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        return {
            "request_id": payload["request_id"],
            "observation_timestamp_s": payload["observation_timestamp_s"],
            "action": self.action,
            "inference_s": 0.0,
        }


class PolicyClient:
    def __init__(
        self,
        embodiment: SynriaEmbodiment,
        transport: PolicyTransport,
        limits: PhysicalLimits,
        config: PolicySafetyConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embodiment, self.transport, self.limits, self.config = (
            embodiment,
            transport,
            limits,
            config,
        )
        self.clock = clock
        self.latencies: list[PolicyDecision] = []
        self._pending: Future[dict[str, Any] | None] | None = None
        if limits.joint_names != DRIVER_JOINT_NAMES:
            raise ValueError("limit joint ordering differs from physical contract")
        self.bounds = (
            *limits.joint_limits_rad,
            limits.gripper_limits_m[embodiment.contract.gripper_type],
        )
        if len(self.bounds) != 7 or any(
            not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in self.bounds
        ):
            raise ValueError("invalid physical limits")
        if self.bounds[-1] != (0, embodiment.contract.gripper_stroke_m):
            raise ValueError("gripper limits differ from contract")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if self._pending is not None and not self._pending.done():
            raise TimeoutError("previous policy request still pending")
        future: Future[dict[str, Any] | None] = Future()
        self._pending = future

        def invoke() -> None:
            try:
                future.set_result(self.transport.request(payload, self.config.response_timeout_s))
            except BaseException as exc:
                future.set_exception(exc)

        # At most one outstanding request; a stuck server cannot prevent hold or process exit.
        threading.Thread(target=invoke, daemon=True).start()
        return future.result(timeout=self.config.response_timeout_s)

    def step(self, observation: SynriaObservation, *, reset: bool = False) -> PolicyDecision:
        start = self.clock()
        request_id = uuid.uuid4().hex
        inference: float | None = None
        action: tuple[float, ...] | None = None
        reason = "accepted"
        try:
            payload = self.embodiment.observation(observation)
            stamps = [
                observation.state.monotonic_timestamp_s,
                observation.wrist.monotonic_timestamp_s,
                observation.front.monotonic_timestamp_s,
            ]
            if any(not 0 <= start - stamp <= self.config.max_observation_age_s for stamp in stamps):
                raise ValueError("stale or future observation")
            current = (*observation.state.joint_positions_rad, observation.state.gripper_m)
            if any(not lo <= v <= hi for v, (lo, hi) in zip(current, self.bounds, strict=True)):
                raise ValueError("current state outside limits")
            payload.update(
                request_id=request_id,
                reset=reset,
                observation_timestamp_s=observation.state.monotonic_timestamp_s,
            )
            response = self._request(payload)
            elapsed = self.clock() - start
            if not 0 <= elapsed <= self.config.response_timeout_s:
                raise ValueError("stale policy response")
            if response is None:
                raise ValueError("missing policy response")
            if (
                response["request_id"] != request_id
                or response["observation_timestamp_s"] != stamps[0]
            ):
                raise ValueError("response does not match current observation")
            if any(self.clock() - stamp > self.config.max_observation_age_s for stamp in stamps):
                raise ValueError("observation expired during inference")
            inference = float(response["inference_s"])
            if not math.isfinite(inference) or inference < 0:
                raise ValueError("invalid inference latency")
            raw = tuple(float(x) for x in response["action"])
            if len(raw) != 7 or any(not math.isfinite(x) for x in raw):
                raise ValueError("invalid action dimensionality or nonfinite value")
            action = tuple(
                max(lo, v - delta, min(hi, v + delta, target))
                for target, v, delta, (lo, hi) in zip(
                    raw, current, self.config.max_delta, self.bounds, strict=True
                )
            )
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            if inference is not None and (not math.isfinite(inference) or inference < 0):
                inference = None
        decision = PolicyDecision(
            action, action is None, reason, inference, max(0.0, self.clock() - start), request_id
        )
        self.latencies.append(decision)
        return decision

    def write_latencies(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as stream:
            for decision in self.latencies:
                stream.write(json.dumps(asdict(decision), allow_nan=False) + "\n")


class GuardedCommandPath:
    """The sink factory requires opt-in plus verified position and speed limits."""

    def __init__(
        self,
        client: PolicyClient,
        sink_factory: Callable[[], CommandSink],
        *,
        enable_motion: bool = False,
    ) -> None:
        self.client = client
        verified = client.limits.verified and client.config.verified
        self.enabled = enable_motion and verified
        if enable_motion and not verified:
            raise ValueError("motion requires operator-verified limits and policy safety config")
        self.sink = sink_factory() if self.enabled else None

    def step(self, observation: SynriaObservation, *, reset: bool = False) -> PolicyDecision:
        start = self.client.clock()
        decision = self.client.step(observation, reset=reset)
        if self.sink is not None:
            try:
                if decision.hold:
                    self.sink.hold()
                else:
                    assert decision.action is not None
                    self.sink.offer(decision.action)
            except BaseException:
                self.sink.hold()
                raise
        decision = replace(decision, end_to_end_s=max(0.0, self.client.clock() - start))
        self.client.latencies[-1] = decision
        return decision

    def hold(self) -> None:
        if self.sink is not None:
            self.sink.hold()


def add_motion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        default=False,
        help="Allow a command sink only with verified limits and policy safety config.",
    )
