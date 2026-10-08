from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from synria_lerobot.embodiment import SynriaEmbodiment, SynriaObservation
from synria_lerobot.physical_contract import (
    ActionSource,
    ImageFrame,
    PhysicalDatasetContract,
    PhysicalState,
)
from synria_lerobot.policy_client import (
    FakePolicy,
    GuardedCommandPath,
    PolicyClient,
    PolicySafetyConfig,
    add_motion_arguments,
)
from synria_lerobot.quality_gates import load_limits


def observation() -> SynriaObservation:
    return SynriaObservation(
        PhysicalState((0.0,) * 6, 0.01, 10, 100),
        ImageFrame(np.ones((4, 4, 3)), 10),
        ImageFrame(np.ones((4, 4, 3)), 10),
        "token A to B",
    )


class Sink:
    def __init__(self) -> None:
        self.offers: list[tuple[float, ...]] = []
        self.holds = 0

    def offer(self, action: tuple[float, ...]) -> None:
        self.offers.append(action)

    def hold(self) -> None:
        self.holds += 1


def client(transport: Any, clock: Any = lambda: 10.01) -> PolicyClient:
    return PolicyClient(
        SynriaEmbodiment(PhysicalDatasetContract("50mm", ActionSource.LEADER, False)),
        transport,
        load_limits(Path("config/synria_limits.yaml")),
        PolicySafetyConfig.load(
            Path("config/synria_policy.json"), command_period_s=0.4, response_timeout_s=0.1
        ),
        clock=clock,
    )


def test_absolute_and_delta_clamps_and_latency(tmp_path: Path) -> None:
    c = client(FakePolicy((100,) * 7))
    d = c.step(observation())
    assert d.action == (0.03,) * 6 + (0.011,)
    near = replace(observation(), state=PhysicalState((2.50, 0, 0, 0, 0, 0), 0.0245, 10, 100))
    d = c.step(near)
    assert d.action is not None and d.action[0] == 2.51 and d.action[6] == 0.025
    c.write_latencies(tmp_path / "latency.jsonl")
    assert json.loads((tmp_path / "latency.jsonl").read_text().splitlines()[0])["inference_s"] == 0


@pytest.mark.parametrize(
    "fault", ["missing", "stale", "mismatch", "nan", "short", "timeout", "latency", "expired"]
)
def test_bad_policy_responses_hold(fault: str) -> None:
    now = [10.01]

    class Broken(FakePolicy):
        def request(self, payload: dict, timeout_s: float) -> Any:
            reply = super().request(payload, timeout_s)
            if fault == "missing":
                return None
            if fault == "stale":
                now[0] += 1
            if fault == "expired":
                now[0] += 0.09
            if fault == "mismatch":
                reply["request_id"] = "previous"
            if fault == "nan":
                reply["action"] = (float("nan"),) * 7
            if fault == "short":
                reply["action"] = (0,) * 6
            if fault == "latency":
                reply["inference_s"] = -1
            if fault == "timeout":
                raise TimeoutError("timeout")
            return reply

    c = client(Broken((0,) * 7), lambda: now[0])
    if fault == "expired":
        c.config = replace(c.config, max_observation_age_s=0.05)
    c.limits = replace(c.limits, verified_by="operator", verified_on="2026-10-08")
    c.config = replace(c.config, verified_by="operator", verified_on="2026-10-08")
    sink = Sink()
    path = GuardedCommandPath(c, lambda: sink, enable_motion=True)
    assert path.step(observation()).hold
    assert sink.holds == 1 and not sink.offers


@pytest.mark.parametrize("fault", ["stale", "future", "range", "camera", "velocity"])
def test_bad_observations_hold(fault: str) -> None:
    c = client(FakePolicy((0,) * 7))
    obs = observation()
    if fault in {"stale", "future"}:
        obs = replace(
            obs, state=replace(obs.state, monotonic_timestamp_s=0 if fault == "stale" else 20)
        )
    if fault == "range":
        obs = replace(obs, state=replace(obs.state, joint_positions_rad=(100,) * 6))
    if fault == "camera":
        obs = replace(obs, front=ImageFrame(None, 10))
    if fault == "velocity":
        obs = replace(obs, state=replace(obs.state, joint_velocities_rad_s=(0,) * 6))
    assert c.step(obs).hold


@pytest.mark.parametrize("limits_verified", [False, True])
@pytest.mark.parametrize("policy_verified", [False, True])
def test_motion_requires_explicit_flag_and_both_verified_configs(
    limits_verified: bool, policy_verified: bool
) -> None:
    parser = argparse.ArgumentParser()
    add_motion_arguments(parser)
    assert not parser.parse_args([]).enable_motion
    assert parser.parse_args(["--enable-motion"]).enable_motion
    c = client(FakePolicy((0,) * 7))
    if limits_verified:
        c.limits = replace(c.limits, verified_by="operator", verified_on="date")
    if policy_verified:
        c.config = replace(c.config, verified_by="operator", verified_on="date")
    created = []

    def factory() -> Sink:
        created.append(True)
        return Sink()

    assert GuardedCommandPath(c, factory).sink is None
    assert not created
    if limits_verified and policy_verified:
        assert GuardedCommandPath(c, factory, enable_motion=True).sink is not None
        assert created == [True]
    else:
        with pytest.raises(ValueError, match="verified"):
            GuardedCommandPath(c, factory, enable_motion=True)
        assert not created


def test_invalid_bounds_and_speed_are_rejected() -> None:
    with pytest.raises(ValueError):
        PolicySafetyConfig((float("nan"),) * 6, 0.0025, 0.4, 0.1, 0.2)
    c = client(FakePolicy((0,) * 7))
    with pytest.raises(ValueError, match="ordering"):
        PolicyClient(c.embodiment, c.transport, replace(c.limits, joint_names=("bad",)), c.config)


def test_end_to_end_latency_includes_command_offer() -> None:
    now = [10.01]
    c = client(FakePolicy((0,) * 7), lambda: now[0])
    c.limits = replace(c.limits, verified_by="operator", verified_on="date")
    c.config = replace(c.config, verified_by="operator", verified_on="date")

    class SlowSink(Sink):
        def offer(self, action: tuple[float, ...]) -> None:
            now[0] += 0.025
            super().offer(action)

    decision = GuardedCommandPath(c, SlowSink, enable_motion=True).step(observation())
    assert decision.end_to_end_s == pytest.approx(0.025)
    assert c.latencies[-1] == decision


def test_stalled_transport_returns_hold_and_cannot_queue_late_actions() -> None:
    release = threading.Event()
    calls = []

    class Stalled(FakePolicy):
        def request(self, payload: dict, timeout_s: float) -> dict:
            calls.append(payload["request_id"])
            release.wait(2)
            return super().request(payload, timeout_s)

    c = client(Stalled((0,) * 7))
    c.config = replace(c.config, response_timeout_s=0.01)
    try:
        assert c.step(observation()).hold
        second = c.step(observation())
        assert second.hold and "pending" in second.reason
        assert len(calls) == 1
    finally:
        release.set()


@pytest.mark.parametrize("period_s", [0.1, 0.4, 0.8])
@pytest.mark.parametrize("direction", [-1, 1])
def test_speed_limits_scale_with_required_period_and_each_axis(
    period_s: float, direction: int
) -> None:
    c = client(FakePolicy((100 * direction,) * 7))
    speeds = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    c.config = replace(c.config, max_joint_speed_rad_s=speeds, command_period_s=period_s)
    decision = c.step(observation())
    expected = tuple(direction * speed * period_s for speed in speeds)
    assert decision.action == pytest.approx((*expected, 0.01 + direction * 0.0025 * period_s))
    assert c.config.max_delta == pytest.approx((*[speed * period_s for speed in speeds],
                                              0.0025 * period_s))


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, float("nan"), float("inf"), "0.1"])
@pytest.mark.parametrize("field", [
    "max_joint_speed_rad_s", "max_gripper_speed_m_s", "command_period_s",
    "response_timeout_s", "max_observation_age_s",
])
def test_policy_safety_rejects_invalid_numbers(field: str, bad: Any) -> None:
    config = client(FakePolicy((0,) * 7)).config
    value = (bad,) * 6 if field == "max_joint_speed_rad_s" else bad
    with pytest.raises(ValueError):
        replace(config, **{field: value})


@pytest.mark.parametrize("speeds", [(), (0.1,) * 7, None, "0.1"])
def test_policy_safety_requires_exactly_six_joint_speeds(speeds: Any) -> None:
    with pytest.raises(ValueError, match="six joint speeds"):
        replace(client(FakePolicy((0,) * 7)).config, max_joint_speed_rad_s=speeds)


def test_derived_delta_cannot_overflow_or_underflow() -> None:
    config = client(FakePolicy((0,) * 7)).config
    for speed, period in ((1e308, 1e308), (1e-300, 1e-300)):
        with pytest.raises(ValueError, match="speed times command period"):
            replace(config, max_joint_speed_rad_s=(speed,) * 6, command_period_s=period)


@pytest.mark.parametrize("field", ["verified_by", "verified_on"])
@pytest.mark.parametrize("bad", [None, True, False, 0, 1])
def test_invalid_verification_markers_never_create_a_sink(
    tmp_path: Path, field: str, bad: Any
) -> None:
    c = client(FakePolicy((0,) * 7))
    created = []
    limits = json.loads(Path("config/synria_limits.yaml").read_text())
    limits.update(verified_by="operator", verified_on="date")
    limits[field] = bad
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(limits))
    with pytest.raises(ValueError, match="verification fields"):
        c.limits = load_limits(path)
        GuardedCommandPath(c, lambda: created.append(True), enable_motion=True)
    with pytest.raises(ValueError, match="verification fields"):
        c.config = replace(c.config, **{field: bad})
        GuardedCommandPath(c, lambda: created.append(True), enable_motion=True)
    assert not created


@pytest.mark.parametrize("field", ["verified_by", "verified_on"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_each_policy_verification_marker_is_required(field: str, blank: str) -> None:
    c = client(FakePolicy((0,) * 7))
    c.limits = replace(c.limits, verified_by="operator", verified_on="date")
    c.config = replace(c.config, verified_by="operator", verified_on="date")
    c.config = replace(c.config, **{field: blank})
    created = []
    with pytest.raises(ValueError, match="verified"):
        GuardedCommandPath(c, lambda: created.append(True), enable_motion=True)
    assert not created


def test_loader_requires_period_and_policy_timeout_and_rejects_shared_defaults(
    tmp_path: Path,
) -> None:
    path = Path("config/synria_policy.json")
    with pytest.raises(TypeError):
        PolicySafetyConfig.load(path, command_period_s=0.4)
    with pytest.raises(TypeError):
        PolicySafetyConfig.load(path, response_timeout_s=0.1)
    original = json.loads(path.read_text())
    assert "response_timeout_s" not in original
    assert original["verified_by"] == original["verified_on"] == ""
    for field in ("max_delta", "response_timeout_s", "command_period_s"):
        altered = tmp_path / "shared.json"
        altered.write_text(json.dumps({**original, field: 0.1}))
        with pytest.raises(ValueError, match="shared config"):
            PolicySafetyConfig.load(altered, command_period_s=0.4, response_timeout_s=0.1)


@pytest.mark.parametrize("period_s", [True, 0, float("nan"), 0.1])
def test_motion_loops_refuse_a_period_different_from_the_clamp(period_s: float) -> None:
    from synria_lerobot.game_runner import PolicyRollArm
    from synria_lerobot.turn_executor import PolicyMoveExecutor

    path = GuardedCommandPath(client(FakePolicy((0,) * 7)), Sink)
    observations = []

    def observe(task: str) -> SynriaObservation:
        observations.append(task)
        return observation()

    with pytest.raises(ValueError, match="command period"):
        PolicyMoveExecutor(path, observe, lambda: True, max_steps=1, period_s=period_s)
    roller = PolicyRollArm(path, observe, lambda phase: True)
    with pytest.raises(ValueError, match="command period"):
        roller.perform("pick_die", {"max_steps": 1, "period_s": period_s})
    assert not observations


def test_each_policy_transport_receives_its_explicit_timeout() -> None:
    observed_timeouts = []

    class TimedPolicy(FakePolicy):
        def request(self, payload: dict, timeout_s: float) -> dict:
            observed_timeouts.append(timeout_s)
            return super().request(payload, timeout_s)

    for timeout in (0.05, 0.35):
        c = client(TimedPolicy((0,) * 7))
        c.config = replace(c.config, response_timeout_s=timeout)
        assert not c.step(observation()).hold
    assert observed_timeouts == [0.05, 0.35]
