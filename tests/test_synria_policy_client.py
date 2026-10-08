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
        PolicySafetyConfig.load(Path("config/synria_policy.json")),
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


def test_motion_requires_explicit_flag_and_verified_limits() -> None:
    parser = argparse.ArgumentParser()
    add_motion_arguments(parser)
    assert not parser.parse_args([]).enable_motion
    assert parser.parse_args(["--enable-motion"]).enable_motion
    c = client(FakePolicy((0,) * 7))
    created = []

    def factory() -> Sink:
        created.append(True)
        return Sink()

    assert GuardedCommandPath(c, factory).sink is None
    with pytest.raises(ValueError, match="verified"):
        GuardedCommandPath(c, factory, enable_motion=True)
    assert not created
    c.limits = replace(c.limits, verified_by="operator", verified_on="date")
    assert GuardedCommandPath(c, factory).sink is None
    assert not created
    assert GuardedCommandPath(c, factory, enable_motion=True).sink is not None


def test_invalid_bounds_and_delta_are_rejected() -> None:
    with pytest.raises(ValueError):
        PolicySafetyConfig((float("nan"),) * 7, 0.1, 0.2)
    c = client(FakePolicy((0,) * 7))
    with pytest.raises(ValueError, match="ordering"):
        PolicyClient(c.embodiment, c.transport, replace(c.limits, joint_names=("bad",)), c.config)


def test_end_to_end_latency_includes_command_offer() -> None:
    now = [10.01]
    c = client(FakePolicy((0,) * 7), lambda: now[0])
    c.limits = replace(c.limits, verified_by="operator", verified_on="date")

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
