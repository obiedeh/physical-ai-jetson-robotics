from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from synria_lerobot.embodiment import SynriaObservation
from synria_lerobot.evaluation import OperatorGrade
from synria_lerobot.physical_contract import ImageFrame, PhysicalState
from synria_lerobot.policy_client import FakePolicy
from synria_lerobot.sessions import run_session


def ready_config(tmp_path: Path) -> dict:
    config = json.loads(Path("config/synria_session.json").read_text())
    limits = json.loads(Path(config["limits"]).read_text())
    limits.update(verified_by="synthetic fixture", verified_on="2026-10-08")
    limits_path = tmp_path / "limits.json"
    limits_path.write_text(json.dumps(limits))
    policy = json.loads(Path(config["policy_config"]).read_text())
    policy.update(verified_by="synthetic fixture", verified_on="2026-10-08")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    squares = {"red:yard:0": [0, 0, 0], "track:0": [0.1, 0, 0], "track:1": [0.2, 0, 0]}
    for key, content in (("calibration", squares), ("reachable", list(squares))):
        path = tmp_path / f"{key}.json"
        path.write_text(
            json.dumps(
                dict(
                    frame="test",
                    squares=content,
                    verified_by="synthetic fixture",
                    verified_on="2026-10-08",
                )
            )
        )
        config[key] = str(path)
    config.update(
        limits=str(limits_path),
        adapter="never.imported:factory",
        gripper_type="50mm",
        max_steps=1,
        command_period_s=0.01,
        response_timeout_s=0.15,
        policy_config=str(policy_path),
        max_turns=2,
        policy_endpoint="http://unused.test",
    )
    config["provenance"] = {key: "synthetic fixture" for key in config["provenance"]}
    return config


class FakeIO:
    def __init__(self, tmp_path: Path) -> None:
        self.offers = []
        self.holds = 0
        self.closed = False
        self.rolls = iter([6, 1])
        self.still = tmp_path / "image.ppm"
        self.still.write_bytes(b"P6\n1 1\n255\n\x80\x80\x80")

    def observe(self, task: str) -> SynriaObservation:
        stamp = time.monotonic()
        frame = ImageFrame(np.ones((4, 4, 3), np.uint8), stamp)
        return SynriaObservation(PhysicalState((0,) * 6, 0, stamp, 0), frame, frame, task)

    def completed(self) -> bool:
        return True

    def command_sink(self) -> FakeIO:
        return self

    def offer(self, action: tuple) -> None:
        self.offers.append(action)

    def hold(self) -> None:
        self.holds += 1

    def grade(self, move: object) -> OperatorGrade:
        return OperatorGrade(
            "success",
            "fake operator",
            str(self.still),
            time.monotonic(),
            True,
            True,
            True,
            True,
            True,
        )

    def recover(self, move: object) -> bool:
        return False

    def operator_roll(self) -> int:
        return next(self.rolls)

    def abort_requested(self) -> bool:
        return False

    def power_state_end(self) -> str:
        return "synthetic fixture"

    def close(self) -> None:
        self.closed = True


def test_session_preflight_never_loads_device_adapter_without_flag(tmp_path: Path) -> None:
    config = ready_config(tmp_path)
    calls = []
    output = tmp_path / "session"
    result = run_session(config, "d3", Path.cwd(), output, factory=lambda conf: calls.append(conf))
    assert result["motion_enabled"] is False and not calls and not output.exists()
    limits = json.loads(Path(config["limits"]).read_text())
    limits["verified_by"] = ""
    Path(config["limits"]).write_text(json.dumps(limits))
    with pytest.raises(ValueError, match="verified"):
        run_session(
            config,
            "d3",
            Path.cwd(),
            output,
            enable_motion=True,
            factory=lambda conf: calls.append(conf),
        )
    assert not calls


def test_session_entry_point_runs_two_turns_with_only_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ready_config(tmp_path)
    io = FakeIO(tmp_path)
    monkeypatch.setattr(
        "synria_lerobot.sessions.HttpPolicyTransport", lambda endpoint: FakePolicy((0,) * 7)
    )
    output = tmp_path / "session"
    stats = run_session(
        config, "d3", Path.cwd(), output, enable_motion=True, factory=lambda conf: io
    )
    assert stats["turn_level"]["ok"] == 2 and stats["stage_status"] == "planned"
    assert len(io.offers) == 2 and io.holds >= 2 and io.closed
    assert (output / "latencies.jsonl").is_file() and (output / "session_end.json").is_file()
    evidence = json.loads((output / "provenance.json").read_text())["policy_safety"]
    assert evidence["response_timeout_s"] == 0.15
    assert evidence["command_period_s"] == 0.01
    assert evidence["max_joint_speed_rad_s"] == [0.075] * 6


@pytest.mark.parametrize("field", ["command_period_s", "response_timeout_s"])
@pytest.mark.parametrize("value", [None, True, 0, -1, float("nan"), float("inf"), "missing"])
def test_session_requires_valid_period_and_per_policy_timeout_before_adapter(
    tmp_path: Path, field: str, value: object
) -> None:
    config = ready_config(tmp_path)
    if value == "missing":
        del config[field]
    else:
        config[field] = value
    opened = []
    output = tmp_path / "session"
    with pytest.raises(ValueError):
        run_session(config, "d3", Path.cwd(), output, enable_motion=True,
                    factory=lambda conf: opened.append(conf))
    assert not opened and not output.exists()


@pytest.mark.parametrize("field", ["verified_by", "verified_on"])
def test_session_refuses_unverified_policy_config_before_adapter(
    tmp_path: Path, field: str,
) -> None:
    config = ready_config(tmp_path)
    path = Path(config["policy_config"])
    policy = json.loads(path.read_text())
    policy[field] = ""
    path.write_text(json.dumps(policy))
    opened = []
    output = tmp_path / "session"
    with pytest.raises(ValueError, match="verified policy"):
        run_session(config, "d3", Path.cwd(), output, enable_motion=True,
                    factory=lambda conf: opened.append(conf))
    assert not opened and not output.exists()


def test_session_rejects_old_duplicate_period_setting(tmp_path: Path) -> None:
    config = ready_config(tmp_path)
    config["period_s"] = 0.001
    with pytest.raises(ValueError, match="single session command period"):
        run_session(config, "d3", Path.cwd(), tmp_path / "session")
