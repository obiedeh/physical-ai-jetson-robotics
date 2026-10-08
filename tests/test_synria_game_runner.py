from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synria_lerobot.embodiment import SynriaEmbodiment, SynriaObservation
from synria_lerobot.evaluation import OperatorGrade
from synria_lerobot.game_runner import DieRead, GameRunner, PolicyRollArm, RollMachine
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
)
from synria_lerobot.quality_gates import load_limits
from synria_lerobot.turn_executor import (
    BoardCalibration,
    PhysicalLudoGame,
    PolicyMoveExecutor,
    TurnExecutor,
)


class FakeArm:
    def __init__(self) -> None:
        self.actions: list[tuple[float, ...]] = []
        self.holds = 0

    def offer(self, action: tuple[float, ...]) -> None:
        self.actions.append(action)

    def hold(self) -> None:
        self.holds += 1


def setup_game(tmp_path: Path, *, fail_trial: bool = False, bad_die: bool = False) -> GameRunner:
    clock = [10.0]
    arm = FakeArm()
    limits = replace(
        load_limits(Path("config/synria_limits.yaml")), verified_by="fake", verified_on="test"
    )
    client = PolicyClient(
        SynriaEmbodiment(PhysicalDatasetContract("50mm", ActionSource.LEADER, False)),
        FakePolicy((0,) * 7),
        limits,
        PolicySafetyConfig((0.075,) * 6, 0.0025, 0.01, 0.1, 0.2, "fake", "test"),
        clock=lambda: clock[0],
    )
    path = GuardedCommandPath(client, lambda: arm, enable_motion=True)

    def observe(task: str) -> SynriaObservation:
        image = ImageFrame(np.ones((4, 4, 3), np.uint8), clock[0])
        return SynriaObservation(PhysicalState((0,) * 6, 0, clock[0], 0), image, image, task)

    squares = {f"track:{i}": [i / 100, 0, 0] for i in range(52)}
    for colour in ("red", "blue", "green", "yellow"):
        for kind, count in (("yard", 4), ("home", 6)):
            squares.update({f"{colour}:{kind}:{i}": [i / 100, 0.1, 0] for i in range(count)})
    cal_path, reach_path = tmp_path / "cal.json", tmp_path / "reachable.json"
    cal_path.write_text(json.dumps({"frame": "test", "squares": squares}))
    reach_path.write_text(json.dumps({"frame": "test", "squares": list(squares)}))
    calibration = BoardCalibration(cal_path, reach_path, require_verified=False)
    config = json.loads(Path("config/synria_roll.json").read_text())
    for value in config["phases"].values():
        value.update(task="synthetic motion", target_xyz_m=[0, 0, 0], max_steps=1, period_s=0.01)
    config_path = tmp_path / "roll.json"
    config_path.write_text(json.dumps(config))
    still = tmp_path / "front.ppm"
    still.write_bytes(b"P6\n1 1\n255\n\x80\x80\x80")
    rng = random.Random(25)
    read_count, die_value = [0], [1]

    def read_die() -> DieRead:
        if read_count[0] % 3 == 0:
            die_value[0] = rng.randint(1, 6)
        read_count[0] += 1
        clock[0] += 0.01
        return DieRead(None if bad_die else die_value[0], str(still), clock[0])

    roll_arm = PolicyRollArm(path, observe, lambda phase: True, sleep=lambda seconds: None)
    roller = RollMachine(
        roll_arm,
        read_die,
        config_path,
        synthetic=True,
        clock=lambda: clock[0],
        sleep=lambda seconds: None,
    )
    labels = [0]

    def grade(move: object) -> OperatorGrade:
        labels[0] += 1
        success = not fail_trial or labels[0] != 4
        return OperatorGrade(
            "success" if success else "failure",
            "fake operator",
            str(still),
            clock[0],
            True,
            True,
            True,
            success,
            success,
        )

    executor = TurnExecutor(
        PhysicalLudoGame(),
        calibration,
        PolicyMoveExecutor(
            path, observe, lambda: True, max_steps=1, period_s=0.01, sleep=lambda seconds: None
        ),
        grade,
        tmp_path / "turns.jsonl",
        data_kind="synthetic",
        max_attempts=2,
        recover=lambda move: True,
    )
    return GameRunner(
        executor,
        roller,
        tmp_path,
        dict(
            operator="fake",
            policy_id="fake",
            git_sha="test",
            host="test",
            utc_date="2026-10-08",
            power_state_start="fake",
            follower_serial="fake",
            camera_ids=["fake-wrist", "fake-front"],
        ),
        max_turns=3000,
    )


def test_full_game_to_winner_with_failed_attempt_and_retry(tmp_path: Path) -> None:
    runner = setup_game(tmp_path, fail_trial=True)
    stats = runner.run()
    assert stats["outcome"] == "winner" and stats["winner"]
    assert stats["misses"] == 1 and stats["retries_used"] == 1
    assert stats["d4_sequence_observed"] and stats["stage_status"] == "planned"
    records = [json.loads(line) for line in (tmp_path / "turns.jsonl").read_text().splitlines()]
    assert len(records) == stats["turns"] and all("attempts" in row for row in records)
    assert json.loads((tmp_path / "frozen_stats.json").read_text())["data_kind"] == "synthetic"
    assert (tmp_path / "rolls.jsonl").is_file()


def test_game_abort_on_unreadable_die_writes_records_and_aggregate(tmp_path: Path) -> None:
    runner = setup_game(tmp_path, bad_die=True)
    stats = runner.run()
    assert stats["outcome"] == "abort" and stats["consecutive_successes"] == 0
    record = json.loads((tmp_path / "turns.jsonl").read_text())
    assert record["status"] == "failed" and record["roll_result"]["events"][-1]["state"] == "failed"
    assert json.loads((tmp_path / "frozen_stats.json").read_text())["outcome"] == "abort"


def test_operator_abort_and_turn_budget(tmp_path: Path) -> None:
    runner = setup_game(tmp_path)
    runner.abort_requested = lambda: True
    assert runner.run()["outcome"] == "abort"
    assert (tmp_path / "turns.jsonl").is_file()


def test_roll_rejects_stale_frames_and_incomplete_physical_config(tmp_path: Path) -> None:
    runner = setup_game(tmp_path)
    runner.roller.read_die = lambda: DieRead(6, str(tmp_path / "front.ppm"), 0)
    assert "stale" in str(runner.roller.run().error)
    with pytest.raises(ValueError, match="verification"):
        RollMachine(runner.roller.arm, runner.roller.read_die, Path("config/synria_roll.json"))


def test_any_failed_attempt_resets_three_turn_success_sequence(tmp_path: Path) -> None:
    runner = setup_game(tmp_path, fail_trial=True)
    runner.max_turns = 4
    values = iter([6] * 3 + [1] * 3 + [6] * 3 + [1] * 3)
    ticks = [10.0]

    def read() -> DieRead:
        ticks[0] += 0.001
        return DieRead(next(values), str(tmp_path / "front.ppm"), ticks[0])

    runner.roller.read_die = read
    runner.roller.clock = lambda: ticks[0]
    stats = runner.run()
    assert stats["peak_consecutive_successes"] == 3
    assert stats["consecutive_successes"] == 0
    assert stats["misses"] == 1 and stats["retries_used"] == 1
    assert stats["outcome"] == "abort"


def test_roll_phase_failure_holds_and_records_started_phase(tmp_path: Path) -> None:
    runner = setup_game(tmp_path)
    calls = []

    class BrokenArm:
        def perform(self, phase: str, parameters: dict) -> None:
            calls.append(phase)
            if phase == "shake":
                raise RuntimeError("injected shake fault")

        def hold(self) -> None:
            calls.append("hold")

    runner.roller.arm = BrokenArm()
    result = runner.roller.run()
    assert calls == ["pick_die", "drop_into_cup", "shake", "hold"]
    assert result.value is None and result.error == "injected shake fault"
    assert result.events[-1]["state"] == "failed"


def test_keyboard_abort_preserves_game_statistics(tmp_path: Path) -> None:
    runner = setup_game(tmp_path)

    def interrupt() -> bool:
        raise KeyboardInterrupt

    runner.abort_requested = interrupt
    stats = runner.run()
    assert stats["outcome"] == "abort"
    assert json.loads((tmp_path / "turns.jsonl").read_text())["status"] == "failed"
    assert (tmp_path / "frozen_stats.json").exists()
