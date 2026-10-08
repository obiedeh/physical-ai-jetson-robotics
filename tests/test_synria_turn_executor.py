from __future__ import annotations

import json
from pathlib import Path

import pytest

from synria_lerobot.evaluation import OperatorGrade
from synria_lerobot.turn_executor import BoardCalibration, PhysicalLudoGame, TaskMove, TurnExecutor


class FakeMover:
    def __init__(self) -> None:
        self.moves: list[TaskMove] = []
        self.holds = 0

    def execute(self, move: TaskMove) -> int:
        self.moves.append(move)
        return 1

    def hold(self) -> None:
        self.holds += 1


def calibration(tmp_path: Path, reachable: bool = True) -> BoardCalibration:
    squares = {"red:yard:0": [0.1, 0.2, 0.3], "track:0": [0.2, 0.2, 0.3]}
    path, reach = tmp_path / "cal.json", tmp_path / "reach.json"
    path.write_text(json.dumps(dict(frame="test", squares=squares)))
    reach.write_text(json.dumps(dict(frame="test", squares=list(squares) if reachable else [])))
    return BoardCalibration(path, reach, require_verified=False)


def grader(tmp_path: Path, success: bool) -> OperatorGrade:
    still = tmp_path / "front.ppm"
    still.write_bytes(b"P6\n1 1\n255\n\x80\x80\x80")
    return OperatorGrade(
        "success" if success else "failure",
        "test operator",
        str(still),
        1,
        True,
        True,
        True,
        success,
        success,
    )


@pytest.mark.parametrize("success", [True, False])
def test_logical_state_changes_only_after_camera_grade(tmp_path: Path, success: bool) -> None:
    game, mover = PhysicalLudoGame(), FakeMover()
    executor = TurnExecutor(
        game,
        calibration(tmp_path),
        mover,
        lambda move: grader(tmp_path, success),
        tmp_path / "turns.jsonl",
        data_kind="synthetic",
    )
    record = executor.execute(6)
    assert record["status"] == ("success" if success else "failed")
    assert game.game.positions[("red", 0)][0] == ("track" if success else "yard")
    assert mover.moves[0].pick_xyz == (0.1, 0.2, 0.3)
    assert json.loads(executor.log.read_text())["attempts"][0]["still_sha256"]
    assert record["stage_status"] == "planned" and mover.holds == 1


def test_unreachable_and_no_legal_moves_never_count_as_attempts(tmp_path: Path) -> None:
    game, mover = PhysicalLudoGame(), FakeMover()
    executor = TurnExecutor(
        game,
        calibration(tmp_path, False),
        mover,
        lambda move: grader(tmp_path, True),
        tmp_path / "turns.jsonl",
        data_kind="synthetic",
    )
    record = executor.execute(6)
    assert record["status"] == "skipped" and record["attempts"] == []
    assert game.game.positions[("red", 0)] == ("yard", 0) and not mover.moves
    assert executor.execute(1)["reason"] == "no legal move"


def test_retries_require_confirmed_reset_and_are_ledgered(tmp_path: Path) -> None:
    game, mover = PhysicalLudoGame(), FakeMover()
    labels = iter([False, True])
    executor = TurnExecutor(
        game,
        calibration(tmp_path),
        mover,
        lambda move: grader(tmp_path, next(labels)),
        tmp_path / "turns.jsonl",
        data_kind="synthetic",
        max_attempts=2,
        recover=lambda move: True,
    )
    record = executor.execute(6)
    assert [a["ok"] for a in record["attempts"]] == [False, True]
    assert record["status"] == "success"


def test_policy_fault_ledgered_and_calibration_unverified(tmp_path: Path) -> None:
    class Broken(FakeMover):
        def execute(self, move: TaskMove) -> int:
            raise TimeoutError("policy stale")

    executor = TurnExecutor(
        PhysicalLudoGame(),
        calibration(tmp_path),
        Broken(),
        lambda move: grader(tmp_path, True),
        tmp_path / "turns.jsonl",
        data_kind="synthetic",
    )
    record = executor.execute(6)
    assert record["status"] == "failed" and record["attempts"][0]["error"] == "policy stale"
    with pytest.raises(ValueError, match="verification"):
        BoardCalibration(tmp_path / "cal.json", tmp_path / "reach.json")
