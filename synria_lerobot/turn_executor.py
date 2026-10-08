"""Transactional physical Ludo planning, calibrated tasks and append-only turns."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ludo_engine.game import LudoGame

from .embodiment import SynriaObservation
from .evaluation import OperatorGrade, attempt_record
from .policy_client import GuardedCommandPath
from .quality_gates import OBJECT_SUCCESS_LIMITATION


def square_id(color: str, position: tuple[Any, ...]) -> str:
    kind, index = position
    if kind == "done":
        kind, index = "home", 5
    return f"track:{index}" if kind == "track" else f"{color}:{kind}:{index}"


@dataclass(frozen=True)
class SquareMove:
    source: str
    target: str
    piece: str
    reason: str


@dataclass(frozen=True)
class TurnPlan:
    description: str
    moves: tuple[SquareMove, ...]


class PhysicalLudoGame:
    """Plan on a copy; only a graded successful turn changes logical piece state."""

    def __init__(self, game: LudoGame | None = None) -> None:
        self.game = game or LudoGame()
        self.pending: LudoGame | None = None

    def plan_turn(self, roll: int) -> TurnPlan:
        if type(roll) is not int or not 1 <= roll <= 6 or self.pending is not None:
            raise ValueError("valid roll required with no uncommitted plan")
        self.pending = copy.deepcopy(self.game)
        move = self.pending.take_turn(roll)
        if move is None:
            return TurnPlan(f"roll {roll}: no legal move", ())
        moves = [
            SquareMove(
                square_id(color, move.dst),
                square_id(color, self.pending.positions[(color, token)]),
                f"{color}:{token}",
                "capture-return",
            )
            for color, token in move.captures
        ]
        moves.append(
            SquareMove(
                square_id(move.color, move.src),
                square_id(move.color, move.dst),
                f"{move.color}:{move.token}",
                "move",
            )
        )
        return TurnPlan(f"{move.color} rolls {roll}", tuple(moves))

    def commit(self) -> None:
        if self.pending is None:
            raise ValueError("no pending turn")
        self.game, self.pending = self.pending, None

    def reject(self) -> None:
        self.pending = None

    @property
    def winner(self) -> str | None:
        return str(self.game.finished_order[0]) if self.game.finished_order else None


@dataclass(frozen=True)
class TaskMove:
    source: str
    target: str
    pick_xyz: tuple[float, ...]
    place_xyz: tuple[float, ...]
    piece: str
    reason: str


class BoardCalibration:
    def __init__(self, path: Path, reachable_path: Path, *, require_verified: bool = True) -> None:
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.reachable_sha256 = hashlib.sha256(reachable_path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding="utf-8"))
        reachable = json.loads(reachable_path.read_text(encoding="utf-8"))
        if require_verified and any(
            not d.get("verified_by") or not d.get("verified_on") for d in (data, reachable)
        ):
            raise ValueError("calibration and reachable squares require operator verification")
        if data["frame"] != reachable["frame"]:
            raise ValueError("calibration frame mismatch")
        self.frame = data["frame"]
        self.squares = {key: tuple(float(v) for v in xyz) for key, xyz in data["squares"].items()}
        if any(
            len(xyz) != 3 or not all(math.isfinite(v) for v in xyz) for xyz in self.squares.values()
        ):
            raise ValueError("square coordinates must be finite XYZ metres")
        self.reachable = set(reachable["squares"])
        if not self.reachable <= self.squares.keys():
            raise ValueError("reachable square lacks calibration")

    def task(self, move: SquareMove) -> TaskMove | None:
        if not {move.source, move.target} <= self.reachable:
            return None
        return TaskMove(
            move.source,
            move.target,
            self.squares[move.source],
            self.squares[move.target],
            move.piece,
            move.reason,
        )


class MoveExecutor(Protocol):
    def execute(self, move: TaskMove) -> int: ...
    def hold(self) -> None: ...


class PolicyMoveExecutor:
    def __init__(
        self,
        path: GuardedCommandPath,
        observe: Callable[[str], SynriaObservation],
        finished: Callable[[], bool],
        *,
        max_steps: int,
        period_s: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_steps <= 0 or not math.isfinite(period_s) or period_s <= 0:
            raise ValueError("bounded positive step count and period required")
        self.path, self.observe, self.finished = path, observe, finished
        self.max_steps, self.period_s, self.sleep = max_steps, period_s, sleep

    def execute(self, move: TaskMove) -> int:
        task = (
            f"Move {move.piece} from {move.pick_xyz} to {move.place_xyz} metres; "
            "release and retract"
        )
        try:
            for step in range(self.max_steps):
                decision = self.path.step(self.observe(task), reset=step == 0)
                if decision.hold:
                    raise RuntimeError(decision.reason)
                if self.finished():
                    return step + 1
                self.sleep(self.period_s)
            raise TimeoutError("policy step budget exhausted")
        finally:
            self.hold()

    def hold(self) -> None:
        self.path.hold()


class TurnExecutor:
    def __init__(
        self,
        game: PhysicalLudoGame,
        calibration: BoardCalibration,
        policy: MoveExecutor,
        grade: Callable[[TaskMove], OperatorGrade],
        log: Path,
        *,
        data_kind: str,
        max_attempts: int = 1,
        recover: Callable[[TaskMove], bool] = lambda move: False,
    ) -> None:
        if data_kind not in {"physical", "synthetic"} or max_attempts <= 0:
            raise ValueError("data kind and positive attempt budget required")
        if log.exists():
            raise FileExistsError("use a new session log; resuming needs board reconciliation")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.touch(exist_ok=False)
        self.game, self.calibration, self.policy, self.grade = game, calibration, policy, grade
        self.log, self.data_kind, self.max_attempts, self.recover = (
            log,
            data_kind,
            max_attempts,
            recover,
        )
        self.records: list[dict[str, Any]] = []

    def record(self, record: dict[str, Any]) -> dict[str, Any]:
        record.update(
            data_kind=self.data_kind,
            stage_status="planned",
            calibration_sha256=self.calibration.sha256,
            reachable_sha256=self.calibration.reachable_sha256,
            object_success_limitation=OBJECT_SUCCESS_LIMITATION,
        )
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.records.append(record)
        return record

    def execute(self, roll: int, *, roll_source: str = "operator") -> dict[str, Any]:
        number = len(self.records) + 1
        record: dict[str, Any] = {
            "turn": number,
            "roll": roll,
            "roll_source": roll_source,
            "attempts": [],
            "ok": [],
            "commands": 0,
        }
        try:
            plan = self.game.plan_turn(roll=roll)
            record.update(desc=plan.description, commands=len(plan.moves))
            tasks = [self.calibration.task(move) for move in plan.moves]
            if not tasks:
                self.policy.hold()
                self.game.commit()
                return self.record({**record, "status": "skipped", "reason": "no legal move"})
            if any(task is None for task in tasks):
                self.policy.hold()
                self.game.reject()
                return self.record({**record, "status": "skipped", "reason": "unreachable square"})
            for command, task in enumerate(tasks):
                assert task is not None
                succeeded = False
                for attempt in range(1, self.max_attempts + 1):
                    detail: dict[str, Any] = {
                        "attempt": attempt,
                        "command": command,
                        "source": task.source,
                        "target": task.target,
                        "ok": False,
                        "steps": 0,
                        "lane": "synria_physical_policy",
                        "err_mm": None,
                        "final_tilt_deg": None,
                        "grasped": False,
                    }
                    try:
                        detail["steps"] = self.policy.execute(task)
                        detail.update(attempt_record(self.grade(task), attempt, detail["steps"]))
                    except Exception as exc:
                        detail["error"] = str(exc)
                    record["attempts"].append(detail)
                    succeeded = detail["ok"]
                    if succeeded or attempt == self.max_attempts or not self.recover(task):
                        break
                record["ok"].append(succeeded)
                if not succeeded:
                    break
            success = len(record["ok"]) == len(tasks) and all(record["ok"])
            self.policy.hold()
            if success:
                self.game.commit()
            else:
                self.game.reject()
            record["status"] = "success" if success else "failed"
        except Exception as exc:
            self.game.reject()
            record.update(status="failed", error=str(exc))
            try:
                self.policy.hold()
            except Exception as hold_error:
                record["hold_error"] = str(hold_error)
        return self.record(record)
