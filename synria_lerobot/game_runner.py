"""Configured die-roll state machine and camera-graded physical Ludo sessions."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from isaac.scripts.ludo_stats import aggregate_records

from .embodiment import SynriaObservation
from .perception import GatedPerception
from .policy_client import GuardedCommandPath
from .quality_gates import OBJECT_SUCCESS_LIMITATION
from .turn_executor import TurnExecutor


class RollState(str, Enum):
    PICK_DIE = "pick_die"
    DROP_INTO_CUP = "drop_into_cup"
    SHAKE = "shake"
    INVERT = "invert"
    READ = "read"
    DONE = "done"
    FAILED = "failed"


MOTION_PHASES = (RollState.PICK_DIE, RollState.DROP_INTO_CUP, RollState.SHAKE, RollState.INVERT)


class RollArm(Protocol):
    def perform(self, phase: str, parameters: dict[str, Any]) -> None: ...
    def hold(self) -> None: ...


class PolicyRollArm:
    def __init__(
        self,
        path: GuardedCommandPath,
        observe: Callable[[str], SynriaObservation],
        completed: Callable[[str], bool],
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.path, self.observe, self.completed, self.sleep = path, observe, completed, sleep

    def perform(self, phase: str, parameters: dict[str, Any]) -> None:
        task = json.dumps({"phase": phase, **parameters}, sort_keys=True)
        for step in range(parameters["max_steps"]):
            decision = self.path.step(self.observe(task), reset=step == 0)
            if decision.hold:
                raise RuntimeError(decision.reason)
            if self.completed(phase):
                return
            self.sleep(parameters["period_s"])
        raise TimeoutError(f"roll phase exhausted step budget: {phase}")

    def hold(self) -> None:
        self.path.hold()


@dataclass(frozen=True)
class DieRead:
    value: int | None
    still: str
    monotonic_s: float


class GatedDieReader:
    """Bind runtime die reads to the component's committed accuracy evidence."""

    def __init__(
        self, perception: GatedPerception, capture: Callable[[], tuple[Any, str, float]]
    ) -> None:
        if "die" not in perception.enabled:
            raise ValueError("die accuracy report required")
        self.perception, self.capture = perception, capture

    def __call__(self) -> DieRead:
        image, still, stamp = self.capture()
        return DieRead(self.perception.die(image), still, stamp)


@dataclass(frozen=True)
class RollResult:
    value: int | None
    events: list[dict[str, Any]]
    config_sha256: str
    error: str | None


class RollMachine:
    def __init__(
        self,
        arm: RollArm,
        read_die: Callable[[], DieRead],
        config_path: Path,
        *,
        synthetic: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if not synthetic and not isinstance(read_die, GatedDieReader):
            raise ValueError("physical die reader requires accuracy verification")
        if not synthetic and isinstance(read_die, GatedDieReader) and read_die.perception.synthetic:
            raise ValueError("synthetic accuracy cannot enable a physical roll")
        self.synthetic = synthetic
        self.config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if not synthetic and (not self.config["verified_by"] or not self.config["verified_on"]):
            raise ValueError("roll parameters require physical operator verification")
        for phase in MOTION_PHASES:
            params = self.config["phases"][phase.value]
            if (
                not params["task"]
                or type(params["max_steps"]) is not int
                or params["max_steps"] <= 0
                or not math.isfinite(params["period_s"])
                or params["period_s"] <= 0
                or len(params["target_xyz_m"]) != 3
                or not all(math.isfinite(x) for x in params["target_xyz_m"])
            ):
                raise ValueError(f"incomplete parameters for {phase.value}")
        reads = self.config["read"]
        if not 1 <= reads["stable_reads"] <= reads["max_reads"] or any(
            not math.isfinite(reads[k]) or reads[k] <= 0 for k in ("period_s", "max_age_s")
        ):
            raise ValueError("invalid die-read bounds")
        self.arm, self.read_die, self.clock, self.sleep = arm, read_die, clock, sleep
        self.state: RollState | None = None

    def run(self) -> RollResult:
        events: list[dict[str, Any]] = []
        value, error = None, None
        try:
            for phase in MOTION_PHASES:
                self.state = phase
                events.append(
                    {"state": phase.value, "status": "started", "monotonic_s": self.clock()}
                )
                self.arm.perform(phase.value, self.config["phases"][phase.value])
                events.append(
                    {"state": phase.value, "status": "completed", "monotonic_s": self.clock()}
                )
            self.state = RollState.READ
            previous, consecutive = None, 0
            last_stamp = -math.inf
            reads = self.config["read"]
            for _ in range(reads["max_reads"]):
                reading = self.read_die()
                if (
                    not math.isfinite(reading.monotonic_s)
                    or not 0 <= self.clock() - reading.monotonic_s <= reads["max_age_s"]
                    or reading.monotonic_s <= last_stamp
                ):
                    raise ValueError("stale or repeated die frame")
                last_stamp = reading.monotonic_s
                still_hash = hashlib.sha256(Path(reading.still).read_bytes()).hexdigest()
                events.append({"state": "read", **asdict(reading), "still_sha256": still_hash})
                valid = type(reading.value) is int and 1 <= reading.value <= 6
                consecutive = consecutive + 1 if valid and reading.value == previous else int(valid)
                previous = reading.value if valid else None
                if consecutive >= reads["stable_reads"]:
                    value = reading.value
                    break
                self.sleep(reads["period_s"])
            if value is None:
                raise ValueError("die reading failed to settle")
            self.state = RollState.DONE
        except Exception as exc:
            self.state, error = RollState.FAILED, str(exc)
        finally:
            try:
                self.arm.hold()
            except Exception as exc:
                self.state, error = RollState.FAILED, f"hold failed: {exc}"
                value = None
        events.append({"state": self.state.value, "error": error, "monotonic_s": self.clock()})
        return RollResult(value if error is None else None, events, self.config_sha256, error)


class GameRunner:
    def __init__(
        self,
        executor: TurnExecutor,
        roller: RollMachine,
        output: Path,
        provenance: dict[str, Any],
        *,
        max_turns: int,
        abort_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("positive game turn budget required")
        if executor.data_kind == "physical" and roller.synthetic:
            raise ValueError("synthetic roll cannot be recorded as physical")
        required = {
            "operator",
            "policy_id",
            "git_sha",
            "host",
            "utc_date",
            "power_state_start",
            "follower_serial",
            "camera_ids",
        }
        if not required <= provenance.keys():
            raise ValueError("incomplete game provenance")
        output.mkdir(parents=True, exist_ok=True)
        self.executor, self.roller, self.output = executor, roller, output
        self.max_turns, self.abort_requested = max_turns, abort_requested
        provenance.update(
            data_kind=executor.data_kind, object_success_limitation=OBJECT_SUCCESS_LIMITATION
        )
        with (output / "provenance.json").open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, indent=2)

    def run(self) -> dict[str, Any]:
        consecutive, peak = 0, 0
        outcome, reason = "abort", "turn budget exhausted"
        try:
            for _ in range(self.max_turns):
                if self.abort_requested():
                    reason = "operator abort"
                    break
                roll = self.roller.run()
                if roll.error or roll.value is None:
                    self.executor.record(
                        {
                            "turn": len(self.executor.records) + 1,
                            "commands": 0,
                            "status": "failed",
                            "desc": "physical roll failed",
                            "attempts": [],
                            "ok": [],
                            "roll": None,
                            "roll_source": "physical",
                            "roll_result": asdict(roll),
                        }
                    )
                    consecutive, reason = 0, roll.error or "missing roll"
                    break
                record = self.executor.execute(roll.value, roll_source="physical")
                # Preserve roll evidence in a separate append-only stream linked by turn number.
                with (self.output / "rolls.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"turn": record["turn"], **asdict(roll)}) + "\n")
                clean = record["status"] == "success" and all(a["ok"] for a in record["attempts"])
                consecutive = consecutive + 1 if clean else 0
                peak = max(peak, consecutive)
                if record["status"] == "failed" or record.get("reason") == "unreachable square":
                    reason = "turn failed or unreachable; reconcile board before resuming"
                    break
                if self.executor.game.winner:
                    outcome, reason = "winner", "game reached a winner"
                    break
        except Exception as exc:
            consecutive, reason = 0, f"session fault: {exc}"
            self.executor.record(
                {
                    "turn": len(self.executor.records) + 1,
                    "commands": 0,
                    "status": "failed",
                    "desc": reason,
                    "attempts": [],
                    "ok": [],
                }
            )
        finally:
            try:
                self.executor.policy.hold()
            except Exception as exc:
                outcome, reason, consecutive = "abort", f"final hold failed: {exc}", 0
        stats = aggregate_records(self.executor.records, data_kind=self.executor.data_kind)
        stats.update(
            outcome=outcome,
            reason=reason,
            winner=self.executor.game.winner,
            turns=len(self.executor.records),
            consecutive_successes=consecutive,
            peak_consecutive_successes=peak,
            d4_sequence_observed=peak >= 3,
            stage_status="planned",
            object_success_limitation=OBJECT_SUCCESS_LIMITATION,
        )
        with (self.output / "frozen_stats.json").open("x", encoding="utf-8") as stream:
            json.dump(stats, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return stats
