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
from .evaluation import FixedTaskGrade
from .perception import GatedPerception
from .policy_client import GuardedCommandPath
from .quality_gates import OBJECT_SUCCESS_LIMITATION
from .task_registry import TASK_IDS, TaskDefinition
from .turn_executor import FixedSkillExecutor, OperatorAbort, TurnExecutor


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
        self.path.client.config.require_command_period(parameters["period_s"])
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


class OperatorDieReader:
    """An operator's camera-linked value, not a perception accuracy claim."""

    def __init__(
        self,
        capture: Callable[[], tuple[Any, str, float]],
        read_value: Callable[[str], int],
    ) -> None:
        self.capture, self.read_value = capture, read_value

    def __call__(self) -> DieRead:
        _, still, stamp = self.capture()
        value = self.read_value(still)
        if type(value) is not int or not 1 <= value <= 6:
            raise ValueError("operator die value must be an integer from 1 through 6")
        return DieRead(value, still, stamp)


class FixedSkillRollArm:
    """One shared arm executes three independently identified policies."""

    def __init__(
        self,
        skills: dict[str, FixedSkillExecutor],
        confirm: Callable[[TaskDefinition], bool],
        grade: Callable[[TaskDefinition], FixedTaskGrade],
        record: Callable[[dict[str, Any]], None],
        capture: Callable[[], tuple[Any, str, float]],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if tuple(skills) != TASK_IDS or any(
            executor.task.task_id != name for name, executor in skills.items()
        ):
            raise ValueError("exactly three ordered task-bound skill executors required")
        self.skills, self.grade, self.record, self.capture, self.clock = (
            skills,
            grade,
            record,
            capture,
            clock,
        )
        self.active: FixedSkillExecutor | None = None
        self.confirm = confirm

    def perform(self, phase: str, parameters: dict[str, Any]) -> None:
        skill = self.skills[phase]
        self.active = skill
        attempt: dict[str, Any] = {
            "kind": "skill_attempt",
            "task_id": phase,
            "status": "started",
            "started_monotonic_s": self.clock(),
            "operator_grade": None,
            "scene_confirmed": False,
            "offered_steps": 0,
            "max_steps": skill.max_steps,
            "period_s": skill.period_s,
            "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
            **skill.task.metadata(),
        }
        self.record(dict(attempt))
        failure: BaseException | None = None
        try:
            skill.hold()
            skill.check_abort()
            confirmed = self.confirm(skill.task)
            if type(confirmed) is not bool or not confirmed:
                raise OperatorAbort("operator did not confirm the next fixed skill")
            attempt["scene_confirmed"] = True
            skill.execute()
            skill.check_abort()
            # execute has already held; a command budget never grades itself.
            grade = self.grade(skill.task)
            if grade.task_definition.sha256 != skill.task.sha256:
                raise ValueError("grade task differs from the executed skill")
            attempt["operator_grade"] = grade.evidence()
            attempt["status"] = grade.label
            if grade.label != "success":
                raise RuntimeError(f"operator marked {phase} failure")
        except BaseException as error:
            failure = error
            if attempt["status"] == "started":
                attempt["status"] = (
                    "aborted"
                    if isinstance(error, (OperatorAbort, KeyboardInterrupt, EOFError))
                    else "failed"
                )
            attempt["error"] = str(error) or type(error).__name__
        finally:
            try:
                skill.hold()
            except BaseException as error:
                attempt["status"] = "failed"
                attempt["hold_error"] = str(error) or type(error).__name__
                failure = failure or error
            if attempt["operator_grade"] is None:
                try:
                    _, still, stamp = self.capture()
                    content = Path(still).read_bytes()
                    if not content:
                        raise ValueError("empty failure camera still")
                    attempt["failure_still"] = {
                        "path": still,
                        "source_monotonic_s": stamp,
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                except BaseException as error:
                    attempt["missing_still_reason"] = str(error) or type(error).__name__
            attempt["offered_steps"] = skill.offered_steps
            attempt["ended_monotonic_s"] = self.clock()
            try:
                self.record(attempt)
            except BaseException as error:
                if failure is not None:
                    raise RuntimeError(
                        f"attempt failed: {failure}; attempt ledger also failed: {error}"
                    ) from failure
                raise
        if failure is not None:
            raise failure

    def hold(self) -> None:
        if self.active is not None:
            self.active.hold()


@dataclass(frozen=True)
class RollResult:
    value: int | None
    events: list[dict[str, Any]]
    config_sha256: str
    error: str | None


def validate_roll_config(config: dict[str, Any], *, synthetic: bool = False) -> None:
    if config.get("mode") == "fixed_scene_skills":
        if config.get("skills") != list(TASK_IDS):
            raise ValueError("fixed-scene roll must use the three registered skills in order")
        if config.get("read", {}).get("source") not in {"operator", "perception"}:
            raise ValueError("explicit operator or gated-perception die source required")
        if config["read"]["source"] == "operator":
            return
        reads = config["read"]
        if (
            any(type(reads.get(k)) is not int for k in ("stable_reads", "max_reads"))
            or not (1 <= reads["stable_reads"] <= reads["max_reads"])
            or any(
                type(reads.get(k)) not in (int, float)
                or not math.isfinite(reads[k])
                or reads[k] <= 0
                for k in ("period_s", "max_age_s")
            )
        ):
            raise ValueError("invalid fixed-scene perception read bounds")
        return
    if not synthetic and (not config["verified_by"] or not config["verified_on"]):
        raise ValueError("roll parameters require physical operator verification")
    for phase in MOTION_PHASES:
        params = config["phases"][phase.value]
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
    reads = config["read"]
    if (
        any(type(reads[k]) is not int for k in ("stable_reads", "max_reads"))
        or not 1 <= reads["stable_reads"] <= reads["max_reads"]
        or any(not math.isfinite(reads[k]) or reads[k] <= 0 for k in ("period_s", "max_age_s"))
    ):
        raise ValueError("invalid die-read bounds")


class RollMachine:
    def __init__(
        self,
        arm: RollArm,
        read_die: Callable[[], DieRead],
        config_path: Path | dict[str, Any],
        *,
        synthetic: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(config_path, Path)
            else json.loads(json.dumps(config_path, allow_nan=False))
        )
        self.fixed_skills = self.config.get("mode") == "fixed_scene_skills"
        if self.fixed_skills and not isinstance(arm, FixedSkillRollArm):
            raise ValueError("fixed-scene roll requires task-bound executors")
        operator_read = self.fixed_skills and isinstance(read_die, OperatorDieReader)
        if self.fixed_skills and operator_read != (self.config["read"]["source"] == "operator"):
            raise ValueError("declared die source differs from reader")
        if not synthetic and not operator_read and not isinstance(read_die, GatedDieReader):
            raise ValueError("physical die reader requires accuracy verification")
        if not synthetic and isinstance(read_die, GatedDieReader) and read_die.perception.synthetic:
            raise ValueError("synthetic accuracy cannot enable a physical roll")
        self.synthetic = synthetic
        encoded = (
            config_path.read_bytes()
            if isinstance(config_path, Path)
            else json.dumps(self.config, sort_keys=True, allow_nan=False).encode()
        )
        self.config_sha256 = hashlib.sha256(encoded).hexdigest()
        validate_roll_config(self.config, synthetic=synthetic)
        self.arm, self.read_die, self.clock, self.sleep = arm, read_die, clock, sleep
        self.state: RollState | None = None

    def run(self) -> RollResult:
        events: list[dict[str, Any]] = []
        value, error = None, None
        try:
            phases = (
                TASK_IDS if self.fixed_skills else tuple(phase.value for phase in MOTION_PHASES)
            )
            for phase in phases:
                self.state = RollState(phase) if not self.fixed_skills else None
                events.append({"state": phase, "status": "started", "monotonic_s": self.clock()})
                self.arm.perform(phase, {} if self.fixed_skills else self.config["phases"][phase])
                events.append({"state": phase, "status": "completed", "monotonic_s": self.clock()})
            self.state = RollState.READ
            previous, consecutive = None, 0
            last_stamp = -math.inf
            reads = self.config["read"]
            operator_read = isinstance(self.read_die, OperatorDieReader)
            for _ in range(1 if operator_read else reads["max_reads"]):
                reading = self.read_die()
                if (
                    not math.isfinite(reading.monotonic_s)
                    or not 0 <= self.clock() - reading.monotonic_s
                    or (
                        not operator_read
                        and self.clock() - reading.monotonic_s > reads["max_age_s"]
                    )
                    or reading.monotonic_s <= last_stamp
                ):
                    raise ValueError("stale or repeated die frame")
                last_stamp = reading.monotonic_s
                still_hash = hashlib.sha256(Path(reading.still).read_bytes()).hexdigest()
                events.append(
                    {
                        "state": "read",
                        **asdict(reading),
                        "still_sha256": still_hash,
                        "source": "operator" if operator_read else "perception",
                    }
                )
                valid = type(reading.value) is int and 1 <= reading.value <= 6
                consecutive = consecutive + 1 if valid and reading.value == previous else int(valid)
                previous = reading.value if valid else None
                if consecutive >= (1 if operator_read else reads["stable_reads"]):
                    value = reading.value
                    break
                self.sleep(reads["period_s"])
            if value is None:
                raise ValueError("die reading failed to settle")
            self.state = RollState.DONE
        except (Exception, KeyboardInterrupt) as exc:
            self.state, error = RollState.FAILED, str(exc) or type(exc).__name__
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
        except (Exception, KeyboardInterrupt) as exc:
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
