"""Committed protocols, camera-linked operator grading and frozen Synria scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from isaac.scripts.ludo_stats import aggregate_records

from .physical_contract import CONTRACT_VERSION, PhysicalDatasetContract
from .quality_gates import OBJECT_SUCCESS_LIMITATION
from .task_registry import TaskDefinition, validate_fixed_scene_schedule

FUNNEL = ("reached", "grasped", "lifted", "placed", "released")
HASH_FIELD = re.compile(r'"protocol_sha256": "[a-f0-9]*"')
FIXED_D2_KIND = "synria_fixed_skill_d2_v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_fixed_protocol(config: dict[str, Any]) -> PhysicalDatasetContract:
    if config.get("kind") != FIXED_D2_KIND or config.get("task_id") != "die_into_cup":
        raise ValueError("the first fixed-scene D2 protocol must be die_into_cup")
    physical = config.get("physical_contract")
    if type(physical) is not dict or type(physical.get("state_has_velocity")) is not bool:
        raise ValueError("complete frozen physical contract required")
    contract = PhysicalDatasetContract.from_dict(physical)
    contract.require_qualifying()
    if contract.task_id != config["task_id"] or config.get("task_definition_sha256") != (
        contract.task_definition.sha256
    ):
        raise ValueError("protocol task/definition differs from its physical contract")
    for key in ("policy_id", "registered_by", "registered_on", "selection_rule"):
        if type(config.get(key)) is not str or not config[key].strip():
            raise ValueError("prospective operator, policy and independent selection rule required")
    if (
        type(config.get("checkpoint_content_sha256")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}",
            config["checkpoint_content_sha256"],
        )
        is None
    ):
        raise ValueError("frozen checkpoint content hash required")
    for key in ("command_period_s", "response_timeout_s"):
        value = config.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not math.isfinite(value) or value <= 0)
        ):
            raise ValueError("frozen command period and per-policy timeout required")
    if type(config.get("max_steps")) is not int or config["max_steps"] <= 0:
        raise ValueError("frozen positive command budget required")
    if type(config.get("randomisation_seed")) is not int:
        raise ValueError("prospective scene-order seed required")
    validate_fixed_scene_schedule(config.get("scene_schedule"), contract.task_id)
    if len(config["scene_schedule"]) != config["trials"]:
        raise ValueError("frozen scene schedule must match the protocol trial count")
    return contract


def load_fixed_protocol(path: Path, repository: Path, expected_hash: str) -> dict[str, Any]:
    config = load_protocol(path, repository, expected_hash)
    validate_fixed_protocol(config)
    return config


def protocol_digest(text: str) -> str:
    """Hash UTF-8 text with the self-referential hash value cleared."""
    if len(HASH_FIELD.findall(text)) != 1:
        raise ValueError("protocol needs exactly one protocol_sha256 field")
    return hashlib.sha256(HASH_FIELD.sub('"protocol_sha256": ""', text).encode()).hexdigest()


def committed_bytes(path: Path, repository: Path) -> bytes:
    """Require committed text, allowing only Git's LF/CRLF checkout conversion."""
    relative = path.resolve().relative_to(repository.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repository, capture_output=True, check=False
    )
    if result.returncode or result.stdout.replace(b"\r\n", b"\n") != path.read_bytes().replace(
        b"\r\n", b"\n"
    ):
        raise ValueError(f"file must be committed and unchanged: {relative}")
    return result.stdout


def load_protocol(path: Path, repository: Path, expected_hash: str) -> dict[str, Any]:
    committed_bytes(path, repository)
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match is None:
        raise ValueError("protocol needs a JSON configuration block")
    config: dict[str, Any] = json.loads(match.group(1))
    digest = protocol_digest(text)
    if config.get("protocol_sha256") != digest or expected_hash != digest:
        raise ValueError("protocol hash mismatch")
    trials, threshold = config["trials"], config["success_threshold"]
    if type(trials) is not int or type(threshold) is not int or not 0 < threshold <= trials:
        raise ValueError("invalid trial count or threshold")
    if type(config["max_attempts"]) is not int or config["max_attempts"] <= 0:
        raise ValueError("invalid protocol attempt budget")
    return config


@dataclass(frozen=True)
class OperatorGrade:
    label: str
    operator: str
    still: str
    still_timestamp_s: float
    reached: bool
    grasped: bool
    lifted: bool
    placed: bool
    released: bool

    def evidence(self) -> dict[str, Any]:
        if self.label not in {"success", "failure"} or not self.operator.strip():
            raise ValueError("operator and success/failure label required")
        if not math.isfinite(self.still_timestamp_s) or self.still_timestamp_s < 0:
            raise ValueError("invalid camera timestamp")
        stages = [getattr(self, name) for name in FUNNEL]
        if any(type(value) is not bool for value in stages):
            raise ValueError("funnel fields must be booleans")
        if any(stages[i] and not stages[i - 1] for i in range(1, len(stages))):
            raise ValueError("funnel stages must be cumulative")
        if self.label == "success" and not all(stages):
            raise ValueError("success requires every funnel stage")
        image = Path(self.still).read_bytes()
        if not image:
            raise ValueError("empty camera still")
        return {**asdict(self), "still_sha256": hashlib.sha256(image).hexdigest()}


@dataclass(frozen=True)
class FixedTaskGrade:
    """Registry outcome label; token funnel observations never decide success."""

    label: str
    operator: str
    still: str
    still_timestamp_s: float
    task_definition: TaskDefinition
    native_resolution: tuple[int, int]
    funnel: dict[str, bool | None] | None = None

    def evidence(self) -> dict[str, Any]:
        if self.label not in {"success", "failure"} or not self.operator.strip():
            raise ValueError("operator and success/failure label required")
        if (
            type(self.still_timestamp_s) not in (int, float)
            or not math.isfinite(self.still_timestamp_s)
            or self.still_timestamp_s < 0
        ):
            raise ValueError("invalid camera timestamp")
        if len(self.native_resolution) != 2 or any(
            type(side) is not int or side <= 0 for side in self.native_resolution
        ):
            raise ValueError("native camera dimensions are required")
        funnel = self.funnel if self.funnel is not None else dict.fromkeys(FUNNEL)
        if set(funnel) != set(FUNNEL) or any(
            value is not None and type(value) is not bool for value in funnel.values()
        ):
            raise ValueError("diagnostic funnel values must be boolean or unknown")
        image = Path(self.still).read_bytes()
        if not image:
            raise ValueError("empty camera still")
        return {
            **self.task_definition.metadata(),
            "grading_rule": "fixed_scene_task_outcome",
            "label": self.label,
            "operator": self.operator,
            "still": self.still,
            "still_timestamp_s": self.still_timestamp_s,
            "native_resolution": list(self.native_resolution),
            "funnel": dict(funnel),
            "still_sha256": hashlib.sha256(image).hexdigest(),
        }


def attempt_record(grade: OperatorGrade, attempt: int, steps: int = 0) -> dict[str, Any]:
    return {
        **grade.evidence(),
        "attempt": attempt,
        "ok": grade.label == "success",
        "steps": steps,
        "lane": "synria_physical_policy",
        "err_mm": None,
        "final_tilt_deg": None,
        "hold_armed": False,
    }


class EvalLogWriter:
    def __init__(
        self,
        path: Path,
        protocol: Path,
        repository: Path,
        protocol_hash: str,
        policy_id: str,
        *,
        data_kind: str,
    ) -> None:
        self.config = load_protocol(protocol, repository, protocol_hash)
        if self.config.get("kind") == FIXED_D2_KIND:
            raise ValueError("fixed-task protocols require camera-linked fixed-task records")
        if data_kind not in {"physical", "synthetic"} or not policy_id.strip():
            raise ValueError("policy id and explicit data kind required")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.count = 0
        with path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "kind": "metadata",
                        "protocol_sha256": protocol_hash,
                        "policy_id": policy_id,
                        "data_kind": data_kind,
                        "contract_version": CONTRACT_VERSION,
                        "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
                    }
                )
                + "\n"
            )

    def trial(self, grades: list[OperatorGrade], scene: str) -> None:
        if self.count >= self.config["trials"] or not grades or not scene.strip():
            raise ValueError("trial budget exhausted, empty trial, or missing scene")
        if any(g.label == "success" for g in grades[:-1]):
            raise ValueError("cannot retry after success")
        if len(grades) > self.config["max_attempts"]:
            raise ValueError("trial exceeds pre-registered attempt budget")
        attempts = [attempt_record(grade, i + 1) for i, grade in enumerate(grades)]
        record = {
            "turn": self.count + 1,
            "desc": scene,
            "commands": 1,
            "ok": [attempts[-1]["ok"]],
            "attempts": attempts,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.count += 1


def score_evaluation(log: Path, protocol: Path, repository: Path, output: Path) -> dict[str, Any]:
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    meta, records = lines[0], lines[1:]
    config = load_protocol(protocol, repository, meta["protocol_sha256"])
    if config.get("kind") == FIXED_D2_KIND:
        return _score_fixed_evaluation(lines, config, output)
    if meta["contract_version"] != CONTRACT_VERSION or meta["data_kind"] not in {
        "physical",
        "synthetic",
    }:
        raise ValueError("wrong embodiment contract or data kind")
    if [r["turn"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("duplicate or out-of-order trial ids")
    if len(records) > config["trials"]:
        raise ValueError("too many trials")
    funnel = dict.fromkeys(FUNNEL, 0)
    for record in records:
        attempts = record["attempts"]
        if len(attempts) > config["max_attempts"]:
            raise ValueError("trial exceeds pre-registered attempt budget")
        if not attempts or [a["attempt"] for a in attempts] != list(range(1, len(attempts) + 1)):
            raise ValueError("invalid attempt sequence")
        for i, attempt in enumerate(attempts):
            grade = OperatorGrade(**{k: attempt[k] for k in OperatorGrade.__dataclass_fields__})
            evidence = grade.evidence()
            if evidence["still_sha256"] != attempt["still_sha256"]:
                raise ValueError("camera still hash mismatch")
            if attempt["ok"] != (grade.label == "success"):
                raise ValueError("score differs from operator label")
            if attempt["ok"] and i < len(attempts) - 1:
                raise ValueError("attempt after success")
        for name in FUNNEL:
            funnel[name] += int(any(a[name] for a in attempts))
    stats = aggregate_records(records, data_kind=meta["data_kind"])
    successes = sum(record["attempts"][-1]["ok"] for record in records)
    stats.update(
        {
            "protocol_sha256": meta["protocol_sha256"],
            "policy_id": meta["policy_id"],
            "funnel": funnel,
            "trial_count": len(records),
            "successes": successes,
            "required_trials": config["trials"],
            "threshold": config["success_threshold"],
            "threshold_met": len(records) == config["trials"]
            and successes >= config["success_threshold"],
            "stage_status": "planned",
            "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return stats


def fixed_attempt_record(
    grade: FixedTaskGrade | None,
    attempt: int,
    *,
    execution_status: str,
    steps: int,
    scene_confirmed: bool,
    errors: list[str],
) -> dict[str, Any]:
    evidence = grade.evidence() if grade is not None else None
    success = grade.label == "success" if grade is not None else None
    return {
        "attempt": attempt,
        "operator_grade": evidence,
        "operator_object_success": success,
        "execution_status": execution_status,
        "scene_confirmed": scene_confirmed,
        "steps": steps,
        "errors": list(errors),
        "ok": execution_status == "completed" and success is True,
        "funnel": evidence["funnel"] if evidence else dict.fromkeys(FUNNEL),
        "lane": "synria_physical_policy",
        "err_mm": None,
        "final_tilt_deg": None,
        "hold_armed": False,
    }


def _validate_fixed_attempt(attempt: dict[str, Any], config: dict[str, Any]) -> None:
    if type(attempt.get("steps")) is not int or not 0 <= attempt["steps"] <= config["max_steps"]:
        raise ValueError("attempt command count differs from frozen budget")
    if (
        type(attempt.get("scene_confirmed")) is not bool
        or attempt.get("execution_status")
        not in {
            "completed",
            "fault",
            "policy_hold",
            "aborted",
            "declined",
        }
        or type(attempt.get("errors")) is not list
        or any(type(error) is not str for error in attempt["errors"])
    ):
        raise ValueError("invalid execution status, scene confirmation or fault evidence")
    evidence = attempt.get("operator_grade")
    success = None
    if evidence is not None:
        task = TaskDefinition.from_metadata(evidence)
        if _canonical(task.as_dict()) != _canonical(config["physical_contract"]["task_definition"]):
            raise ValueError("operator grade task differs from the frozen protocol")
        grade = FixedTaskGrade(
            evidence["label"],
            evidence["operator"],
            evidence["still"],
            evidence["still_timestamp_s"],
            task,
            tuple(evidence["native_resolution"]),
            evidence["funnel"],
        )
        if _canonical(grade.evidence()) != _canonical(evidence):
            raise ValueError("camera still hash or fixed-task evidence mismatch")
        success = grade.label == "success"
    if attempt.get("operator_object_success") is not success:
        raise ValueError("raw object score differs from operator label")
    completed = attempt["execution_status"] == "completed"
    if completed and (
        not attempt["scene_confirmed"]
        or evidence is None
        or (attempt["steps"] != config["max_steps"] or attempt["errors"])
    ):
        raise ValueError("completed attempt requires its confirmed budget and label")
    if type(attempt.get("ok")) is not bool or attempt["ok"] != (completed and success is True):
        raise ValueError("score differs from completed execution and operator label")
    expected_funnel = evidence["funnel"] if evidence else dict.fromkeys(FUNNEL)
    if _canonical(attempt.get("funnel")) != _canonical(expected_funnel):
        raise ValueError("funnel diagnostics differ from their operator evidence")


def _validate_fixed_trial(
    attempts: list[dict[str, Any]],
    scene: dict[str, Any],
    config: dict[str, Any],
    index: int,
) -> None:
    if index >= config["trials"] or not attempts or len(attempts) > config["max_attempts"]:
        raise ValueError("trial exceeds the frozen budget")
    if _canonical(scene) != _canonical(config["scene_schedule"][index]):
        raise ValueError("trial scene differs from its frozen order")
    for number, attempt in enumerate(attempts):
        if type(attempt["attempt"]) is not int or attempt["attempt"] != number + 1:
            raise ValueError("invalid attempt sequence")
        _validate_fixed_attempt(attempt, config)
        if attempt["ok"] and number < len(attempts) - 1:
            raise ValueError("cannot retry after qualified success")
        if attempt["execution_status"] != "completed" and number < len(attempts) - 1:
            raise ValueError("cannot retry after a stopped execution")


class FixedEvalLogWriter:
    def __init__(
        self,
        path: Path,
        protocol: Path,
        repository: Path,
        protocol_hash: str,
        *,
        data_kind: str,
    ) -> None:
        self.config = load_fixed_protocol(protocol, repository, protocol_hash)
        if data_kind not in {"physical", "synthetic"}:
            raise ValueError("explicit physical or synthetic evaluation kind required")
        self.path, self.count, self.ended = path, 0, False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._append(
            {
                "kind": "metadata",
                "grading_rule": FIXED_D2_KIND,
                "protocol_sha256": protocol_hash,
                "data_kind": data_kind,
                "contract_version": CONTRACT_VERSION,
                **{
                    key: self.config[key]
                    for key in (
                        "task_id",
                        "task_definition_sha256",
                        "physical_contract",
                        "policy_id",
                        "checkpoint_content_sha256",
                        "command_period_s",
                        "response_timeout_s",
                        "max_steps",
                    )
                },
                "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
            },
            create=True,
        )

    def _append(self, record: dict[str, Any], *, create: bool = False) -> None:
        with self.path.open("x" if create else "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def trial(self, attempts: list[dict[str, Any]], scene: dict[str, Any]) -> None:
        if self.ended:
            raise ValueError("evaluation has already ended")
        _validate_fixed_trial(attempts, scene, self.config, self.count)
        self._append(
            {
                "kind": "trial",
                "turn": self.count + 1,
                "trial": scene,
                "desc": json.dumps(scene, sort_keys=True),
                "commands": 1,
                "ok": [attempts[-1]["ok"]],
                "attempts": attempts,
            }
        )
        self.count += 1

    def finish(self, status: str, errors: list[str]) -> None:
        if self.ended or status not in {"completed", "aborted", "failed"}:
            raise ValueError("evaluation must end once with an explicit status")
        self._append({"kind": "session_end", "status": status, "errors": errors})
        self.ended = True


def _score_fixed_evaluation(
    lines: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    validate_fixed_protocol(config)
    meta = lines[0]
    if (
        meta.get("kind") != "metadata"
        or meta.get("grading_rule") != FIXED_D2_KIND
        or meta.get("data_kind")
        not in {
            "physical",
            "synthetic",
        }
        or meta.get("contract_version") != CONTRACT_VERSION
    ):
        raise ValueError("wrong fixed-task log contract or data kind")
    for key in (
        "task_id",
        "task_definition_sha256",
        "physical_contract",
        "policy_id",
        "checkpoint_content_sha256",
        "command_period_s",
        "response_timeout_s",
        "max_steps",
    ):
        if _canonical(meta.get(key)) != _canonical(config[key]):
            raise ValueError("EvalLog task/checkpoint binding differs from its committed protocol")
    if (
        len(lines) < 2
        or lines[-1].get("kind") != "session_end"
        or (lines[-1].get("status") not in {"completed", "aborted", "failed"})
    ):
        raise ValueError("fixed evaluation requires its final session status")
    records = lines[1:-1]
    terminal = lines[-1]
    if (
        type(terminal.get("errors")) is not list
        or any(type(error) is not str for error in terminal["errors"])
        or (terminal["status"] == "completed" and terminal["errors"])
    ):
        raise ValueError("invalid or inconsistent terminal error evidence")
    if len(records) > config["trials"]:
        raise ValueError("too many fixed evaluation trials")
    for index, record in enumerate(records):
        if (
            record.get("kind") != "trial"
            or type(record.get("turn")) is not int
            or (record["turn"] != index + 1)
        ):
            raise ValueError("out-of-order fixed trial ids")
        _validate_fixed_trial(record["attempts"], record["trial"], config, index)
        stopped = any(a["execution_status"] != "completed" for a in record["attempts"])
        if stopped and (index != len(records) - 1 or terminal["status"] == "completed"):
            raise ValueError("evaluation cannot continue or complete after a stopped execution")
        if record.get("ok") != [record["attempts"][-1]["ok"]]:
            raise ValueError("trial score differs from its final attempt")
    stats = aggregate_records(records, data_kind=meta["data_kind"])
    attempts = [attempt for record in records for attempt in record["attempts"]]
    funnel = {}
    unknown = {}
    for name in FUNNEL:
        observed = [
            any(a["funnel"][name] is True for a in r["attempts"])
            for r in records
            if any(a["funnel"][name] is not None for a in r["attempts"])
        ]
        funnel[name] = sum(observed) if observed else None
        unknown[name] = len(records) - len(observed)
    successes = sum(record["attempts"][-1]["ok"] for record in records)
    stats.update(
        {
            "protocol_sha256": meta["protocol_sha256"],
            "policy_id": meta["policy_id"],
            "task_id": meta["task_id"],
            "task_definition_sha256": meta["task_definition_sha256"],
            "checkpoint_content_sha256": meta["checkpoint_content_sha256"],
            "funnel": funnel,
            "funnel_unknown_trials": unknown,
            "trial_count": len(records),
            "successes": successes,
            "operator_object_successes": sum(
                r["attempts"][-1]["operator_object_success"] is True for r in records
            ),
            "operator_object_success_attempts": sum(
                a["operator_object_success"] is True for a in attempts
            ),
            "required_trials": config["trials"],
            "threshold": config["success_threshold"],
            "threshold_met": len(records) == config["trials"]
            and lines[-1]["status"] == "completed"
            and successes >= config["success_threshold"],
            "session_status": lines[-1]["status"],
            "incomplete": len(records) < config["trials"],
            "stage_status": "planned",
            "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
            "failure_taxonomy_counts": {
                "operator-labeled object failure": sum(
                    a["operator_object_success"] is False for a in attempts
                ),
                "execution fault or policy hold": sum(
                    a["execution_status"] in {"fault", "policy_hold"} for a in attempts
                ),
                "aborted or declined": sum(
                    a["execution_status"] in {"aborted", "declined"} for a in attempts
                ),
                "object outcome unknown": sum(
                    a["operator_object_success"] is None for a in attempts
                ),
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return stats


def append_policy_row(ledger: Path, fields: list[str], *, response_timeout_s: float) -> None:
    if (
        type(response_timeout_s) not in (int, float)
        or not math.isfinite(response_timeout_s)
        or response_timeout_s <= 0
    ):
        raise ValueError("per-policy response timeout must be positive, finite and not boolean")
    if len(fields) != 8 or any(not value.strip() for value in fields):
        raise ValueError("eight nonempty policy ledger fields required")
    if any("|" in value or "\n" in value or "\r" in value for value in fields):
        raise ValueError("ledger fields must be single-line table cells")
    text = ledger.read_text(encoding="utf-8")
    if any(
        line.split("|")[2].strip() == fields[1]
        for line in text.splitlines()
        if line.startswith("|") and len(line.split("|")) >= 4
    ):
        raise ValueError("policy id already recorded")
    with ledger.open("a", encoding="utf-8") as stream:
        row = list(fields)
        row[3] += f"; response_timeout_s={response_timeout_s!r}"
        stream.write("| " + " | ".join(row) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register-protocol")
    register.add_argument("protocol", type=Path)
    score = sub.add_parser("score")
    score.add_argument("--log", type=Path, required=True)
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--repository", type=Path, default=Path.cwd())
    score.add_argument("--output", type=Path, required=True)
    row = sub.add_parser("policy-row")
    row.add_argument("--ledger", type=Path, required=True)
    row.add_argument("--response-timeout-s", type=float, required=True)
    row.add_argument("fields", nargs=8)
    args = parser.parse_args()
    if args.command == "register-protocol":
        text = args.protocol.read_text(encoding="utf-8")
        digest = protocol_digest(text)
        args.protocol.write_text(
            HASH_FIELD.sub(f'"protocol_sha256": "{digest}"', text), encoding="utf-8", newline="\n"
        )
        print(digest)
    elif args.command == "policy-row":
        append_policy_row(args.ledger, args.fields, response_timeout_s=args.response_timeout_s)
    else:
        print(json.dumps(score_evaluation(args.log, args.protocol, args.repository, args.output)))


if __name__ == "__main__":
    main()
