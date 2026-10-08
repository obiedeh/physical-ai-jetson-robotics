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

from .physical_contract import CONTRACT_VERSION
from .quality_gates import OBJECT_SUCCESS_LIMITATION

FUNNEL = ("reached", "grasped", "lifted", "placed", "released")
HASH_FIELD = re.compile(r'"protocol_sha256": "[a-f0-9]*"')


def protocol_digest(text: str) -> str:
    """Hash UTF-8 text with the self-referential hash value cleared."""
    if len(HASH_FIELD.findall(text)) != 1:
        raise ValueError("protocol needs exactly one protocol_sha256 field")
    return hashlib.sha256(HASH_FIELD.sub('"protocol_sha256": ""', text).encode()).hexdigest()


def committed_bytes(path: Path, repository: Path) -> bytes:
    relative = path.resolve().relative_to(repository.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=repository, capture_output=True, check=False
    )
    if result.returncode or result.stdout != path.read_bytes():
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


def append_policy_row(ledger: Path, fields: list[str]) -> None:
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
        stream.write("| " + " | ".join(fields) + " |\n")


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
        append_policy_row(args.ledger, args.fields)
    else:
        print(json.dumps(score_evaluation(args.log, args.protocol, args.repository, args.output)))


if __name__ == "__main__":
    main()
