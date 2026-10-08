"""Operator sessions with explicit site-owned device adapter bindings."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from isaac.scripts.ludo_stats import aggregate_records

from .embodiment import SynriaEmbodiment, SynriaObservation
from .evaluation import EvalLogWriter, OperatorGrade, load_protocol, score_evaluation
from .game_runner import (
    GameRunner,
    GatedDieReader,
    PolicyRollArm,
    RollMachine,
    validate_roll_config,
)
from .perception import GatedPerception
from .physical_contract import ActionSource, PhysicalDatasetContract
from .policy_client import (
    CommandSink,
    GuardedCommandPath,
    HttpPolicyTransport,
    PolicyClient,
    PolicySafetyConfig,
    add_motion_arguments,
)
from .quality_gates import OBJECT_SUCCESS_LIMITATION, load_limits
from .recorder import _close_all
from .turn_executor import (
    BoardCalibration,
    PhysicalLudoGame,
    PolicyMoveExecutor,
    SquareMove,
    TaskMove,
    TurnExecutor,
)


class OperatorAbort(RuntimeError):
    """Explicit operator cancellation, distinct from a policy or device failure."""


class SessionIO(Protocol):
    """Factory opens read-only sources; command_sink alone creates a publisher.

    The sink must implement the site's verified controlled hold. Completion
    means motion ended; grade independently assesses camera-linked object success.
    """

    def observe(self, task: str) -> SynriaObservation: ...
    def preflight(self) -> dict[str, Any]: ...
    def authorize_motion(self) -> dict[str, Any]: ...
    def completed(self) -> bool: ...
    def roll_completed(self, phase: str) -> bool: ...
    def command_sink(self) -> CommandSink: ...
    def grade(self, move: TaskMove) -> OperatorGrade: ...
    def recover(self, move: TaskMove) -> bool: ...
    def operator_roll(self) -> int: ...
    def capture_front(self) -> tuple[Any, str, float]: ...
    def abort_requested(self) -> bool: ...
    def power_state_end(self) -> str: ...
    def session_evidence(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


def validate_session(
    config: dict[str, Any],
    mode: str,
    repository: Path,
    *,
    enable_motion: bool = False,
) -> PolicySafetyConfig:
    if mode not in {"d2", "d3", "d4", "d5"}:
        raise ValueError("unknown session mode")
    limits = load_limits(Path(config["limits"]))
    if enable_motion and not limits.verified:
        raise ValueError("operator-verified limits required")
    PhysicalDatasetContract(
        config["gripper_type"],
        ActionSource(config["action_source"]),
        config["state_has_velocity"],
        action_lookahead_steps=config["action_lookahead_steps"],
    )
    if not {"command_period_s", "response_timeout_s"} <= config.keys():
        raise ValueError("command period and per-policy response timeout are required")
    safety = PolicySafetyConfig.load(
        Path(config["policy_config"]),
        command_period_s=config["command_period_s"],
        response_timeout_s=config["response_timeout_s"],
    )
    if enable_motion and not safety.verified:
        raise ValueError("operator-verified policy safety config required")
    if "period_s" in config:
        raise ValueError("use command_period_s as the single session command period")
    calibration = BoardCalibration(Path(config["calibration"]), Path(config["reachable"]))
    required = {
        "operator",
        "policy_id",
        "checkpoint_sha256",
        "git_sha",
        "host",
        "utc_date",
        "power_state_start",
        "follower_serial",
        "camera_ids",
    }
    if (
        not config["adapter"]
        or not config["policy_endpoint"]
        or any(
            type(config[k]) is not int or config[k] <= 0
            for k in ("max_steps", "max_turns", "max_attempts")
        )
        or not required <= config["provenance"].keys()
        or not all(config["provenance"][key] for key in required)
    ):
        raise ValueError("complete session configuration and provenance required")
    if mode == "d2":
        protocol = load_protocol(Path(config["protocol"]), repository, config["protocol_sha256"])
        if config["max_attempts"] != protocol["max_attempts"]:
            raise ValueError("retry budget must match the committed protocol")
        if len(config["d2_trials"]) != protocol["trials"]:
            raise ValueError("D2 schedule must match the protocol trial count")
        for trial in config["d2_trials"]:
            square = SquareMove(trial["source"], trial["target"], trial["piece"], "eval")
            if calibration.task(square) is None:
                raise ValueError("D2 scene contains unreachable squares")
        if protocol.get("scene_schedule") != config["d2_trials"]:
            raise ValueError("scene schedule must be frozen in the committed protocol")
    if mode in {"d4", "d5"}:
        if set(config["accuracy_reports"]) != {"board", "tokens", "die"}:
            raise ValueError("board, token and die accuracy reports required")
        GatedPerception(
            Path(config["perception_config"]),
            {key: Path(value) for key, value in config["accuracy_reports"].items()},
            repository,
        )
        roll = json.loads(Path(config["roll_config"]).read_text(encoding="utf-8"))
        validate_roll_config(roll)
        for phase in roll["phases"].values():
            safety.require_command_period(phase["period_s"])
    return safety


def run_session(
    config: dict[str, Any],
    mode: str,
    repository: Path,
    output: Path,
    *,
    enable_motion: bool = False,
    factory: Callable[[dict[str, Any]], SessionIO] | None = None,
) -> dict[str, Any]:
    safety = validate_session(config, mode, repository, enable_motion=enable_motion)
    if output.exists():
        raise FileExistsError("use a new output directory for each session")
    if factory is None:
        module, name = config["adapter"].split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    assert factory is not None
    io = factory({**config, "enable_motion": enable_motion, "session_output": str(output)})
    path: GuardedCommandPath | None = None
    client: PolicyClient | None = None
    provenance: dict[str, Any] | None = None
    try:
        source_evidence = io.preflight()
        if not enable_motion:
            return {
                "motion_enabled": False,
                "validated": True,
                "source_preflight": source_evidence,
                "status": "implemented, unmeasured",
            }
        output.mkdir(parents=True)
        provenance = {
            **config["provenance"],
            "policy_safety": asdict(safety),
            "source_preflight": source_evidence,
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        provenance["source_preflight"] = io.authorize_motion()
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        contract = PhysicalDatasetContract(
            config["gripper_type"],
            ActionSource(config["action_source"]),
            config["state_has_velocity"],
            action_lookahead_steps=config["action_lookahead_steps"],
        )
        client = PolicyClient(
            SynriaEmbodiment(contract),
            HttpPolicyTransport(config["policy_endpoint"]),
            load_limits(Path(config["limits"])),
            safety,
        )
        path = GuardedCommandPath(client, io.command_sink, enable_motion=enable_motion)
        mover = PolicyMoveExecutor(
            path,
            io.observe,
            io.completed,
            max_steps=config["max_steps"],
            period_s=safety.command_period_s,
        )
        calibration = BoardCalibration(Path(config["calibration"]), Path(config["reachable"]))
        if mode == "d2":
            writer = EvalLogWriter(
                output / "EvalLog.jsonl",
                Path(config["protocol"]),
                repository,
                config["protocol_sha256"],
                config["provenance"]["policy_id"],
                data_kind="physical",
            )
            (output / "provenance.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8"
            )
            try:
                for trial in config["d2_trials"]:
                    if io.abort_requested():
                        break
                    square = SquareMove(trial["source"], trial["target"], trial["piece"], "eval")
                    move = calibration.task(square)
                    assert move is not None
                    grades = []
                    for attempt in range(config["max_attempts"]):
                        try:
                            mover.execute(move)
                        except Exception as exc:
                            with (output / "faults.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(
                                    json.dumps(
                                        {
                                            "trial": writer.count + 1,
                                            "attempt": attempt + 1,
                                            "error": str(exc),
                                        }
                                    )
                                    + "\n"
                                )
                            path.hold()
                        grades.append(io.grade(move))
                        if (
                            grades[-1].label == "success"
                            or attempt + 1 == config["max_attempts"]
                            or not io.recover(move)
                        ):
                            break
                    writer.trial(grades, json.dumps(trial, sort_keys=True))
            finally:
                stats = score_evaluation(
                    writer.path, Path(config["protocol"]), repository, output / "frozen_stats.json"
                )
        else:
            executor = TurnExecutor(
                PhysicalLudoGame(),
                calibration,
                mover,
                io.grade,
                output / "turns.jsonl",
                data_kind="physical",
                max_attempts=config["max_attempts"],
                recover=io.recover,
            )
            if mode == "d3":
                try:
                    for _ in range(config["max_turns"]):
                        if io.abort_requested():
                            break
                        record = executor.execute(io.operator_roll())
                        if (
                            record["status"] == "failed"
                            or record.get("reason") == "unreachable square"
                        ):
                            break
                finally:
                    stats = aggregate_records(executor.records, data_kind="physical")
                    stats.update(
                        stage_status="planned", object_success_limitation=OBJECT_SUCCESS_LIMITATION
                    )
                    (output / "provenance.json").write_text(
                        json.dumps(provenance, indent=2), encoding="utf-8"
                    )
                    (output / "frozen_stats.json").write_text(
                        json.dumps(stats, indent=2), encoding="utf-8"
                    )
            else:
                perception = GatedPerception(
                    Path(config["perception_config"]),
                    {key: Path(value) for key, value in config["accuracy_reports"].items()},
                    repository,
                )

                def capture_die() -> tuple[Any, str, float]:
                    image, still, stamp = io.capture_front()
                    if perception.board(image) is None or any(
                        value in {"unlocated", "ambiguous"}
                        for value in perception.tokens(image).values()
                    ):
                        raise ValueError("board/token perception unavailable for turn")
                    return image, still, stamp

                roller = RollMachine(
                    PolicyRollArm(path, io.observe, io.roll_completed),
                    GatedDieReader(perception, capture_die),
                    Path(config["roll_config"]),
                )
                stats = GameRunner(
                    executor,
                    roller,
                    output,
                    provenance,
                    max_turns=config["max_turns"],
                    abort_requested=io.abort_requested,
                ).run()
        return stats
    finally:

        def end_evidence() -> None:
            if output.exists():
                (output / "session_end.json").write_text(
                    json.dumps({"power_state_end": io.power_state_end()}), encoding="utf-8"
                )

        def final_provenance() -> None:
            if provenance is not None:
                provenance["source_preflight"] = io.session_evidence()
                (output / "provenance.json").write_text(
                    json.dumps(provenance, indent=2), encoding="utf-8"
                )

        callbacks = [io.close, final_provenance, end_evidence]
        if client is not None:
            callbacks.insert(0, lambda: client.write_latencies(output / "latencies.jsonl"))
        if path is not None:
            callbacks.insert(0, path.hold)
        _close_all(*callbacks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["d2", "d3", "d4", "d5"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    add_motion_arguments(parser)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run_session(
                config, args.mode, args.repository, args.output, enable_motion=args.enable_motion
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
