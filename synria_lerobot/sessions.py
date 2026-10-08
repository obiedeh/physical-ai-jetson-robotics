"""Operator sessions with explicit site-owned device adapter bindings."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from isaac.scripts.ludo_stats import aggregate_records

from .embodiment import SynriaEmbodiment, SynriaObservation
from .evaluation import (
    EvalLogWriter,
    FixedTaskGrade,
    OperatorGrade,
    load_protocol,
    score_evaluation,
)
from .game_runner import (
    FixedSkillRollArm,
    GameRunner,
    GatedDieReader,
    OperatorDieReader,
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
    PolicyTransport,
    add_motion_arguments,
)
from .quality_gates import OBJECT_SUCCESS_LIMITATION, load_limits
from .recorder import _close_all
from .task_registry import TASK_IDS, TaskDefinition
from .turn_executor import (
    BoardCalibration,
    FixedSkillExecutor,
    OperatorAbort,
    PhysicalLudoGame,
    PolicyMoveExecutor,
    SquareMove,
    TaskMove,
    TurnExecutor,
)


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
    def grade_skill(self, task: TaskDefinition) -> FixedTaskGrade: ...
    def confirm_skill(self, task: TaskDefinition) -> bool: ...
    def recover(self, move: TaskMove) -> bool: ...
    def recover_skill(self, task: TaskDefinition) -> bool: ...
    def operator_roll(self, *, still: str | None = None) -> int: ...
    def capture_front(self, *, require_native: bool = False) -> tuple[Any, str, float]: ...
    def abort_requested(self) -> bool: ...
    def abort_pending(self) -> bool: ...
    def power_state_end(self) -> str: ...
    def session_evidence(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class FixedSkillBinding:
    contract: PhysicalDatasetContract
    metadata: dict[str, Any]
    safety: PolicySafetyConfig
    transport: PolicyTransport
    max_steps: int


def bind_fixed_skill(
    config: dict[str, Any],
    specification: dict[str, Any],
    task_id: str,
    *,
    enable_motion: bool = False,
    transport: Any = None,
) -> FixedSkillBinding:
    """Verify local receipt and endpoint identity before opening any device source."""
    from .policy_server import CheckpointIdentity

    identity = CheckpointIdentity.load(
        Path(specification["checkpoint"]),
        Path(specification["receipt"]),
        Path(config["task_registry"]),
    )
    metadata = identity.metadata()
    if identity.contract.task_id != task_id:
        raise ValueError("checkpoint task differs from the requested fixed skill")
    safety = PolicySafetyConfig.load(
        Path(config["policy_config"]),
        command_period_s=config["command_period_s"],
        response_timeout_s=specification["response_timeout_s"],
    )
    if safety.response_timeout_s != metadata["response_timeout_s"]:
        raise ValueError("per-policy timeout differs from checkpoint receipt")
    safety.require_command_period(metadata["cadence"]["command_period_s"])
    if type(specification.get("max_steps")) is not int or specification["max_steps"] <= 0:
        raise ValueError("explicit positive per-skill command budget required")
    limits = load_limits(Path(config["limits"]))
    if enable_motion and (not safety.verified or not limits.verified):
        raise ValueError("operator-verified policy safety and limits required")
    endpoint = specification.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("explicit policy endpoint required")
    selected = transport if transport is not None else HttpPolicyTransport(endpoint)
    remote = selected.metadata(safety.response_timeout_s)
    if json.dumps(remote, sort_keys=True, allow_nan=False) != json.dumps(
        metadata,
        sort_keys=True,
        allow_nan=False,
    ):
        raise ValueError("policy endpoint metadata differs from frozen checkpoint receipt")
    return FixedSkillBinding(
        identity.contract,
        metadata,
        safety,
        selected,
        specification["max_steps"],
    )


def fixed_io_configuration(config: dict[str, Any], binding: FixedSkillBinding) -> dict[str, Any]:
    """Share physical sources without substituting one skill's task identity for another."""
    from .ros_adapter import validate_adapter_config

    settings = json.loads(Path(config["adapter_config"]).read_text(encoding="utf-8"))
    validate_adapter_config(settings, binding.safety)
    shapes = binding.metadata["model_feature_shapes"]
    expected = [3, settings["image_height"], settings["image_width"]]
    if any(shapes["observation.images." + camera] != expected for camera in ("wrist", "front")):
        raise ValueError("adapter image dimensions differ from checkpoint feature shapes")
    required = {
        "operator",
        "git_sha",
        "host",
        "utc_date",
        "power_state_start",
        "follower_serial",
        "camera_ids",
        "scene",
    }
    if not config.get("adapter") or not all(config.get("provenance", {}).get(k) for k in required):
        raise ValueError("complete fixed-scene session provenance required")
    return {
        **config,
        **binding.contract.as_dict(),
        "response_timeout_s": binding.safety.response_timeout_s,
        "provenance": {
            **config["provenance"],
            "policy_id": binding.metadata["policy_id"],
            "checkpoint_sha256": binding.metadata["checkpoint_content_sha256"],
        },
    }


def run_roll_skills(
    config: dict[str, Any],
    repository: Path,
    output: Path,
    *,
    enable_motion: bool = False,
    factory: Callable[[dict[str, Any]], SessionIO] | None = None,
    transports: dict[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """A standalone fixed-scene roll; no board move, game or milestone advancement."""
    if output.exists():
        raise FileExistsError("use a new output directory for each roll session")
    specifications = config["roll_skills"]
    if set(specifications) != set(TASK_IDS):
        raise ValueError("exactly three registered skill policies required")
    from .checkpoint_eval import _artifact_destination

    _artifact_destination(
        output,
        tuple(Path(specifications[task]["checkpoint"]) for task in TASK_IDS),
        (
            Path(config["task_registry"]),
            *(Path(specifications[task]["receipt"]) for task in TASK_IDS),
        ),
    )
    bindings = {
        task: bind_fixed_skill(
            config,
            specifications[task],
            task,
            enable_motion=enable_motion,
            transport=(transports or {}).get(task),
        )
        for task in TASK_IDS
    }
    first = bindings[TASK_IDS[0]]
    for binding in bindings.values():
        if (
            any(
                getattr(binding.contract, key) != getattr(first.contract, key)
                for key in (
                    "gripper_type",
                    "state_has_velocity",
                    "action_source",
                )
            )
            or binding.metadata["model_feature_shapes"] != first.metadata["model_feature_shapes"]
        ):
            raise ValueError(
                "skill policies require matching physical conventions and image shapes"
            )
    for key in ("policy_id", "checkpoint_content_sha256"):
        if len({binding.metadata[key] for binding in bindings.values()}) != len(TASK_IDS):
            raise ValueError("three distinct task-bound policies/checkpoints required")
    io_config = fixed_io_configuration(config, first)
    roll_config = config["fixed_roll"]
    validate_roll_config(roll_config)
    if roll_config.get("mode") != "fixed_scene_skills":
        raise ValueError("explicit fixed-scene roll mode required")
    perception = None
    if roll_config["read"]["source"] == "perception":
        perception = GatedPerception(
            Path(config["perception_config"]),
            {"die": Path(config["accuracy_reports"]["die"])},
            repository,
        )
        # The existing harness operates on dataset-sized RGB, never native still pixels.
        if (
            roll_config["read"].get("image_shape")
            != first.metadata["model_feature_shapes"]["observation.images.front"]
        ):
            raise ValueError("die accuracy input shape must explicitly match stored front images")
        channels, height, width = first.metadata["model_feature_shapes"]["observation.images.front"]
        perception.require_input_shape("die", (height, width, channels))
    if factory is None:
        module, name = config["adapter"].split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    assert factory is not None
    io = factory({**io_config, "enable_motion": enable_motion, "session_output": str(output)})
    clients: dict[str, PolicyClient] = {}
    shared_sink: CommandSink | None = None
    report: dict[str, Any] = {
        "kind": "fixed_scene_roll",
        "status": "failed",
        "motion_enabled": enable_motion,
        "stage_status": "planned",
        "d4_qualifying": False,
        "errors": [],
        "object_success_limitation": OBJECT_SUCCESS_LIMITATION,
        "provenance": config["provenance"],
        "skills": {
            task: {
                **binding.metadata,
                "max_steps": binding.max_steps,
                "policy_safety": asdict(binding.safety),
            }
            for task, binding in bindings.items()
        },
    }
    try:
        report["source_preflight"] = io.preflight()
        if perception is not None:
            report["perception_report_hashes"] = perception.report_hashes
        output.mkdir(parents=True)
        if not enable_motion:
            report["status"] = "read_only"
        else:
            report["source_preflight"] = io.authorize_motion()
            shared_sink = io.command_sink()
            executors = {}
            for task, binding in bindings.items():
                client = PolicyClient(
                    SynriaEmbodiment(binding.contract),
                    binding.transport,
                    load_limits(Path(config["limits"])),
                    binding.safety,
                )
                clients[task] = client
                path = GuardedCommandPath(client, lambda: shared_sink, enable_motion=True)
                executors[task] = FixedSkillExecutor(
                    path,
                    io.observe,
                    io.abort_pending,
                    max_steps=binding.max_steps,
                    period_s=binding.safety.command_period_s,
                    sleep=sleep,
                )

            def record(attempt: dict[str, Any]) -> None:
                with (output / "skill_attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {**attempt, "policy": report["skills"][attempt["task_id"]]},
                            allow_nan=False,
                        )
                        + "\n"
                    )

            def capture() -> tuple[Any, str, float]:
                return io.capture_front(require_native=True)

            arm = FixedSkillRollArm(executors, io.confirm_skill, io.grade_skill, record, capture)

            def capture_perception() -> tuple[Any, str, float]:
                import cv2

                image, still, stamp = io.capture_front()
                if getattr(image, "shape", None) != (height, width, channels):
                    raise ValueError("die image dimensions differ from committed accuracy input")
                # Accuracy images are loaded by cv2.imread (BGR), not policy RGB.
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), still, stamp

            reader = (
                OperatorDieReader(capture, lambda still: io.operator_roll(still=still))
                if perception is None
                else GatedDieReader(perception, capture_perception)
            )
            result = RollMachine(arm, reader, roll_config, sleep=sleep).run()
            report["roll"] = asdict(result)
            report["status"] = "completed" if result.error is None else "failed"
    except BaseException as error:
        report["status"] = (
            "aborted"
            if isinstance(error, (OperatorAbort, KeyboardInterrupt, EOFError))
            else "failed"
        )
        report["errors"].append(str(error) or type(error).__name__)
    finally:
        callbacks: list[Callable[[], Any]] = []
        if shared_sink is not None:
            callbacks.append(shared_sink.hold)
        callbacks.append(io.close)
        for task, client in clients.items():
            callbacks.append(partial(client.write_latencies, output / f"{task}_latencies.jsonl"))
        for callback in callbacks:
            try:
                callback()
            except BaseException as error:
                report["status"] = "failed"
                report["errors"].append(str(error) or type(error).__name__)
        try:
            report["source_preflight"] = io.session_evidence()
            report["power_state_end"] = io.power_state_end()
        except BaseException as error:
            report["status"] = "failed"
            report["errors"].append(str(error) or type(error).__name__)
        output.mkdir(parents=True, exist_ok=True)
        (output / "roll_result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return report


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
    contract = PhysicalDatasetContract(
        config["gripper_type"],
        ActionSource(config["action_source"]),
        config["state_has_velocity"],
        action_lookahead_steps=config["action_lookahead_steps"],
        task_id=config["task_id"],
        task_definition=TaskDefinition.from_metadata(config),
        recording_purpose=config["recording_purpose"],
    )
    contract.require_qualifying()
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
    if enable_motion and mode in {"d3", "d4", "d5"}:
        raise ValueError("token moves and goal conditioning are deferred; use fixed roll skills")
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
            task_id=config["task_id"],
            task_definition=TaskDefinition.from_metadata(config),
            recording_purpose=config["recording_purpose"],
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


def roll_skills_main() -> None:
    parser = argparse.ArgumentParser(description="Run three fixed-scene roll skills without a game")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    add_motion_arguments(parser)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run_roll_skills(
                config,
                args.repository,
                args.output,
                enable_motion=args.enable_motion,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
