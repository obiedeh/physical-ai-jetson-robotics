"""Fixed D2 registration, scoring and sessions use synthetic evidence only."""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest
from test_synria_checkpoint_eval import commit, fixed_probe_trial
from test_synria_roll_skills import RollIO, roll_config

from synria_lerobot.evaluation import (
    FUNNEL,
    HASH_FIELD,
    FixedEvalLogWriter,
    FixedTaskGrade,
    fixed_attempt_record,
    load_fixed_protocol,
    protocol_digest,
    score_evaluation,
)
from synria_lerobot.sessions import OperatorAbort, run_fixed_evaluation


def fixed_setup(tmp_path: Path, *, trials: int = 20, threshold: int = 14) -> tuple:
    config, transports = roll_config(tmp_path)
    task = "die_into_cup"
    config["d2_policy"] = dict(config["roll_skills"][task], max_steps=1)
    config["data_kind"] = "synthetic"
    transport = transports[task]
    repository = tmp_path / "protocol-repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    register_fixed_protocol(config, transport.identity.metadata(), repository, trials, threshold)
    return config, transport, repository


def register_fixed_protocol(
    config: dict, metadata: dict, repository: Path, trials: int = 20, threshold: int = 14,
    *, selection_rule: str = "Preselected checkpoint; diagnostic probes never select D2 policy",
) -> None:
    """Commit a synthetic campaign using the explicitly supplied checkpoint identity."""
    path = repository / "protocol.md"
    text = Path("docs/ludo_flagship/D2_EVAL_PROTOCOL.md").read_text(encoding="utf-8")
    protocol = json.loads(re.search(r"```json\s*(.*?)\s*```", text, re.S).group(1))
    scenes = []
    for index in range(trials):
        scene = fixed_probe_trial(f"trial-{index + 1}")
        scene["scene"].update(
            die_position_in_zone=f"marked position {index % 3}", die_face_up=index % 6 + 1
        )
        scenes.append(scene)
    protocol.update(
        trials=trials,
        success_threshold=threshold,
        max_steps=1,
        registered_by="synthetic operator",
        registered_on="2026-10-08",
        selection_rule=selection_rule,
        randomisation_seed=7,
        scene_schedule=scenes,
        **{
            key: metadata[key]
            for key in (
                "task_id",
                "physical_contract",
                "policy_id",
                "checkpoint_content_sha256",
                "response_timeout_s",
            )
        },
        task_definition_sha256=metadata["physical_contract"]["task_definition_sha256"],
        command_period_s=metadata["cadence"]["command_period_s"],
    )
    text = re.sub(
        r"```json\s*.*?\s*```",
        "```json\n" + json.dumps(protocol, indent=2) + "\n```",
        text,
        flags=re.S,
    )
    digest = protocol_digest(text)
    path.write_text(HASH_FIELD.sub(f'"protocol_sha256": "{digest}"', text), encoding="utf-8")
    commit(repository, path)
    config.update(protocol=str(path), protocol_sha256=digest)


class EvalIO(RollIO):
    def __init__(self, root: Path, *, successes: int = 14, unknown: bool = False) -> None:
        super().__init__(root)
        self.successes, self.unknown = successes, unknown

    def confirm_skill(self, task) -> bool:
        return True

    def grade_skill(self, task) -> FixedTaskGrade:
        grade = super().grade_skill(task)
        from dataclasses import replace

        return replace(grade, label="success" if len(self.labels) <= self.successes else "failure")

    def funnel_observations(self) -> dict:
        return dict.fromkeys(FUNNEL, None if self.unknown else False)

    def recover_skill(self, task) -> bool:
        return True


def test_twenty_operator_labels_pass_without_extra_funnel_gate(tmp_path: Path) -> None:
    config, transport, repository = fixed_setup(tmp_path)
    io = EvalIO(tmp_path)
    stats = run_fixed_evaluation(
        config,
        repository,
        tmp_path / "evaluation",
        enable_motion=True,
        factory=lambda _: io,
        transport=transport,
        sleep=lambda _: None,
    )
    assert stats["threshold_met"] and stats["successes"] == stats["operator_object_successes"] == 14
    assert stats["first_try"]["rate"] == 0.7
    assert stats["funnel"] == dict.fromkeys(FUNNEL, 0)
    assert stats["funnel_unknown_trials"] == dict.fromkeys(FUNNEL, 0)
    assert io.sinks == 1 and len(io.offers) == 20 and io.closed
    assert transport.model.reset_count == 20
    assert stats["stage_status"] == "planned" and stats["data_kind"] == "synthetic"
    assert stats["failure_taxonomy_counts"]["operator-labeled object failure"] == 6


@pytest.mark.parametrize("change", ["hash", "task", "checkpoint", "window", "scene", "uncommitted"])
def test_protocol_or_checkpoint_mismatch_refuses_before_io(tmp_path: Path, change: str) -> None:
    config, transport, repository = fixed_setup(tmp_path)
    path = Path(config["protocol"])
    text = path.read_text(encoding="utf-8")
    if change == "hash":
        config["protocol_sha256"] = "0" * 64
    elif change == "uncommitted":
        path.write_text(text + "changed\n", encoding="utf-8")
    else:
        payload = json.loads(re.search(r"```json\s*(.*?)\s*```", text, re.S).group(1))
        if change == "task":
            payload["task_id"] = "cup_return"
        elif change == "checkpoint":
            payload["checkpoint_content_sha256"] = "0" * 64
        elif change == "window":
            payload["physical_contract"]["max_episode_s"] = 5
        else:
            payload["scene_schedule"][1]["scene"]["cup_mark"] = "new goal mark"
        text = re.sub(
            r"```json\s*.*?\s*```", "```json\n" + json.dumps(payload) + "\n```", text, flags=re.S
        )
        config["protocol_sha256"] = protocol_digest(text)
        path.write_text(
            HASH_FIELD.sub(f'"protocol_sha256": "{config["protocol_sha256"]}"', text),
            encoding="utf-8",
        )
        commit(repository, path)
    calls = []
    with pytest.raises(ValueError):
        run_fixed_evaluation(
            config,
            repository,
            tmp_path / "output",
            enable_motion=True,
            factory=lambda _: calls.append(True),
            transport=transport,
        )
    assert not calls and not (tmp_path / "output").exists()


def test_unknown_diagnostics_remain_unknown_and_still_tampering_refuses(tmp_path: Path) -> None:
    config, transport, repository = fixed_setup(tmp_path, trials=2, threshold=1)
    io = EvalIO(tmp_path, successes=1, unknown=True)
    output = tmp_path / "evaluation"
    stats = run_fixed_evaluation(
        config,
        repository,
        output,
        enable_motion=True,
        factory=lambda _: io,
        transport=transport,
        sleep=lambda _: None,
    )
    assert stats["threshold_met"] and stats["threshold"] == 1
    assert stats["funnel"] == dict.fromkeys(FUNNEL)
    assert stats["funnel_unknown_trials"] == dict.fromkeys(FUNNEL, 2)
    io.still.write_bytes(b"modified still")
    with pytest.raises(ValueError, match="still hash"):
        score_evaluation(
            output / "EvalLog.jsonl",
            Path(config["protocol"]),
            repository,
            tmp_path / "tampered.json",
        )


@pytest.mark.parametrize("status", ["aborted", "failed"])
def test_twenty_labels_cannot_pass_noncompleted_terminal_session(
    tmp_path: Path, status: str
) -> None:
    config, transport, repository = fixed_setup(tmp_path)
    writer = FixedEvalLogWriter(
        tmp_path / "log.jsonl",
        Path(config["protocol"]),
        repository,
        config["protocol_sha256"],
        data_kind="synthetic",
    )
    io = EvalIO(tmp_path, successes=20)
    for scene in writer.config["scene_schedule"]:
        grade = FixedTaskGrade(
            "success",
            "operator",
            str(io.still),
            1,
            transport.identity.contract.task_definition,
            (1, 1),
        )
        writer.trial(
            [
                fixed_attempt_record(
                    grade, 1, execution_status="completed", steps=1, scene_confirmed=True, errors=[]
                )
            ],
            scene,
        )
    writer.finish(status, ["terminal fault or abort"])
    stats = score_evaluation(
        writer.path, Path(config["protocol"]), repository, tmp_path / "stats.json"
    )
    assert stats["successes"] == 20 and not stats["threshold_met"]


@pytest.mark.parametrize("graded", [True, False])
def test_fault_preserves_raw_object_label_and_does_not_call_it_operator_failure(
    tmp_path: Path,
    graded: bool,
) -> None:
    config, transport, repository = fixed_setup(tmp_path, trials=2, threshold=1)
    writer = FixedEvalLogWriter(
        tmp_path / "log.jsonl",
        Path(config["protocol"]),
        repository,
        config["protocol_sha256"],
        data_kind="synthetic",
    )
    io = EvalIO(tmp_path)
    grade = (
        FixedTaskGrade(
            "success",
            "operator",
            str(io.still),
            1,
            transport.identity.contract.task_definition,
            (1, 1),
        )
        if graded
        else None
    )
    attempt = fixed_attempt_record(
        grade,
        1,
        execution_status="fault",
        steps=1,
        scene_confirmed=True,
        errors=["synthetic fault"],
    )
    writer.trial([attempt], writer.config["scene_schedule"][0])
    writer.finish("failed", ["synthetic fault"])
    stats = score_evaluation(
        writer.path, Path(config["protocol"]), repository, tmp_path / "stats.json"
    )
    assert stats["successes"] == 0 and stats["operator_object_successes"] == int(graded)
    assert stats["incomplete"] and not stats["threshold_met"]
    assert stats["failure_taxonomy_counts"]["operator-labeled object failure"] == 0
    assert stats["failure_taxonomy_counts"]["execution fault or policy hold"] == 1
    assert stats["failure_taxonomy_counts"]["object outcome unknown"] == int(not graded)
    assert attempt["operator_grade"] is None or attempt["operator_grade"]["label"] == "success"


@pytest.mark.parametrize("reset", ["abort", "decline"])
def test_failed_attempt_is_saved_when_operator_stops_at_reset(tmp_path: Path, reset: str) -> None:
    config, transport, repository = fixed_setup(tmp_path, trials=2, threshold=1)
    path = Path(config["protocol"])
    text = path.read_text(encoding="utf-8").replace('"max_attempts": 1', '"max_attempts": 2')
    config["max_attempts"] = 2
    config["protocol_sha256"] = protocol_digest(text)
    path.write_text(
        HASH_FIELD.sub(f'"protocol_sha256": "{config["protocol_sha256"]}"', text),
        encoding="utf-8",
    )
    commit(repository, path)
    io = EvalIO(tmp_path, successes=0)

    def recover(task):
        if reset == "abort":
            raise OperatorAbort("operator aborted reset")
        return False

    io.recover_skill = recover
    stats = run_fixed_evaluation(
        config,
        repository,
        tmp_path / "evaluation",
        enable_motion=True,
        factory=lambda _: io,
        transport=transport,
        sleep=lambda _: None,
    )
    assert stats["session_status"] == "aborted" and stats["trial_count"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "evaluation/EvalLog.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[1]["attempts"][0]["operator_grade"]["label"] == "failure"
    assert records[1]["attempts"][0]["execution_status"] == "completed"
    assert not stats["threshold_met"] and io.closed


def test_cleanup_interrupt_keeps_trial_and_attempts_remaining_evidence(tmp_path: Path) -> None:
    config, transport, repository = fixed_setup(tmp_path, trials=2, threshold=1)
    io = EvalIO(tmp_path, successes=2)
    closed = []

    def interrupt_close():
        closed.append(True)
        raise KeyboardInterrupt("synthetic cleanup interruption")

    io.close = interrupt_close
    output = tmp_path / "evaluation"
    stats = run_fixed_evaluation(
        config,
        repository,
        output,
        enable_motion=True,
        factory=lambda _: io,
        transport=transport,
        sleep=lambda _: None,
    )
    assert closed and stats["trial_count"] == 2 and stats["session_status"] == "aborted"
    assert not stats["threshold_met"]
    assert (output / "latencies.jsonl").is_file() and (output / "provenance.json").is_file()
    terminal = (output / "EvalLog.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(terminal)["errors"]


@pytest.mark.parametrize("change", ["terminal_errors", "false_completed", "continued_after_fault"])
def test_fixed_scorer_refuses_inconsistent_terminal_or_stopped_history(
    tmp_path: Path,
    change: str,
) -> None:
    config, transport, repository = fixed_setup(tmp_path, trials=2, threshold=1)
    writer = FixedEvalLogWriter(
        tmp_path / "log.jsonl",
        Path(config["protocol"]),
        repository,
        config["protocol_sha256"],
        data_kind="synthetic",
    )
    io = EvalIO(tmp_path)
    grade = FixedTaskGrade(
        "success", "operator", str(io.still), 1, transport.identity.contract.task_definition, (1, 1)
    )
    attempt = fixed_attempt_record(
        grade, 1, execution_status="fault", steps=1, scene_confirmed=True, errors=["fault"]
    )
    writer.trial([attempt], writer.config["scene_schedule"][0])
    writer.finish("failed", ["fault"])
    rows = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    if change == "terminal_errors":
        rows[-1]["errors"] = "not a list"
    elif change == "false_completed":
        rows[-1].update(status="completed", errors=[])
    else:
        second = copy.deepcopy(rows[1])
        second.update(turn=2, trial=writer.config["scene_schedule"][1])
        rows.insert(2, second)
    writer.path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal|stopped execution"):
        score_evaluation(writer.path, Path(config["protocol"]), repository, tmp_path / "stats.json")


def test_protocol_fixture_uses_utf8_under_a_windows_legacy_text_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def legacy_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return original_open(self, mode, buffering, encoding, errors, newline)

    template = Path("docs/ludo_flagship/D2_EVAL_PROTOCOL.md").read_bytes()
    assert template.decode("cp1252") != template.decode("utf-8")
    monkeypatch.setattr(Path, "open", legacy_open)
    config, _, repository = fixed_setup(tmp_path)
    protocol = load_fixed_protocol(Path(config["protocol"]), repository, config["protocol_sha256"])
    assert protocol["trials"] == 20 and protocol["success_threshold"] == 14
