from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from synria_lerobot.evaluation import (
    HASH_FIELD,
    EvalLogWriter,
    OperatorGrade,
    append_policy_row,
    load_protocol,
    protocol_digest,
    score_evaluation,
)


@pytest.fixture
def protocol_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    path = repository / "protocol.md"
    # Historical token logs remain readable after the prospective task change.
    text = (
        "# Legacy token evaluation fixture\n\n```json\n"
        + json.dumps(
            dict(
                trials=20,
                success_threshold=14,
                max_attempts=1,
                scene_schedule=[],
                protocol_sha256="",
            ),
            indent=2,
        )
        + "\n```\n"
    )
    digest = protocol_digest(text)
    path.write_text(HASH_FIELD.sub(f'"protocol_sha256": "{digest}"', text), encoding="utf-8")
    subprocess.run(["git", "add", "protocol.md"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Operator",
            "-c",
            "user.email=operator@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Register evaluation",
        ],
        cwd=repository,
        check=True,
    )
    return repository, path, digest


def grade(tmp_path: Path, success: bool = True) -> OperatorGrade:
    still = tmp_path / "still.ppm"
    still.write_bytes(b"P6\n1 1\n255\n\x80\x80\x80")
    return OperatorGrade(
        "success" if success else "failure",
        "operator",
        str(still),
        10.0,
        True,
        True,
        True,
        success,
        success,
    )


def test_protocol_requires_committed_unchanged_hash(protocol_repo: tuple, tmp_path: Path) -> None:
    repo, protocol, digest = protocol_repo
    assert load_protocol(protocol, repo, digest)["success_threshold"] == 14
    with pytest.raises(ValueError, match="hash"):
        load_protocol(protocol, repo, "0" * 64)
    untracked = repo / "untracked.md"
    untracked.write_bytes(protocol.read_bytes())
    with pytest.raises(ValueError, match="committed"):
        load_protocol(untracked, repo, digest)
    protocol.write_text(protocol.read_text() + "changed\n")
    with pytest.raises(ValueError, match="committed"):
        load_protocol(protocol, repo, digest)


def test_twenty_trials_frozen_stats_and_still_integrity(
    protocol_repo: tuple, tmp_path: Path
) -> None:
    repo, protocol, digest = protocol_repo
    log = tmp_path / "EvalLog.jsonl"
    writer = EvalLogWriter(log, protocol, repo, digest, "policy-1", data_kind="synthetic")
    for index in range(20):
        writer.trial([grade(tmp_path, index < 14)], f"square schedule {index}")
    stats = score_evaluation(log, protocol, repo, tmp_path / "stats.json")
    assert stats["threshold_met"] and stats["first_try"]["rate"] == 0.7
    assert stats["funnel"] == dict(reached=20, grasped=20, lifted=20, placed=14, released=14)
    assert stats["stage_status"] == "planned"
    with pytest.raises(FileExistsError):
        score_evaluation(log, protocol, repo, tmp_path / "stats.json")
    (tmp_path / "still.ppm").write_bytes(b"changed")
    with pytest.raises(ValueError, match="still hash"):
        score_evaluation(log, protocol, repo, tmp_path / "other.json")


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_checkout_line_endings_preserve_content_gate(
    protocol_repo: tuple, line_ending: bytes
) -> None:
    repo, protocol, digest = protocol_repo
    text = protocol.read_bytes().replace(b"\r\n", b"\n")
    protocol.write_bytes(text.replace(b"\n", line_ending))
    assert load_protocol(protocol, repo, digest)["success_threshold"] == 14
    with pytest.raises(ValueError, match="hash"):
        load_protocol(protocol, repo, "0" * 64)
    protocol.write_bytes(protocol.read_bytes().replace(b'"trials": 20', b'"trials": 21'))
    with pytest.raises(ValueError, match="committed"):
        load_protocol(protocol, repo, digest)


def test_threshold_is_protocol_driven_and_incomplete_cannot_pass(
    protocol_repo: tuple,
    tmp_path: Path,
) -> None:
    repo, protocol, _ = protocol_repo
    text = (
        protocol.read_text()
        .replace('"trials": 20', '"trials": 2')
        .replace('"success_threshold": 14', '"success_threshold": 1')
    )
    text = text.replace('"max_attempts": 1', '"max_attempts": 2')
    digest = protocol_digest(text)
    protocol.write_text(HASH_FIELD.sub(f'"protocol_sha256": "{digest}"', text))
    subprocess.run(["git", "add", "protocol.md"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test Operator",
            "-c",
            "user.email=operator@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Register next campaign",
        ],
        cwd=repo,
        check=True,
    )
    writer = EvalLogWriter(
        tmp_path / "eval.jsonl", protocol, repo, digest, "policy-2", data_kind="synthetic"
    )
    writer.trial([grade(tmp_path, False), grade(tmp_path)], "scene")
    stats = score_evaluation(writer.path, protocol, repo, tmp_path / "partial.json")
    assert not stats["threshold_met"] and stats["first_try"]["rate"] == 0
    writer.trial([grade(tmp_path, False)], "scene")
    assert score_evaluation(writer.path, protocol, repo, tmp_path / "final.json")["threshold_met"]
    lines = writer.path.read_text().splitlines()
    meta = json.loads(lines[0])
    meta["protocol_sha256"] = "0" * 64
    writer.path.write_text("\n".join([json.dumps(meta), *lines[1:]]))
    with pytest.raises(ValueError, match="hash"):
        score_evaluation(writer.path, protocol, repo, tmp_path / "wrong.json")


def test_operator_funnel_and_policy_ledger(tmp_path: Path) -> None:
    g = grade(tmp_path)
    with pytest.raises(ValueError, match="cumulative"):
        OperatorGrade(g.label, g.operator, g.still, 1, False, True, True, True, True).evidence()
    ledger = tmp_path / "ledger.md"
    ledger.write_text("# Policy ledger\n")
    row = ["2026-10-08", "policy", "data/hash/100", "recipe", "sha", "eval", "0/20", "retry"]
    append_policy_row(ledger, row, response_timeout_s=0.35)
    assert "recipe; response_timeout_s=0.35" in ledger.read_text()
    with pytest.raises(ValueError, match="already"):
        append_policy_row(ledger, row, response_timeout_s=0.35)


@pytest.mark.parametrize("timeout", [None, True, 0, -1, float("nan"), float("inf"), "0.1"])
def test_policy_ledger_rejects_invalid_timeout_without_changing_history(
    tmp_path: Path, timeout: object
) -> None:
    ledger = tmp_path / "ledger.md"
    history = "# Ledger\n\n| earlier | historical policy | original evidence |\n"
    ledger.write_text(history)
    fields = ["date", "new policy", "data", "recipe", "sha", "eval", "result", "decision"]
    with pytest.raises(ValueError, match="per-policy response timeout"):
        append_policy_row(ledger, fields, response_timeout_s=timeout)
    assert ledger.read_text() == history


def test_policy_ledger_requires_explicit_timeout_and_preserves_existing_rows(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.md"
    history = Path("docs/ludo_flagship/POLICY_LEDGER.md").read_text()
    ledger.write_text(history)
    fields = ["date", "new policy", "data", "recipe", "sha", "eval", "result", "decision"]
    with pytest.raises(TypeError):
        append_policy_row(ledger, fields)
    assert ledger.read_text() == history
    append_policy_row(ledger, fields, response_timeout_s=0.45)
    assert ledger.read_text() == history + (
        "| date | new policy | data | recipe; response_timeout_s=0.45 "
        "| sha | eval | result | decision |\n"
    )
    assert fields[3] == "recipe"
