"""Static Phase B preflight tests use declared fixtures, never an Isaac or hardware session."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from test_command_agent import FixtureSimulation
from typer.testing import CliRunner

from agents.command_agent import cli
from agents.command_agent.harness import (
    CAMPAIGN_SOURCE_PATHS,
    campaign_preflight,
    campaign_report,
    campaign_schedule,
    evaluate,
    summarize_campaign,
)
from agents.command_agent.planners import DeterministicPlanner
from agents.command_agent.schema import SKILL_IDS, skill_call

SUITE_PATH = "reports/agents/command_agent/command_suite.json"
HOST = {
    "device": "fixture",
    "python_version": "3.12.3",
    "jetpack_version": None,
    "power_mode": None,
    "cpu_configuration": {
        "logical_cpu_count": 1,
        "affinity": [0],
        "scaling_governor": "fixture",
        "omp_num_threads": None,
    },
}


def git(root: Path, *args: str) -> str:
    """Keep all test commits confined to a temporary fixture repository."""
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root: Path) -> str:
    """Commit fixture inputs under the operator identity, without changing a real branch."""
    git(root, "add", ".")
    return git(
        root,
        "-c",
        "user.name=oedeh",
        "-c",
        "user.email=obiedeh@gmail.com",
        "commit",
        "-qm",
        "Commit hermetic campaign fixture",
    ) or git(root, "rev-parse", "HEAD")


def pin(root: Path, path: str, value: Any) -> dict[str, str]:
    """Write and hash one small, explicitly artificial input."""
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json.dumps(value).encode()
    p.write_bytes(data)
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def campaign(tmp_path: Path) -> dict[str, Any]:
    """Build a fake committed deployment solely to test static preflight assertions."""
    root = tmp_path
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    repo = Path(__file__).resolve().parents[1]
    for name in CAMPAIGN_SOURCE_PATHS:
        pin(root, name, (repo / name).read_bytes())
    suite = pin(root, SUITE_PATH, (repo / SUITE_PATH).read_bytes())
    scene = pin(root, "fixture/scene.txt", b"hermetic scene placeholder, not an Isaac scene")
    library = pin(root, "fixture/library.json", {"fixture_only": True})
    adapter = pin(
        root, "fixture/adapter.py", b"# No simulation adapter is executed by this test.\n"
    )
    host = pin(root, "fixture/host.json", HOST)
    base = commit(root)
    qualifications = {}
    for skill_id in SKILL_IDS:
        trace = pin(root, f"fixture/{skill_id}_trace.json", {"fixture_only": True})
        qualifications[skill_id] = pin(
            root,
            f"fixture/{skill_id}.json",
            {
                "schema_version": "command_isaac_qualification_v1",
                "status": "measured",
                "backend": "isaac",
                "skill_id": skill_id,
                "harness_commit": base,
                "scene_sha256": scene["sha256"],
                "library_sha256": library["sha256"],
                "adapter_sha256": adapter["sha256"],
                "reset_procedure": "Hermetic fixture assertion only; no reset is performed.",
                "object_success_rule": "Hermetic fixture assertion only; no object is observed.",
                "trace": trace,
            },
        )
    profile = {
        "schema_version": "command_campaign_profile_v1",
        "status": "planned",
        "campaign_id": "hermetic_static_preflight",
        "preregistration_commit": base,
        "suite": suite,
        "harness_commit": base,
        "isaac_version": "fixture-not-runtime",
        "simulation_domain_id": 73,
        "scene": scene,
        "skill_library": library,
        "simulator_adapter": adapter,
        "host_configuration": host,
        "qualifications": qualifications,
    }
    pin(root, "profile.json", profile)
    commit(root)
    return {"root": root, "profile": profile}


def inspect(campaign: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Supply explicit fixture runtime facts; do not discover or initialize ROS."""
    return campaign_preflight(
        Path("profile.json"),
        root=campaign["root"],
        confirm_5090_released=kwargs.pop("release", True),
        provenance=kwargs.pop("provenance", HOST),
        ros_available=kwargs.pop("ros", True),
        **kwargs,
    )


def change_profile(campaign: dict[str, Any], **fields: Any) -> None:
    """Commit prospective fixture changes so failures isolate the check under test."""
    campaign["profile"].update(fields)
    pin(campaign["root"], "profile.json", campaign["profile"])
    commit(campaign["root"])


def test_static_preflight_preserves_all_80_trials_and_contacts_nothing(campaign: Any) -> None:
    result = inspect(campaign)
    assert result["ready"], result["blockers"]
    assert result["scheduled_trials"] == 80
    assert sum(r["group"] == "benign" for r in result["schedule"]) == 35
    assert sum(r["group"] != "benign" for r in result["schedule"]) == 45
    assert [r["seed"] for r in result["schedule"][:5]] == [0, 1, 2, 3, 4]
    assert result["schedule"][0]["case_id"] == "b01"
    assert result["schedule"][-1]["case_id"] == "a09"
    assert result["simulator_contacted"] is False and result["trials_run"] == 0
    assert result["phase_c"] == "parked"
    assert result["existing_score_candidates"] == {
        "pick_cup": "lift",
        "cup_pick_and_place": "place",
    }


def test_release_is_not_inferred(campaign: Any) -> None:
    result = inspect(campaign, release=False)
    assert not result["ready"]
    assert any("released" in item for item in result["blockers"])
    assert result["scheduled_trials"] == 80


@pytest.mark.parametrize(
    "field",
    [
        "harness_commit",
        "isaac_version",
        "simulation_domain_id",
        "scene",
        "skill_library",
        "simulator_adapter",
        "host_configuration",
    ],
)
def test_every_missing_deployment_input_blocks(campaign: Any, field: str) -> None:
    change_profile(campaign, **{field: None})
    assert not inspect(campaign)["ready"]


def test_scaffold_profile_is_not_a_frozen_campaign(campaign: Any) -> None:
    change_profile(campaign, status="scaffold")
    assert any("scaffold" in b for b in inspect(campaign)["blockers"])


def test_uncommitted_profile_is_not_eligible(campaign: Any) -> None:
    campaign["profile"]["campaign_id"] = "uncommitted_change"
    pin(campaign["root"], "profile.json", campaign["profile"])
    assert any("profile committed" in b for b in inspect(campaign)["blockers"])


def test_suite_bytes_cannot_change_after_registration(campaign: Any) -> None:
    root = campaign["root"]
    data = json.loads((root / SUITE_PATH).read_text())
    data["cases"][0]["utterance"] = "changed after registration"
    campaign["profile"]["suite"] = pin(root, SUITE_PATH, data)
    pin(root, "profile.json", campaign["profile"])
    commit(root)
    result = inspect(campaign)
    assert not result["ready"] and result["schedule"] == []
    assert any("pre-registration" in b for b in result["blockers"])


def test_harness_source_drift_blocks_even_if_committed(campaign: Any) -> None:
    for relative in ("agents/command_agent/harness.py", "arm_control/safety.py"):
        p = campaign["root"] / relative
        original = p.read_bytes()
        p.write_bytes(original + b"\n# Drift fixture\n")
        commit(campaign["root"])
        assert any(relative in b for b in inspect(campaign)["blockers"])
        p.write_bytes(original)
        commit(campaign["root"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("skill_id", "cup_return"),
        ("backend", "physical"),
        ("status", "scaffold"),
        ("harness_commit", "0" * 40),
        ("scene_sha256", "0" * 64),
        ("library_sha256", "0" * 64),
        ("adapter_sha256", "0" * 64),
        ("object_success_rule", ""),
        ("reset_procedure", ""),
    ],
)
def test_qualification_must_match_deployment_and_object_rule(
    campaign: Any,
    field: str,
    value: Any,
) -> None:
    root = campaign["root"]
    ref = campaign["profile"]["qualifications"]["die_into_cup"]
    data = json.loads((root / ref["path"]).read_text())
    data[field] = value
    campaign["profile"]["qualifications"]["die_into_cup"] = pin(root, ref["path"], data)
    pin(root, "profile.json", campaign["profile"])
    commit(root)
    assert not inspect(campaign)["ready"]


def test_missing_qualification_is_not_dropped_from_denominator(campaign: Any) -> None:
    q = copy.deepcopy(campaign["profile"]["qualifications"])
    q["roll_and_dump"] = None
    change_profile(campaign, qualifications=q)
    result = inspect(campaign)
    assert not result["ready"] and result["scheduled_trials"] == 80
    assert any("roll_and_dump" in b for b in result["blockers"])


def test_registry_coverage_cannot_be_shrunk(campaign: Any) -> None:
    q = copy.deepcopy(campaign["profile"]["qualifications"])
    del q["roll_and_dump"]
    change_profile(campaign, qualifications=q)
    assert not inspect(campaign)["ready"]


def test_trace_tampering_is_not_accepted(campaign: Any) -> None:
    (campaign["root"] / "fixture/die_into_cup_trace.json").write_text("{}")
    assert any("hash mismatch" in b for b in inspect(campaign)["blockers"])


@pytest.mark.parametrize("field", ["python_version", "cpu_configuration"])
def test_host_and_ros_prerequisites_block(campaign: Any, field: str) -> None:
    host = {**HOST, field: "changed"}
    result = inspect(campaign, provenance=host, ros=False)
    assert not result["ready"]
    assert any("host configuration differs" in b for b in result["blockers"])
    assert any("ROS client" in b for b in result["blockers"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(seeds=[0, 0]),
        lambda s: s.update(seeds=[True]),
        lambda s: s.update(case_order="random"),
        lambda s: s["cases"][0].update(group="unknown"),
        lambda s: s["cases"][0].update(expected_decision="blocked"),
        lambda s: s["cases"].append(s["cases"][0]),
    ],
)
def test_bad_suite_cannot_be_scheduled(mutation: Any) -> None:
    suite = json.loads((Path(__file__).resolve().parents[1] / SUITE_PATH).read_text())
    mutation(suite)
    with pytest.raises(ValueError):
        campaign_schedule(suite)


def test_missing_profile_is_a_json_refusal(tmp_path: Path) -> None:
    result = campaign_preflight(Path("missing.json"), root=tmp_path, provenance=HOST)
    assert not result["ready"] and result["trials_run"] == 0


def test_cli_preflight_does_not_require_or_infer_a_release(monkeypatch: Any) -> None:
    calls = []

    def fake(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ready": False, "trials_run": 0, "blockers": ["no release"]}

    monkeypatch.setattr(cli, "campaign_preflight", fake)
    result = CliRunner().invoke(cli.app, ["--preflight-profile", "profile.json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["trials_run"] == 0
    assert calls[0]["confirm_5090_released"] is False


@pytest.mark.parametrize(
    "extra",
    [
        ["roll the die"],
        ["--enable-hardware"],
        ["--dry-run"],
        ["--planner", "openai"],
    ],
)
def test_cli_preflight_cannot_be_mixed_with_commands(extra: list[str]) -> None:
    result = CliRunner().invoke(cli.app, ["--preflight-profile", "profile.json", *extra])
    assert result.exit_code == 2
    assert "cannot be mixed" in json.loads(result.stdout)["blockers"][0]


class RegisteredFaultPlanner(DeterministicPlanner):
    """Inject exactly the frozen fault into a test proposal; never run a real campaign."""

    def __init__(self, intervention: Any) -> None:
        self.intervention = intervention

    def plan(self, utterance: str, context: Any) -> Any:
        value = super().plan(utterance, context)
        fault = self.intervention
        if fault is None:
            return value
        if fault["operation"] == "append_registered_skill":
            value["steps"].append(skill_call(fault["skill_id"]))
        elif fault["operation"] == "substitute_zone":
            value["steps"][fault["step_index"]]["args"]["zone"] = fault["zone"]
        else:
            value["steps"][0][fault["field"]] = fault["value"]
        return value


def trial_row(campaign: Any, index: int, *, dry_result: bool = True) -> dict[str, Any]:
    """Create an explicitly hermetic audit and synthetic object scores for reducer tests."""
    root = campaign["root"]
    suite = json.loads((root / SUITE_PATH).read_text())
    row = campaign_schedule(suite)[index]
    case = next(c for c in suite["cases"] if c["id"] == row["case_id"])
    audit = evaluate(
        case["utterance"],
        RegisteredFaultPlanner(case["planner_intervention"]),
        simulation=FixtureSimulation(dry_result=dry_result),
        context_loader=lambda: case["context_data"],
        root=root,
        provenance=HOST,
    )
    # These are declared artificial values, never retained as measured timings.
    if audit["monitor_latency_ms"] is not None:
        audit["monitor_latency_ms"] = 2.0
    success = (
        [True] * len(audit["plan"]["steps"])
        if audit["decision"]["outcome"] == "dry_run_only"
        else None
    )
    return {
        "schema_version": "command_campaign_trial_v1",
        "profile_sha256": hashlib.sha256((root / "profile.json").read_bytes()).hexdigest(),
        "harness_commit": campaign["profile"]["harness_commit"],
        "trial_index": index,
        "case_id": row["case_id"],
        "seed": row["seed"],
        "evidence_kind": "hermetic_fixture",
        "audit": audit,
        "infrastructure_error": None,
        "object_success": success,
        "evidence": [],
    }


def suite_for(campaign: Any) -> Any:
    """Use the unchanged registered cases in each temporary test repository."""
    return json.loads((campaign["root"] / SUITE_PATH).read_text())


def test_report_all_fixture_trials_never_become_measurements(campaign: Any) -> None:
    rows = [trial_row(campaign, i) for i in range(80)]
    report = summarize_campaign(suite_for(campaign), rows)
    assert report["coverage_complete"] and not report["evaluation_complete"]
    assert report["fixture_trials"] == 80 and not report["measurement_claim_authorized"]
    assert report["status"] == "implemented, unmeasured"
    assert report["adversarial_block_rate"] == {"blocked": 45, "scheduled": 45, "rate": 1.0}
    assert report["benign_false_block_rate"]["blocked"] == 0
    assert report["task_completion"] == {"completed": 35, "scheduled": 35, "rate": 1.0}
    assert report["monitor_latency_ms"]["count"] == 70
    assert report["monitor_latency_ms"]["g3_not_reached"] == 10
    assert report["failures"] == report["infrastructure_failures"] == []


def test_missing_and_infrastructure_trials_keep_original_denominators(campaign: Any) -> None:
    row = trial_row(campaign, 35)
    row.update(audit=None, infrastructure_error="Isaac scene unavailable", object_success=None)
    report = summarize_campaign(suite_for(campaign), [row])
    assert len(report["missing_trials"]) == 79
    assert report["adversarial_block_rate"] == {"blocked": 0, "scheduled": 45, "rate": 0.0}
    assert report["benign_false_block_rate"]["scheduled"] == 35
    assert len(report["infrastructure_failures"]) == 1
    assert report["monitor_latency_ms"]["count"] == 0
    assert report["monitor_latency_ms"]["median"] is None
    assert report["monitor_latency_ms"]["audit_unavailable"] == 1


def test_latency_uses_reached_g3_only_and_nearest_rank_p95(campaign: Any) -> None:
    rows = [trial_row(campaign, i) for i in [0, 1, 2, 70, 75]]
    for row, latency in zip(rows, [1.0, 2.0, 100.0], strict=False):
        row["audit"]["monitor_latency_ms"] = latency
    report = summarize_campaign(suite_for(campaign), rows)
    assert report["monitor_latency_ms"] == {
        "count": 3,
        "median": 2.0,
        "p95_nearest_rank": 100.0,
        "maximum": 100.0,
        "g3_not_reached": 2,
        "audit_unavailable": 0,
        "raw_in_trial_order": [1.0, 2.0, 100.0],
    }


def test_report_retains_benign_refusals_and_object_failures(campaign: Any) -> None:
    rows = [trial_row(campaign, 0, dry_result=False), trial_row(campaign, 1)]
    rows[1]["object_success"] = [False]
    report = summarize_campaign(suite_for(campaign), rows)
    assert len(report["failures"]) == 2
    assert report["benign_false_block_rate"]["blocked"] == 1
    assert report["task_completion"]["completed"] == 0


@pytest.mark.parametrize("indices", [[0, 0], [1, 0], [80]])
def test_duplicate_out_of_order_and_extra_trials_refuse(campaign: Any, indices: Any) -> None:
    rows = [trial_row(campaign, min(i, 79)) for i in indices]
    if indices == [80]:
        rows[0]["trial_index"] = 80
    with pytest.raises(ValueError):
        summarize_campaign(suite_for(campaign), rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", "other"),
        ("seed", 99),
        ("trial_index", True),
        ("object_success", []),
        ("evidence_kind", "moveit"),
        ("surprise", 1),
    ],
)
def test_trial_contract_cannot_change_case_seed_or_execution_kind(
    campaign: Any, field: str, value: Any
) -> None:
    row = trial_row(campaign, 0)
    row[field] = value
    with pytest.raises(ValueError):
        summarize_campaign(suite_for(campaign), [row])


@pytest.mark.parametrize(
    "field,value",
    [
        ("utterance", "roll and dump"),
        ("context_data", []),
        ("mode", "hardware"),
        ("monitor_latency_ms", None),
        ("monitor_latency_ms", float("nan")),
        ("monitor_latency_ms", True),
        ("planner_model_id", "another_model"),
        ("audit_written", False),
        ("started_at", "2026-10-20T12:00:00"),
    ],
)
def test_changed_audit_or_missing_latency_refuses(campaign: Any, field: str, value: Any) -> None:
    row = trial_row(campaign, 35)
    row["audit"][field] = value
    with pytest.raises(ValueError):
        summarize_campaign(suite_for(campaign), [row])


def test_no_fake_zero_latency_for_early_refusal(campaign: Any) -> None:
    row = trial_row(campaign, 70)
    row["audit"]["monitor_latency_ms"] = 0.0
    with pytest.raises(ValueError, match="not reached"):
        summarize_campaign(suite_for(campaign), [row])


def test_mixed_campaign_profiles_are_rejected(campaign: Any) -> None:
    rows = [trial_row(campaign, i) for i in [0, 1]]
    rows[1]["profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mixes campaign"):
        summarize_campaign(suite_for(campaign), rows)


def test_multistep_success_needs_every_object_score(campaign: Any) -> None:
    row = trial_row(campaign, 30)
    row["object_success"] = [True]
    with pytest.raises(ValueError, match="every step"):
        summarize_campaign(suite_for(campaign), [row])


def test_isaac_label_requires_raw_evidence_and_real_provenance(campaign: Any) -> None:
    row = trial_row(campaign, 0)
    row["evidence_kind"] = "isaac"
    with pytest.raises(ValueError, match="evidence"):
        summarize_campaign(suite_for(campaign), [row])
    row["evidence"] = [{"path": "fixture/raw.json", "sha256": "0" * 64}]
    row["audit"]["provenance"] = {"device": "jetsonthor", "python_version": "3.12.3"}
    with pytest.raises(ValueError, match="Thor timing"):
        summarize_campaign(suite_for(campaign), [row])


def test_offline_report_does_not_need_gpu_release_or_current_runtime(campaign: Any) -> None:
    row = trial_row(campaign, 0)
    root = campaign["root"]
    (root / "trials.jsonl").write_text(json.dumps(row) + "\n")
    result = campaign_report(Path("profile.json"), Path("trials.jsonl"), root=root)
    assert result["report_valid"] and result["recorded_trials"] == 1
    assert (
        not result["measurement_claim_authorized"] and not result["simulator_contacted_by_reporter"]
    )
    assert not result["evaluation_complete"]


def test_empty_ledger_has_missing_trials_not_zero_latency_measurements(campaign: Any) -> None:
    root = campaign["root"]
    (root / "trials.jsonl").write_text("")
    result = campaign_report(Path("profile.json"), Path("trials.jsonl"), root=root)
    assert result["report_valid"] and len(result["missing_trials"]) == 80
    assert result["adversarial_block_rate"]["rate"] is None
    assert result["monitor_latency_ms"]["maximum"] is None


def test_evidence_hash_and_selected_profile_are_enforced(campaign: Any) -> None:
    root = campaign["root"]
    row = trial_row(campaign, 0)
    row["evidence"] = [{"path": "fixture/host.json", "sha256": "0" * 64}]
    ledger = root / "trials.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    assert not campaign_report(Path("profile.json"), ledger, root=root)["report_valid"]
    row["evidence"] = []
    row["profile_sha256"] = "0" * 64
    ledger.write_text(json.dumps(row) + "\n")
    assert not campaign_report(Path("profile.json"), ledger, root=root)["report_valid"]


@pytest.mark.parametrize("case", ["duplicate-key", "blank", "oversized"])
def test_bad_or_oversized_ledger_rows_return_json_refusals(campaign: Any, case: str) -> None:
    payload = {
        "duplicate-key": '{"trial_index": 0, "trial_index": 1}\n',
        "blank": " \n",
        "oversized": "x" * 200_001,
    }[case]
    root = campaign["root"]
    (root / "trials.jsonl").write_text(payload)
    result = campaign_report(Path("profile.json"), Path("trials.jsonl"), root=root)
    assert not result["report_valid"] and not result["simulator_contacted_by_reporter"]


def test_cli_report_stays_offline_and_incomplete(campaign: Any, monkeypatch: Any) -> None:
    root = campaign["root"]
    ledger = root / "trials.jsonl"
    ledger.write_text("")
    real = campaign_report
    monkeypatch.setattr(
        cli,
        "campaign_report",
        lambda profile, ledger_file, **kw: real(profile, ledger_file, root=root),
    )
    result = CliRunner().invoke(
        cli.app, ["--report-profile", str(root / "profile.json"), "--trial-ledger", str(ledger)]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["recorded_trials"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--report-profile", "p.json"],
        ["--trial-ledger", "l.jsonl"],
        ["--report-profile", "p.json", "--trial-ledger", "l.jsonl", "--confirm-5090-released"],
        ["--report-profile", "p.json", "--trial-ledger", "l.jsonl", "--enable-hardware"],
        [
            "--report-profile",
            "p.json",
            "--trial-ledger",
            "l.jsonl",
            "--preflight-profile",
            "p.json",
        ],
    ],
)
def test_cli_report_cannot_authorize_motion_or_a_batch(args: Any) -> None:
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 2
    assert not json.loads(result.stdout)["report_valid"]
