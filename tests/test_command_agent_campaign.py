"""Static Phase B preflight tests use declared fixtures, never an Isaac or hardware session."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agents.command_agent import cli
from agents.command_agent.harness import (
    CAMPAIGN_SOURCE_PATHS,
    campaign_preflight,
    campaign_schedule,
)
from agents.command_agent.schema import SKILL_IDS

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
