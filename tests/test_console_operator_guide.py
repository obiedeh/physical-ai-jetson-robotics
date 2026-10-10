"""Exercise offline operator help and refusal guidance without opening hardware or ROS runtimes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_console_service import ConsoleFixture

from synria_lerobot.console.app import GUIDE_DOCUMENTS, REPOSITORY
from synria_lerobot.console.readiness import ReadinessReport


@pytest.mark.parametrize("document", list(GUIDE_DOCUMENTS))
def test_help_reads_only_committed_allowlisted_documents(tmp_path: Path, document: str) -> None:
    """Embedded procedures match their governed sources rather than a divergent copy."""
    with ConsoleFixture(tmp_path) as service:
        result = service.read(["help", document], {})
        assert result["text"] == (REPOSITORY / GUIDE_DOCUMENTS[document]).read_text("utf-8")
        assert service.catalog.events() == []


@pytest.mark.parametrize("document", [
    "../AGENTS.md", "/etc/passwd", "runbook/../limits", "unknown",
])
def test_help_cannot_choose_a_filesystem_path(tmp_path: Path, document: str) -> None:
    """Only server-owned help identifiers are readable; client paths are never evaluated."""
    with ConsoleFixture(tmp_path) as service:
        with pytest.raises(KeyError, match="unknown read-only"):
            service.read(["help", document], {})


def test_guide_exposes_only_selected_environment_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators can compare ROS settings without dumping unrelated process environment secrets."""
    monkeypatch.setenv("ROS_DOMAIN_ID", "23")
    monkeypatch.setenv("PRIVATE_TEST_SECRET", "must-not-appear")
    with ConsoleFixture(
        tmp_path, builder=lambda *args, **kwargs: pytest.fail("opened sources"),
    ) as service:
        guide = service.read(["operator-guide"], {})
        assert guide["environment"]["ros_domain"] == "23"
        assert set(guide["environment"]) == {
            "python", "host", "ros_domain", "ros_distribution", "middleware",
        }
        assert "must-not-appear" not in json.dumps(guide)
        assert service.catalog.events() == []


def test_subscription_creation_is_not_reported_as_receiving_state() -> None:
    """Successful subscriber construction remains distinct from rate measurement and samples."""
    report = ReadinessReport(lambda: 0, demo=False)
    report.begin()
    report.observe("state_source", {"passed": True, "topic": "/fake_states"})
    assert report.checks["state_subscription"]["status"] == "configured"
    assert "state_source" not in report.checks
    assert report.checks["follower_sample"]["status"] == "not_checked"
    report.observe("state_rate", {"passed": False, "message": "No state sample received"})
    report.finish()
    assert report.snapshot()["status"] == "blocked"


def test_operator_guidance_survives_failed_and_contradictory_inputs() -> None:
    """Run shipped guidance against deliberately bad snapshots without a browser or devices."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime unavailable")
    subprocess.run([
        executable, str(Path(__file__).with_name("console_operator_guide.cjs")),
        str(REPOSITORY / "synria_lerobot/console/static/operator-guide.js"),
        str(REPOSITORY / "synria_lerobot/console/static/app.js"),
    ], check=True, capture_output=True, text=True, timeout=15)


def test_guided_browser_handles_faults_without_implicit_device_actions(tmp_path: Path) -> None:
    """Exercise real browser interactions against a loopback fake API with no hardware runtime."""
    node = shutil.which("node")
    browser = (
        shutil.which("google-chrome") or shutil.which("google-chrome-stable")
        or shutil.which("chromium") or shutil.which("chromium-browser")
    )
    if node is None or browser is None:
        pytest.skip("optional existing browser and JavaScript runtime unavailable")
    result = subprocess.run([
        node, str(Path(__file__).with_name("console_guided_browser.cjs")),
        str(REPOSITORY / "synria_lerobot/console/static"), browser,
        str(tmp_path / "synthetic-guided-console.png"),
    ], check=False, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_browser_startup_failure_is_reported_without_unhandled_pipe_error(tmp_path: Path) -> None:
    """An executable rejecting browser flags must fail the harness with useful child diagnostics."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("optional JavaScript runtime unavailable")
    result = subprocess.run([
        node, str(Path(__file__).with_name("console_guided_browser.cjs")),
        str(REPOSITORY / "synria_lerobot/console/static"), node,
        str(tmp_path / "must-not-exist.png"),
    ], check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Isolated browser transport failed" in result.stderr
    assert "Unhandled 'error' event" not in result.stderr
    assert not (tmp_path / "must-not-exist.png").exists()
