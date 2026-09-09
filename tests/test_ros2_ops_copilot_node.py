"""Tests for the physical_ai_ops_copilot ROS 2 package.

The rclpy-dependent ``OpsCopilotNode`` class is NOT instantiated in CI
(rclpy is unavailable without a sourced ROS 2 workspace). All tests target:

    - Package structure and manifest
    - Module-level import (rclpy guard)
    - ``parse_telemetry_json`` — pure JSON → RobotTelemetrySample conversion
    - ``process_telemetry_window`` — pure triage + enrich without ROS 2
    - ``OpsCopilotNode`` constructor guard (raises without rclpy)
    - Launch file: ``generate_launch_description`` function exists
"""

from __future__ import annotations

import ast
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "ros2_ws" / "src" / "physical_ai_ops_copilot"

# Make the ament_python package importable in CI without colcon build.
sys.path.insert(0, str(PKG_ROOT))


# ---------------------------------------------------------------------------
# Package structure
# ---------------------------------------------------------------------------


def test_ops_copilot_package_files_exist() -> None:
    expected = [
        "package.xml",
        "setup.py",
        "setup.cfg",
        "resource/physical_ai_ops_copilot",
        "physical_ai_ops_copilot/__init__.py",
        "physical_ai_ops_copilot/ops_copilot_node.py",
        "launch/ops_copilot.launch.py",
    ]
    missing = [f for f in expected if not (PKG_ROOT / f).exists()]
    assert missing == [], f"Missing files: {missing}"


def test_ops_copilot_package_xml_name() -> None:
    root = ET.parse(PKG_ROOT / "package.xml").getroot()
    assert root.findtext("name") == "physical_ai_ops_copilot"


def test_ops_copilot_package_xml_build_type_is_ament_python() -> None:
    root = ET.parse(PKG_ROOT / "package.xml").getroot()
    build_type = root.findtext("export/build_type")
    assert build_type == "ament_python"


def test_launch_file_defines_generate_launch_description() -> None:
    src = (PKG_ROOT / "launch" / "ops_copilot.launch.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "generate_launch_description" in fn_names


# ---------------------------------------------------------------------------
# Module import and rclpy guard
# ---------------------------------------------------------------------------


def test_ops_copilot_node_module_imports_cleanly() -> None:
    """The module imports without error regardless of rclpy availability."""
    from physical_ai_ops_copilot.ops_copilot_node import (  # noqa: F401
        _RCLPY_AVAILABLE,
        OpsCopilotNode,
        parse_telemetry_json,
        process_telemetry_window,
    )
    # _RCLPY_AVAILABLE is a bool: True when rclpy is installed, False otherwise.
    assert isinstance(_RCLPY_AVAILABLE, bool)


def test_ops_copilot_node_constructor_raises_without_rclpy() -> None:
    """Instantiating OpsCopilotNode without rclpy raises RuntimeError.

    This test only runs when rclpy is absent (the import guard code path).
    In environments where rclpy is installed the test is skipped — the node
    can be constructed normally there (but requires rclpy.init() first).
    """
    from physical_ai_ops_copilot.ops_copilot_node import _RCLPY_AVAILABLE, OpsCopilotNode

    if _RCLPY_AVAILABLE:
        pytest.skip("rclpy is installed — RuntimeError guard only applies when rclpy is absent")

    with pytest.raises(RuntimeError, match="rclpy"):
        OpsCopilotNode()


# ---------------------------------------------------------------------------
# parse_telemetry_json
# ---------------------------------------------------------------------------


def _sample_dict(
    robot_id: str = "synria-01",
    battery_percent: float = 85.0,
    motor_temp_c: float = 45.0,
    edge_latency_ms: float = 15.0,
    network_latency_ms: float = 10.0,
    localization_quality: float = 0.92,
    task_success_probability: float = 0.88,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "robot_id": robot_id,
        "battery_percent": battery_percent,
        "motor_temp_c": motor_temp_c,
        "edge_latency_ms": edge_latency_ms,
        "network_latency_ms": network_latency_ms,
        "localization_quality": localization_quality,
        "task_success_probability": task_success_probability,
    }


def test_parse_telemetry_json_robot_id() -> None:
    from physical_ai_ops_copilot.ops_copilot_node import parse_telemetry_json

    sample = parse_telemetry_json(_sample_dict(robot_id="synria-arm-01"))
    assert sample.robot_id == "synria-arm-01"


def test_parse_telemetry_json_numeric_fields() -> None:
    from physical_ai_ops_copilot.ops_copilot_node import parse_telemetry_json

    sample = parse_telemetry_json(_sample_dict(battery_percent=72.5, motor_temp_c=61.0))
    assert sample.battery_percent == pytest.approx(72.5)
    assert sample.motor_temp_c == pytest.approx(61.0)


def test_parse_telemetry_json_missing_key_raises() -> None:
    from physical_ai_ops_copilot.ops_copilot_node import parse_telemetry_json

    bad = _sample_dict()
    del bad["battery_percent"]  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        parse_telemetry_json(bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# process_telemetry_window
# ---------------------------------------------------------------------------


def test_process_telemetry_window_nominal_status() -> None:
    from physical_ai_ops_copilot.ops_copilot_node import (
        parse_telemetry_json,
        process_telemetry_window,
    )

    from agents.ops_copilot.copilot import DeterministicClassifier

    sample = parse_telemetry_json(_sample_dict())
    report = process_telemetry_window([sample], DeterministicClassifier())

    assert report["robot_id"] == "synria-01"
    assert report["classifier_name"] == "deterministic"
    assert isinstance(report["escalate_to_human"], bool)
    assert "triage" in report


def test_process_telemetry_window_critical_battery_escalates() -> None:
    from physical_ai_ops_copilot.ops_copilot_node import (
        parse_telemetry_json,
        process_telemetry_window,
    )

    from agents.ops_copilot.copilot import DeterministicClassifier

    sample = parse_telemetry_json(_sample_dict(battery_percent=5.0))
    report = process_telemetry_window([sample], DeterministicClassifier())

    # DeterministicClassifier.enrich() sets escalate_to_human=True when there
    # are critical findings, regardless of the triage.status field (which is
    # computed separately by summarize_operational_risk).
    assert report["escalate_to_human"] is True
    triage = report["triage"]
    severities = {f["severity"] for f in triage["findings"]}  # type: ignore[index]
    assert "critical" in severities


def test_process_telemetry_window_multi_sample_window() -> None:
    from physical_ai_ops_copilot.ops_copilot_node import (
        parse_telemetry_json,
        process_telemetry_window,
    )

    from agents.ops_copilot.copilot import DeterministicClassifier

    samples = [parse_telemetry_json(_sample_dict()) for _ in range(5)]
    report = process_telemetry_window(samples, DeterministicClassifier())

    assert report["status"] in {"nominal", "watch", "critical"}


def test_process_telemetry_window_report_is_json_serialisable() -> None:
    import json as _json

    from physical_ai_ops_copilot.ops_copilot_node import (
        parse_telemetry_json,
        process_telemetry_window,
    )

    from agents.ops_copilot.copilot import DeterministicClassifier

    sample = parse_telemetry_json(_sample_dict())
    report = process_telemetry_window([sample], DeterministicClassifier())
    # Must not raise
    serialised = _json.dumps(report)
    assert len(serialised) > 0
