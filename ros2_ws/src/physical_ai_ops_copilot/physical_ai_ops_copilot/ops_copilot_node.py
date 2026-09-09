"""ROS 2 ops copilot node.

Subscribes to ``/robot_telemetry`` (``std_msgs/String``, JSON payload),
runs deterministic triage and optional LLM enrichment, and publishes the
``OpsCopilotReport`` to ``/ops_copilot/report`` (``std_msgs/String``, JSON).

This module is designed for hermetic testing: the pure functions
``parse_telemetry_json`` and ``process_telemetry_window`` have no ROS 2
dependency and are exercised in CI without a sourced workspace.

Deploy with::

    ros2 launch physical_ai_ops_copilot ops_copilot.launch.py \\
        window_size:=12 use_llm:=true

Or with ``ANTHROPIC_API_KEY`` set to activate LLM enrichment::

    ANTHROPIC_API_KEY=sk-... ros2 run physical_ai_ops_copilot ops_copilot_node
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from agents.ops_copilot.copilot import (
    AnthropicClassifier,
    Classifier,
    DeterministicClassifier,
    enrich,
)
from physical_ai_lab.ops_copilot import triage_telemetry
from physical_ai_lab.telemetry import RobotTelemetrySample

try:
    import rclpy  # type: ignore[import-not-found]
    from rclpy.node import Node as _RclpyNode  # type: ignore[import-not-found]
    from std_msgs.msg import String as _StringMsg  # type: ignore[import-not-found]

    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False
    _RclpyNode = object  # type: ignore[misc, assignment]
    _StringMsg = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Pure functions — testable without rclpy
# ---------------------------------------------------------------------------


def parse_telemetry_json(data: dict[str, Any]) -> RobotTelemetrySample:
    """Parse a telemetry dict (from JSON) into a ``RobotTelemetrySample``.

    Args:
        data: A dict with keys matching ``RobotTelemetrySample`` fields.
              ``timestamp`` must be an ISO 8601 string.

    Returns:
        A ``RobotTelemetrySample`` instance.

    Raises:
        KeyError: If a required field is absent.
        ValueError: If a field cannot be coerced to its expected type.
    """
    return RobotTelemetrySample(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        robot_id=str(data["robot_id"]),
        battery_percent=float(data["battery_percent"]),
        motor_temp_c=float(data["motor_temp_c"]),
        edge_latency_ms=float(data["edge_latency_ms"]),
        network_latency_ms=float(data["network_latency_ms"]),
        localization_quality=float(data["localization_quality"]),
        task_success_probability=float(data["task_success_probability"]),
    )


def process_telemetry_window(
    window: list[RobotTelemetrySample],
    classifier: Classifier,
) -> dict[str, object]:
    """Run triage + enrichment on a telemetry window and return the report dict.

    This is the pure logic layer used by ``OpsCopilotNode``; it is testable
    in CI without rclpy, Isaac Lab, or real hardware.

    Args:
        window:     Non-empty list of ``RobotTelemetrySample`` objects.
        classifier: Any ``Classifier`` implementation — deterministic or LLM.

    Returns:
        A JSON-serialisable dict in the shape of ``OpsCopilotReport.to_dict()``.
    """
    triage = triage_telemetry(window)
    report = enrich(triage, classifier)
    return report.to_dict()


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------


class OpsCopilotNode(_RclpyNode):  # type: ignore[misc, valid-type]
    """ROS 2 node that subscribes to robot telemetry and publishes ops reports.

    Topics
    ------
    Subscribes:
        ``/robot_telemetry``  (``std_msgs/String``) — JSON-serialised telemetry.
    Publishes:
        ``/ops_copilot/report`` (``std_msgs/String``) — JSON ``OpsCopilotReport``.

    Parameters
    ----------
    window_size : int (default 12)
        Number of telemetry samples to buffer before triaging.
    use_llm : bool (default false)
        Use ``AnthropicClassifier`` if ``ANTHROPIC_API_KEY`` is set; otherwise
        warn and fall back to ``DeterministicClassifier``.
    """

    TELEMETRY_TOPIC = "/robot_telemetry"
    REPORT_TOPIC = "/ops_copilot/report"
    DEFAULT_WINDOW_SIZE = 12

    def __init__(self) -> None:
        if not _RCLPY_AVAILABLE:
            raise RuntimeError(
                "rclpy is not available. "
                "Source a ROS 2 workspace before running OpsCopilotNode."
            )
        super().__init__("ops_copilot")  # type: ignore[call-arg]

        self._window_size: int = (
            self.declare_parameter("window_size", self.DEFAULT_WINDOW_SIZE)  # type: ignore[attr-defined]
            .value  # type: ignore[union-attr]
        )
        use_llm: bool = (
            self.declare_parameter("use_llm", False)  # type: ignore[attr-defined]
            .value  # type: ignore[union-attr]
        )

        self._classifier: Classifier
        if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
            self._classifier = AnthropicClassifier()
            self.get_logger().info(  # type: ignore[attr-defined]
                "AnthropicClassifier active for telemetry enrichment."
            )
        else:
            self._classifier = DeterministicClassifier()
            if use_llm:
                self.get_logger().warn(  # type: ignore[attr-defined]
                    "use_llm=True but ANTHROPIC_API_KEY is not set; "
                    "falling back to DeterministicClassifier."
                )

        self._buffer: list[RobotTelemetrySample] = []

        self._sub = self.create_subscription(  # type: ignore[attr-defined]
            _StringMsg,
            self.TELEMETRY_TOPIC,
            self._on_telemetry,
            10,
        )
        self._pub = self.create_publisher(  # type: ignore[attr-defined]
            _StringMsg,
            self.REPORT_TOPIC,
            10,
        )

    def _on_telemetry(self, msg: Any) -> None:
        """Handle an incoming ``/robot_telemetry`` JSON message."""
        try:
            data = json.loads(msg.data)
            sample = parse_telemetry_json(data)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"Malformed telemetry message: {exc}")  # type: ignore[attr-defined]
            return

        self._buffer.append(sample)
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

        report_dict = process_telemetry_window(self._buffer, self._classifier)
        out = _StringMsg()  # type: ignore[call-arg]
        out.data = json.dumps(report_dict)
        self._pub.publish(out)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover — rclpy-gated
    """Entry point for ``ros2 run physical_ai_ops_copilot ops_copilot_node``."""
    if not _RCLPY_AVAILABLE:
        raise SystemExit("rclpy not available — source a ROS 2 workspace first.")
    rclpy.init()
    node = OpsCopilotNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
