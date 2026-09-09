"""Launch the physical_ai_ops_copilot node.

Usage::

    # Deterministic triage (no API key required):
    ros2 launch physical_ai_ops_copilot ops_copilot.launch.py

    # LLM enrichment via AnthropicClassifier (requires ANTHROPIC_API_KEY):
    ANTHROPIC_API_KEY=sk-... \\
        ros2 launch physical_ai_ops_copilot ops_copilot.launch.py use_llm:=true

The node subscribes to /robot_telemetry (std_msgs/String, JSON) and
publishes enriched reports to /ops_copilot/report (std_msgs/String, JSON).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "window_size",
                default_value="12",
                description=(
                    "Number of telemetry samples to buffer before each triage pass."
                ),
            ),
            DeclareLaunchArgument(
                "use_llm",
                default_value="false",
                description=(
                    "Set to 'true' to use AnthropicClassifier. "
                    "Requires ANTHROPIC_API_KEY env var and the 'llm' optional dep. "
                    "Falls back to DeterministicClassifier if the key is absent."
                ),
            ),
            Node(
                package="physical_ai_ops_copilot",
                executable="ops_copilot_node",
                name="ops_copilot",
                output="screen",
                parameters=[
                    {
                        "window_size": LaunchConfiguration("window_size"),
                        "use_llm": LaunchConfiguration("use_llm"),
                    }
                ],
            ),
        ]
    )
