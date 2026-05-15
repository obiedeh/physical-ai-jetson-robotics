"""High-impact RTX simulation and training project catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RtxProject:
    """One RTX workstation project with runnable evidence targets."""

    identifier: str
    title: str
    objective: str
    command: str
    evidence: str
    impact: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def high_impact_rtx_projects() -> list[RtxProject]:
    """Return the RTX-first simulation/training backlog in execution order."""
    return [
        RtxProject(
            identifier="openusd-factory-render",
            title="OpenUSD Factory Cell Rendering",
            objective="Render the factory-cell digital twin and validate camera viewpoints.",
            command="bash scripts/linux_rtx/render_factory_cell.sh",
            evidence="isaac/outputs/factory_cell_v0.png",
            impact="Portfolio-grade visual proof and synthetic-data starting point.",
        ),
        RtxProject(
            identifier="synria-reach-policy",
            title="Synria Synthetic Reaching Policy",
            objective=(
                "Train a CUDA-backed policy that maps arm state and target pose "
                "to joint deltas."
            ),
            command="physical-ai-lab train-synria-reach --device cuda",
            evidence="reports/training/synria_reach_policy.json",
            impact="First RTX training workload for manipulation before real-arm data collection.",
        ),
        RtxProject(
            identifier="ros-gazebo-factory-cell",
            title="ROS 2 Gazebo Factory Cell",
            objective="Launch the factory-cell world and validate ROS/Gazebo integration.",
            command="bash scripts/linux_rtx/run_ros_gazebo_factory_cell.sh",
            evidence="reports/simulation/ros_gazebo_factory_cell.json",
            impact=(
                "Navigation and mobile-manipulation simulation path independent "
                "of Isaac runtime."
            ),
        ),
        RtxProject(
            identifier="moveit-synria-planning",
            title="MoveIt Synria Planning Smoke Test",
            objective="Validate approximate Synria arm kinematics and planning configuration.",
            command="bash scripts/linux_rtx/run_synria_moveit_smoke.sh",
            evidence="reports/simulation/synria_moveit_smoke.json",
            impact="Planning proof before hardware motion and before imitation-learning demos.",
        ),
        RtxProject(
            identifier="synthetic-perception-benchmark",
            title="Synthetic Perception Benchmark",
            objective="Benchmark object-detection style inference on rendered factory-cell frames.",
            command="physical-ai-lab edge-plan --compute workstation",
            evidence="reports/edge_ai/linux_rtx_benchmarks.json",
            impact="RTX baseline for later Jetson Orin/Thor comparisons.",
        ),
    ]


def rtx_project_by_id(identifier: str) -> RtxProject:
    """Return one RTX project by identifier."""
    for project in high_impact_rtx_projects():
        if project.identifier == identifier:
            return project
    known = ", ".join(project.identifier for project in high_impact_rtx_projects())
    raise ValueError(f"Unknown RTX project '{identifier}'. Known projects: {known}")
