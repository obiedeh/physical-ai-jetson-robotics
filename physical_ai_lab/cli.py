"""Command-line interface for the Physical AI lab."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from physical_ai_lab.config import ComputeTarget, LabProfile, RobotPlatform
from physical_ai_lab.edge_ai import default_edge_benchmark_plan
from physical_ai_lab.inventory import write_inventory
from physical_ai_lab.mission import scenario_by_id, signature_scenarios
from physical_ai_lab.ops_copilot import triage_telemetry
from physical_ai_lab.rtx_projects import high_impact_rtx_projects, rtx_project_by_id
from physical_ai_lab.rtx_training import TrainingRunConfig, train_synria_reach_policy
from physical_ai_lab.slam import demo_mecanum_trials, summarize_mecanum_calibration
from physical_ai_lab.synria_upstream import (
    ActivityPhase,
    synria_upstream_activities,
    synria_upstream_activity_by_id,
)
from physical_ai_lab.telemetry import generate_demo_telemetry, summarize_operational_risk

app = typer.Typer(help="Physical AI Jetson Robotics Lab CLI")
console = Console()
DEFAULT_INVENTORY_OUTPUT = Path("reports/inventory/latest_inventory.json")


@app.command()
def profile(
    name: str = "local-dev",
    compute: ComputeTarget = ComputeTarget.workstation,
    ros_domain_id: int = 42,
) -> None:
    """Print a lab deployment profile."""
    lab_profile = LabProfile(
        name=name,
        compute_target=compute,
        robot_platforms=[
            RobotPlatform.digital_twin,
            RobotPlatform.robo_car,
            RobotPlatform.robotic_arm,
        ],
        ros_domain_id=ros_domain_id,
    )
    console.print(lab_profile.summary(), markup=False)


@app.command("demo-telemetry")
def demo_telemetry(
    robot_id: str = "robo-car-01",
    samples: int = typer.Option(12, min=1, max=288),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Generate synthetic robot telemetry and summarize operational risk."""
    telemetry = generate_demo_telemetry(robot_id=robot_id, samples=samples)
    summary = summarize_operational_risk(telemetry)

    if json_output:
        console.print(json.dumps(summary, indent=2, sort_keys=True))
        return

    table = Table(title=f"Robot Telemetry Summary: {robot_id}")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in summary.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command("collect-inventory")
def collect_inventory_command(
    target: Annotated[
        str,
        typer.Option(
            help="Target label, such as windows-dev, linux-rtx, jetson-orin, or jetson-thor."
        ),
    ] = "windows-dev",
    output: Annotated[
        Path,
        typer.Option(help="Output JSON path."),
    ] = DEFAULT_INVENTORY_OUTPUT,
) -> None:
    """Collect machine, GPU, ROS, camera, and Jetson inventory evidence."""
    written = write_inventory(target=target, output=output)
    console.print(f"Wrote inventory report to {written}", markup=False)


@app.command("list-scenarios")
def list_scenarios() -> None:
    """List the signature non-generic demo scenarios."""
    table = Table(title="Signature Physical AI Scenarios")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Readiness")

    for scenario in signature_scenarios():
        table.add_row(scenario.identifier, scenario.title, f"{scenario.readiness_score:.0%}")

    console.print(table)


@app.command("scenario")
def show_scenario(
    identifier: str = typer.Argument(..., help="Scenario ID from list-scenarios."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show a signature demo scenario and the evidence it needs."""
    scenario = scenario_by_id(identifier)

    if json_output:
        console.print(json.dumps(scenario.to_dict(), indent=2, sort_keys=True))
        return

    console.print(f"{scenario.title}", style="bold")
    console.print(scenario.objective)
    console.print()
    console.print("Hardware: " + ", ".join(scenario.hardware), markup=False)
    console.print("Software: " + ", ".join(scenario.software), markup=False)

    table = Table(title="Evidence Requirements")
    table.add_column("Requirement")
    table.add_column("Evidence")
    table.add_column("Business Value")

    for requirement in scenario.requirements:
        table.add_row(requirement.name, requirement.evidence, requirement.business_value)

    console.print(table)


@app.command("edge-plan")
def edge_plan(
    compute: ComputeTarget = ComputeTarget.jetson_orin,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show the first edge AI inference benchmark plan."""
    plan = default_edge_benchmark_plan(compute)

    if json_output:
        console.print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return

    table = Table(title=f"Edge AI Benchmark Plan: {compute.value}")
    table.add_column("Workload")
    table.add_column("Format")
    table.add_column("Input")
    table.add_column("Targets")

    for workload in plan.workloads:
        table.add_row(
            workload.name,
            workload.model_format,
            workload.input_source,
            (
                f">={workload.minimum_fps:g} FPS, "
                f"<={workload.maximum_latency_ms:g} ms, "
                f"<={workload.maximum_power_w:g} W, "
                f"<={workload.maximum_memory_mb} MB"
            ),
        )

    console.print(table)
    console.print(f"Evidence: {plan.evidence_path}", markup=False)


@app.command("ops-triage")
def ops_triage(
    robot_id: str = "robo-car-01",
    samples: int = typer.Option(12, min=1, max=288),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a deterministic robot operations triage report."""
    telemetry = generate_demo_telemetry(robot_id=robot_id, samples=samples)
    report = triage_telemetry(telemetry)

    if json_output:
        console.print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    table = Table(title=f"Operations Triage: {report.robot_id}")
    table.add_column("Severity")
    table.add_column("Signal")
    table.add_column("Finding")
    table.add_column("Action")

    for finding in report.findings:
        table.add_row(
            finding.severity,
            finding.signal,
            finding.summary,
            finding.recommended_action,
        )

    console.print(table)


@app.command("mecanum-calibration")
def mecanum_calibration(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Summarize a sample Yahboom mecanum-base calibration run."""
    summary = summarize_mecanum_calibration(demo_mecanum_trials())

    if json_output:
        console.print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return

    table = Table(title=f"Mecanum Calibration: {summary.status}")
    table.add_column("Command")
    table.add_column("Commanded")
    table.add_column("Measured")
    table.add_column("Drift")
    table.add_column("Error")

    for trial in summary.trials:
        table.add_row(
            trial.command,
            f"{trial.commanded_m:.2f} m",
            f"{trial.measured_m:.2f} m",
            f"{trial.drift_m:.3f} m",
            f"{trial.distance_error_percent:.2f}%",
        )

    console.print(table)


@app.command("rtx-projects")
def rtx_projects(
    identifier: str | None = typer.Argument(None, help="Optional RTX project ID."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List high-impact RTX simulation and training projects."""
    projects = [rtx_project_by_id(identifier)] if identifier else high_impact_rtx_projects()

    if json_output:
        payload = [project.to_dict() for project in projects]
        console.print(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title="High-Impact RTX Simulation and Training Projects")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Command")
    table.add_column("Evidence")

    for project in projects:
        table.add_row(project.identifier, project.title, project.command, project.evidence)

    console.print(table)


@app.command("train-synria-reach")
def train_synria_reach(
    samples: int = typer.Option(4096, min=512, max=200_000),
    epochs: int = typer.Option(25, min=1, max=10_000),
    batch_size: int = typer.Option(256, min=16, max=8192),
    learning_rate: float = typer.Option(0.001, min=0.000001, max=1.0),
    seed: int = 7,
    device: str = typer.Option("auto", help="auto, cuda, or cpu."),
    report: Path = Path("reports/training/synria_reach_policy.json"),
    checkpoint: Path = Path("runs/rtx_training/synria_reach_policy.pt"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Train the synthetic Synria reaching policy on the RTX workstation."""
    result = train_synria_reach_policy(
        TrainingRunConfig(
            samples=samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            report_path=report,
            checkpoint_path=checkpoint,
        )
    )

    if json_output:
        console.print(json.dumps(result, indent=2, sort_keys=True))
        return

    table = Table(title="Synria Synthetic Reaching Policy Training")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in result.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command("synria-upstream-todos")
def synria_upstream_todos(
    identifier: str | None = typer.Argument(None, help="Optional upstream activity ID."),
    phase: Annotated[
        ActivityPhase | None,
        typer.Option(help="Filter by rtx-now or hardware-later."),
    ] = None,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List Synria vendor-alignment TODO activities."""
    activities = (
        [synria_upstream_activity_by_id(identifier)]
        if identifier
        else synria_upstream_activities()
    )
    if phase is not None:
        activities = [activity for activity in activities if activity.phase == phase]

    if json_output:
        console.print(json.dumps([activity.to_dict() for activity in activities], indent=2))
        return

    table = Table(title="Synria Upstream TODO Activities")
    table.add_column("ID")
    table.add_column("Phase")
    table.add_column("Title")
    table.add_column("Source")
    table.add_column("Deliverable")

    for activity in activities:
        table.add_row(
            activity.identifier,
            activity.phase.value,
            activity.title,
            activity.source_repo,
            activity.deliverable,
        )

    console.print(table)
