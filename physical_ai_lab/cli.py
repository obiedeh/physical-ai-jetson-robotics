"""Command-line interface for the Physical AI lab."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from physical_ai_lab.config import ComputeTarget, LabProfile, RobotPlatform
from physical_ai_lab.inventory import write_inventory
from physical_ai_lab.mission import scenario_by_id, signature_scenarios
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
