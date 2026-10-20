"""CLI for proposal inspection and mandatory simulation checks, with JSON-only output.

An explicit launch-owned skill library and ROS simulation domain enable the
MoveIt service path. Missing dependencies, model selection or context files are
audited refusals. Physical execution is deliberately unavailable in Phase A.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from agents.command_agent.harness import (
    MoveItSimulation,
    campaign_preflight,
    evaluate,
    host_provenance,
)
from agents.command_agent.planners import (
    AnthropicPlanner,
    DeterministicPlanner,
    ExternalPlanner,
    OpenAIPlanner,
)
from agents.command_agent.schema import ContextData, parse_json


class LaunchPlanner:
    """Defer launch validation into G1 so configuration failures produce an audited decision."""

    def __init__(self, name: str) -> None:
        """Record the selected identity without loading SDKs or consulting an alternate backend."""
        self.name = self.provider = name
        self.model_id = os.environ.get(f"COMMAND_AGENT_{name.upper()}_MODEL", "")
        if name == "deterministic":
            self.model_id = "keyword_rules_v1"
        if name == "external":
            self.model_id = "external_command"

    def plan(self, utterance: str, context: tuple[ContextData, ...]) -> Any:
        """Construct only the requested backend; missing settings fail closed."""
        if self.name == "deterministic":
            return DeterministicPlanner().plan(utterance, context)
        if self.name == "anthropic":
            return AnthropicPlanner(self.model_id).plan(utterance, context)
        if self.name == "openai":
            return OpenAIPlanner(self.model_id).plan(utterance, context)
        if self.name == "external":
            command = os.environ.get("SYNRIA_TASK_PLANNER_COMMAND", "")
            return ExternalPlanner(command).plan(utterance, context)
        raise ValueError("planner must be deterministic, anthropic, openai or external")


def command(
    request: Annotated[
        str | None, typer.Argument(help="Operator request; context is separate data.")
    ] = None,
    planner: Annotated[str, typer.Option(help="deterministic, anthropic, openai or external")] = (
        "deterministic"
    ),
    monitor: Annotated[
        str, typer.Option(help="deterministic, anthropic or openai")
    ] = "deterministic",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Mandatory simulation-only mode.")
    ] = False,
    enable_hardware: Annotated[
        bool,
        typer.Option("--enable-hardware", help="Reserved eligibility flag; Phase C is parked."),
    ] = False,
    context: Annotated[
        Path | None, typer.Option(help="JSON array of labelled context data.")
    ] = None,
    keepout_config: Annotated[
        Path | None,
        typer.Option(help="JSON list of keep-out boxes in simulation base coordinates."),
    ] = None,
    simulation_library: Annotated[
        Path | None, typer.Option(help="Trusted fixed-scene skill-path JSON, never model output.")
    ] = None,
    simulation_domain_id: Annotated[
        int | None,
        typer.Option(help="Dedicated simulation ROS domain; never a physical-arm domain."),
    ] = None,
    preflight_profile: Annotated[
        Path | None, typer.Option(help="Inspect a frozen Phase B profile without running trials.")
    ] = None,
    confirm_5090_released: Annotated[
        bool, typer.Option("--confirm-5090-released", help="Operator release for this batch only.")
    ] = False,
) -> None:
    """Print an audited command decision or a static Phase B preflight; refuse on failure."""
    if preflight_profile is not None:
        if (
            request is not None
            or dry_run
            or enable_hardware
            or context is not None
            or keepout_config is not None
            or simulation_library is not None
            or simulation_domain_id is not None
            or planner != "deterministic"
            or monitor != "deterministic"
        ):
            result: dict[str, Any] = {
                "ready": False,
                "trials_run": 0,
                "simulator_contacted": False,
                "blockers": ["campaign preflight cannot be mixed with a command or motion options"],
            }
        else:
            result = campaign_preflight(
                preflight_profile,
                root=Path(__file__).resolve().parents[2],
                confirm_5090_released=confirm_5090_released,
            )
        typer.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        raise typer.Exit(0 if result["ready"] else 2)

    def load_context() -> Any:
        """Load data inside G0 so malformed or unavailable context cannot be ignored."""
        if confirm_5090_released:
            raise ValueError("release confirmation applies only to campaign preflight")
        if dry_run and enable_hardware:
            raise ValueError("dry-run and hardware enable cannot be combined")
        if monitor not in ("deterministic", "anthropic", "openai"):
            raise ValueError("monitor must be deterministic or a known provider")
        return parse_json(context.read_bytes()) if context else []

    root = Path(__file__).resolve().parents[2]
    result = evaluate(
        request or "",
        LaunchPlanner(planner),
        simulation=MoveItSimulation(simulation_library, simulation_domain_id),
        context_loader=load_context,
        keepout_loader=(lambda: parse_json(keepout_config.read_bytes()))
        if keepout_config
        else None,
        monitor=None if monitor == "deterministic" else LaunchPlanner(monitor),
        mode="dry_run" if dry_run else "hardware",
        enable_hardware=enable_hardware,
        root=root,
        provenance=host_provenance(),
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise typer.Exit(2 if result["decision"]["outcome"] == "blocked" else 0)


app = typer.Typer(help="Inspect command plans behind fail-closed safety gates.")
app.command()(command)


if __name__ == "__main__":
    app()
