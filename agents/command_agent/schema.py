"""Strict command and evidence contracts keep language output away from motion values.

The registry owns skill identities. Fixed-scene arguments are enumerated rather
than interpreted as coordinates. Simulation trajectories are a separate trusted
contract and never appear in the planner's tool schema.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synria_lerobot.task_planner.schema import MAX_STEPS, MAX_TEXT
from synria_lerobot.task_registry import TASK_IDS

SKILL_IDS = (*TASK_IDS, "return_to_ready")
SCHEMA_VERSION = "command_plan_v1"
Gate = Literal["G0", "G1", "G2", "G3", "G4"]


class Contract(BaseModel):
    """Reject coercion and extra fields so ambiguous model output fails closed."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class SkillArgs(Contract):
    """Name fixed objects and zones; bounded board squares remain unsupported by execution."""

    object: Literal["die", "cup", "arm"]
    zone: Literal["cup", "cup_mark", "landing_tray", "ready"]
    board_square: Annotated[int, Field(ge=1, le=64)] | None


class SkillCall(Contract):
    """Select one registry skill with its exact fixed-scene argument contract."""

    skill_id: str
    args: SkillArgs

    @model_validator(mode="after")
    def validate_skill(self) -> SkillCall:
        """Reject unknown identities and object/zone substitutions before resolution."""
        if self.skill_id not in SKILL_IDS:
            raise ValueError("skill_id is outside the task registry allowlist")
        expected = {
            "die_into_cup": ("die", "cup"),
            "pick_cup": ("cup", "cup_mark"),
            "cup_pick_and_place": ("cup", "cup_mark"),
            "roll_and_dump": ("cup", "landing_tray"),
            "cup_return": ("cup", "cup_mark"),
            "return_to_ready": ("arm", "ready"),
        }[self.skill_id]
        if (self.args.object, self.args.zone) != expected:
            raise ValueError("object and zone differ from the registered fixed-scene skill")
        return self


class CommandPlan(Contract):
    """Bind an ordered, bounded skill proposal to the original operator utterance."""

    schema_version: Literal["command_plan_v1"]
    utterance: Annotated[str, Field(min_length=1, max_length=MAX_TEXT)]
    planner_name: Annotated[str, Field(min_length=1, max_length=80)]
    steps: Annotated[list[SkillCall], Field(min_length=1, max_length=MAX_STEPS)]
    rationale: Annotated[str, Field(min_length=1, max_length=MAX_TEXT)]


class ContextData(Contract):
    """Carry untrusted observations in labelled data fields, never system instructions."""

    source: Literal["camera", "retrieved_note", "other"]
    text: Annotated[str, Field(max_length=4000)]


class GateResult(Contract):
    """Retain the gate, outcome and reason for every attempted check."""

    gate: Gate
    passed: bool
    reason: str


class Decision(Contract):
    """Expose review authority without representing permission to actuate hardware."""

    outcome: Literal["blocked", "dry_run_only", "operator_confirm"]
    gate: Gate
    escalate_to_human: bool

    @model_validator(mode="after")
    def validate_escalation(self) -> Decision:
        """A refusal must always retain the human escalation path."""
        if self.outcome == "blocked" and not self.escalate_to_human:
            raise ValueError("blocked decisions must escalate to a human")
        return self


def command_schema() -> dict[str, Any]:
    """Return a fresh tool schema with a registry-derived allowlist."""
    schema = CommandPlan.model_json_schema()
    schema["$defs"]["SkillCall"]["properties"]["skill_id"] = {
        "type": "string",
        "enum": list(SKILL_IDS),
    }
    return schema


def parse_json(text: str | bytes) -> Any:
    """Reject oversized JSON, duplicate keys and nonfinite constants at all boundaries."""
    if len(text) > 200_000:
        raise ValueError("JSON exceeds the 200000 byte/character boundary")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """A duplicated field has no unambiguous safety meaning."""
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON field: {key}")
            value[key] = item
        return value

    def invalid(value: str) -> Any:
        """Nonfinite numbers must never pass numerical safety comparisons."""
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def plan_digest(plan: CommandPlan) -> str:
    """Bind simulator evidence to the exact validated proposal, including its order."""
    data = json.dumps(plan.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def skill_call(skill_id: str) -> dict[str, Any]:
    """Build canonical fixed-scene arguments without accepting language-supplied coordinates."""
    obj, zone = {
        "die_into_cup": ("die", "cup"),
        "pick_cup": ("cup", "cup_mark"),
        "cup_pick_and_place": ("cup", "cup_mark"),
        "roll_and_dump": ("cup", "landing_tray"),
        "cup_return": ("cup", "cup_mark"),
        "return_to_ready": ("arm", "ready"),
    }[skill_id]
    return {"skill_id": skill_id, "args": {"object": obj, "zone": zone, "board_square": None}}


class ArtifactRef(Contract):
    """Pin operator-owned campaign inputs by bytes; paths never come from command plans."""

    path: Annotated[str, Field(min_length=1, max_length=1000)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CampaignProfile(Contract):
    """Freeze Phase B deployment facts separately from the pre-registered command suite."""

    schema_version: Literal["command_campaign_profile_v1"]
    status: Literal["scaffold", "planned"]
    campaign_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")]
    preregistration_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    suite: ArtifactRef
    harness_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None
    isaac_version: Annotated[str, Field(min_length=1, max_length=100)] | None
    simulation_domain_id: Annotated[int, Field(ge=1, le=232)] | None
    scene: ArtifactRef | None
    skill_library: ArtifactRef | None
    simulator_adapter: ArtifactRef | None
    host_configuration: ArtifactRef | None
    qualifications: dict[str, ArtifactRef | None]

    @model_validator(mode="after")
    def require_registry_coverage(self) -> CampaignProfile:
        """Refuse omitted or extra skills instead of quietly shrinking the benign denominator."""
        if set(self.qualifications) != set(SKILL_IDS):
            raise ValueError("qualification slots must cover exactly the registry plus ready skill")
        return self


class SkillQualification(Contract):
    """Bind one committed Isaac integration check to its reset, scorer and raw evidence."""

    schema_version: Literal["command_isaac_qualification_v1"]
    status: Literal["measured"]
    backend: Literal["isaac"]
    skill_id: str
    harness_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    scene_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    library_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    adapter_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    reset_procedure: Annotated[str, Field(min_length=1, max_length=2000)]
    object_success_rule: Annotated[str, Field(min_length=1, max_length=2000)]
    trace: ArtifactRef
