"""Signature Physical AI mission scenarios and readiness scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

_PROGRESS_FILE = Path(__file__).parent.parent / "reports" / "progress.json"


def _load_progress() -> dict[str, dict[str, bool]]:
    """Read reports/progress.json; return empty dict if absent or malformed."""
    if not _PROGRESS_FILE.exists():
        return {}
    try:
        data = json.loads(_PROGRESS_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass(frozen=True)
class MissionRequirement:
    """One measurable requirement for a signature demo."""

    name: str
    evidence: str
    business_value: str
    complete: bool = False


@dataclass(frozen=True)
class MissionScenario:
    """A standout portfolio scenario with technical and business evidence."""

    identifier: str
    title: str
    objective: str
    hardware: list[str]
    software: list[str]
    requirements: list[MissionRequirement] = field(default_factory=list)

    @property
    def readiness_score(self) -> float:
        if not self.requirements:
            return 0.0
        completed = sum(1 for requirement in self.requirements if requirement.complete)
        return round(completed / len(self.requirements), 3)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["readiness_score"] = self.readiness_score
        return data


def _apply_progress(
    scenarios: list[MissionScenario], progress: dict[str, dict[str, bool]]
) -> list[MissionScenario]:
    result = []
    for scenario in scenarios:
        done = progress.get(scenario.identifier, {})
        updated = [
            replace(req, complete=done.get(req.name, req.complete))
            for req in scenario.requirements
        ]
        result.append(replace(scenario, requirements=updated))
    return result


def signature_scenarios() -> list[MissionScenario]:
    """Return the project's signature non-generic demo scenarios."""
    scenarios = [
        MissionScenario(
            identifier="yahboom-mobile-manipulation",
            title="Yahboom Orin NX Mobile Manipulation Bring-Up",
            objective=(
                "Prove the robot car can expose ROS 2 topics, stream vision, move safely, "
                "and provide measured Jetson evidence for mobile robotics."
            ),
            hardware=[
                "Yahboom Orin NX 8GB robot car",
                "mecanum base",
                "onboard 6DOF arm",
                "camera/display/audio stack",
            ],
            software=["ROS 2", "JetPack", "Nav2 or vendor navigation stack", "physical-ai-lab CLI"],
            requirements=[
                MissionRequirement(
                    name="inventory",
                    evidence="reports/inventory/yahboom_orin_nx.json",
                    business_value="Confirms the deployed hardware/software stack.",
                ),
                MissionRequirement(
                    name="ros-topic-map",
                    evidence="ROS 2 topic list captured in reports/yahboom",
                    business_value="Shows integration with the real robot runtime.",
                ),
                MissionRequirement(
                    name="mecanum-calibration",
                    evidence="motion accuracy and drift table",
                    business_value="Quantifies navigation reliability.",
                ),
                MissionRequirement(
                    name="camera-proof",
                    evidence="camera stream screenshot",
                    business_value="Enables inspection and embodied perception.",
                ),
                MissionRequirement(
                    name="slam-proof",
                    evidence="map image or localization report",
                    business_value="Supports autonomous indoor operations.",
                ),
            ],
        ),
        MissionScenario(
            identifier="synria-eye-in-hand",
            title="Synria Eye-In-Hand Manipulation",
            objective=(
                "Prove safe arm bring-up with C10 wrist-camera perception, MoveIt planning, "
                "and a path toward LeRobot/ALOHA-style demonstrations."
            ),
            hardware=["Synria 6DOF arm", "Synria C10 camera", "Jetson or Linux control host"],
            software=[
                "ROS 2",
                "MoveIt 2",
                "camera calibration tools",
                "LeRobot-compatible datasets",
            ],
            requirements=[
                MissionRequirement(
                    name="inventory",
                    evidence="reports/inventory/synria_arm.json",
                    business_value="Confirms the manipulation host and camera environment.",
                ),
                MissionRequirement(
                    name="camera-calibration",
                    evidence="C10 calibration file and screenshot",
                    business_value="Enables reliable object localization.",
                ),
                MissionRequirement(
                    name="moveit-mock-plan",
                    evidence="MoveIt planning screenshot",
                    business_value="Reduces risk before real robot motion.",
                ),
                MissionRequirement(
                    name="joint-state-proof",
                    evidence="joint states and first reduced-speed motion notes",
                    business_value="Shows hardware control readiness.",
                ),
                MissionRequirement(
                    name="demo-dataset",
                    evidence="LeRobot/ALOHA-style dataset schema",
                    business_value="Shows learning-from-demonstration capability.",
                ),
            ],
        ),
        MissionScenario(
            identifier="openusd-digital-twin",
            title="OpenUSD Digital Twin Factory Cell",
            objective=(
                "Mirror the real lab in OpenUSD/Isaac with robot assets, cameras, objects, "
                "collision zones, and synthetic-data-ready views."
            ),
            hardware=["Linux RTX workstation", "Yahboom robot reference", "Synria arm reference"],
            software=["OpenUSD", "Isaac Sim", "Isaac Lab", "RTX rendering"],
            requirements=[
                MissionRequirement(
                    name="rtx-inventory",
                    evidence="reports/inventory/linux_rtx.json",
                    business_value="Confirms simulation compute environment.",
                ),
                MissionRequirement(
                    name="usd-scene",
                    evidence="isaac/usd factory-cell scene",
                    business_value="Creates a reusable digital twin.",
                ),
                MissionRequirement(
                    name="render-proof",
                    evidence="rendered screenshot in reports/isaac",
                    business_value="Makes the portfolio visually inspectable.",
                ),
                MissionRequirement(
                    name="synthetic-data-sample",
                    evidence="sample camera output and labels",
                    business_value="Supports data generation for perception.",
                ),
                MissionRequirement(
                    name="sim-real-notes",
                    evidence="comparison report",
                    business_value="Connects simulation to real robot deployment.",
                ),
            ],
        ),
    ]
    return _apply_progress(scenarios, _load_progress())


def scenario_by_id(identifier: str) -> MissionScenario:
    """Return one scenario by identifier."""
    for scenario in signature_scenarios():
        if scenario.identifier == identifier:
            return scenario
    known = ", ".join(scenario.identifier for scenario in signature_scenarios())
    raise ValueError(f"Unknown scenario '{identifier}'. Known scenarios: {known}")
