"""Serve the staged what-where-how-plan-execute executor on the policy protocol.

Simulation lane only. The executor reads the scene's published object and mark
poses, plans with inverse kinematics on the committed chain inside the operator
limits, and answers every policy request with the next command of its current
phase. Start the scene in policy mode first, then::

    python scripts/serve_staged_executor.py --task-id pick_cup --port 8091 \\
        --command-period-s 0.2 --stage-log reports/sim_policy/staged/stages.jsonl

and run ``scripts/run_sim_policy_trials.py`` against ``http://127.0.0.1:8091``. The
stage log receives one line per phase change and one record per trial with the
what, where, how and plan answers. Nothing here publishes on a real-arm topic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.arm_kinematics import KinematicChain  # noqa: E402
from synria_lerobot.physical_contract import (  # noqa: E402
    SIMULATION_LANE,
    ActionSource,
    PhysicalDatasetContract,
)
from synria_lerobot.policy_client import PolicySafetyConfig  # noqa: E402
from synria_lerobot.policy_server import make_http_server  # noqa: E402
from synria_lerobot.quality_gates import load_limits  # noqa: E402
from synria_lerobot.sim_policy_runner import (  # noqa: E402
    PLACE_SCORE,
    SCORE_BY_TASK,
    SIM_CUP_POSE_TOPIC,
    SIM_PLACE_TARGET_POSE_TOPIC,
    RosPoseSource,
)
from synria_lerobot.staged_executor import OBJECT_BY_TASK, StagedPolicyService  # noqa: E402
from synria_lerobot.task_registry import load_task_registry  # noqa: E402

DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"
DEFAULT_KINEMATICS = REPOSITORY / "config" / "synria_kinematics.json"
DEFAULT_REGISTRY = REPOSITORY / "config" / "synria_tasks.json"
DEFAULT_POLICY_CONFIG = REPOSITORY / "config" / "synria_sim_policy.json"
OBJECT_TOPICS = {"cup": SIM_CUP_POSE_TOPIC}


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; the command period is explicit because it sets the step size."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-id", required=True, choices=sorted(SCORE_BY_TASK))
    parser.add_argument("--task-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--gripper", default="50mm", choices=("50mm", "100mm"))
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG,
                        help="Safety file whose speed limits size the executor's steps.")
    parser.add_argument("--command-period-s", type=float, required=True)
    parser.add_argument("--speed-fraction", type=float, default=0.8,
                        help="Fraction of the client's per-step limit the executor uses.")
    parser.add_argument("--margin-deg", type=float, default=3.0,
                        help="Distance every planned joint keeps from its operator bound.")
    parser.add_argument("--goal-channel", action="append", default=[], choices=("object", "target"),
                        help="Declare a goal channel in the served contract, so a campaign or "
                             "a recording driven by this executor carries it; repeatable.")
    parser.add_argument("--stage-log", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    return parser


def contract_for_task(task_id: str, registry: Path, gripper: str,
                      goal_channels: tuple[str, ...] = ()) -> PhysicalDatasetContract:
    """The simulation-lane contract the executor declares for one registry task."""
    tasks = load_task_registry(registry)
    if task_id not in tasks:
        raise SystemExit(f"task {task_id} is not in {registry}")
    return PhysicalDatasetContract(
        gripper, ActionSource.LEADER, False, task_id=task_id, task_definition=tasks[task_id],
        recording_purpose="simulation", goal_channels=goal_channels,
    )


def main(argv: list[str] | None = None) -> int:
    """Open the pose readers, build the service and serve until interrupted."""
    import math

    args = build_parser().parse_args(argv)
    contract = contract_for_task(args.task_id, args.task_registry, args.gripper,
                                 tuple(args.goal_channel))
    safety = PolicySafetyConfig.load(
        args.policy_config, command_period_s=args.command_period_s,
        response_timeout_s=0.5, lane=SIMULATION_LANE,
    )
    object_source = RosPoseSource(OBJECT_TOPICS[OBJECT_BY_TASK[args.task_id]],
                                  node_name="synria_staged_object")
    target_source = None
    if SCORE_BY_TASK[args.task_id] == PLACE_SCORE:
        target_source = RosPoseSource(SIM_PLACE_TARGET_POSE_TOPIC, node_name="synria_staged_mark")
    service = StagedPolicyService(
        contract, KinematicChain.load(args.kinematics), load_limits(args.limits),
        object_source.read, None if target_source is None else target_source.read,
        command_period_s=args.command_period_s,
        max_joint_speed_rad_s=min(safety.max_joint_speed_rad_s),
        max_gripper_speed_m_s=safety.max_gripper_speed_m_s,
        speed_fraction=args.speed_fraction, margin_rad=math.radians(args.margin_deg),
        stage_log=args.stage_log,
    )
    try:
        with make_http_server(service, args.host, args.port) as server:
            print(f"staged executor serving {args.task_id} on http://{args.host}:{args.port}",
                  flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        object_source.close()
        if target_source is not None:
            target_source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
