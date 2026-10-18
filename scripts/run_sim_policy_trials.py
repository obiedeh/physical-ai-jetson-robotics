"""Run scored closed-loop policy trials in the Isaac scene's policy mode.

Start the scene first, in the teleop repository, with the leader arm out of the loop::

    SKIP_DRIVER=1 bash scripts/run_alicia_stack.sh policy

and a simulation-lane policy server in this repository::

    python scripts/serve_synria_policy.py --checkpoint "$CHECKPOINT" \\
        --checkpoint-record "$RECORD" --device cuda --lane simulation

Then::

    python scripts/run_sim_policy_trials.py --trials 20 --dataset-root "$DATASET_ROOT" \\
        --command-period-s 0.2 --response-timeout-s 0.5 \\
        --output reports/sim_policy/<policy_id>/trials_<utc>.json

The runner reads ``/isaac_joint_states`` and the two recorded camera topics, publishes
targets on ``/isaac/policy_joint_targets`` (never a real-arm topic), walks the arm to the
start pose and then asks the scene for a Randomize and reset before every trial (at
``--randomize-scale``, seeded from ``--placement-seed`` plus the trial index when given, so
campaigns sharing a seed face the same placements), and scores each trial from the scene's poses
by the served task: ``pick_cup`` succeeds when the cup is ``--lift-m`` above its rest
height for ``--hold-s``; ``cup_pick_and_place`` when the cup, lifted at least once, rests
within ``--place-radius-m`` of the scene's mark and within ``--place-height-m`` of its
rest height with the gripper centre ``--gripper-clear-m`` away, for ``--hold-s``. The
scene's gripper pose gives each trial its closest and final gripper-to-cup distance. The
start pose is the mean first frame of the dataset unless ``--start-pose`` gives seven
values. The report is written once and says it is simulation evidence only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.embodiment import SynriaEmbodiment  # noqa: E402
from synria_lerobot.physical_contract import (  # noqa: E402
    SIMULATION_LANE,
    PhysicalDatasetContract,
)
from synria_lerobot.policy_client import (  # noqa: E402
    HttpPolicyTransport,
    PolicyClient,
    PolicySafetyConfig,
)
from synria_lerobot.quality_gates import load_limits  # noqa: E402
from synria_lerobot.recorder import RosImageFrameSource, RosJointStateSource  # noqa: E402
from synria_lerobot.sim_policy_runner import (  # noqa: E402
    PLACE_SCORE,
    SCORE_BY_TASK,
    SIM_CUP_POSE_TOPIC,
    SIM_GRIPPER_POSE_TOPIC,
    SIM_OVERHEAD_TOPIC,
    SIM_PLACE_TARGET_POSE_TOPIC,
    SIM_RANDOMIZE_TOPIC,
    SIM_STATE_TOPIC,
    SIM_TARGET_TOPIC,
    SIM_WRIST_TOPIC,
    RosPoseSource,
    RosSimPublisher,
    SimulationTrialRunner,
    TrialConfig,
    load_placement_tolerance,
    start_pose_from_dataset,
    write_report,
)

DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"
DEFAULT_POLICY_CONFIG = REPOSITORY / "config" / "synria_sim_policy.json"
DEFAULT_TOLERANCES = REPOSITORY / "config" / "synria_placement_tolerances.json"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; every timing value is explicit."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--command-period-s", type=float, required=True)
    parser.add_argument("--response-timeout-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        help="LeRobot dataset whose mean first frame is the start pose.")
    parser.add_argument("--start-pose", type=float, nargs=7, metavar="V",
                        help="Six joint angles in radians and a gripper width in metres.")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--max-trial-s", type=float, default=30.0)
    parser.add_argument("--lift-m", type=float, default=0.03)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--approach-s", type=float, default=2.0)
    parser.add_argument("--randomize-scale", type=float, default=1.0,
                        help="The scene's Randomize scale for every reset (0.5 to 3).")
    parser.add_argument("--placement-seed", type=int, default=None,
                        help="Seed of trial 0's layout draw; trial i uses seed + i.")
    parser.add_argument("--placement-tolerances", type=Path, default=DEFAULT_TOLERANCES,
                        help="Committed per-task placement radii; the place radius default.")
    parser.add_argument("--place-radius-m", type=float, default=None,
                        help="Place score: the cup's centre within this of the mark's centre.")
    parser.add_argument("--place-height-m", type=float, default=0.01,
                        help="Place score: the cup within this of its rest height.")
    parser.add_argument("--gripper-clear-m", type=float, default=0.06,
                        help="Place score: the gripper centre at least this far from the cup.")
    parser.add_argument("--target-topic", default=SIM_TARGET_TOPIC)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Wire the ROS sources, the policy client and the runner, then write the report."""
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if (args.dataset_root is None) == (args.start_pose is None):
        raise SystemExit("give exactly one of --dataset-root and --start-pose")
    start_pose = (tuple(args.start_pose) if args.start_pose is not None
                  else start_pose_from_dataset(args.dataset_root))
    transport = HttpPolicyTransport(args.endpoint)
    metadata = transport.metadata(args.response_timeout_s)
    if metadata.get("lane") != SIMULATION_LANE:
        raise SystemExit("the policy server is not serving the simulation lane; "
                         "start it with --lane simulation")
    contract = PhysicalDatasetContract.from_dict(metadata["physical_contract"])
    score = SCORE_BY_TASK.get(contract.task_id)
    if score is None:
        raise SystemExit(f"task {contract.task_id} has no scene score; trials run for "
                         f"{sorted(SCORE_BY_TASK)}")
    place_radius = args.place_radius_m
    if place_radius is None:
        place_radius = (load_placement_tolerance(args.placement_tolerances, contract.task_id)
                        if score == PLACE_SCORE else 0.03)
    safety = PolicySafetyConfig.load(
        args.policy_config, command_period_s=args.command_period_s,
        response_timeout_s=args.response_timeout_s, lane=SIMULATION_LANE,
    )
    client = PolicyClient(SynriaEmbodiment(contract), transport, load_limits(args.limits), safety)
    config = TrialConfig(
        trials=args.trials, command_period_s=args.command_period_s, start_pose=start_pose,
        max_trial_s=args.max_trial_s, lift_m=args.lift_m, hold_s=args.hold_s,
        settle_s=args.settle_s, approach_s=args.approach_s, score=score,
        place_radius_m=place_radius, place_height_m=args.place_height_m,
        gripper_clear_m=args.gripper_clear_m, randomize_scale=args.randomize_scale,
        placement_seed=args.placement_seed,
    )
    with ExitStack() as stack:
        publisher = stack.enter_context(_closing(RosSimPublisher(
            target_topic=args.target_topic, randomize_topic=SIM_RANDOMIZE_TOPIC,
        )))
        state = stack.enter_context(_closing(RosJointStateSource(
            SIM_STATE_TOPIC, node_name="synria_sim_policy_state",
            state_has_velocity=contract.state_has_velocity,
        )))
        wrist = stack.enter_context(_closing(RosImageFrameSource(
            SIM_WRIST_TOPIC, node_name="synria_sim_policy_wrist",
            width=args.image_size, height=args.image_size,
        )))
        front = stack.enter_context(_closing(RosImageFrameSource(
            SIM_OVERHEAD_TOPIC, node_name="synria_sim_policy_front",
            width=args.image_size, height=args.image_size,
        )))
        cup = stack.enter_context(_closing(RosPoseSource(
            SIM_CUP_POSE_TOPIC, node_name="synria_sim_policy_cup",
        )))
        gripper = stack.enter_context(_closing(RosPoseSource(
            SIM_GRIPPER_POSE_TOPIC, node_name="synria_sim_policy_gripper",
        )))
        target = None
        if score == PLACE_SCORE:
            target = stack.enter_context(_closing(RosPoseSource(
                SIM_PLACE_TARGET_POSE_TOPIC, node_name="synria_sim_policy_mark",
            )))
        runner = SimulationTrialRunner(
            client, publisher, state, wrist, front, cup, publisher.request_randomize, config,
            gripper_source=gripper, target_source=target,
        )
        report = runner.run(metadata)
    report["endpoint"] = args.endpoint
    report["start_pose_source"] = (str(args.dataset_root) if args.dataset_root is not None
                                   else "command line")
    write_report(report, args.output)
    rate = report["success_rate"]
    if not math.isfinite(rate):
        raise SystemExit("report has no finite success rate")
    print(json.dumps({"successes": report["successes"], "trials": args.trials,
                      "success_rate": rate, "report": str(args.output)}))
    return 0


class _closing:
    """A context manager that closes a source on exit, whatever happened."""

    def __init__(self, source: object) -> None:
        """Hold the source."""
        self.source = source

    def __enter__(self) -> object:
        """Hand the source back."""
        return self.source

    def __exit__(self, *exc: object) -> None:
        """Close it."""
        self.source.close()  # type: ignore[attr-defined]


if __name__ == "__main__":
    sys.exit(main())
