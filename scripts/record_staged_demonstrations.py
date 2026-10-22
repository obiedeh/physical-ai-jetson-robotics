"""Record staged-executor demonstrations in the Isaac scene as a simulation dataset.

Start the scene in policy mode first (teleop repository)::

    SKIP_DRIVER=1 bash scripts/run_alicia_stack.sh policy

then::

    python scripts/record_staged_demonstrations.py --task-id cup_pick_and_place --takes 200 \\
        --placement-seed 3026 --dataset-path ~/synria-sim-datasets/staged_place_01/dataset \\
        --repo-id local/staged_place_01 --report-dir reports/sim_policy/staged_demos

The recorder samples the sim state, both cameras and the task's goal channels at
``--fps``; on every sample the staged executor plans or steps from the sampled state
and its command is published on the sim-only target topic and stored as the frame's
leader-type action. With ``--offset-fraction`` that share of takes ends its first
descent short of the grasp and to one side, drawn per take from the placement seed,
so the corpus shows the open gripper arriving off the cup, the seat check failing and
the correction that seats it before the close. The scene's scorer labels each take;
failures are labelled and appended to the dataset's exclusion log. Use a placement seed far from the
evaluation seeds (2026 and 2027) so no evaluation placement is in the training set.
Simulation only; nothing here touches the physical arm.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from synria_lerobot.arm_kinematics import KinematicChain  # noqa: E402
from synria_lerobot.console.isaac import SIM_TOPICS, sim_goal_channels  # noqa: E402
from synria_lerobot.demo_engine import (  # noqa: E402
    DemonstrationConfig,
    DemonstrationSession,
    ExecutorActionSource,
    write_summary,
)
from synria_lerobot.quality_gates import load_limits  # noqa: E402
from synria_lerobot.recorder import build_recording_session  # noqa: E402
from synria_lerobot.sim_policy_runner import (  # noqa: E402
    PLACE_SCORE,
    SCORE_BY_TASK,
    SIM_CUP_POSE_TOPIC,
    SIM_GRIPPER_POSE_TOPIC,
    SIM_PLACE_TARGET_POSE_TOPIC,
    SIM_RANDOMIZE_TOPIC,
    RosPoseSource,
    RosSimPublisher,
    load_placement_tolerance,
    start_pose_from_dataset,
)
from synria_lerobot.staged_executor import OBJECT_BY_TASK, STRATEGIES, StagedExecutor  # noqa: E402

DEFAULT_LIMITS = REPOSITORY / "config" / "synria_limits.yaml"
DEFAULT_KINEMATICS = REPOSITORY / "config" / "synria_kinematics.json"
DEFAULT_REGISTRY = REPOSITORY / "config" / "synria_tasks.json"
DEFAULT_TOLERANCES = REPOSITORY / "config" / "synria_placement_tolerances.json"
EVALUATION_SEEDS = (2026, 2027)


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; the dataset path must lie outside the repository."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-id", required=True, choices=sorted(SCORE_BY_TASK))
    parser.add_argument("--takes", type=int, required=True)
    parser.add_argument("--placement-seed", type=int, required=True)
    parser.add_argument("--randomize-scale", type=float, default=1.0)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--start-pose-dataset", type=Path, default=None,
                        help="A dataset whose mean first frame is the start pose.")
    parser.add_argument("--start-pose", type=float, nargs=7, metavar="V", default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-take-s", type=float, default=30.0)
    parser.add_argument("--gripper", default="50mm", choices=("50mm", "100mm"))
    parser.add_argument("--task-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--placement-tolerances", type=Path, default=DEFAULT_TOLERANCES)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=1.0)
    parser.add_argument("--speed-fraction", type=float, default=0.8)
    parser.add_argument("--margin-deg", type=float, default=3.0)
    parser.add_argument("--start-jitter-deg", type=float, nargs=6, default=None,
                        metavar="D",
                        help="Per-joint half-ranges of a seeded offset to each take's start pose.")
    parser.add_argument("--offset-fraction", type=float, default=0.0,
                        help="Share of takes whose first descent ends off the grasp.")
    parser.add_argument("--offset-back-m", type=float, nargs=2, default=(0.01, 0.03),
                        metavar=("LOW", "HIGH"),
                        help="Range of the arrival's shortfall along the approach axis.")
    parser.add_argument("--offset-side-m", type=float, default=0.01,
                        help="Largest sideways arrival offset along the finger axis.")
    return parser


def arrival_offsets(seed: int, fraction: float, back_m: tuple[float, float],
                    side_m: float) -> random.Random | None:
    """The per-take offset generator, or None when no take is offset."""
    if not 0 <= fraction <= 1:
        raise SystemExit("--offset-fraction must lie between 0 and 1")
    low, high = back_m
    if not 0 <= low <= high or side_m < 0:
        raise SystemExit("offset ranges must be non-negative and ordered")
    return None if fraction == 0 else random.Random(seed)


def draw_offset(rng: random.Random | None, fraction: float, back_m: tuple[float, float],
                side_m: float) -> tuple[float, float]:
    """One take's arrival offset: (back, sideways), or zero for an unperturbed take."""
    if rng is None or rng.random() >= fraction:
        return (0.0, 0.0)
    return (rng.uniform(*back_m), rng.uniform(-side_m, side_m))


def recorder_args(args: argparse.Namespace) -> argparse.Namespace:
    """The recorder's option namespace for a simulation session driven by the executor."""
    return argparse.Namespace(
        task_id=args.task_id, task_registry=args.task_registry, dataset_path=args.dataset_path,
        repo_id=args.repo_id, gripper_type=args.gripper, action_source="leader",
        action_lookahead_steps=1, follower_topic=SIM_TOPICS["state"],
        state_source="isaac_simulation", guard_command_topic=[],
        leader_topic="/joint_states", wrist_camera=SIM_TOPICS["wrist"],
        front_camera=SIM_TOPICS["front"], fps=args.fps, image_width=args.image_size,
        image_height=args.image_size, state_has_velocity=False, smoke=False,
        practice_seconds=None, state_startup_timeout_s=10.0,
        goal_channel=sim_goal_channels(args.task_id),
    )


def main(argv: list[str] | None = None) -> int:
    """Open the scene sources and the recorder, run the takes, write the summary."""
    args = build_parser().parse_args(argv)
    if args.placement_seed in EVALUATION_SEEDS or any(
        abs(args.placement_seed - seed) < args.takes for seed in EVALUATION_SEEDS
    ):
        raise SystemExit("choose a placement seed whose takes cannot overlap the evaluation "
                         f"seeds {EVALUATION_SEEDS}")
    if (args.start_pose_dataset is None) == (args.start_pose is None):
        raise SystemExit("give exactly one of --start-pose-dataset and --start-pose")
    start_pose = (tuple(args.start_pose) if args.start_pose is not None
                  else start_pose_from_dataset(args.start_pose_dataset))
    score = SCORE_BY_TASK[args.task_id]
    place_radius = (load_placement_tolerance(args.placement_tolerances, args.task_id)
                    if score == PLACE_SCORE else 0.03)
    chain = KinematicChain.load(args.kinematics)
    limits = load_limits(args.limits)
    strategy = STRATEGIES[OBJECT_BY_TASK[args.task_id]]
    period = 1.0 / args.fps
    max_step = args.max_joint_speed_rad_s * period * args.speed_fraction
    config = DemonstrationConfig(
        takes=args.takes, fps=args.fps, start_pose=start_pose, score=score,
        placement_seed=args.placement_seed, randomize_scale=args.randomize_scale,
        max_take_s=args.max_take_s, place_radius_m=place_radius, max_step_rad=max_step,
        start_jitter_rad=(() if args.start_jitter_deg is None
                          else tuple(math.radians(v) for v in args.start_jitter_deg)),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with ExitStack() as stack:
        publisher = stack.enter_context(_closing(RosSimPublisher(
            randomize_topic=SIM_RANDOMIZE_TOPIC, node_name="synria_demo_engine",
        )))
        cup = stack.enter_context(_closing(RosPoseSource(
            SIM_CUP_POSE_TOPIC, node_name="synria_demo_cup")))
        gripper = stack.enter_context(_closing(RosPoseSource(
            SIM_GRIPPER_POSE_TOPIC, node_name="synria_demo_gripper")))
        target = None
        if score == PLACE_SCORE:
            target = stack.enter_context(_closing(RosPoseSource(
                SIM_PLACE_TARGET_POSE_TOPIC, node_name="synria_demo_mark")))

        offsets = arrival_offsets(args.placement_seed, args.offset_fraction,
                                  tuple(args.offset_back_m), args.offset_side_m)

        def factory() -> StagedExecutor:
            return StagedExecutor(
                chain, "simulation", limits, args.gripper, strategy, score, cup.read,
                None if target is None else target.read, max_step_rad=max_step,
                max_gripper_step_m=0.05 * period * args.speed_fraction,
                margin_rad=math.radians(args.margin_deg),
                arrival_offset_m=draw_offset(offsets, args.offset_fraction,
                                             tuple(args.offset_back_m), args.offset_side_m),
            )

        actions = ExecutorActionSource(
            factory, publisher, (*limits.joint_limits_rad, limits.gripper_limits_m[args.gripper]),
        )
        recorder = stack.enter_context(build_recording_session(
            recorder_args(args), simulation=True, injected_action_source=actions,
        ))
        session = DemonstrationSession(
            recorder, actions, publisher, recorder.state_source, cup.read, gripper.read,
            None if target is None else target.read, publisher.request_randomize, config,
            dataset_root=args.dataset_path,
        )
        summary = session.run()
    summary["dataset_contract"] = json.loads(
        (args.dataset_path / "physical_contract.json").read_text(encoding="utf-8"))
    write_summary(summary, args.report_dir / f"demonstrations_{args.task_id}_{stamp}.json")
    print(json.dumps({"takes": args.takes, "successes": summary["successes"],
                      "dataset": str(args.dataset_path)}), flush=True)
    return 0


class _closing:
    """Context manager that closes a source on exit."""

    def __init__(self, source: object) -> None:
        self.source = source

    def __enter__(self) -> object:
        return self.source

    def __exit__(self, *exc: object) -> None:
        close = getattr(self.source, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    sys.exit(main())
