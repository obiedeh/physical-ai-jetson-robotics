"""Record MimicGen-style demonstrations in the Isaac scene from teleoperated source takes.

Start the scene in policy mode first (teleop repository)::

    SKIP_DRIVER=1 bash scripts/run_alicia_stack.sh policy

then::

    python scripts/record_mimic_demonstrations.py --source-dataset "$SOURCE_DATASET" \\
        --held-out 12 24 36 44 --takes 20 --placement-seed 5026 \\
        --dataset-path ~/synria-sim-datasets/mimic_cup_01/dataset \\
        --repo-id local/mimic_cup_01 --report-dir reports/sim_policy/mimic_cup_01

Every take: the arm walks to the start pose, the scene draws a seeded placement, the
cup's pose is read, a source take is chosen and moved to it
(``synria_lerobot.mimic_generation``), and the moved take is replayed through the
sim-only target topic while the recorder samples state, both cameras and the task's
goal channels. The scene's scorer labels each take; failures (including takes whose
move could not be solved) are labelled and appended to the dataset's exclusion log.
The trainer's held-out episodes are never used as sources. Use a placement seed far
from the evaluation seeds (2026 and 2027). Simulation only; nothing here touches the
physical arm.
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

from scripts.record_staged_demonstrations import (  # noqa: E402
    DEFAULT_KINEMATICS,
    DEFAULT_LIMITS,
    DEFAULT_REGISTRY,
    EVALUATION_SEEDS,
    _closing,
    recorder_args,
)
from synria_lerobot.arm_kinematics import KinematicChain  # noqa: E402
from synria_lerobot.demo_engine import (  # noqa: E402
    DemonstrationConfig,
    DemonstrationSession,
    ExecutorActionSource,
    write_summary,
)
from synria_lerobot.mimic_generation import (  # noqa: E402
    MimicReplayExecutor,
    load_source_takes,
)
from synria_lerobot.quality_gates import load_limits  # noqa: E402
from synria_lerobot.recorder import build_recording_session  # noqa: E402
from synria_lerobot.sim_policy_runner import (  # noqa: E402
    LIFT_SCORE,
    SIM_CUP_POSE_TOPIC,
    SIM_GRIPPER_POSE_TOPIC,
    SIM_RANDOMIZE_TOPIC,
    RosPoseSource,
    RosSimPublisher,
    start_pose_from_dataset,
)

LANE = "simulation"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options; the dataset path must lie outside the repository."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-dataset", type=Path, required=True,
                        help="The teleoperated LeRobot dataset whose takes are moved.")
    parser.add_argument("--held-out", type=int, nargs="*", default=[],
                        help="Source episodes the trainer holds out; never used as sources.")
    parser.add_argument("--task-id", default="pick_cup", choices=("pick_cup",))
    parser.add_argument("--takes", type=int, required=True)
    parser.add_argument("--placement-seed", type=int, required=True)
    parser.add_argument("--randomize-scale", type=float, default=1.0)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--start-pose", type=float, nargs=7, metavar="V", default=None,
                        help="Default: the source dataset's mean first frame.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-take-s", type=float, default=45.0)
    parser.add_argument("--gripper", default="50mm", choices=("50mm", "100mm"))
    parser.add_argument("--task-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=1.0,
                        help="Joint speed of the walk to the moved take's first frame.")
    parser.add_argument("--speed-fraction", type=float, default=0.8)
    parser.add_argument("--margin-deg", type=float, default=0.5,
                        help="Inverse-kinematics margin inside the joint bounds; the human "
                             "takes run within 2 degrees of some bounds, so the staged "
                             "executor's 3 degrees rejects them.")
    parser.add_argument("--nearest", type=int, default=3,
                        help="Choose each take's source among this many nearest by radius.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load the sources, open the scene and the recorder, run the takes, write the summary."""
    args = build_parser().parse_args(argv)
    if any(abs(args.placement_seed - seed) < args.takes for seed in EVALUATION_SEEDS):
        raise SystemExit("choose a placement seed whose takes cannot overlap the evaluation "
                         f"seeds {EVALUATION_SEEDS}")
    chain = KinematicChain.load(args.kinematics)
    limits = load_limits(args.limits)
    sources, skipped = load_source_takes(args.source_dataset, chain, LANE,
                                         exclude=args.held_out)
    if not sources:
        raise SystemExit(f"no usable source takes: {skipped}")
    print(json.dumps({"sources": [take.episode_index for take in sources],
                      "skipped": {str(k): v for k, v in skipped.items()}}), flush=True)
    start_pose = (tuple(args.start_pose) if args.start_pose is not None
                  else start_pose_from_dataset(args.source_dataset))
    period = 1.0 / args.fps
    max_step = args.max_joint_speed_rad_s * period * args.speed_fraction
    config = DemonstrationConfig(
        takes=args.takes, fps=args.fps, start_pose=start_pose, score=LIFT_SCORE,
        placement_seed=args.placement_seed, randomize_scale=args.randomize_scale,
        max_take_s=args.max_take_s, max_step_rad=max_step,
    )
    rng = random.Random(args.placement_seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with ExitStack() as stack:
        publisher = stack.enter_context(_closing(RosSimPublisher(
            randomize_topic=SIM_RANDOMIZE_TOPIC, node_name="synria_mimic_engine",
        )))
        cup = stack.enter_context(_closing(RosPoseSource(
            SIM_CUP_POSE_TOPIC, node_name="synria_mimic_cup")))
        gripper = stack.enter_context(_closing(RosPoseSource(
            SIM_GRIPPER_POSE_TOPIC, node_name="synria_mimic_gripper")))

        def factory() -> MimicReplayExecutor:
            return MimicReplayExecutor(
                sources, cup.read, chain, LANE, limits.joint_limits_rad, rng,
                margin_rad=math.radians(args.margin_deg), max_step_rad=max_step,
                max_gripper_step_m=0.05 * period * args.speed_fraction,
                nearest=args.nearest,
            )

        actions = ExecutorActionSource(
            factory, publisher,  # type: ignore[arg-type]
            (*limits.joint_limits_rad, limits.gripper_limits_m[args.gripper]),
        )
        recorder = stack.enter_context(build_recording_session(
            recorder_args(args), simulation=True, injected_action_source=actions,
        ))
        session = DemonstrationSession(
            recorder, actions, publisher, recorder.state_source, cup.read, gripper.read,
            None, publisher.request_randomize, config, dataset_root=args.dataset_path,
        )
        summary = session.run()
    summary["kind"] = "simulation_mimic_demonstrations"
    summary["note"] = ("Teleoperated takes moved to new cup placements (MimicGen method) and "
                       "replayed in the Isaac scene; failures are labelled and excluded, "
                       "successes train; never physical evidence.")
    summary["source_dataset"] = str(args.source_dataset)
    summary["source_episodes"] = [take.episode_index for take in sources]
    summary["sources_skipped"] = {str(k): v for k, v in skipped.items()}
    summary["dataset_contract"] = json.loads(
        (args.dataset_path / "physical_contract.json").read_text(encoding="utf-8"))
    write_summary(summary, args.report_dir / f"demonstrations_mimic_{args.task_id}_{stamp}.json")
    print(json.dumps({"takes": args.takes, "successes": summary["successes"],
                      "dataset": str(args.dataset_path)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
