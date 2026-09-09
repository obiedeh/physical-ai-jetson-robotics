"""Command-line interface for the Physical AI lab."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from physical_ai_lab.config import ComputeTarget, LabProfile, RobotPlatform
from physical_ai_lab.inventory import write_inventory
from physical_ai_lab.ops_copilot import triage_telemetry
from physical_ai_lab.rtx_training import TrainingRunConfig, train_synria_reach_policy
from physical_ai_lab.slam import demo_mecanum_trials, summarize_mecanum_calibration
from physical_ai_lab.telemetry import generate_demo_telemetry, summarize_operational_risk

try:
    from agents.ops_copilot.copilot import (
        AnthropicClassifier,
        Classifier,
        DeterministicClassifier,
        enrich,
    )

    _AGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AGENTS_AVAILABLE = False

try:
    from arm_control.demo import full_pick_place_sequence, validate_all_demo_trajectories
    from arm_control.safety import ArmSafetyGate as _ArmSafetyGate  # noqa: F401

    _ARM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ARM_AVAILABLE = False

try:
    from edge_ai.benchmark import run_latency_benchmark, write_benchmark_report
    from edge_ai.camera_inference import CameraInferenceLoop, MockFrameSource

    _EDGE_AI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EDGE_AI_AVAILABLE = False

try:
    from slam.calibration import demo_calibration_trials, run_calibration_session

    _SLAM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SLAM_AVAILABLE = False

try:
    from lerobot.cube_sort import CubeSortSimulation
    from lerobot.dataset import (
        SynriaEpisodeDataset,
        generate_synthetic_episode,
        write_dataset_summary,
    )
    from lerobot.policy_eval import DeterministicArmPolicy, evaluate_policy_on_dataset
    from lerobot.recorder import RecordingSession

    _LEROBOT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LEROBOT_AVAILABLE = False

try:
    from isaac.isaaclab_tasks.synria_pickplace.gr00t.embodiment import (
        SYNRIA_EMBODIMENT_TAG,
        SynriaEmbodimentConfig,
        synria_modality_config,
    )

    _GR00T_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GR00T_AVAILABLE = False

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


@app.command("ops-copilot")
def ops_copilot(
    robot_id: str = "robo-car-01",
    samples: int = typer.Option(12, min=1, max=288),
    window_size: int = typer.Option(
        12,
        min=1,
        max=288,
        help="Sliding window size (samples) used in --replay mode.",
    ),
    output: Path = Path("reports/agents/ops_copilot/latest.json"),
    json_output: bool = typer.Option(False, "--json"),
    replay: bool = typer.Option(
        False,
        "--replay",
        help=(
            "Feed the full telemetry stream through a sliding window, emitting one report "
            "per step (mirrors the ROS 2 ops_copilot_node behaviour). "
            "Output is a JSON array written to --output."
        ),
    ),
    llm: bool = typer.Option(
        False,
        "--llm",
        help=(
            "Use the Anthropic LLM classifier instead of the deterministic rule-based default. "
            "Requires ANTHROPIC_API_KEY env var and 'pip install ....[llm]'. "
            "Falls back to the deterministic classifier if the API key is absent."
        ),
    ),
) -> None:
    """Run deterministic triage enriched by the agentic ops copilot layer."""
    if not _AGENTS_AVAILABLE:
        console.print(
            "[red]agents package not importable — run from the repo root.[/red]"
        )
        raise typer.Exit(1)

    chosen_classifier: Classifier
    if llm:
        if os.environ.get("ANTHROPIC_API_KEY"):
            chosen_classifier = AnthropicClassifier()
        else:
            console.print(
                "Warning: --llm requested but ANTHROPIC_API_KEY is not set. "
                "Falling back to the deterministic classifier."
            )
            chosen_classifier = DeterministicClassifier()
    else:
        chosen_classifier = DeterministicClassifier()

    telemetry = generate_demo_telemetry(robot_id=robot_id, samples=samples)
    output.parent.mkdir(parents=True, exist_ok=True)

    if replay:
        # Sliding-window replay: mirrors ops_copilot_node._on_telemetry behaviour.
        # After each new sample the oldest sample outside the window is dropped.
        replay_reports: list[dict[str, object]] = []
        buffer: list[object] = []
        for sample in telemetry:
            buffer.append(sample)
            if len(buffer) > window_size:
                buffer.pop(0)
            triage = triage_telemetry(buffer)  # type: ignore[arg-type]
            report = enrich(triage, chosen_classifier)
            replay_reports.append(report.to_dict())

        output.write_text(
            json.dumps(replay_reports, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if json_output:
            console.print(json.dumps(replay_reports, indent=2, sort_keys=True))
            return

        table = Table(title=f"Ops Copilot Replay: {robot_id}  ({len(replay_reports)} steps)")
        table.add_column("Step")
        table.add_column("Status")
        table.add_column("Confidence")
        table.add_column("Escalate")
        table.add_column("Findings")
        for idx, rep in enumerate(replay_reports, 1):
            triage_d: dict[str, object] = rep.get("triage", {})  # type: ignore[assignment]
            findings_list: list[object] = triage_d.get("findings", [])  # type: ignore[assignment]
            n_findings = len(findings_list)
            table.add_row(
                str(idx),
                str(rep.get("status", "")),
                str(rep.get("confidence", "")),
                str(rep.get("escalate_to_human", "")),
                str(n_findings),
            )
        console.print(table)
        console.print(
            f"\n{len(replay_reports)} steps replayed. Report written to {output}",
            markup=False,
        )
        return

    # --- Single-window mode (default) ---
    triage = triage_telemetry(telemetry)
    report = enrich(triage, chosen_classifier)

    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if json_output:
        console.print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    table = Table(title=f"Ops Copilot Report: {report.robot_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("status", report.status)
    table.add_row("classifier", report.classifier_name)
    table.add_row("confidence", report.confidence)
    table.add_row("escalate_to_human", str(report.escalate_to_human))
    table.add_row("root_cause", report.root_cause_hypothesis)
    for idx, rec in enumerate(report.operator_recommendation, 1):
        table.add_row(f"recommendation_{idx}", rec)
    console.print(table)
    console.print(f"\nReport written to {output}", markup=False)


@app.command("arm-safety-demo")
def arm_safety_demo(
    zone: str = typer.Option("left", help="Staging zone: left, right, top, bottom."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate all Synria arm demo trajectories against the safety gate."""
    if not _ARM_AVAILABLE:
        console.print("[red]arm_control package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    validation = validate_all_demo_trajectories()
    sequence = full_pick_place_sequence(zone)

    total_issues = sum(len(v) for v in validation.values())
    result: dict[str, object] = {
        "zone": zone,
        "sequence_trajectories": len(sequence),
        "total_duration_s": sum(t.duration_s() for t in sequence),
        "validation_issues": total_issues,
        "trajectories": {name: issues for name, issues in validation.items()},
    }

    if json_output:
        console.print(json.dumps(result, indent=2, sort_keys=True))
        return

    table = Table(title=f"Synria Arm Safety Demo — zone: {zone}")
    table.add_column("Trajectory")
    table.add_column("Duration (s)")
    table.add_column("Issues")
    for traj in sequence:
        issues = validation.get(traj.name, [])
        table.add_row(traj.name, f"{traj.duration_s():.1f}", str(len(issues)))
    console.print(table)
    status = "✅ All clear" if total_issues == 0 else f"⚠️  {total_issues} issue(s)"
    console.print(f"\n{status} across {len(sequence)}-step pick-place sequence.", markup=False)


@app.command("edge-ai-benchmark")
def edge_ai_benchmark(
    n_runs: int = typer.Option(100, min=10, max=10_000, help="Number of timed inference runs."),
    n_warmup: int = typer.Option(5, min=0, max=100, help="Warmup runs before timing."),
    frames: int = typer.Option(50, min=1, max=1_000, help="Mock camera frames to process."),
    report: Path = Path("reports/edge_ai/mock_benchmark.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run a mock inference latency benchmark and camera loop timing demo."""
    if not _EDGE_AI_AVAILABLE:
        console.print("[red]edge_ai package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    import numpy as np

    def identity(x: object) -> object:
        return x

    bench = run_latency_benchmark(
        inference_fn=identity,
        sample_input=np.zeros((1, 3, 224, 224), dtype=np.float32),
        n_warmup=n_warmup,
        n_runs=n_runs,
        workload_name="identity-224x224-mock",
        backend="mock",
        device="cpu",
        input_shape=[1, 3, 224, 224],
        batch_size=1,
        notes="CLI mock benchmark — replace inference_fn with real model for hardware evidence.",
    )
    written = write_benchmark_report(bench, report)

    loop = CameraInferenceLoop(
        source=MockFrameSource(height=480, width=640),
        inference_fn=lambda f: f,
        max_frames=frames,
    )
    loop.run()
    loop_stats = loop.stats()

    result: dict[str, object] = {
        "benchmark": bench.to_dict(),
        "camera_loop": loop_stats.to_dict() if loop_stats else {},
    }

    if json_output:
        console.print(json.dumps(result, indent=2, sort_keys=True))
        return

    table = Table(title="Edge AI Mock Benchmark")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("backend", bench.backend)
    table.add_row("n_runs", str(bench.latency.n_runs))
    table.add_row("mean_ms", f"{bench.latency.mean_ms:.3f}")
    table.add_row("p95_ms", f"{bench.latency.p95_ms:.3f}")
    table.add_row("throughput_fps", f"{bench.throughput_fps:.1f}")
    if loop_stats:
        table.add_row("camera_frames", str(loop_stats.n_frames))
        table.add_row("camera_mean_fps", f"{loop_stats.mean_fps:.1f}")
    console.print(table)
    console.print(f"\nBenchmark report written to {written}", markup=False)


@app.command("slam-calibration")
def slam_calibration(
    robot_id: str = "yahboom-orin-01",
    output: Path = Path("reports/slam/calibration_demo.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run a demo mecanum odometry calibration session and write evidence."""
    if not _SLAM_AVAILABLE:
        console.print("[red]slam package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    trials = demo_calibration_trials()
    report = run_calibration_session(robot_id=robot_id, trials=trials)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.to_json() + "\n", encoding="utf-8")

    if json_output:
        console.print(report.to_json())
        return

    table = Table(title=f"SLAM Calibration: {report.status.upper()} — {robot_id}")
    table.add_column("Trial")
    table.add_column("Axis")
    table.add_column("Error")
    table.add_column("Drift")
    table.add_column("Pass")
    for trial in report.trials:
        if trial.is_yaw:
            error = f"{trial.angular_error_rad:.4f} rad"
            drift = "—"
        else:
            error = f"{trial.linear_error_pct:.2f}%"
            drift = f"{trial.drift_m:.3f} m"
        table.add_row(
            trial.name, trial.axis, error, drift,
            "✅" if trial.passes() else "❌",
        )
    console.print(table)
    console.print(
        f"\nMax linear error: {report.max_linear_error_pct:.2f}%  "
        f"Max drift: {report.max_drift_m:.3f} m  "
        f"Max angular error: {report.max_angular_error_rad:.4f} rad",
        markup=False,
    )
    console.print(f"Report written to {output}", markup=False)


@app.command("synria-kinematics-benchmark")
def synria_kinematics_benchmark(
    n_fk_cases: int = typer.Option(
        6,
        min=1,
        max=36,
        help="Number of joint-angle cases to evaluate with planar FK.",
    ),
    report: Path = Path("reports/training/synria_kinematics_benchmark.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark Synria arm FK, workspace reachability, and trajectory validation.

    Runs planar forward kinematics across a sweep of joint angles, checks
    Cartesian target reachability for representative staging poses, and validates
    all named demo trajectories against the URDF joint limits.

    Output is written to --report (default: reports/training/synria_kinematics_benchmark.json).
    This fills SYNRIA_UPSTREAM_TODO.md item 8 (RoboCore-style kinematics benchmark).
    """
    import math as _math

    if not _ARM_AVAILABLE:
        console.print("[red]arm_control package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    from arm_control.demo import (
        POSE_BOARD_APPROACH,
        POSE_CARRY,
        POSE_HOME,
        validate_all_demo_trajectories,
    )
    from arm_control.kinematics import (
        LINK_FOREARM_M,
        LINK_UPPER_ARM_M,
        SYNRIA_GEOMETRIC_MAX_REACH_M,
        SYNRIA_VENDOR_MAX_REACH_M,
        CartesianPose,
        check_workspace_reachability,
        forward_kinematics_planar,
    )

    # --- FK sweep: evenly spaced q2 angles, elbow at 0 and -π/4 alternating ---
    fk_results: list[dict[str, object]] = []
    base_cases = [
        ("home",           0.0,              0.0),
        ("shoulder_up_45", _math.pi / 4,     0.0),
        ("elbow_bent_90",  0.0,              -_math.pi / 2),
        ("board_approach", 0.6,              -1.2),
        ("carry",          1.0,              -1.5),
        ("extended",       _math.pi / 4,     _math.pi / 4),
        ("shoulder_30",    _math.pi / 6,     0.0),
        ("shoulder_60",    _math.pi / 3,     0.0),
        ("elbow_minus_45", _math.pi / 4,     -_math.pi / 4),
        ("elbow_minus_60", _math.pi / 3,     -_math.pi / 3),
        ("fold_back",      -_math.pi / 6,    _math.pi / 4),
        ("down_reach",     -_math.pi / 4,    0.0),
    ]
    for case_name, q2, q3 in base_cases[:n_fk_cases]:
        r = forward_kinematics_planar(q2, q3)
        fk_results.append(
            {
                "case": case_name,
                "q2_rad": round(q2, 6),
                "q3_rad": round(q3, 6),
                "x_m": r.x_m,
                "z_m": r.z_m,
                "reach_m": r.reach_m,
                "within_geometric_max": r.reach_m <= SYNRIA_GEOMETRIC_MAX_REACH_M,
                "within_vendor_max": r.reach_m <= SYNRIA_VENDOR_MAX_REACH_M,
            }
        )

    # --- Workspace reachability ---
    workspace_cases: list[tuple[str, CartesianPose]] = [
        ("board_centre",      CartesianPose(0.30,  0.0,    0.15)),
        ("left_staging",      CartesianPose(0.20,  0.25,   0.12)),
        ("right_staging",     CartesianPose(0.20,  -0.25,  0.12)),
        ("overhead_far",      CartesianPose(0.0,   0.0,    0.70)),
        ("too_close",         CartesianPose(0.01,  0.0,    0.0)),
        ("max_extended_safe", CartesianPose(0.55,  0.0,    0.10)),
    ]
    ws_results: list[dict[str, object]] = []
    for case_name, pose in workspace_cases:
        res = check_workspace_reachability(pose)
        ws_results.append(
            {
                "case": case_name,
                "target_xyz_m": [pose.x, pose.y, pose.z],
                "radial_distance_m": res.radial_distance_m,
                "reachable": res.reachable,
                "message": res.message,
            }
        )

    # --- Named pose FK ---
    pose_fk: list[dict[str, object]] = []
    for pose_name, joints in [
        ("HOME", POSE_HOME),
        ("CARRY", POSE_CARRY),
        ("BOARD_APPROACH", POSE_BOARD_APPROACH),
    ]:
        r = forward_kinematics_planar(joints["joint_2"], joints["joint_3"])
        pose_fk.append(
            {
                "pose": pose_name,
                "joint_2_rad": joints["joint_2"],
                "joint_3_rad": joints["joint_3"],
                "planar_x_m": r.x_m,
                "planar_z_m": r.z_m,
                "planar_reach_m": r.reach_m,
            }
        )

    # --- Trajectory validation ---
    traj_results = validate_all_demo_trajectories()
    traj_summary = [
        {
            "trajectory": traj_name,
            "errors": [str(e) for e in errors],
            "passed": len(errors) == 0,
        }
        for traj_name, errors in traj_results.items()
    ]
    traj_pass = sum(1 for t in traj_summary if t["passed"])

    benchmark: dict[str, object] = {
        "meta": {
            "date": "2026-05-28",
            "mode": "rtx_simulation",
            "solver": "planar_fk_geometric",
            "note": (
                "Planar FK covers joint_2 (shoulder) and joint_3 (elbow) in the sagittal plane. "
                "Full 3D FK with the complete DH table is deferred to hardware bringup."
            ),
        },
        "link_lengths_m": {
            "base_height": 0.090,
            "upper_arm": LINK_UPPER_ARM_M,
            "forearm": LINK_FOREARM_M,
            "wrist": 0.105,
            "tool": 0.090,
            "geometric_max_reach": round(SYNRIA_GEOMETRIC_MAX_REACH_M, 4),
            "vendor_stated_max_reach": SYNRIA_VENDOR_MAX_REACH_M,
        },
        "fk_cases": fk_results,
        "workspace_checks": ws_results,
        "named_pose_fk": pose_fk,
        "trajectory_validation": {
            "summary": traj_summary,
            "passed": traj_pass,
            "total": len(traj_summary),
        },
    }

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if json_output:
        console.print(json.dumps(benchmark, indent=2, sort_keys=True))
        return

    # --- Rich summary tables ---
    fk_table = Table(title=f"Synria Planar FK — {len(fk_results)} cases")
    fk_table.add_column("Case")
    fk_table.add_column("q2 (rad)")
    fk_table.add_column("q3 (rad)")
    fk_table.add_column("x (m)")
    fk_table.add_column("z (m)")
    fk_table.add_column("reach (m)")
    fk_table.add_column("≤ geo max")
    for fk_r in fk_results:
        fk_table.add_row(
            str(fk_r["case"]),
            f"{fk_r['q2_rad']:.4f}",
            f"{fk_r['q3_rad']:.4f}",
            f"{fk_r['x_m']:.5f}",
            f"{fk_r['z_m']:.5f}",
            f"{fk_r['reach_m']:.5f}",
            "✅" if fk_r["within_geometric_max"] else "❌",
        )
    console.print(fk_table)

    ws_table = Table(title="Workspace Reachability")
    ws_table.add_column("Case")
    ws_table.add_column("Distance (m)")
    ws_table.add_column("Reachable")
    for w in ws_results:
        ws_table.add_row(
            str(w["case"]),
            str(w["radial_distance_m"]),
            "✅" if w["reachable"] else "❌",
        )
    console.print(ws_table)

    traj_table = Table(title="Trajectory Validation")
    traj_table.add_column("Trajectory")
    traj_table.add_column("Pass")
    for t in traj_summary:
        traj_table.add_row(str(t["trajectory"]), "✅" if t["passed"] else "❌")
    console.print(traj_table)

    console.print(
        f"\nFK cases: {len(fk_results)}  "
        f"Workspace checks: {len(ws_results)}  "
        f"Trajectories: {traj_pass}/{len(traj_summary)} passed",
        markup=False,
    )
    console.print(f"Report written to {report}", markup=False)


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


@app.command("lerobot-dataset-demo")
def lerobot_dataset_demo(
    n_episodes: int = typer.Option(4, min=1, max=64, help="Episodes to generate."),
    zone: str = typer.Option("left", help="Staging zone: left, right, top, bottom."),
    game: str = typer.Option("chess", help="Board game: ludo, chess, checkers."),
    n_steps: int = typer.Option(0, min=0, help="Steps per episode (0 = full trajectory)."),
    output: Path = Path("reports/lerobot/demo_dataset_summary.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Generate synthetic Synria demonstration episodes and write a dataset summary."""
    if not _LEROBOT_AVAILABLE:
        console.print("[red]lerobot package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    session = RecordingSession(
        n_episodes=n_episodes,
        zones=(zone,),
        games=(game,),
        mock=True,
    )
    dataset = session.record_all(n_steps=n_steps)
    written = write_dataset_summary(dataset, output)

    if json_output:
        import json as _json
        summary = {
            "n_episodes": len(dataset),
            "total_steps": dataset.total_steps,
            "output": str(written),
        }
        console.print(_json.dumps(summary, indent=2, sort_keys=True))
        return

    table = Table(title=f"LeRobot Dataset Demo — {n_episodes} episode(s), zone={zone!r}")
    table.add_column("Episode")
    table.add_column("Zone")
    table.add_column("Game")
    table.add_column("Steps")
    table.add_column("Synthetic")
    for ep in dataset:
        table.add_row(
            ep.metadata.episode_id[:8] + "…",
            ep.metadata.staging_zone,
            ep.metadata.game,
            str(ep.metadata.n_steps),
            "✅" if ep.metadata.synthetic else "🔴",
        )
    console.print(table)
    console.print(f"\nDataset summary written to {written}", markup=False)


@app.command("lerobot-policy-eval")
def lerobot_policy_eval(
    n_episodes: int = typer.Option(4, min=1, max=64, help="Episodes to evaluate."),
    n_steps: int = typer.Option(30, min=1, help="Steps per episode."),
    output: Path = Path("reports/lerobot/policy_eval_deterministic.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Evaluate the deterministic oracle policy against synthetic demo episodes."""
    if not _LEROBOT_AVAILABLE:
        console.print("[red]lerobot package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    import json as _json

    from lerobot.schema import VALID_STAGING_ZONES

    zones = list(VALID_STAGING_ZONES)
    episodes = [
        generate_synthetic_episode(zone=zones[i % len(zones)], n_steps=n_steps)
        for i in range(n_episodes)
    ]
    dataset = SynriaEpisodeDataset(episodes)
    policy = DeterministicArmPolicy()
    results = evaluate_policy_on_dataset(policy, dataset)

    total_pass = sum(1 for r in results if r.passed)
    result_dicts = [r.to_dict() for r in results]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json.dumps(result_dicts, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")

    if json_output:
        console.print(_json.dumps(result_dicts, indent=2, sort_keys=True))
        return

    table = Table(title="LeRobot Policy Eval — DeterministicArmPolicy")
    table.add_column("Episode")
    table.add_column("Steps")
    table.add_column("Mean err (rad)")
    table.add_column("Max err (rad)")
    table.add_column("Completion")
    table.add_column("Pass")
    for r in results:
        table.add_row(
            r.episode_id[:8] + "…",
            str(r.n_steps),
            f"{r.mean_joint_tracking_error_rad:.6f}",
            f"{r.max_joint_tracking_error_rad:.6f}",
            f"{r.completion_rate:.1%}",
            "✅" if r.passed else "❌",
        )
    console.print(table)
    console.print(
        f"\n{total_pass}/{n_episodes} episodes passed. Report written to {output}",
        markup=False,
    )


@app.command("lerobot-cube-sort-demo")
def lerobot_cube_sort_demo(
    n_cubes: int = typer.Option(6, min=1, max=24, help="Number of cubes to sort."),
    seed: int = typer.Option(42, help="RNG seed for deterministic cube placement."),
    report: Path = Path("reports/demo/synria_cube_sort_sim.md"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Simulate the Synria colored-cube sorting workflow end-to-end.

    Runs a mock color-detection + arm trajectory planning pipeline for
    N cubes. Each cube is detected (color + board position), assigned
    a staging zone (red→right, green→left, blue→top, yellow→bottom),
    and a full home→approach→staging→home trajectory is validated through
    the safety gate.

    Closes SYNRIA_UPSTREAM_TODO.md item 6 (cube sorting simulation demo).
    Report written to --report (default: reports/demo/synria_cube_sort_sim.md).
    """
    if not _LEROBOT_AVAILABLE:
        console.print("[red]lerobot package not importable — run from repo root.[/red]")
        raise typer.Exit(1)

    sim = CubeSortSimulation(n_cubes=n_cubes, seed=seed)
    results = sim.run()
    summary = sim.summary(results)

    # --- Write markdown report ---
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synria Cube Sorting Simulation Demo",
        "",
        "**Date:** 2026-05-28  ",
        "**Mode:** RTX simulation (no physical arm, no camera)  ",
        f"**Seed:** {seed}  ",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cubes sorted | {summary['n_cubes']} |",
        f"| Passed | {summary['passed']} |",
        f"| Failed | {summary['failed']} |",
        f"| Pass rate | {summary['pass_rate']:.0%} |",
        f"| Total trajectory time | {summary['total_trajectory_duration_s']:.1f} s |",
        "",
        "## Color → Zone Assignments",
        "",
        "| Color | Zone | Count |",
        "|---|---|---|",
    ]
    for color, zone in sorted(
        [("red", "right"), ("green", "left"), ("blue", "top"), ("yellow", "bottom")]
    ):
        count = summary["color_counts"].get(color, 0)
        cubes = summary["zone_assignments"].get(zone, [])
        lines.append(f"| {color} | {zone} | {count} (cubes: {cubes}) |")

    lines += [
        "",
        "## Per-Cube Results",
        "",
        "| Cube | Color | Zone | Confidence | Traj steps | Duration (s) | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['cube_idx']} | {r['color']} | {r['target_zone']} "
            f"| {r['detection_confidence']:.3f} "
            f"| {r['trajectory_steps']} "
            f"| {r['trajectory_duration_s']:.1f} "
            f"| {'✅' if r['passed'] else '❌'} |"
        )

    lines += [
        "",
        "## Pipeline",
        "",
        "1. **MockColorDetector** — seeded RNG assigns color labels and board positions",
        "   (simulates YOLOv8 / OpenCV HSV overhead detection)",
        "2. **CubeSortPlanner** — maps color → staging zone; builds",
        "   `home → board_approach → staging_zone → home` trajectory via `arm_control`",
        "3. **ArmSafetyGate** — validates every waypoint against URDF joint limits",
        "",
        "## Hardware Path",
        "",
        "Replace `MockColorDetector` with the real overhead board camera pipeline",
        "(YOLOv8 detection + ArUco homography for board-to-world transform) and",
        "replace trajectory replay with live MoveIt 2 execution via",
        "`physical_ai_ops_copilot` or the Synria arm ROS 2 driver.",
        "",
        "See `docs/reports/synria_c10_camera_contract.md` for camera integration contract.",
    ]

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if json_output:
        console.print(json.dumps(summary, indent=2, sort_keys=True))
        return

    table = Table(title=f"Synria Cube Sort Simulation — {n_cubes} cubes, seed={seed}")
    table.add_column("Cube")
    table.add_column("Color")
    table.add_column("Zone")
    table.add_column("Conf")
    table.add_column("Duration (s)")
    table.add_column("Pass")
    for r in summary["results"]:
        table.add_row(
            str(r["cube_idx"]),
            r["color"],
            r["target_zone"],
            f"{r['detection_confidence']:.3f}",
            f"{r['trajectory_duration_s']:.1f}",
            "✅" if r["passed"] else "❌",
        )
    console.print(table)
    console.print(
        f"\n{summary['passed']}/{summary['n_cubes']} cubes passed. "
        f"Report written to {report}",
        markup=False,
    )


@app.command("ops-copilot-health")
def ops_copilot_health(
    robot_id: str = "robo-car-01",
    output: Path = Path("reports/agents/ops_copilot/health.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Verify the ops copilot stack is fully importable and functional.

    Runs a single telemetry sample through triage + enrichment and writes
    a health report to --output. Exits 0 on success, 1 on any failure.
    Useful as a pre-flight check before starting the ROS 2 node.
    """
    if not _AGENTS_AVAILABLE:
        console.print(
            "[red]agents package not importable — ops copilot stack is broken.[/red]"
        )
        raise typer.Exit(1)


    checks: dict[str, bool] = {}
    errors: list[str] = []

    # Check 1: telemetry generation
    try:
        telemetry = generate_demo_telemetry(robot_id=robot_id, samples=1)
        checks["telemetry_generation"] = True
    except Exception as exc:  # noqa: BLE001
        checks["telemetry_generation"] = False
        errors.append(f"telemetry_generation: {exc}")

    # Check 2: deterministic triage
    try:
        triage = triage_telemetry(telemetry)  # type: ignore[possibly-undefined]
        checks["triage"] = True
    except Exception as exc:  # noqa: BLE001
        checks["triage"] = False
        errors.append(f"triage: {exc}")

    # Check 3: deterministic enrichment
    try:
        classifier = DeterministicClassifier()
        report_obj = enrich(triage, classifier)  # type: ignore[possibly-undefined]
        checks["enrichment"] = True
    except Exception as exc:  # noqa: BLE001
        checks["enrichment"] = False
        errors.append(f"enrichment: {exc}")

    # Check 4: report serialisation
    try:
        _ = json.dumps(report_obj.to_dict())  # type: ignore[possibly-undefined]
        checks["serialisation"] = True
    except Exception as exc:  # noqa: BLE001
        checks["serialisation"] = False
        errors.append(f"serialisation: {exc}")

    all_passed = all(checks.values())
    health: dict[str, object] = {
        "robot_id": robot_id,
        "status": "healthy" if all_passed else "degraded",
        "checks": checks,
        "errors": errors,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if json_output:
        console.print(json.dumps(health, indent=2, sort_keys=True))
        if not all_passed:
            raise typer.Exit(1)
        return

    table = Table(title="Ops Copilot Health Check")
    table.add_column("Check")
    table.add_column("Status")
    for check_name, passed in checks.items():
        table.add_row(check_name, "✅" if passed else "❌")
    console.print(table)
    console.print(f"\nOverall: {health['status']}  Report written to {output}", markup=False)

    if not all_passed:
        raise typer.Exit(1)


@app.command("gr00t-embodiment")
def gr00t_embodiment(
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON."),
) -> None:
    """Show the Synria GR00T embodiment config (tag, joints, action dim, cameras)."""
    import json as _json

    if not _GR00T_AVAILABLE:
        console.print(
            "[red]isaac.isaaclab_tasks package not importable — run from repo root.[/red]"
        )
        raise typer.Exit(1)

    cfg = SynriaEmbodimentConfig()
    modality = synria_modality_config()
    payload = {
        "embodiment_tag": cfg.embodiment_tag,
        "arm_joints": cfg.arm_joints,
        "gripper_joints": cfg.gripper_joints,
        "action_dim": cfg.action_dim,
        "camera_keys": cfg.camera_keys,
        "image_size": list(cfg.image_size),
        "modality_config": modality,
    }

    if json_output:
        console.print(_json.dumps(payload, indent=2))
        return

    table = Table(title=f"Synria GR00T Embodiment — tag: {SYNRIA_EMBODIMENT_TAG}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("embodiment_tag", cfg.embodiment_tag)
    table.add_row("arm_joints", ", ".join(cfg.arm_joints))
    table.add_row("gripper_joints", ", ".join(cfg.gripper_joints))
    table.add_row("action_dim", str(cfg.action_dim))
    table.add_row("camera_keys", ", ".join(cfg.camera_keys))
    table.add_row("image_size", f"{cfg.image_size[0]}×{cfg.image_size[1]}")
    table.add_row("modality_keys", ", ".join(sorted(modality)))
    console.print(table)
    console.print(
        "\nFull recipe: docs/SYNRIA_GR00T_FINETUNE.md",
        markup=False,
    )


@app.command("gr00t-finetune-dry-run")
def gr00t_finetune_dry_run(
    game: str = typer.Option("chess", help="Game: ludo | chess | checkers"),
    amplified: bool = typer.Option(False, "--amplified", help="Use Mimic-amplified dataset."),
    base_model: str = typer.Option("nvidia/GR00T-N1.7-3B", help="Base model ID."),
    output_dir: Path = Path("runs/gr00t_finetune"),
    max_steps: int = typer.Option(10000, min=1),
    global_batch_size: int = typer.Option(16, min=1),
) -> None:
    """Print the GR00T fine-tune launch command without invoking it (dry run).

    Requires no hardware — useful for reviewing the command before running
    on the RTX 5090.
    """
    import sys

    if not _GR00T_AVAILABLE:
        console.print(
            "[red]isaac.isaaclab_tasks package not importable — run from repo root.[/red]"
        )
        raise typer.Exit(1)

    if game not in ("ludo", "chess", "checkers"):
        console.print(f"[red]Unknown game '{game}'. Choose: ludo, chess, checkers.[/red]")
        raise typer.Exit(1)

    # Build the argv list that finetune.main() would parse, plus --dry-run.
    argv = [
        "--game", game,
        "--base-model", base_model,
        "--output-dir", str(output_dir),
        "--max-steps", str(max_steps),
        "--global-batch-size", str(global_batch_size),
        "--dry-run",
    ]
    if amplified:
        argv.insert(2, "--amplified")

    console.print(f"[bold]GR00T fine-tune dry run[/bold] — game={game}, amplified={amplified}")
    console.print("Passing to gr00t.finetune.main():", " ".join(argv))
    console.print()

    try:
        from isaac.isaaclab_tasks.synria_pickplace.gr00t.finetune import main as finetune_main

        finetune_main(argv)
    except SystemExit as exc:  # pragma: no cover
        if exc.code not in (0, None):
            console.print(f"[red]Finetune wrapper exited with code {exc.code}[/red]")
            raise typer.Exit(int(exc.code) if isinstance(exc.code, (int, str)) else 1) from exc
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)


@app.command("gr00t-inference-dry-run")
def gr00t_inference_dry_run(
    task: str = typer.Option(
        "Synria-Chess-PickPlace-v0",
        help="Isaac Lab task ID: Synria-{Ludo,Chess,Checkers}-PickPlace-v0.",
    ),
    checkpoint: Path = Path("runs/gr00t_finetune"),
    base_model: str = typer.Option("nvidia/GR00T-N1.7-3B", help="Base GR00T model ID."),
    num_episodes: int = typer.Option(20, min=1, help="Evaluation episodes to roll out."),
    instruction: str = typer.Option(
        "pick the target piece and move it to the target staging zone, then return it",
        help="Language instruction passed to the GR00T policy each episode.",
    ),
    report: Path = Path("reports/training/gr00t_eval.json"),
) -> None:
    """Print the GR00T inference rollout config without executing it (dry run).

    Requires no hardware — review the config before running on the RTX 5090.
    """
    if not _GR00T_AVAILABLE:
        console.print(
            "[red]isaac.isaaclab_tasks package not importable — run from repo root.[/red]"
        )
        raise typer.Exit(1)

    valid_tasks = (
        "Synria-Ludo-PickPlace-v0",
        "Synria-Chess-PickPlace-v0",
        "Synria-Checkers-PickPlace-v0",
    )
    if task not in valid_tasks:
        console.print(
            f"[red]Unknown task '{task}'. "
            f"Choose from: {', '.join(valid_tasks)}[/red]"
        )
        raise typer.Exit(1)

    argv = [
        "--task", task,
        "--checkpoint", str(checkpoint),
        "--base-model", base_model,
        "--num-episodes", str(num_episodes),
        "--instruction", instruction,
        "--report", str(report),
    ]

    console.print(f"[bold]GR00T inference dry run[/bold] — task={task}")
    console.print("Would invoke inference.main() with:", " ".join(argv))
    console.print("Not invoking — dry run only.")
    console.print()

    table = Table(title="GR00T Inference Config")
    table.add_column("Parameter")
    table.add_column("Value")
    table.add_row("task", task)
    table.add_row("checkpoint", str(checkpoint))
    table.add_row("base_model", base_model)
    table.add_row("num_episodes", str(num_episodes))
    short_instr = instruction[:72] + "..." if len(instruction) > 72 else instruction
    table.add_row("instruction", short_instr)
    table.add_row("report", str(report))
    console.print(table)
    console.print("\nFull recipe: docs/SYNRIA_GR00T_FINETUNE.md — Phase 6 (Evaluate)", markup=False)


if __name__ == "__main__":
    app()
