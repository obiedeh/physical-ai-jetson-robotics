"""Isaac Lab RL training entry point for Synria pick-and-place tasks.

Wraps RSL-RL's OnPolicyRunner with Synria-specific task registration,
hyperparameter config lookup, and artifact capture. Designed to run inside
Isaac Sim 5.1's Python on the Linux RTX 5090 workstation.

Quick smoke test (validates env + network, ~2 min on RTX 5090):
    $ISAAC_PYTHON isaac/scripts/train_synria_pickplace.py \\
        --task Synria-Chess-PickPlace-v0 \\
        --num_envs 16 \\
        --max_iterations 5 \\
        --headless

Full training run (~2–4 h on RTX 5090 with 4096 envs):
    $ISAAC_PYTHON isaac/scripts/train_synria_pickplace.py \\
        --task Synria-Chess-PickPlace-v0 \\
        --num_envs 4096 \\
        --headless

Resume from checkpoint:
    $ISAAC_PYTHON isaac/scripts/train_synria_pickplace.py \\
        --task Synria-Chess-PickPlace-v0 \\
        --num_envs 4096 \\
        --resume \\
        --checkpoint reports/training/synria_chess_pickplace_v0/model_1000.pt \\
        --headless

Outputs (written to reports/training/<experiment_name>/):
    model_<iter>.pt        — RSL-RL checkpoint (policy + optimizer state)
    model_final.pt         — final checkpoint
    tb_logs/               — TensorBoard event files
    training_results.json  — scalar summary for CI / evidence capture
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Repo root on sys.path — must happen before Isaac Lab / Synria task imports.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TRAIN_OUT_ROOT = _REPO_ROOT / "reports" / "training"

# Sibling module — TensorBoard-only, so importing it costs nothing and works
# outside the Isaac venv too.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_training_run  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_synria_pickplace",
        description="Train a Synria pick-and-place policy with RSL-RL PPO inside Isaac Lab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=(
            "Synria-Ludo-PickPlace-v0",
            "Synria-Chess-PickPlace-v0",
            "Synria-Checkers-PickPlace-v0",
        ),
        required=True,
        help="Which registered Isaac Lab task to train.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="Number of parallel simulation environments. Use 16 for smoke test.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override max_iterations from the task's PPO config. Use 5 for smoke test.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record a short evaluation video after training (requires display or VNC).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint for this experiment.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit path to a .pt checkpoint to resume from (overrides --resume auto-detect).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic training runs.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Log to TensorBoard every N iterations (default: 1).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the loaded checkpoint in pure inference — no training, no "
        "checkpoint writes. Use for GUI viewing so a 16-env viewer doesn't "
        "pollute the experiment directory with its own checkpoints.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Override the experiment name (checkpoint/log directory under "
        "reports/training/). Required when running two trainers side by side "
        "(e.g. an A/B run) so their model_*.pt files do not interleave.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def _assert_gripper_gate(skip: bool = False) -> None:
    """Phase 7 startup guard: refuse to train on an unvalidated gripper.

    The repair brief made Synria policy training conditional on the grasp
    gate passing, and required the launchers to refuse to start until it
    does — otherwise it is far too easy to spend a night of GPU time
    training against a gripper that cannot pick anything up, which is
    exactly what happened before the gate existed.

    Checks the two committed gate artefacts:
      * ``reports/synria_grasp_trials_production.json`` — Phase 9, needs
        >= 9 of 10 successful scripted grasp trials (the brief's threshold).
      * ``reports/synria_blocks_production.json``       — Phase 5, needs
        every non-stress block width measured within tolerance.

    Raises SystemExit(2) with an actionable message if either is missing or
    failing. ``--skip_gripper_gate`` bypasses it for env-only debugging and
    prints a loud warning; never use it for a run whose numbers you intend
    to report.
    """
    import json

    if skip:
        sys.stderr.write(
            "\n*** WARNING: --skip_gripper_gate set. Training against an\n"
            "*** UNVALIDATED gripper. Results from this run must not be\n"
            "*** reported as evidence about task learnability.\n\n"
        )
        return

    repo = Path(__file__).resolve().parents[2]
    trials = repo / "reports/synria_grasp_trials_production.json"
    blocks = repo / "reports/synria_blocks_production.json"
    how = (
        "Run the gate first:\n"
        "  $ISAAC_PYTHON isaac/scripts/synria_grasp_feasibility.py "
        "--headless --trials 10\n"
        "  $ISAAC_PYTHON isaac/scripts/synria_grasp_feasibility.py "
        "--headless --block_test\n"
        "Or pass --skip_gripper_gate to train anyway (results not reportable)."
    )
    for p in (trials, blocks):
        if not p.exists():
            raise SystemExit(f"[gate] BLOCKED: {p.name} missing.\n{how}")

    t = json.loads(trials.read_text())
    n_ok = sum(1 for r in t if r.get("success"))
    if n_ok < 9:
        raise SystemExit(
            f"[gate] BLOCKED: Phase 9 grasp gate is {n_ok}/{len(t)}; "
            f"the brief requires >= 9/10.\n{how}"
        )

    b = json.loads(blocks.read_text())
    gate_rows = [r for r in b if not r.get("stress_case")]
    bad = [r for r in gate_rows if not r.get("ok")]
    if bad or not gate_rows:
        raise SystemExit(
            f"[gate] BLOCKED: Phase 5 width gauge has {len(bad)} failing "
            f"block trial(s) of {len(gate_rows)}.\n{how}"
        )

    print(
        f"[gate] gripper validated: Phase 9 {n_ok}/{len(t)} grasp trials, "
        f"Phase 5 {len(gate_rows) - len(bad)}/{len(gate_rows)} block widths. "
        f"Training may proceed.",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = _build_parser()
    parser.add_argument(
        "--skip_gripper_gate",
        action="store_true",
        help="bypass the Phase 5/9 gripper validation guard (debugging only; "
             "runs so launched are not reportable evidence)",
    )

    # ------------------------------------------------------------------
    # 1. Boot Isaac Sim via Isaac Lab's AppLauncher
    # ------------------------------------------------------------------
    # AppLauncher (not a raw SimulationApp) is the supported way to boot a
    # standalone Isaac Lab training script. It injects --headless/--device/etc.
    # into the parser and, crucially, sets up the standalone-script lifecycle
    # (ISAAC_LAUNCHED_FROM_TERMINAL) so SimulationContext does NOT install its
    # interactive timeline-STOP handler. With a raw SimulationApp that handler
    # busy-loops `while not is_playing(): self.render()` when the timeline
    # stops during env setup, deadlocking headless runs at 100% CPU forever.
    try:
        from isaaclab.app import AppLauncher  # type: ignore[import-not-found]
    except ImportError as exc:
        sys.stderr.write(
            "Isaac Lab (isaaclab.app.AppLauncher) is not importable. "
            "Activate the Isaac Sim 5.1 Python venv before running:\n"
            "  source ~/.venv/isaacsim5/bin/activate\n"
        )
        raise SystemExit(1) from exc

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    # Guard BEFORE booting Isaac Sim: a blocked run should cost a second,
    # not a 40 s simulator boot.
    _assert_gripper_gate(getattr(args, "skip_gripper_gate", False))
    app_launcher = AppLauncher(args)
    app = app_launcher.app

    try:
        return _run_training(args, app)
    finally:
        app.close()


def _run_training(args: argparse.Namespace, app: object) -> int:
    import torch  # type: ignore[import-not-found]

    # ------------------------------------------------------------------
    # 2. Import Isaac Lab + RSL-RL (available after SimulationApp boots)
    # ------------------------------------------------------------------
    try:
        import gymnasium as gym  # type: ignore[import-not-found]
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]
        from rsl_rl.runners import (
            OnPolicyRunner as RslRlOnPolicyRunner,  # type: ignore[import-not-found]
        )
    except ImportError as exc:
        sys.stderr.write(
            f"Isaac Lab or isaaclab_rl not importable: {exc}\n"
            "Install Isaac Lab: see docs/SETUP_RTX.md\n"
        )
        raise SystemExit(1) from exc

    # ------------------------------------------------------------------
    # Neutralize SimulationContext's interactive timeline-STOP handler.
    #
    # `_app_control_on_stop_handle_fn` busy-loops `while not is_playing():
    # self.render()` whenever a timeline STOP event fires while
    # `_disable_app_control_on_stop_handle` is False. `app.close()` closes the
    # stage (UsdContext.closeStage), which fires exactly such a STOP — so ANY
    # exit path, including normal completion and error unwinding through our
    # `finally: app.close()`, spins forever (100% CPU, GPU idle) or, with
    # fastShutdown, kills the process with exit 0 before Python can print a
    # traceback. That masking is how a plain TypeError in this script
    # masqueraded as a "training deadlock" for weeks.
    # The handler only exists to keep a *standalone interactive* app alive when
    # the user stops the sim; a headless training run drives its own
    # reset/step loop and never needs it. Replace it with a no-op so the STOP
    # event returns immediately. The flag ISAAC_LAUNCHED_FROM_TERMINAL must
    # stay False here, otherwise ManagerBasedEnv skips sim.reset() and physics
    # handles never activate.
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = (  # type: ignore[assignment]
        lambda self, event: None
    )

    # ------------------------------------------------------------------
    # 3. Register Synria tasks (imports trigger gym.register calls)
    # ------------------------------------------------------------------
    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace.train_cfg import TASK_TRAIN_CFG

    if args.task not in TASK_TRAIN_CFG:
        sys.stderr.write(f"No training config found for task '{args.task}'.\n")
        raise SystemExit(1)

    runner_cfg_cls = TASK_TRAIN_CFG[args.task]
    runner_cfg = runner_cfg_cls()

    # Apply CLI overrides
    if args.max_iterations is not None:
        runner_cfg.max_iterations = args.max_iterations
    if args.device:
        runner_cfg.device = args.device

    # ------------------------------------------------------------------
    # 4. Create the gym environment
    # ------------------------------------------------------------------
    print(f"\n[train] Task      : {args.task}", flush=True)
    print(f"[train] Num envs  : {args.num_envs}", flush=True)
    print(f"[train] Iterations: {runner_cfg.max_iterations}", flush=True)
    print(f"[train] Device    : {runner_cfg.device}", flush=True)
    print(f"[train] Seed      : {args.seed}", flush=True)

    torch.manual_seed(args.seed)

    env_cfg = parse_env_cfg(args.task, device=runner_cfg.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed  # silences "Seed not set" and makes env resets reproducible
    # Replace the full tabletop scene USD (a whole stage with its own
    # PhysicsScene/cameras/lights — messy to spawn per-env) with a simple
    # ground plane so every cloned env has collision ground to stand on.
    # The plane is raised so its surface sits at TABLE_SURFACE_Z: the target
    # piece spawns at table height and must rest on collision geometry there,
    # not fall 0.79 m to a floor (which would trip the drop termination).
    import isaaclab.sim as _sim_utils  # type: ignore[import-not-found]

    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z

    env_cfg.scene.tabletop.spawn = _sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)

    # ------------------------------------------------------------------
    # 5. Build the RSL-RL runner
    # ------------------------------------------------------------------
    if args.experiment:
        runner_cfg.experiment_name = args.experiment
    log_dir = _TRAIN_OUT_ROOT / runner_cfg.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = RslRlOnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=str(log_dir),
        device=runner_cfg.device,
    )

    # Resume from checkpoint if requested
    if args.checkpoint is not None and args.checkpoint.exists():
        print(f"[train] Resuming from checkpoint: {args.checkpoint}", flush=True)
        runner.load(str(args.checkpoint))
    elif args.resume:
        # Auto-detect the latest checkpoint for this experiment
        ckpts = sorted(log_dir.glob("model_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            print(f"[train] Resuming from latest checkpoint: {latest}", flush=True)
            runner.load(str(latest))
        else:
            print("[train] No existing checkpoint found — starting fresh.", flush=True)

    # ------------------------------------------------------------------
    # 6. Train (or pure-inference eval for GUI viewing)
    # ------------------------------------------------------------------
    if args.eval:
        print("[train] EVAL mode: pure inference, no training, no saves.", flush=True)
        policy = runner.get_inference_policy(device=runner_cfg.device)
        obs = env.get_observations()  # TensorDict (rsl-rl 3.x wrapper)
        while True:
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        return 0  # unreachable; loop runs until the app is closed

    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    elapsed_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 7. Save final checkpoint + results summary
    # ------------------------------------------------------------------
    final_ckpt = log_dir / "model_final.pt"
    runner.save(str(final_ckpt))
    print(f"\n[train] Final checkpoint: {final_ckpt}", flush=True)

    # Recover the final reward statistics from the TensorBoard events this run
    # already wrote. This used to read ``runner.logger`` — an attribute RSL-RL
    # does not have — under a bare ``except Exception: pass``, so every summary
    # written since said ``NaN`` and no one saw why. The statistics cannot be
    # read off the runner at all: ``learn()`` keeps rewbuffer/lenbuffer as
    # function locals. The event file is the authoritative record.
    results = {
        "task": args.task,
        "experiment_name": runner_cfg.experiment_name,
        "num_envs": args.num_envs,
        "max_iterations": runner_cfg.max_iterations,
        "device": runner_cfg.device,
        "seed": args.seed,
        "elapsed_s": round(elapsed_s, 1),
        "final_checkpoint": str(final_ckpt),
    }
    try:
        results.update(summarize_training_run.final_scalars(log_dir))
    except Exception as exc:  # noqa: BLE001 - summary must not fail the run
        # Loud, but never fatal: the checkpoint is the run's real output and a
        # bad summary must not cost hours of training.
        print(f"[train] WARNING: could not summarize from events: {exc}", flush=True)
        print(
            "[train] recover later with: python isaac/scripts/"
            f"summarize_training_run.py {log_dir} --write",
            flush=True,
        )

    results_path = log_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[train] Results: {results_path}", flush=True)

    # Symlink / copy to the canonical path used by CI evidence tests
    canonical = _TRAIN_OUT_ROOT / "synria_reach_policy.json"
    if not canonical.exists() or args.task.startswith("Synria-Chess"):
        canonical.write_text(json.dumps(results, indent=2))

    env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - hardware-gated entry point
    raise SystemExit(main())
