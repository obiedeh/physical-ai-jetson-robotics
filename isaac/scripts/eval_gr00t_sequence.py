"""Fixed-harness in-sim evaluation of a GR00T policy (task #5 comparison).

Runs the SAME task/scene/seed convention as eval_synria_sequence.py
(chess task id, ground-plane override, seed 123) with recorder cameras,
but actions come from a GR00T policy server (gr00t venv) over ZeroMQ —
the Isaac venv needs only pyzmq + msgpack-numpy (installed --no-deps).

Server (gr00t venv):
    HF_TOKEN=$(cat ~/.cache/huggingface/token) ~/.venv/gr00t/bin/python \
      gr00t/eval/run_gr00t_server.py --model-path <checkpoint> \
      --embodiment-tag new_embodiment --modality-config-path <modality_config.py> \
      --port 5591

Client (this script, Isaac venv):
    ~/.venv/isaacsim5/bin/python isaac/scripts/eval_gr00t_sequence.py \
      --headless --port 5591 [--num_envs 16] [--steps 3600] [--seed 123]

Metrics: trial-manager funnel (grasp, carry5s, dock/set-down, full
cycles) totaled over the run and normalized per episode-equivalent —
directly comparable to the RL ledger's stage metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

INSTRUCTION = "pick up the cup, carry it to the plate, and set it down gently"
ACTION_HORIZON = 16
EXEC_HORIZON = 8  # steps of each chunk to execute before re-planning


class MiniGrootClient:
    """Minimal PolicyClient: zmq REQ + msgpack_numpy, no gr00t import."""

    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        import msgpack
        import msgpack_numpy as mnp
        import zmq

        self._msgpack = msgpack
        self._mnp = mnp
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def get_action(self, observation: dict) -> dict:
        # The server calls handler(**data): wrap as the 'observation' kwarg.
        req = self._msgpack.packb(
            {"endpoint": "get_action", "data": {"observation": observation}},
            default=self._mnp.encode,
        )
        self.socket.send(req)
        msg = self.socket.recv()
        if msg == b"ERROR":
            raise RuntimeError("policy server error")
        resp = self._msgpack.unpackb(msg, object_hook=self._mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server: {resp['error']}")
        return resp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5591)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--instruction",
        type=str,
        default=INSTRUCTION,
        help="Language conditioning string; MUST be the evaluated model's "
        "training instruction (paraphrase scored exactly zero in E41).",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Enable ZONE surface-pick staging (the demo distribution) instead "
        "of scratch starts — separates distribution-mismatch from can't-do-task.",
    )
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run(args)
    except BaseException:
        # Isaac's app.close() may os._exit(0), eating the traceback and
        # faking success — print it BEFORE shutdown.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import gymnasium as gym  # type: ignore[import-not-found]
    import isaaclab.sim as sim_utils  # type: ignore[import-not-found]
    import numpy as np
    import torch  # type: ignore[import-not-found]
    from isaaclab.sim import SimulationContext  # type: ignore[import-not-found]
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore[import-not-found]

    SimulationContext._app_control_on_stop_handle_fn = lambda self, event: None  # type: ignore[assignment]

    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401
    from isaac.isaaclab_tasks.synria_pickplace import mdp
    from isaac.isaaclab_tasks.synria_pickplace.recorder_env_cfg import (
        add_recorder_cameras,
    )
    from isaac.isaaclab_tasks.synria_pickplace.task_geometry import TABLE_SURFACE_Z

    # Scratch starts by default (fixed harness); --staged matches the demo
    # distribution (80% zone surface-pick starts).
    mdp.PREGRASP_FRACTION = 0.0
    mdp.ZONE_START_FRACTION = 0.8 if args.staged else 0.0
    mdp.CARRY_ELAPSED_FRACTION = 0.0
    if hasattr(mdp, "RETURN_START_FRACTION"):
        mdp.RETURN_START_FRACTION = 0.0

    env_cfg = parse_env_cfg(
        "Synria-Chess-PickPlace-v0", device="cuda", num_envs=args.num_envs
    )
    env_cfg.seed = args.seed
    env_cfg.scene.tabletop.spawn = sim_utils.GroundPlaneCfg()
    env_cfg.scene.tabletop.prim_path = "/World/ground"
    env_cfg.scene.tabletop.init_state.pos = (0.0, 0.0, TABLE_SURFACE_Z)
    add_recorder_cameras(env_cfg)

    raw = gym.make("Synria-Chess-PickPlace-v0", cfg=env_cfg)
    env = raw.unwrapped
    env.reset()
    robot = env.scene["robot"]
    rids = mdp._robot_ids(env)
    arm_ids = list(rids["arm"])
    finger_ids = [int(rids["left"]), int(rids["right"])]
    n = env.num_envs
    default_arm = robot.data.default_joint_pos[:, arm_ids]
    default_l = robot.data.default_joint_pos[:, rids["left"]]
    default_r = robot.data.default_joint_pos[:, rids["right"]]
    mgr = mdp._get_trial_mgr(env)

    client = MiniGrootClient(args.host, args.port)
    print(f"[eval] connected to policy server {args.host}:{args.port}", flush=True)

    def frames() -> tuple[np.ndarray, np.ndarray]:
        outs = []
        for key in ("wrist", "overhead"):
            rgb = env.scene.sensors[key].data.output["rgb"].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            outs.append(rgb[..., :3])
        return outs[0], outs[1]

    prev_cycles = mgr.cycles.clone()
    totals = {"grasp": 0, "carry5s": 0, "dock": 0, "cycle": 0}
    prev_f = {
        "carry5s": mgr.f_carry5s.clone(),
        "dock": mgr.f_dock.clone(),
    }
    prev_phase = mgr.phase.clone()
    chunk = None
    chunk_i = 0

    for step in range(args.steps):
        if chunk is None or chunk_i >= EXEC_HORIZON:
            joint_state = robot.data.joint_pos
            state_arm = joint_state[:, arm_ids].cpu().numpy().astype(np.float32)
            state_grip = joint_state[:, finger_ids].cpu().numpy().astype(np.float32)
            wrist_f, over_f = frames()
            obs = {
                "video": {
                    "wrist": wrist_f[:, None],       # (B,1,H,W,3)
                    "overhead": over_f[:, None],
                },
                "state": {
                    "arm": state_arm[:, None],       # (B,1,6)
                    "gripper": state_grip[:, None],  # (B,1,2)
                },
                "language": {
                    "annotation.human.task_description": [[args.instruction]] * n,
                },
            }
            resp = client.get_action(obs)
            # get_action returns (action_dict, info_dict) — serialized as a
            # 2-list. Element 0 holds BATCHED arrays: arm (B,T,6), gripper
            # (B,T,2).
            if isinstance(resp, (list, tuple)):
                resp = resp[0]
            arm_chunk = np.asarray(resp["arm"], dtype=np.float32)
            grip_chunk = np.asarray(resp["gripper"], dtype=np.float32)
            if arm_chunk.ndim == 2:
                arm_chunk, grip_chunk = arm_chunk[None], grip_chunk[None]
            if step == 0:
                print(f"[eval] action chunk shapes: arm={arm_chunk.shape} "
                      f"gripper={grip_chunk.shape}", flush=True)
            chunk = np.concatenate([arm_chunk, grip_chunk], axis=-1)   # (B,T,8)
            chunk_i = 0

        act = torch.from_numpy(chunk[:, min(chunk_i, chunk.shape[1] - 1)]).to(env.device)
        chunk_i += 1
        # Predictions are offset-convention already; safety-clip like the env.
        tgt = (default_arm + act[:, :6]).clamp(
            torch.tensor([-2.749, -2.0, -0.5, -2.79, -1.57, -3.14159], device=env.device),
            torch.tensor([2.749, 2.0, 3.14159, 2.79, 1.57, 3.14159], device=env.device),
        )
        # Env action space is 7-dim since E30 (6 arm offsets + binary
        # gripper); map the model's finger-target prediction to the binary
        # command with the same threshold the ACT eval uses.
        l_target = (default_l + act[:, 6]).clamp(0.0, 0.025)
        act7 = torch.cat(
            [
                tgt - default_arm,
                ((l_target > 0.020).float() * 2.0 - 1.0).unsqueeze(-1),
            ],
            dim=1,
        )
        env.step(act7)

        totals["grasp"] += int(
            ((prev_phase == 0) & (mgr.phase == 1)).sum()
        )
        for k in ("carry5s", "dock"):
            f = getattr(mgr, f"f_{k}")
            totals[k] += int((f & ~prev_f[k]).sum())
            prev_f[k] = f.clone()
        totals["cycle"] += int(((mgr.cycles - prev_cycles) > 0).sum())
        prev_cycles = mgr.cycles.clone()
        prev_phase = mgr.phase.clone()

        if step % 300 == 0:
            print(f"[eval] step={step} totals={totals}", flush=True)

    ep_eq = n * args.steps / int(env.max_episode_length)
    print("[eval] ======== GR00T FIXED-HARNESS RESULT ========", flush=True)
    print(f"[eval] episode-equivalents: {ep_eq:.1f}", flush=True)
    for k, v in totals.items():
        print(f"[eval] {k}: {v} total, {v / ep_eq:.3f}/episode", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
