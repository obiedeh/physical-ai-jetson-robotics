"""Closed-loop in-sim eval of a GR00T checkpoint on the Franka+2F-85 env.

The surrogate counterpart of eval_gr00t_sequence.py: the SAME env the
S1 expert ran in (FrankaCupEnvCfg — ground cup, overhead+wrist cameras,
8-dim actions), but every action comes from a GR00T policy server over
ZeroMQ. No stage machine, no scripted help: the policy sees images +
joint state and must produce the whole behavior.

Server (gr00t venv):
    HF_TOKEN=$(cat ~/.cache/huggingface/token) ~/.venv/gr00t/bin/python \
      gr00t/eval/run_gr00t_server.py --model-path <checkpoint> \
      --embodiment-tag new_embodiment --port 5591

Client (this script, Isaac venv):
    ~/.venv/isaacsim5/bin/python isaac/scripts/eval_gr00t_franka.py \
      --headless --port 5591 [--num_envs 4] [--steps 7200]

Metrics per fixed 900-step episode window (mirrors demo length):
    PICK  — cup center exceeded z 0.10 m during the window
    PLACE — after a pick, cup back on the ground (z < 0.05) displaced
            >= 5 cm from its spawn xy at window end
Cup respawns randomized exactly like the expert's recycler.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

INSTRUCTION = "pick up the cup, carry it, and set it back down where it started"
EXEC_HORIZON = 8
EP_STEPS = 900


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
    from isaaclab.app import AppLauncher  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5591)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--snap_dir", type=str, default="",
                        help="dump env0 overhead frames every 30 steps")
    parser.add_argument("--fixed_spawn", action="store_true",
                        help="always respawn the cup at (0.5, 0) — control runs")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        return _run(args)
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        return 1
    finally:
        app.close()


def _run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch  # type: ignore[import-not-found]

    from isaaclab.envs import ManagerBasedEnv  # type: ignore[import-not-found]
    from isaac.isaaclab_tasks.synria_pickplace.franka_cup_env import (
        FrankaCupEnvCfg,
    )

    torch.manual_seed(args.seed)
    cfg = FrankaCupEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedEnv(cfg)
    env.reset()
    robot = env.scene["robot"]
    cup = env.scene["cup"]
    n = env.scene.num_envs
    dev = env.device
    origins = env.scene.env_origins

    fj = robot.joint_names.index("finger_joint")
    arm_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    state_ids = arm_ids + [fj]
    lo = robot.data.joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.joint_pos_limits[:, arm_ids, 1]
    default_arm = robot.data.default_joint_pos[:, arm_ids]

    # settle
    zero = torch.zeros(n, 8, device=dev)
    for _ in range(120):
        env.step(zero)

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

    def respawn(ids: torch.Tensor) -> torch.Tensor:
        root = cup.data.default_root_state.clone()
        rand = (torch.rand(n, 2, device=dev) - 0.5) * 0.16
        if args.fixed_spawn:
            rand[:] = 0.0
        root[:, 0] = 0.5 + rand[:, 0]
        root[:, 1] = rand[:, 1]
        root[:, 2] = 0.021
        root[:, :3] += origins
        root[:, 7:] = 0.0
        cup.write_root_pose_to_sim(root[ids][:, :7], env_ids=ids)
        cup.write_root_velocity_to_sim(root[ids][:, 7:], env_ids=ids)
        # rehome the arm so each window starts from the boot pose
        jp = robot.data.default_joint_pos[ids]
        robot.write_joint_state_to_sim(
            jp, torch.zeros_like(jp), env_ids=ids
        )
        return root[:, :2] - origins[:, :2]

    all_ids = torch.arange(n, device=dev)
    spawn_xy = respawn(all_ids)
    for _ in range(60):
        env.step(zero)

    picks = places = episodes = 0
    lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    ep_t = 0
    chunk = None
    chunk_i = 0

    for step in range(args.steps):
        if chunk is None or chunk_i >= EXEC_HORIZON:
            js = robot.data.joint_pos
            wrist_f, over_f = frames()
            obs = {
                "video": {
                    "wrist": wrist_f[:, None],
                    "overhead": over_f[:, None],
                },
                "state": {
                    "arm": js[:, arm_ids].cpu().numpy().astype(np.float32)[:, None],
                    "gripper": js[:, [fj]].cpu().numpy().astype(np.float32)[:, None],
                },
                "language": {
                    "annotation.human.task_description": [[INSTRUCTION]] * n,
                },
            }
            resp = client.get_action(obs)
            if isinstance(resp, (list, tuple)):
                resp = resp[0]
            arm_chunk = np.asarray(resp["arm"], dtype=np.float32)
            grip_chunk = np.asarray(resp["gripper"], dtype=np.float32)
            if arm_chunk.ndim == 2:
                arm_chunk, grip_chunk = arm_chunk[None], grip_chunk[None]
            if step == 0:
                print(f"[eval] chunk shapes: arm={arm_chunk.shape} "
                      f"gripper={grip_chunk.shape}", flush=True)
            chunk = np.concatenate([arm_chunk, grip_chunk], axis=-1)
            chunk_i = 0

        act = torch.from_numpy(
            chunk[:, min(chunk_i, chunk.shape[1] - 1)]
        ).to(dev)
        chunk_i += 1
        # safety-clip: arm targets inside joint limits, finger 0..0.79
        tgt = (default_arm + act[:, :7]).clamp(lo, hi)
        act8 = torch.cat(
            [tgt - default_arm, act[:, 7:8].clamp(0.0, 0.79)], dim=1
        )
        env.step(act8)

        cup_pos = cup.data.root_pos_w - origins
        lifted |= cup_pos[:, 2] > 0.10
        ep_t += 1
        if ep_t >= EP_STEPS:
            on_ground = cup_pos[:, 2] < 0.05
            displaced = (cup_pos[:, :2] - spawn_xy).norm(dim=1) > 0.05
            placed = lifted & on_ground & displaced
            picks += int(lifted.sum())
            places += int(placed.sum())
            episodes += n
            print(f"[eval] window done: {int(lifted.sum())} picks "
                  f"{int(placed.sum())} places | totals {picks}/{places} "
                  f"over {episodes} episodes", flush=True)
            lifted[:] = False
            spawn_xy = respawn(all_ids)
            for _ in range(60):
                env.step(zero)
            ep_t = 0

        if step % 600 == 0:
            hand_idx = robot.body_names.index("panda_hand")
            hand0 = (robot.data.body_pos_w[0, hand_idx] - origins[0]).tolist()
            print(f"[eval] step={step} cup_z0={float(cup_pos[0,2]):.3f} "
                  f"hand0={[round(v,3) for v in hand0]} "
                  f"fj0={float(robot.data.joint_pos[0, fj]):.3f} "
                  f"picks={picks} places={places}", flush=True)
        if args.snap_dir and step % 30 == 0:
            import cv2

            rgb = env.scene.sensors["overhead"].data.output["rgb"][0]
            rgb = rgb.detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(f"{args.snap_dir}/{step:05d}_overhead.png",
                        cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR))

    print(f"[eval] ======== RESULT: {picks} picks, {places} places "
          f"over {episodes} episodes ========", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
