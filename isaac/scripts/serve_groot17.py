#!/usr/bin/env python3
"""GR00T-1.7 (LeRobot GrootPolicy) policy server, wire-compatible with the
campaign's ZMQ protocol — every existing eval client (eval_gr00t_sequence,
eval_gr00t_system, funnel clients) works unmodified.

Run from OUTSIDE the repo (the repo-local lerobot/ package shadows the
library) in the lerobot17 venv:

    cd /tmp && env -u HF_TOKEN \
      LD_LIBRARY_PATH=$HOME/.venv/lerobot17/lib/python3.12/site-packages/nvidia/cu13/lib \
      $HOME/.venv/lerobot17/bin/python \
      $HOME/github/physical-ai-jetson-robotics/isaac/scripts/serve_groot17.py \
      --checkpoint $HOME/github/physical-ai-jetson-robotics/reports/training/ludo_groot17_v1/checkpoints/last/pretrained_model \
      --port 5591

Wire contract (matches gr00t/eval/run_gr00t_server.py):
  request : msgpack {"endpoint": "get_action", "data": {"observation": {
              "video": {"wrist": (B,1,H,W,3) u8, "overhead": ...},
              "state": {"arm": (B,1,6) f32, "gripper": (B,1,2) f32},
              "language": {"annotation.human.task_description": [[str]]*B}}}}
  response: msgpack [ {"arm": (B,T,6) f32, "gripper": (B,T,2) f32}, {} ]
Actions are the corpus convention: 6 arm offsets from default + 2 finger
target offsets from finger defaults (recorder-verbatim contract).
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=5591)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import msgpack
    import msgpack_numpy as mnp
    import numpy as np
    import torch
    import zmq

    from lerobot.policies.groot.modeling_groot import GrootPolicy

    from pathlib import Path
    ckpt = Path(args.checkpoint)
    if (ckpt / "adapter_config.json").exists():
        # LoRA checkpoint: load the base recorded in the adapter config,
        # apply the adapter, merge for serving speed.
        import json as _json
        base = _json.loads((ckpt / "adapter_config.json").read_text())[
            "base_model_name_or_path"]
        print(f"[serve17] LoRA adapter; loading base {base}", flush=True)
        policy = GrootPolicy.from_pretrained(base)
        from peft import PeftModel
        policy = PeftModel.from_pretrained(policy, str(ckpt))
        policy = policy.merge_and_unload()
        print("[serve17] adapter merged", flush=True)
    else:
        policy = GrootPolicy.from_pretrained(args.checkpoint)
    policy.to(args.device)
    policy.eval()
    # the checkpoint's own processor pipelines (tokenization, input
    # packing, action unpack/unnormalize) — required; raw batches fail
    # with 'input_ids'
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt))
    print(f"[serve17] policy + processors ready from {args.checkpoint}",
          flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[serve17] listening on :{args.port}", flush=True)

    def to_batch(obs: dict) -> dict:
        # (B,1,H,W,3) u8 -> (B,3,H,W) f32 in [0,1]
        def img(key):
            v = np.asarray(obs["video"][key])[:, 0]
            t = torch.from_numpy(v).to(args.device).float() / 255.0
            return t.permute(0, 3, 1, 2)

        arm = np.asarray(obs["state"]["arm"], dtype=np.float32)[:, 0]
        grip = np.asarray(obs["state"]["gripper"], dtype=np.float32)[:, 0]
        state = torch.from_numpy(np.concatenate([arm, grip], -1)).to(args.device)
        lang = obs["language"]["annotation.human.task_description"]
        tasks = [l[0] if isinstance(l, (list, tuple)) else l for l in lang]
        return {
            "observation.images.wrist": img("wrist"),
            "observation.images.overhead": img("overhead"),
            "observation.state": state,
            "task": tasks,
        }

    while True:
        raw = sock.recv()
        try:
            req = msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)
            obs = req["data"]["observation"]
            batch = to_batch(obs)
            with torch.inference_mode():
                batch = pre(batch)
                chunk = policy.predict_action_chunk(batch)  # (B,T,A)
                chunk = post(chunk)
            a = chunk.float().cpu().numpy()
            resp = [{"arm": a[..., :6], "gripper": a[..., 6:8]}, {}]
            sock.send(msgpack.packb(resp, default=mnp.encode))
        except Exception as e:  # keep serving; client raises on error dict
            print(f"[serve17] ERROR: {e}", flush=True)
            try:
                sock.send(msgpack.packb({"error": str(e)}, default=mnp.encode))
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
