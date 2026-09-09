"""Serve a trained lerobot ACT policy over localhost — the project's
policy-server pattern (same architecture as the GR00T port-5594 setup).

Runs in the LEROBOT venv (the Isaac venv must never import lerobot — the
operating notes forbid installs there, and the two torches must not meet):

    cd ~ && HF_HUB_OFFLINE=1 ~/.venv/lerobot/bin/python \
        ~/github/physical-ai-jetson-robotics/isaac/scripts/serve_act_policy.py \
        --checkpoint <run>/checkpoints/020000/pretrained_model [--port 5599]

Protocol (pickle over TCP, localhost only, one client):
    request : {"state": (8,) float32, "wrist": (H,W,3) uint8,
               "overhead": (H,W,3) uint8, "reset": bool, "task": str}
    response: {"action": (8,) float32}   # same convention as the corpus:
              6 arm offsets from default + 2 finger-target offsets

`reset` clears the policy's internal action-chunk queue — REQUIRED at every
episode boundary or temporal ensembling blends actions across episodes.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct


def recv_msg(conn: socket.socket) -> dict | None:
    hdr = conn.recv(8, socket.MSG_WAITALL)
    if len(hdr) < 8:
        return None
    (size,) = struct.unpack("<Q", hdr)
    buf = conn.recv(size, socket.MSG_WAITALL)
    if len(buf) < size:
        return None
    return pickle.loads(buf)


def send_msg(conn: socket.socket, obj: dict) -> None:
    buf = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("<Q", len(buf)) + buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to .../checkpoints/<step>/pretrained_model")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="0.0.0.0 to serve across the LAN — e.g. policy "
                         "served from AGX Thor, sim eval on the RTX box")
    args = ap.parse_args()

    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.eval().cuda()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    print(f"[serve] ACT loaded from {args.checkpoint}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"[serve] listening on {args.bind}:{args.port}", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[serve] client connected", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                if req.get("reset"):
                    policy.reset()
                obs = {
                    "observation.state": torch.from_numpy(
                        req["state"]).float().unsqueeze(0),
                    "observation.images.wrist": torch.from_numpy(
                        req["wrist"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "observation.images.overhead": torch.from_numpy(
                        req["overhead"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    "task": [req.get("task", "")],
                }
                with torch.no_grad():
                    batch = preprocessor(obs)
                    act = policy.select_action(batch)
                    act = postprocessor(act)
                # tolist(), not an ndarray: the Isaac venv runs numpy 1.x
                # and cannot unpickle numpy 2.x arrays (numpy._core)
                send_msg(conn, {"action": act.squeeze(0).cpu().numpy().tolist()})
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print("[serve] client disconnected", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
