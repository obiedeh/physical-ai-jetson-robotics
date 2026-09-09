#!/usr/bin/env bash
# Quick smoke test: import the Synria task, create a tiny env (4 envs),
# reset it, step 10 times, print obs/action shapes, and exit.
#
# Validates the full Isaac Sim → Isaac Lab → Synria task → env stack
# without running a full training loop. Takes ~60 s on the RTX 5090.
#
# Usage:
#   bash scripts/linux_rtx/smoke_isaaclab_env.sh
#   bash scripts/linux_rtx/smoke_isaaclab_env.sh --task Synria-Ludo-PickPlace-v0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"
TASK="${1:-Synria-Chess-PickPlace-v0}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Sim Python not found: $ISAAC_PYTHON" >&2
  echo "Override: ISAAC_PYTHON=/path/to/python bash $0" >&2
  exit 1
fi

echo "========================================"
echo " Isaac Lab Env Smoke Test"
echo " Task  : $TASK"
echo " Envs  : 4 (smoke test)"
echo " Steps : 10"
echo "========================================"

"$ISAAC_PYTHON" - <<PYEOF
import sys
sys.path.insert(0, "$REPO_ROOT")

# Boot Isaac Sim
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

try:
    import torch
    import gymnasium as gym

    # Patch warp.launch to unwrap ProxyArray → warp.array for Warp 1.12.x compatibility.
    # Isaac Lab 3.0 (beta2) passes ProxyArray objects to wp.launch, but Warp 1.12.0 (bundled
    # with Isaac Sim 5.1) requires exact warp.array types for structured dtypes.
    import warp as wp
    _orig_wp_launch = wp.launch
    def _patched_wp_launch(kernel, dim, inputs=None, outputs=None, **kwargs):
        def _unwrap(args):
            if args is None:
                return args
            result = []
            for a in args:
                if hasattr(a, '_warp') and isinstance(a._warp, wp.array):
                    result.append(a._warp)
                else:
                    result.append(a)
            return result
        return _orig_wp_launch(kernel, dim, inputs=_unwrap(inputs), outputs=_unwrap(outputs), **kwargs)
    wp.launch = _patched_wp_launch

    # Register Synria tasks
    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401

    print(f"\n[smoke] Creating env: $TASK (num_envs=4)")
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg("$TASK", device="cuda:0", num_envs=4)
    # Remove the tabletop asset — scene USDs are built separately via
    # build_synria_*_scene.sh and are not required for the env stack smoke test.
    del env_cfg.scene.tabletop
    env = gym.make("$TASK", cfg=env_cfg)

    print(f"[smoke] Obs space  : {env.observation_space}")
    print(f"[smoke] Action space: {env.action_space}")

    obs, _ = env.reset()
    print(f"[smoke] obs shape after reset: {obs['policy'].shape}")

    for step in range(10):
        # Isaac Lab returns action_space.shape = (num_envs, action_dim);
        # take the trailing dim, not [0].
        action = torch.zeros(4, env.action_space.shape[-1])
        obs, rew, terminated, truncated, info = env.step(action)
        if (step + 1) % 5 == 0:
            print(f"[smoke] step {step+1:2d}  reward mean: {rew.mean():.4f}")

    env.close()
    print("\n[smoke] PASSED — env creates, resets, and steps cleanly.")

except Exception as exc:
    print(f"\n[smoke] FAILED: {exc}")
    import traceback; traceback.print_exc()
    raise SystemExit(1)

finally:
    app.close()
PYEOF

echo "========================================"
echo " Smoke test complete."
echo " Next: bash scripts/linux_rtx/train_isaaclab_synria.sh"
echo "========================================"
