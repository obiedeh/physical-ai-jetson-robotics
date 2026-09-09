#!/usr/bin/env bash
# Quick smoke test for Isaac Sim 5.1 + Isaac Lab 2.3.x (stable path).
#
# Isaac Sim 6.0 is not yet production-ready.
# Use this script for the validated, stable Isaac Sim 5.1 stack.
#
# TWO-PHASE VALIDATION:
#   Phase 1: Isaac Lab env stack (Isaac-Cartpole-v0) — no robot USD required
#   Phase 2: Synria Cartpole task — requires import_synria_urdf.sh to have run
#
# NOTE: The existing synria_6dof_arm.usda has rigid-body xformstack issues
# from the old URDF converter. Run import_synria_urdf.sh on the RTX workstation
# to regenerate a correct USD before Phase 2 works.
#
# Usage:
#   bash scripts/linux_rtx/smoke_isaaclab_env_isaacsim5.sh
#   PHASE=2 bash scripts/linux_rtx/smoke_isaaclab_env_isaacsim5.sh  # Synria task (after URDF import)

set -euo pipefail

export ACCEPT_EULA=Y

# Set Isaac Sim environment variables for proper extension + asset path resolution.
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${HOME}/isaacsim}"
export CARB_APP_PATH="$ISAACSIM_ROOT/kit"
export ISAAC_PATH="$ISAACSIM_ROOT"
export EXP_PATH="$ISAACSIM_ROOT/apps"
export LD_PRELOAD="$ISAACSIM_ROOT/kit/libcarb.so"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim5/bin/python}"
PHASE="${PHASE:-1}"
TASK="${1:-Synria-Chess-PickPlace-v0}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Sim 5.1 Python not found: $ISAAC_PYTHON" >&2
  echo "Install: pip install isaacsim==5.1.0 --extra-index-url https://pypi.nvidia.com" >&2
  echo "Override: ISAAC_PYTHON=/path/to/python bash $0" >&2
  exit 1
fi

echo "========================================"
echo " Isaac Sim 5.1 + Isaac Lab 2.3.x Smoke Test"
echo " Phase: $PHASE"
echo " Task : $TASK"
echo " Envs : 4"
echo " Steps: 10"
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
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # Register Synria tasks (Phase 2 requires import_synria_urdf.sh to have run)
    import isaac.isaaclab_tasks.synria_pickplace  # noqa: F401

    task = "$TASK"
    import sys
    print(f"\n[smoke] Creating env: {task} (num_envs=4)", flush=True)
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
    del env_cfg.scene.tabletop

    env_raw = gym.make(task, cfg=env_cfg).unwrapped
    env = RslRlVecEnvWrapper(env_raw)

    print(f"[smoke] Obs space  : {env.observation_space}", flush=True)
    print(f"[smoke] Action space: {env.action_space}", flush=True)

    obs, _ = env.reset()
    print(f"[smoke] obs shape after reset: {obs.shape}", flush=True)

    num_envs = obs.shape[0]
    action_dim = env.unwrapped.action_manager.total_action_dim
    for step in range(10):
        action = torch.zeros(num_envs, action_dim)
        obs, rew, dones, info = env.step(action)
        if (step + 1) % 5 == 0:
            print(f"[smoke] step {step+1:2d}  reward mean: {rew.mean():.4f}", flush=True)

    env.close()
    sys.stdout.flush()
    print(f"\n[smoke] PASSED — {task} env creates, resets, and steps cleanly.", flush=True)

except Exception as exc:
    print(f"\n[smoke] FAILED: {exc}", flush=True)
    import traceback; traceback.print_exc()
    raise SystemExit(1)

finally:
    app.close()
PYEOF

echo "========================================"
echo " Smoke test complete."
echo " NOTE: If robot USD fails, run import_synria_urdf.sh first."
echo " Next: ISAAC_PYTHON=~/.venv/isaacsim5/bin/python bash scripts/linux_rtx/train_isaaclab_synria.sh"
echo "========================================"
