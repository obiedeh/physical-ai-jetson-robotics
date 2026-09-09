# LeRobot Evidence Reports

This directory holds demonstration dataset summaries and policy evaluation
evidence for the Synria 6DOF arm LeRobot / ALOHA-compatible track.

See ``docs/LEROBOT_ALOHA.md`` for the full workflow description.

## Report Types

| File pattern | Contents |
|---|---|
| `demo_dataset_summary.json` | CLI demo run — synthetic dataset summary (no per-step data) |
| `dataset_*.json` | Full episode datasets including per-step obs/action pairs |
| `policy_eval_*.json` | Policy evaluation results: tracking error, completion rate |

## Dataset Summary Schema

```json
{
  "n_episodes": 4,
  "total_steps": 482,
  "staging_zones": ["bottom", "left", "right", "top"],
  "games": ["chess"],
  "fps_values": [30.0],
  "episodes_meta": [
    {
      "episode_id": "uuid",
      "task_variant": "V1",
      "game": "chess",
      "staging_zone": "left",
      "robot_id": "synria-arm-01",
      "fps": 30.0,
      "n_steps": 120,
      "captured_at": "2026-05-27T10:00:00+00:00",
      "synthetic": true,
      "notes": "Generated from arm_control.demo ..."
    }
  ]
}
```

## Episode Schema (per-step)

Each `dataset_*.json` contains the full `steps` array:

```json
{
  "metadata": { ... },
  "steps": [
    {
      "step_index": 0,
      "observation": {
        "frame_index": 0,
        "timestamp_s": 0.0,
        "joint_state": {
          "positions_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          "velocities_rad_s": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          "gripper_open_m": 0.085,
          "timestamp_s": 0.0
        },
        "ee_pose": {
          "x_m": 0.0, "y_m": 0.0, "z_m": 0.415,
          "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
          "timestamp_s": 0.0
        }
      },
      "action": {
        "joint_deltas_rad": [0.001, 0.012, -0.008, 0.0, 0.0, 0.0],
        "gripper_command": 0.0,
        "timestamp_s": 0.0
      }
    }
  ]
}
```

## Policy Evaluation Schema

```json
{
  "episode_id": "uuid",
  "n_steps": 120,
  "mean_joint_tracking_error_rad": 0.0,
  "max_joint_tracking_error_rad": 0.0,
  "completion_rate": 1.0,
  "passed": true,
  "notes": "Evaluated with DeterministicArmPolicy on zone='left', game='chess'."
}
```

## Run CLI Demo

```bash
physical-ai-lab lerobot-dataset-demo
physical-ai-lab lerobot-policy-eval
```

## Observation / Action Contract

| Signal | Shape | Source |
|---|---|---|
| `joint_state.positions_rad` | (6,) | `/joint_states` topic |
| `joint_state.velocities_rad_s` | (6,) | `/joint_states` topic |
| `joint_state.gripper_open_m` | scalar | gripper feedback |
| `ee_pose` | (7,) pos+quat | TF `tool0` frame |
| `action.joint_deltas_rad` | (6,) | policy output |
| `action.gripper_command` | scalar ∈ [−1, +1] | policy output |

## Hardware Evidence Gate

Reports in this directory are **synthetic evidence** until the Synria arm
and C10 camera are live and real demonstration recordings are committed
alongside them.

LeRobot version: 0.5.1 (installed in `.venv`).
Real training command once data is collected:

```bash
lerobot-train \
  --dataset.repo_id obiedeh/synria-c10-aloha-demo \
  --env.type aloha \
  --policy.type act
```
