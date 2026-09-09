# Synria Pick-and-Place Isaac Lab Task

Isaac Lab RL environment for the Synria board ↔ staging pick-and-place V1
task across three game scenes (Ludo, chess, checkers).

## Registered task IDs

| Gym ID | Scene |
|---|---|
| `Synria-Ludo-PickPlace-v0` | `isaac/usd/scenes/synria_ludo/synria_ludo_v0.usda` |
| `Synria-Chess-PickPlace-v0` | `isaac/usd/scenes/synria_chess/synria_chess_v0.usda` |
| `Synria-Checkers-PickPlace-v0` | `isaac/usd/scenes/synria_checkers/synria_checkers_v0.usda` |

All three share the same env structure; only the scene USD and per-game
initial-state seeding differ. For the full task specification see
[`docs/SYNRIA_PICK_AND_PLACE_TASK.md`](../../../docs/SYNRIA_PICK_AND_PLACE_TASK.md).

---

## V1 Task

```
PICK_FROM_BOARD → PLACE_ON_ZONE → PICK_FROM_ZONE → RETURN_TO_BOARD → SUCCESS
                                                                    ↓
                                                                  FAILED
```

1. Pick the target piece off the board.
2. Carry it to one of the four staging zones (left / right / top / bottom).
3. Pick it back up from the zone.
4. Return it to its original board square.

---

## File layout

```
isaac/isaaclab_tasks/synria_pickplace/
├── __init__.py        # gymnasium.register() for the three task IDs
├── env_cfg.py         # ManagerBasedRLEnvCfg + per-game subclasses,
│                      #   actuator limits, ActionsCfg
├── mdp.py             # Isaac Lab-compatible term functions
│                      #   (obs / reward / termination / action)
├── mdp_logic.py       # Pure-Python MDP kernels — no torch, no GPU,
│                      #   fully unit-testable on any machine
├── task_geometry.py   # Scene geometry constants and board helpers
│                      #   (zone centres, piece positions, tolerances)
├── trial_state.py     # Per-env trial state machine (TaskPhase enum,
│                      #   TrialStateManager, TrialInfo)
├── gr00t/             # NVIDIA Isaac GR00T integration (see below)
│   ├── __init__.py
│   ├── embodiment.py  #   joint names, action dim, modality config
│   ├── dataset.py     #   LeRobot dataset adapter + path conventions
│   ├── finetune.py    #   CLI wrapper — python -m ...gr00t.finetune
│   ├── inference.py   #   rollout eval — python -m ...gr00t.inference
│   └── README.md
└── README.md          # this file
```

---

## Architecture

The MDP is split into two layers so that the core logic is testable
without a GPU:

```
mdp.py (Isaac Lab wrappers)
  │  reads tensors from env.scene["robot"] and env.extras
  │  calls ↓
mdp_logic.py (pure-Python kernels)
  │  imports constants from ↓
task_geometry.py (geometry constants)

trial_state.py (TrialStateManager — lives in env.extras["trial_state"])
  │  queried every step by mdp.py wrappers
```

### Observation space (57 floats, concatenated)

| Term | Shape | Description |
|---|---|---|
| `joint_state_synria` | (13,) | 6 joint pos + 6 joint vel + gripper width (m) |
| `ee_pose_synria` | (7,) | tool0 XYZ + quaternion wxyz |
| `target_piece_id` | (32,) | one-hot: which chess/checkers/ludo piece |
| `target_zone_id` | (4,) | one-hot: left / right / top / bottom zone |

### Action space (8 floats)

| Indices | Description |
|---|---|
| [0–5] | Absolute joint position targets (rad), clipped to URDF limits |
| [6] | Left-finger target position (0 = closed, 0.025 m = open) |
| [7] | Right-finger target position (0 = closed, −0.025 m = open) |

### Reward structure

| Term | Weight | Type |
|---|---|---|
| `ee_to_target_distance` | −0.5 | Dense — negative L2 to phase-dependent target |
| `gripper_to_piece_when_grasping` | −0.3 | Dense — penalty when fingers close far from piece |
| `action_norm_penalty` | −0.01 | Dense — discourages large joint jumps |
| `grasp_confirmed` | +1.0 | Sparse — fires once on confirmation (3 consecutive steps) |
| `piece_in_zone` | +5.0 | Sparse — piece centred on zone (±10 mm) |
| `piece_returned` | +5.0 | Sparse — piece back at origin (±10 mm) |
| `piece_dropped` | −2.0 | Sparse — piece below table surface |
| `arm_collision` | −5.0 | Sparse — contact sensor detects unsafe collision |

### Termination conditions

| Term | Condition |
|---|---|
| `task_success_v1` | Phase reached SUCCESS (piece returned to origin) |
| `time_out` | Episode length ≥ `episode_length_s / dt` steps |
| `piece_dropped_below_table_term` | Piece Z < TABLE_SURFACE_Z − 0.05 m |
| `arm_collision_term` | Contact sensor force > 0.1 N on non-target links |

### Actuator limits (from `synria_6dof_arm.urdf`)

| Joint | Type | Range (rad / m) | Effort | Velocity | PD gains |
|---|---|---|---|---|---|
| Joint1 | revolute | ±2.749 rad | 5 Nm | 12 rad/s | k=400, d=40 |
| Joint2 | revolute | ±2.000 rad | 5 Nm | 12 rad/s | k=400, d=40 |
| Joint3 | revolute | −0.5 … +π | 5 Nm | 12 rad/s | k=400, d=40 |
| Joint4 | revolute | ±2.790 rad | 5 Nm | 12 rad/s | k=400, d=40 |
| Joint5 | revolute | ±1.570 rad | 5 Nm | 12 rad/s | k=400, d=40 |
| Joint6 | revolute | ±π rad | 5 Nm | 12 rad/s | k=400, d=40 |
| left_finger | prismatic | 0 … +0.025 m | 5 N | 0.05 m/s | k=2000, d=100 |
| right_finger | prismatic | −0.025 … 0 m | 5 N | 0.05 m/s | k=2000, d=100 |

---

## Key geometry constants (task_geometry.py)

```python
TABLE_SURFACE_Z       = 0.79 m   # TABLE_HEIGHT(0.75) + TABLE_THICKNESS(0.04)
ARM_BASE_X            = −0.50 m
BOARD_CENTER_X        = 0.10 m
BOARD_SIZE            = 0.44 m   # all three games
STAGING_ZONE_OFFSET   = 0.06 m   # inner edge distance from board edge
STAGING_ZONE_DEPTH    = 0.10 m
GRIPPER_OPEN_M        = 0.05 m   # URDF prismatic travel, 25 mm per finger
GRIPPER_GRASP_WIDTH_M = 0.045 m  # upper edge of the holding band
GRASP_APPROACH_RADIUS = 0.15 m   # "piece in gripper" distance, from PAD CENTRE
GRASP_CONFIRM_STEPS   = 1        # 1, not 3: airborne criteria are unfakeable
                                 # in one step, and a 3-step window lost a
                                 # race at curriculum resets
```

These are checked against `task_geometry.py` by
`tests/test_gripper_geometry.py::test_readme_constants_match_source`. Every
number above was stale at some point — `GRIPPER_OPEN_M` carried 0.085 from a
servo datasheet that does not describe this arm, and the approach radius was
listed as 0.020 while the code used 0.15 — so the table is now enforced rather
than maintained by hand.

The grasp reference point is the **pad centre**, not the wrist flange and not
the finger link frames. See `gripper_geometry.py` for why that distinction cost
this project three separate multi-day debugging rounds.

Zone centre calculation:
```
edge_clearance = board_size/2 + STAGING_ZONE_OFFSET + STAGING_ZONE_DEPTH/2
left:   (BOARD_CENTER_X,                BOARD_CENTER_Y − edge_clearance)
right:  (BOARD_CENTER_X,                BOARD_CENTER_Y + edge_clearance)
top:    (BOARD_CENTER_X − edge_clearance, BOARD_CENTER_Y)
bottom: (BOARD_CENTER_X + edge_clearance, BOARD_CENTER_Y)
```

---

## Per-env trial state (trial_state.py)

```python
mgr = TrialStateManager(num_envs=4096, game="chess", seed=0)

# Called by the env's __post_init__:
env.extras["trial_state"] = mgr

# Every step, mdp.py queries:
mgr.current_target_xy(env_id)   # phase-dependent EE target XY
mgr.get_phase(env_id)           # TaskPhase enum value

# On phase completion, mdp.py calls:
mgr.advance_phase(env_id)       # PICK_FROM_BOARD → PLACE_ON_ZONE → …
mgr.mark_failed(env_id)         # on drop or collision
```

---

## GR00T integration (`gr00t/`)

NVIDIA Isaac GR00T N1.7 fine-tuning scaffold. Follows NVIDIA's published
[LeRobot SO-101 post-training recipe](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning)
with Synria-specific embodiment values.

| Module | Purpose |
|---|---|
| `embodiment.py` | `SYNRIA_EMBODIMENT_TAG = "synria_6dof_v0"`, joint names, 8-DOF action dim, camera config, `synria_modality_config()` |
| `dataset.py` | `lerobot_dataset(game, amplified)` — wraps GR00T's `LeRobotSingleDataset`; defines seed + GR00T-Mimic output paths |
| `finetune.py` | `python -m ...gr00t.finetune --game chess --amplified` — wraps `gr00t/experiment/launch_finetune.py` |
| `inference.py` | `python -m ...gr00t.inference --task Synria-Chess-PickPlace-v0 --checkpoint …` — rolls out inside Isaac Lab |

Two integration paths:
- **Path A** — fine-tune GR00T N1.7 directly on ~100 Synria seed demonstrations.
- **Path C** — amplify seed → 100K+ synthetic trajectories via GR00T-Mimic, then fine-tune on the amplified set.

Full recipe: [`docs/SYNRIA_GR00T_FINETUNE.md`](../../../docs/SYNRIA_GR00T_FINETUNE.md)

Quick CLI inspection (no GPU required):

```bash
physical-ai-lab gr00t-embodiment        # show embodiment config table
physical-ai-lab gr00t-finetune-dry-run --game chess   # print launch command
```

---

## Run on the RTX 5090

```bash
# 1. Activate Isaac Lab's Python environment
source ~/.venv/isaacsim5/bin/activate

# 2. Build the scene USD if not already committed
bash scripts/linux_rtx/build_synria_chess_scene.sh

# 3. Launch PPO training (Isaac Lab entry point)
python -m isaaclab.scripts.train \
    --task Synria-Chess-PickPlace-v0 \
    --headless \
    --num_envs 4096

# 4. Evaluate a checkpoint
python -m isaaclab.scripts.play \
    --task Synria-Chess-PickPlace-v0 \
    --checkpoint path/to/model.pt
```

Before first run, verify on the RTX box:
1. **Confirm joint names** against the loaded URDF — the config assumes `Joint1`..`Joint6` (capital J, matching the SolidWorks export) and `left_finger`/`right_finger` for the gripper.
2. **Confirm body name** — the EE pose observation looks up `tool0`; check that name exists in the loaded USD.
3. **Wire `piece_pos_w`** — the env's `__post_init__` must populate `env.extras["piece_pos_w"]` at reset and update it each step from the scene's tracked rigid-body.
4. **Add cameras** — `OverheadBoardCamera` and wrist RGB observations are not yet wired in `ObservationsCfg`. Add `Camera` sensor configs to the scene and new `ObsTerm` entries when available.

---

## Testing (no GPU required)

The pure-Python layer (`task_geometry.py`, `trial_state.py`, `mdp_logic.py`) is fully unit-tested on any machine:

```bash
pytest tests/test_isaaclab_mdp.py -v           # 98 tests
pytest tests/test_isaaclab_tasks_import.py -v  # 5 tests
```

---

## Windows / CI behaviour

The package is import-safe on any environment without Isaac Lab:

- `import isaac.isaaclab_tasks.synria_pickplace` — always succeeds
- Gym task IDs are NOT registered (no `gymnasium` installed in CI)
- Instantiating any config class raises `RuntimeError("Isaac Lab is not available")`
- All pure-Python helpers import and run normally on Windows, CI, and Jetson

---

## What's out of scope until RTX first-run

- **Training runs** — needs RTX 5090 and PPO convergence (~1–2 days compute)
- **Imitation-learning data** — handled by the LeRobot track (`lerobot/`)
- **Camera observations** — needs `Camera` sensor config wired to the scene USD
- **Sim-to-real transfer** — post-training, via Jetson AGX Thor deployment
- **PPO hyperparameter sweep** — after first successful sim episode
