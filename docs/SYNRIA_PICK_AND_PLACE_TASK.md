# Synria Primary Training Task: Board ↔ Staging Pick-and-Place

Status correction, 2026-09-07: the task outline below is the initial design,
not a statement of completed physical capability. MDP terms and simulation
execution are implemented; current behavior lives in the task code and ledgers.
Vendor-dependent runs now require [local asset setup](VENDOR_INTEGRATION_MAP.md).

The first training task shared across all three Synria tabletop scenes (Ludo, chess, checkers) is **board-to-staging pick-and-place and return**: the arm picks a game piece from its position on the board, moves it to one of four staging zones flanking the board, then returns it to the board.

This task is the foundation skill the policy must master before any game-specific behavior (legal-move enforcement, chess engine integration, Ludo turn order) is layered on.

## Staging zones

Four rectangular pads flank every board on all four sides, sized to hold one row of pieces from the largest game (chess back rank). Pad positions are returned by the `add_staging_zones(stage, board_size=...)` helper in [`isaac/scripts/_synria_scene_common.py`](../isaac/scripts/_synria_scene_common.py) and are color-coded for overhead-camera identification:

| Zone | World direction (relative to board center) | Color | Pad size | Notes |
|---|---|---|---|---|
| `left` | −Y (operator camera's left) | blue | board edge × 0.10 m | side-pick |
| `right` | +Y (operator camera's right) | orange | board edge × 0.10 m | side-pick |
| `top` | −X (toward the arm base) | purple | 0.10 m × board edge | shortest motion from board to staging — closest to arm base |
| `bottom` | +X (away from arm, toward operator) | green | 0.10 m × board edge | longest motion from board to staging — far edge of arm reach |

Pads sit on the table surface at 1.2 mm thickness so a token / piece / disc placed on the pad rests on the colored area rather than the bare table. Having all four sides gives the policy a wider range of motion distances and approach angles to learn against.

## Task variants

The same scene supports several task shapes; pick one per training run.

### V1 — single-piece pick-and-place

- **Initial state**: all pieces in their game's opening position on the board
- **Trial**: a target piece and a target zone (`left` / `right` / `top` / `bottom`) are randomly sampled
- **Step 1**: pick the target piece up off the board
- **Step 2**: place it inside the target zone
- **Step 3**: pick it back up
- **Step 4**: return it to its original square
- **Success**: piece centered within ±0.01 m of original square center and standing upright
- **Failure**: piece dropped outside any pad / outside board, gripper collision with another piece, arm collision with table, episode timeout

### V2 — multi-piece sequence

Same as V1 but with N successive pick-and-place targets in one episode. Reward shaped to favor minimum-time clear-and-restore.

### V3 — single-direction (open-ended)

Same as V1, step 1 + step 2 only (no return). Useful for early reward-shaping sanity checks before adding the return phase.

## Starting positions per game

The opening state on the board differs by game; the staging zones are always empty at episode start.

| Game | Opening state on the board |
|---|---|
| Ludo | 16 tokens in the four colored home corners (4 per color); die placed adjacent to the board |
| Chess | 32 pieces in standard chess opening — white on ranks 1–2, black on ranks 7–8 |
| Checkers | 24 discs on dark squares of ranks 1–3 (white) and 6–8 (black) |

## Observation space (proposed)

- **Joint state**: 6 joint positions + 6 joint velocities + gripper open width (13 floats)
- **End-effector pose**: 3 position + 4 quaternion (7 floats)
- **Overhead RGB**: 256×256×3, board-state view (the `OverheadBoardCamera` prim)
- **Wrist RGB**: 224×224×3 from the Synria C10 wrist camera (per the vendor email, mount frames are defined in the URDF)
- **Target piece + target zone**: encoded as either one-hot vectors or as a target XY position

## Action space (proposed)

Pick one of:

- **Joint deltas**: 6-DOF Δjoint per step + 1-DOF gripper command (7 floats)
- **End-effector deltas**: 3 position Δ + 3 axis-angle Δ + 1-DOF gripper command (7 floats)
- **Pose targets** (when running with MoveIt2 in the loop): single 6-DOF target pose, gripper open/close discrete (7 floats + 1 discrete)

URDF joint / velocity / torque limits are the action-space bounds. Per the vendor email, those limits are calibrated against firmware with a built-in safety margin — use them as-is for sim, and re-read firmware limits from the SDK as ground truth before deploying to the physical arm.

## Reward sketch

Per-step shaping (dense):

- distance from end-effector to current target (board piece or staging zone center)
- penalty proportional to gripper-piece distance once grasping
- mild action-norm penalty

Sparse milestones:

- +1 on successful grasp confirmed (gripper closed + piece position tracks gripper)
- +5 on piece centered within target zone tolerance
- +5 on piece returned to original square (V1) or +5 per piece in the cleared/restored sequence (V2)
- −2 on drop (piece falls below table surface)
- −5 on collision (arm vs table, arm vs another piece)
- −1 per step beyond a per-task timeout cap

Tune the magnitudes during training; the structure above is a starting point, not final.

## Termination

- Success: V1 completes step 4, or V2 completes the sequence, or V3 completes step 2
- Truncation: episode timeout (e.g. 30 s wall clock per phase)
- Reset on: any collision, piece dropped outside pads, gripper attempts to grip empty air for N consecutive steps

## Reference implementation

The Isaac Lab task config skeleton lives at [`isaac/isaaclab_tasks/synria_pickplace/`](../isaac/isaaclab_tasks/synria_pickplace/):

- `__init__.py` — registers `Synria-Ludo-PickPlace-v0`, `Synria-Chess-PickPlace-v0`, `Synria-Checkers-PickPlace-v0` with `gymnasium`
- `env_cfg.py` — `SynriaTabletopPickPlaceEnvCfg` (`ManagerBasedRLEnvCfg` subclass) with three per-game subclasses, plus observation / action / reward / termination managers
- `mdp.py` — implemented observation/reward/termination, trial-state and attach/reset logic;
  collision terms return zero when their contact sensor is absent
- `README.md` — RTX run instructions + RTX-side fill-in checklist

The scaffold is **Windows-import-safe** — `isaaclab` and `gymnasium` imports are gated behind `try / except`, so static parsing / ruff / mypy / pytest all pass without Isaac Lab installed. Runtime validation belongs on the Linux RTX 5090 box. Recorded simulation
runs exist; this does not establish physical-arm task success.

## What this doc deliberately does NOT specify

- Concrete reward coefficients — depends on the action-space choice and the policy class
- The exact PPO / SAC hyperparameters — those land with the agent config
- Imitation-learning data collection — handled separately via the LeRobot / ALOHA track
- Real-robot transfer details — those are part of the Jetson AGX Thor deployment plan, not the sim task
