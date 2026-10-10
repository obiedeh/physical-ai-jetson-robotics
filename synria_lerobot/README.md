# LeRobot / ALOHA-Compatible Data and Policy Track

This package implements the Synria arm LeRobot track: episode schema,
synthetic dataset generation, episode recording, policy evaluation,
and CLI commands.

Primary guide:
[`docs/LEROBOT_ALOHA.md`](../docs/LEROBOT_ALOHA.md)

---

## Package layout

```text
synria_lerobot/
├── schema.py        Data model — JointState, EEPose, ActionFrame,
│                    ObservationFrame, EpisodeStep, Episode, EpisodeMetadata
├── dataset.py       Synthetic episode generator and SynriaEpisodeDataset
├── recorder.py      Shared read-only physical session builder and terminal recorder
├── console/         Local recording, offline review and audited session catalog
├── exclusions.py    Shared sibling-log curation replay for summaries and training
├── policy_eval.py   DeterministicArmPolicy (oracle baseline) and evaluation helpers
└── README.md        this file
```

---

## Data model

All data structures are frozen dataclasses with no external dependencies.

### Observation

| Field | Type | Description |
|---|---|---|
| `joint_state` | `JointState` | 6 positions (rad) + 6 velocities (rad/s) + gripper width (m) |
| `ee_pose` | `EEPose` | XYZ (m) + quaternion wxyz |
| `frame_index` | `int` | Step index within the episode |
| `timestamp_s` | `float` | Wall-clock time in seconds |

`JointState.as_vector()` → 13-float tuple `[pos×6, vel×6, gripper_m]`
`EEPose.as_vector()` → 7-float tuple `[x, y, z, qw, qx, qy, qz]`

### Action

```python
ActionFrame(
    joint_deltas_rad: tuple[float, ...]  # 6 Δjoint commands
    gripper_command: float               # −1.0 (close) … +1.0 (open)
    timestamp_s: float
)
```

---

## Synthetic dataset generation

Samples the demo trajectories from `arm_control/demo.py` at a given fps
via linear interpolation and finite-difference velocities. EE pose is
computed with a planar forward-kinematics approximation.

```python
from synria_lerobot.dataset import generate_synthetic_episode, SynriaEpisodeDataset

ep = generate_synthetic_episode(zone="left", game="chess", fps=30.0)
print(len(ep.steps))        # number of steps
print(ep.metadata.game)     # "chess"

dataset = SynriaEpisodeDataset([
    generate_synthetic_episode(zone=z, game="chess")
    for z in ("left", "right", "top", "bottom")
])
```

Valid staging zones: `"left"`, `"right"`, `"top"`, `"bottom"`
Valid games: `"chess"`, `"checkers"`, `"ludo"`

---

## Episode recording

```python
from synria_lerobot.recorder import EpisodeRecorder, RecordingSession

# Single mock episode (synthetic, no hardware required)
rec = EpisodeRecorder(zone="left", game="chess", mock=True)
episode = rec.record(n_steps=120)

# Full 4-zone session
session = RecordingSession(n_episodes=4, zones=("left","right","top","bottom"))
dataset = session.record_all(n_steps=120)

# Hardware recording (Jetson AGX Thor)
# rec = EpisodeRecorder(zone="left", game="chess", mock=False)
# episode = rec.record()     # reads live from ROS 2 + cv2 — requires Jetson
```

---

## Policy evaluation

```python
from synria_lerobot.policy_eval import DeterministicArmPolicy, evaluate_policy_on_dataset
from synria_lerobot.dataset import generate_synthetic_episode, SynriaEpisodeDataset

dataset = SynriaEpisodeDataset([generate_synthetic_episode(zone="left")])
policy = DeterministicArmPolicy()
results = evaluate_policy_on_dataset(policy, dataset)

for r in results:
    print(r.passed, r.mean_joint_tracking_error_rad)
```

`DeterministicArmPolicy` is an oracle baseline that replays ground-truth
deltas. It achieves zero tracking error on synthetic episodes, providing
a deterministic upper bound for the pipeline.

`PolicyEvalResult` fields:

| Field | Type | Description |
|---|---|---|
| `episode_id` | `str` | Unique episode identifier |
| `n_steps` | `int` | Number of evaluated steps |
| `mean_joint_tracking_error_rad` | `float` | Mean absolute Δjoint error |
| `max_joint_tracking_error_rad` | `float` | Worst-case Δjoint error |
| `completion_rate` | `float` | Fraction of steps within threshold |
| `passed` | `bool` | `mean ≤ threshold` AND `max ≤ 5×threshold` |

---

## CLI commands

```bash
# Generate and summarise a synthetic 4-zone dataset
physical-ai-lab lerobot-dataset-demo

# Evaluate the oracle policy on the synthetic dataset
physical-ai-lab lerobot-policy-eval
```

---

## Tests

```bash
pytest tests/test_lerobot_schema.py      # 30 tests — data model
pytest tests/test_lerobot_dataset.py     # 30 tests — synthetic generation
pytest tests/test_lerobot_recorder.py    # 16 tests — recording session
pytest tests/test_lerobot_policy_eval.py # 11 tests — evaluation pipeline
```

All 87 LeRobot tests run without torch, Isaac Lab, or hardware.

---

## Connection to Isaac Lab

The LeRobot and Isaac Lab tracks share geometry constants via
`isaac/isaaclab_tasks/synria_pickplace/task_geometry.py`.
Synthetic episodes are generated from the same waypoint trajectories
used by the Isaac Lab demo arm. When a real ACT / diffusion policy is
trained, replace `DeterministicArmPolicy` with a wrapper around the
checkpoint:

```python
class MyACTPolicy:
    def predict(self, obs: ObservationFrame) -> ActionFrame:
        ...  # call ACT model

results = evaluate_policy_on_dataset(MyACTPolicy(), dataset)
```

---

## Physical recording and console

The separate `synria_physical_v1` contract and read-only multi-episode recorder
are **implemented, unmeasured**; the schema examples above describe the legacy
simulation helpers, not the physical contract. Follow the
[D1 operator runbook](../docs/ludo_flagship/D1_OPERATOR_RUNBOOK.md) for authorized
physical setup, or use the [console guide](../docs/ludo_flagship/TELEOP_CONSOLE.md)
for synthetic demo, named recording sessions and offline review. Both surfaces
use the same builder, transactions, contract and gates. Dataset roots and console
workspaces remain outside git; there is no console upload or motion control.
