# Physical-contract ACT training

Software: **implemented, unmeasured**. D1–D5 remain **planned**. Validation uses
temporary synthetic datasets and tiny CPU updates, not a trained physical
policy or a physical evaluation.

## Fixed-scene task eligibility

This ACT policy consumes **state and images only**. It ignores task text. The
existing move executor expresses the planned target XYZ and token identity in
that text, and this dataset contract declares no visual goal cue or other
goal input. Changing a planned target while keeping the observations identical
therefore cannot condition this model on the new target.

These checkpoints are **not cleared for varying-target D2, D3, D4 or D5 motion**.
Run and checkpoint manifests explicitly record the input features,
`task_text_conditioning: false`, no declared visual goal cue, and
`varying_target_motion_eligible: false`. The prospective 2026-10-08 decision
allows the three [fixed-scene roll skills](../../config/synria_tasks.json).
Each new checkpoint records `fixed_scene_motion_eligible: true` and
`eligible_task_ids` containing only its trained task. Use a separate dataset
and checkpoint for each skill. Token moves and goal-conditioning design remain
deferred until the roll works. Fixed-scene eligibility does not verify limits,
the safety bridge or physical motion. Do not treat
finite predictions, low held-out error or a successful local reload as resolving
this limitation.

Choose `TASK_ID` as `die_into_cup`, `roll_and_dump` or `cup_return` for the
workflow's paths and identifiers. Training does not accept a task override:
`expected_contract` must exactly match that dataset's full task snapshot/hash
and operator-configured episode window. All checked-in real windows remain
null; first time the physical skill and record a prospective choice in
[DECISIONS.md](DECISIONS.md), then collect and review its D1 data. Keep the
marked cup position, die start zone and fixed tray from its
[collection protocol](D1_DATASET_PROTOCOL.md).

Serve a finalized checkpoint with its trainer-written completion receipt:

```bash
python scripts/serve_synria_policy.py \
  --checkpoint "$CHECKPOINT_ROOT" \
  --checkpoint-record "$CHECKPOINT_RECORD" \
  --task-registry "$TASK_REGISTRY" --device cpu
```

The default address is `127.0.0.1:8080`. Set these variables to the chosen
checkpoint directory, its committed `checkpoint-step-<step>.json` evidence and
the operator-configured registry. The server verifies the receipt, content
hash, task snapshot and saved cadence, then loads only local model weights and
saved normalizers. Its `evaluated` receipt status confirms completed diagnostic
capture, not a metric threshold or policy selection. Missing/partial or changed
checkpoints are refused. Loading accepts only the wrapper's default registered
ACT processor steps with tensor state confined to the hashed checkpoint;
custom processor imports and pretrained-backbone downloads are refused before
model construction. Each request must carry the exact trained task text
and contract. Reset at each trial; chunk slots follow the saved command-period
stride. `/metadata` exposes the serving identity for inspection. This command
starts no ROS, camera or command path.

## Inputs and isolation

Use a separate Python 3.12 environment and install this repository with
`python3 -m pip install -e '.[robot-learning,vision]'`. The optional dependency
is upstream LeRobot 0.6's `training` extra, including dataset and trainer
dependencies; it is not a vendor fork. Do not modify the working teleoperation
environment. This wrapper opens no robot interface and starts no ROS process,
camera, driver, bridge or teleoperation.

Checkpoint compatibility is tested with released upstream 0.6.1 and the local
0.6.2 development checkout; these are distinct installations, not two claimed
package-index releases. The wrapper unwraps the model and inspects the installed
save function, supplying `accelerator` only when that named keyword is supported.
Save errors are not retried under another signature. The installed package
version is recorded in every run and checkpoint manifest. The existing optional
dependency remains `lerobot[training]>=0.6,<0.7` on Python 3.12 or newer.

Finish the D1 quality and visual review, then follow the
[frozen probe runbook](CHECKPOINT_PROBE_RUNBOOK.md) **before training**. Preserve
the finalized dataset outside git. The committed probe set specifies held-out
episodes and physical trial identities; training uses exactly the remaining
episodes. Both sets must be nonempty. Diagnostic probe results must never be
used for D2 policy selection; register the D2 selection rule separately before
viewing them. No operator dataset, held-out IDs or physical probes are supplied
by this software.

Complete an operator-owned copy of the
[training configuration](../../config/synria_act_training.json):

- `policy_id`: a new identifier absent from the policy ledger; `dataset_repo_id`
  identifies the existing local dataset.
- `expected_contract`: copy the complete dataset `physical_contract.json`
  object, including task ID/definition/hash/window, recording purpose, action
  source, velocity flag, gripper, lookahead and rate. Smoke cannot train.
- `repository`, committed `probe_set` and `probe_sha256`; new
  `evidence_directory` inside the repository; separate existing `policy_ledger`
  and `timeline` destinations. Relative config paths resolve under `repository`.
- Required `command_period_s`, `chunk_size` and this policy's
  `response_timeout_s`; explicit batch size, save frequency, optimizer settings,
  architecture, and `device` (`cpu` or `cuda`). Template recipe values are
  unmeasured starting points, not a recommended physical policy.

The command period multiplied by dataset FPS must be an integer stride `q`
of at least one, and `chunk_size >= q`. Recorded action lookahead is preserved,
not shifted again. The serving convention is absolute targets offered now and
held until the next command tick, using chunk slots `0,q,2q,...`; this does not
guarantee reaching a pose at the end of a period. The future physical session
must also honor the bridge-duration floor and verified motion limits. A
training configuration never authorizes motion.

## Train locally

From the repository root, choose an explicit seed and positive step count,
an absolute dataset path, its contract path, and a **new external** raw-output
directory. Existing output/evidence directories are refused, not overwritten.
For a CPU run, hide accelerators before starting the interpreter:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
  python3 scripts/train_synria_act.py \
  --dataset-root "$DATASET_ROOT" --contract "$DATASET_ROOT/physical_contract.json" \
  --output "$TRAINING_OUTPUT" --config "$TRAINING_CONFIG" \
  --seed "$TRAINING_SEED" --steps "$TRAINING_STEPS"
```

The wrapper also disables remote downloads/uploads and experiment-service
logging. Backbones start with `pretrained_backbone_weights=None`; no pretrained
weights are fetched. Supported execution is single-process, full-precision
training. This initial path starts a new run; it does not resume partially
failed optimization.

The [wrapper](../../synria_lerobot/act_training.py) calls upstream ACT,
`update_policy` and `save_checkpoint`, and saves both normalization pipelines.
Normalization aggregates only the training episodes' persisted statistics;
the dataset-global statistics are not used. Camera features must match the
D1 named CHW metadata, with equal stored wrist/front sizes; loaded tensors are
CHW RGB. The saved model state is seven positions/gripper values or thirteen
values including six velocities, exactly as declared by the physical contract.

Every periodic save and the final unique step invoke the frozen held-out
evaluation hook. Raw checkpoint directories include an immutable physical
manifest; separate small checkpoint records hold the complete content hash.
Dataset, contract, split/probe hash, step, recipe, seed, host, git SHA and
upstream version are recorded. A full content hash is checked again after each
evaluation; hashing and training performance are unmeasured.

## Simulation lane

Takes recorded by the simulation console carry the `simulation` purpose, and the
physical lane above refuses them. A policy for the Isaac scene trains in the
simulation lane instead: set `"lane": "simulation"` in the training
configuration and `"lane": "simulation"` in the probe set. The lane changes four
things and nothing else:

- the dataset contract must carry the `simulation` purpose, and a qualifying
  dataset is refused, so the two lanes can never be mixed by mistake;
- the probe set holds out episodes but carries no physical trials
  (`physical_trials` is `[]`); closed-loop trials in the scene are the
  simulation runner's business;
- curated exclusions are kept: an excluded take is left out of training (it may
  not be held out), and `training_episode_limit` may name how many of the
  remaining takes to train on, in recording order, so each larger subset
  contains every smaller one. Every omitted index is listed in the run and
  checkpoint manifests as `omitted_episode_indices`; nothing goes unused silently;
- the policy ledger must be the [simulation ledger](SIM_POLICY_LEDGER.md), whose
  first line names the lane. Rows carry `lane=simulation` and
  `N of M episodes trained`.

Every manifest, checkpoint record and evaluation record from this lane says
`"lane": "simulation"`, `"physical_eligible": false`,
`"fixed_scene_motion_eligible": false` and `"simulation_motion_eligible": true`.
The policy server loads such a checkpoint only with `--lane simulation`, and a
physical server refuses it; the physical probe runner refuses simulation probes.
A configuration or probe set without a `lane` field is physical.

The data-scaling study trains the same dataset at several
`training_episode_limit` values with the same held-out set, seed and step count,
and compares the held-out action error per checkpoint and the closed-loop
success rate in the scene against the trained episode count.

### Closed-loop trials in the scene

`scripts/run_sim_policy_trials.py` runs scored trials against a server started
with `--lane simulation`, with the Isaac scene in its policy launch mode
(`SKIP_DRIVER=1 bash scripts/run_alicia_stack.sh policy` in the teleop
repository), which mirrors `/isaac/policy_joint_targets` instead of the real
arm's feedback. Before every trial the runner walks the arm from wherever it
is to the start pose (the dataset's mean first frame unless `--start-pose` is
given) within the per-command step bound, then asks the scene for a Randomize
and reset at `--randomize-scale` (1x by default; the layout is drawn for the
arm at the start pose), then resets the policy and steps it at the command
period. With `--placement-seed N` trial i's draw is seeded with N plus i, so
two campaigns sharing a seed face the same placements and can be compared
trial by trial; without it the scene draws unseeded. The score
follows the served task and comes from the scene's own poses. `pick_cup`:
success when the cup has been `--lift-m` (3 cm) above its rest height for
`--hold-s` (1 s) without a break. `cup_pick_and_place`: success when the cup,
lifted at least once, rests within `--place-radius-m` (3 cm) of the scene's
mark (`/isaac/pick_objects/place_target/pose`) and within `--place-height-m`
(1 cm) of its rest height with the gripper centre `--gripper-clear-m` (6 cm)
away, for `--hold-s`. Any other task is refused. Every trial records where the
cup stood after the reset; the trial otherwise ends at `--max-trial-s`. Speed
bounds come from
[`config/synria_sim_policy.json`](../../config/synria_sim_policy.json), a
simulation-lane file the physical command path refuses. The runner never
publishes on `/joint_states`, `/joint_commands`, `/policy_joint_targets` or
`/policy_arm`, and refuses a server or contract outside the simulation lane.
The report, written once under `reports/sim_policy/`, carries the rule, every
trial, the success rate and `physical_evidence: false`.

`scripts/run_sim_checkpoint_campaigns.py` serves several saved checkpoints in
turn (each named with its receipt) and runs one campaign against each with the
same `--placement-seed` and `--randomize-scale`, one report per checkpoint in
`--output-dir`. That is how a run's intermediate checkpoints, its averaged
checkpoint and another run's are compared on the same placements; the study
driver's `--trials-only` mode takes the same two options for a row's final
checkpoint.

### Approach levers: state noise, action smoothing, grasp-phase sampling, gripper guard

The first study's failures were approach and grasp failures: the policy descended,
hesitated, and closed by timing rather than by sight. Four levers aim there; all
default to off, so every recorded run is unchanged, and each is recorded in the
manifest or the campaign report.

- `state_noise` in the training configuration adds Gaussian noise (`joint_rad`,
  `gripper_m`, driver units) to every training sample's joint state and replaces
  the state of a `drop_probability` share of samples with the training mean, so the
  model cannot read the grasp timing off its own joints and has to use the cameras.
  Held-out evaluation sees clean states.
- `action_smoothing_window` (odd, 3 or more) replaces the six joint actions of
  every training chunk with a centred moving average over the chunk, edge padded;
  the gripper action is untouched. The recorded hand shake of the leader arm is
  what this removes. Held-out error is still measured against the raw actions.
- `grasp_phase` oversamples the frames around each take's final close: the close is
  the frame after the last gripper action below `close_above_m` (0.016 m sits
  between the leader's idle squeeze, about 0.011 m in the first dataset, and a
  close on the cup, 0.018 m and above); frames `seconds_before_close` before to
  `seconds_after_close` after it are drawn `weight` times as often, multiplied
  with any `episode_weights`. The manifest records how many frames were weighted.
- `gripper_guard` in the policy safety file (`close_above_m`, `consecutive_steps`)
  makes the client honour a change between open and closed only after that many
  consecutive requests, holding the gripper where it was meanwhile; joints are
  never touched. `config/synria_sim_policy.json` carries 0.016 m and 3 steps (0.6 s
  at the 0.2 s period); the physical file carries none. Every trial record counts
  its guarded steps and the report carries the safety settings.

The Help page's recording rules ask for a continuous approach from any direction
that suits the cup, the gripper fully open until the cup is between the fingers,
and one deliberate squeeze, with one approach style per session; that is the
recording-side fix for the same failure. An earlier wording asked for a hover and
a stop before the descent; it was withdrawn because a pause in a demonstration is
an ambiguous state the model imitates as a stall, and a top-down descent hides the
cup from the wrist camera in the last centimetres.

### Failure modes of the campaigns

`scripts/summarize_sim_trial_failures.py --study <study directory>` sorts every
trial of every `trials_*.json` into one mode (success; held by the safety
client; lifted then dropped; reached the mark but not held there, place score
only; moved the cup without a lift; reached the cup without a grasp; never
reached the cup; cup untouched with no gripper pose recorded) and writes
`FAILURE_MODES.md` and `failure_modes.json` beside them, with the campaigns the
study superseded marked and left out of the totals. The first study's table
says the cup was untouched in 164 of 180 current trials: the approach and the
grasp are the bottleneck, not the lift, so the next changes aim there. The
table's last column is the share of all steps the safety client held, whatever
the mode: a hold short of half a trial does not make a mode, but a policy pulled
back from a joint bound on a sixth of its steps is pressing the operator limits.

### Placement tolerance per task

`config/synria_placement_tolerances.json` holds the radius within which a released
object counts as placed, per task: 3 cm for the cup mark, 1 cm for a Ludo token
and a checkers piece, 1.2 cm for a chess piece. The trial script takes it as the
default `--place-radius-m` for the served task, and every place campaign's
failure-mode table ends with the final cup-to-mark distance of the trials that
lifted the cup, median and 90th percentile, against that radius, so a near miss
is measured and not only counted.

### Goal channels in training and serving

When the dataset's contract declares goal channels, the trainer requires the
`observation.environment_state` feature of three values per channel, trains it
as the policy's environment-state input, normalises it from the training
episodes' own statistics, records `goal_channels` and the input key in the
manifest, and the held-out evaluation feeds it. A dataset with the feature but
no declared channel, or the reverse, is refused. The served checkpoint requires
the goal in every request and the trial runner fills it from the scene's poses
by channel name. Without channels nothing changes: contracts, manifests and
requests are byte for byte what they were.

### The staged executor: what, where, how, plan, execute

`scripts/serve_staged_executor.py --task-id pick_cup --command-period-s 0.2
--stage-log STAGES.jsonl` serves a scripted expert on the policy protocol, and
`scripts/run_sim_staged_campaign.py` runs a seeded campaign against it with the
same client, guard, limits and scorer as a learned policy's. Before it moves it
answers four questions and writes them to the stage log: what (the registry
task and its object), where (the object's and the mark's positions from the
scene's published poses; a camera locator replaces that on hardware), how (a
grasp strategy per object class; the cup's is the median of the operator's own
takes at the squeeze: gripper centre 30 mm up, approach pitched 54 degrees from
vertical, fingers horizontal, lift 80 mm) and plan (inverse kinematics on the
committed chain in `config/synria_kinematics.json`, every waypoint 3 degrees
inside the operator bounds, on the operator's own elbow branch; when a group of
waypoints does not solve at the strategy's pitch, steeper pitches of 40, 25 and
10 degrees are tried and the pitch used is recorded). A plan that does not solve
at any pitch is reported as infeasible with the waypoint that failed, and the
arm holds. Offline, all 50 seeded placements of seed 2026 solve for both tasks,
seven place marks at the 40 degree fallback. Execution is approach, descend along the approach axis, squeeze,
lift, and for the place score carry, lower, release and retreat; each phase
entry, every stall and the final phase are in the record, so a failed trial
names its stage. The chain was validated against the scene's published gripper
centre to 0.0 mm with the simulation calibration offsets; the physical offsets
are unmeasured and the executor refuses the physical lane. It is the ceiling
measurement for the staged design and the demonstration engine of the strategy
review, not a learned policy, and its campaigns say so in their metadata.

### Serving and client ablations on a fixed seed

Two switches let a paired campaign measure one serving-side change alone, with
everything else fixed by the placement seed:

- `--policy-config config/synria_sim_policy_noguard.json` on
  `run_sim_checkpoint_campaigns.py` runs the trial client without the gripper
  guard; the file is the simulation safety file minus the guard and is never
  served to the physical arm.
- `--temporal-ensemble off` serves a checkpoint trained with a
  `temporal_ensemble_coeff` chunk by chunk instead; the server refuses it outside
  the simulation lane and names the override under `serving_overrides` in the
  metadata every trial report copies, so such a run can never pass as the
  manifest's own serving mode.

The first paired campaigns at seed 2026 with the guard on scored the plain
45-take final checkpoint 28 of 50 against 1 of 20 in the earlier unseeded
campaign, and both v2 checkpoints 0 of 50 with the gripper never within 13 cm of
the cup. The guard, the seeded placements and the scene with the place mark all
changed between those runs, so the chain queued after the approach recipe runs
the plain checkpoint at seed 2026 without the guard, the v2 checkpoint without
ensembling, and the plain checkpoint at seed 2027 with the guard, each 50
trials, into `reports/sim_policy/paired_seed2026_noguard`,
`paired_seed2026_noensemble` and `paired_seed2027`. Unmeasured until they land.

### Which takes press the operator limits

`scripts/audit_dataset_limits.py --dataset-root <dataset> --output <study directory>`
compares every commanded and observed value of a dataset with
`config/synria_limits.yaml` and writes `LIMITS_AUDIT.md` and `limits_audit.json`:
per joint and per episode, the frames within half a degree of a bound, the
frames past the exact bound and the frames past the recording tolerance. The
recording gate accepts a joint up to that tolerance beyond a whole-degree bound;
the policy client does not, so it clamps every such command and, in the
simulation lane, holds and pulls the arm back whenever the sagging sim joint
reads past the bound. On the first dataset, 41 of 50 takes press a bound, 17 of
them Joint2's upper stop on the far reach and 21 of them Joint3's lower stop,
and 1010 frames are past a bound without any past the tolerance. A take that
presses a bound teaches a target the client will clip, so the next recording
batch keeps cups inside the reach the limits allow, and the audit is run before
a dataset is frozen.

### Augmentation, temporal ensembling and checkpoint averaging

Three optional levers, each recorded in the manifest and refused when malformed:

- Geometric transforms (`affine`) are never applied to a fixed-camera policy: they
  move the apparent object in the image without moving the action label and
  destroy the pixel-to-action mapping (strategy review of 2026-10-18, G7). Use
  colour, brightness, contrast, saturation, hue and sharpness only.
- `image_transforms` in the training configuration names upstream transforms
  (brightness, contrast, saturation, hue, sharpness; the trainer refuses affine)
  applied to training frames only, never to held-out frames, with the upstream
  default ranges.
  Augmentation runs on the CPU inside the data loader, so set
  `dataloader_workers` (default 0, every frame decoded and augmented in the
  training process) to a few worker processes when it is enabled. The first
  augmented run on the simulation study host trained about seven times slower
  per step than the plain baseline with 0 workers. The count is part of the
  recorded configuration; held-out evaluation always uses 0.
- `episode_weights` in the training configuration maps episode indices
  (`"31"`, or an inclusive range `"30-49"`) to positive weights; frames of a
  weighted episode are drawn that many times as often as frames of an
  unweighted one, with replacement. The training set and the held-out frames
  are unchanged, and the manifest records the weight that took effect for every
  training episode. The first study showed the randomized takes were the only
  ones that produced picks from new spots, so weighting them up is the cheap
  way to make them count more without recording.
- `temporal_ensemble_coeff` makes the policy server predict a chunk on every
  request and return, for the current step, the exp-weighted mean of what every
  still-overlapping chunk predicted for it (weight `exp(-coeff * age)`, oldest
  chunk age 0, as in the ACT paper). Null keeps the one-prediction-per-chunk queue.
- `scripts/average_synria_checkpoints.py --evidence <run> --last 4` averages the
  float weights of a completed run's last evaluated checkpoints into one
  checkpoint beside them, evaluates it on the frozen held-out set with the run's
  evaluator, and writes a receipt the policy server accepts. The manifest names
  the averaged steps. Nothing trains.

### The data-scaling study

`scripts/run_sim_data_scaling_study.py --spec <study>.json` trains the frozen
dataset at each size in `subset_sizes` (nested by recording order, so each
run contains every smaller one) with one seed, step count and held-out set,
through the simulation-lane trainer, so every run has its own evidence
directory, checkpoint hash and ledger row. The template is
[`config/synria_sim_data_scaling.json`](../../config/synria_sim_data_scaling.json).
With `--trials N` the final checkpoint of each run is served on a local port
and the trial campaign above runs against the scene before the next size
trains. The study record `study.json` and the table `STUDY.md` are written
into the evidence root after every row, so a failed study leaves what it
measured marked failed. The table lists, per episode count, the held-out
joint error of the final checkpoint and the closed-loop success rate; its
caveats (recording order is not random, one seed, simulation only) are part
of the document.

## Failures and evidence

Review the evidence directory's start/run manifests, checkpoint records and
evaluation records, plus the appended timeline events. Each run that begins
optimization appends one policy-ledger row with its required response timeout;
checkpoints do not add extra rows. Failure rows retain the last finalized
checkpoint hash, or explicit `N/A` when none exists. Incomplete saves are listed
as partial paths without a qualifying hash; never load or evaluate them.
Do not reinterpret a failed optimizer step as a valid checkpoint.

Evaluation errors retain their failed record and stop training. Manifest and
ledger finalization are both attempted; secondary write errors do not replace
the primary training/evaluation failure. Reconcile any reported evidence-write
error manually before another run, using a new policy id/output destination;
do not erase negative results or suppress failed diagnostics.

Commit the small evidence, ledger, activity entry and timeline with the run.
The trainer creates exactly one policy-ledger row for this optimization run;
checkpoint saves and later D2 sessions must not create duplicate rows for it.
Keep raw datasets and checkpoints outside git. Use the
[playback script](../../scripts/playback_synria_training.py) for diagnostic
curves; neither its metrics nor the latest checkpoint automatically select a
D2 policy or advance a stage. Physical probe execution and server attestation
remain separate, explicitly authorized workflows.

For the [standalone three-skill roll](D4_OPERATOR_RUNBOOK.md#standalone-fixed-scene-roll),
repeat collection, frozen probes and training separately for all three task IDs.
Serve three finalized task-bound checkpoints on distinct local ports, each with
its matching completion receipt and task registry:

```bash
python3 scripts/serve_synria_policy.py \
  --checkpoint "$DIE_INTO_CUP_CHECKPOINT" --checkpoint-record "$DIE_INTO_CUP_RECEIPT" \
  --task-registry "$TASK_REGISTRY" --device cpu --host 127.0.0.1 --port 8080
python3 scripts/serve_synria_policy.py \
  --checkpoint "$ROLL_AND_DUMP_CHECKPOINT" --checkpoint-record "$ROLL_AND_DUMP_RECEIPT" \
  --task-registry "$TASK_REGISTRY" --device cpu --host 127.0.0.1 --port 8081
python3 scripts/serve_synria_policy.py \
  --checkpoint "$CUP_RETURN_CHECKPOINT" --checkpoint-record "$CUP_RETURN_RECEIPT" \
  --task-registry "$TASK_REGISTRY" --device cpu --host 127.0.0.1 --port 8082
```

Run each server in its own terminal after setting the six explicit checkpoint
and receipt variables; these are separate long-running commands, not a shell
pipeline. Set the corresponding session `roll_skills` endpoints to the three
addresses. Their command periods and stored image sizes must match the shared
adapter; retain each policy's own trained response timeout and command budget.
Metadata refuses a substituted task, checkpoint or cadence. Serving makes no
arm offers; use D2's disarmed preflight/manual arming procedure before the roll
runner. For D2, serve only the prospectively selected `die_into_cup` checkpoint;
probe metrics and the latest save never make that selection automatically.
