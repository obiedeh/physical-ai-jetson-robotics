# Physical-contract ACT training

Software: **implemented, unmeasured**. D1–D5 remain **planned**. Validation uses
temporary synthetic datasets and tiny CPU updates, not a trained physical
policy or a physical evaluation.

## Known goal-observability blocker

This ACT policy consumes **state and images only**. It ignores task text. The
existing move executor expresses the planned target XYZ and token identity in
that text, and this dataset contract declares no visual goal cue or other
goal input. Changing a planned target while keeping the observations identical
therefore cannot condition this model on the new target.

These checkpoints are **not cleared for varying-target D2, D3, D4 or D5 motion**.
Run and checkpoint manifests explicitly record the input features,
`task_text_conditioning: false`, no declared visual goal cue, and
`varying_target_motion_eligible: false`. Serving and physical integration must
wait for an operator-approved observable-goal design or a prospectively defined
restricted task scope. Neither is implemented or assumed here. Do not treat
finite predictions, low held-out error or a successful local reload as resolving
this limitation.

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
  object, including action source, velocity flag, gripper, lookahead and rate.
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
Keep raw datasets and checkpoints outside git. Use the
[playback script](../../scripts/playback_synria_training.py) for diagnostic
curves; neither its metrics nor the latest checkpoint automatically select a
D2 policy or advance a stage. Physical probe execution and server attestation
remain separate, explicitly authorized workflows.
