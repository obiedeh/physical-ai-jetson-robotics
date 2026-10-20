# Phase B preflight preparation

Status: **implemented, unmeasured** for static preflight. Phase B's Isaac
evaluation and monitor benchmark remain **planned**. No trial has been run.

Phase A and its final results were merged in PR #6 at
d13216a91e194050898e9cc6c9a10fa2723fb76f. This change advances only the
preflight work on the isolated Thor command-agent branch.

The command now accepts a separate preparation path:

    physical-ai-lab command-agent --preflight-profile reports/agents/command_agent/phase_b_profile.template.json

Exit code 2 means prerequisites are missing or changed. This command does not
invoke a planner, initialize ROS, contact Isaac or execute a trajectory. A static
ready result only permits proceeding to live simulator preflight; it is not a
task outcome or authority to run hardware. Preflight options cannot be mixed
with a command request or hardware flags.

The template intentionally stays **scaffold**. Copy it to a campaign-specific
profile and supply observed site-owned deployment facts before declaring that
new profile planned. Its original suite pin points at the pre-registration commit
f3005f7da51cb382b8b4a35cce51a06c92c6402a. The preflight expands the exact listed
case/seed order into 80 scheduled trials (35 benign, 45 adversarial); missing
bindings never shrink those denominators. The original suite was not edited.
No scheduled row is a measured trial.

A prospective planned profile must be committed before trials. It pins the
preregistered suite, exact harness source commit (including reused safety, registry, scoring and
kinematics dependencies), Isaac version, isolated ROS
domain, scene, skill library, simulator integration code and host configuration.
All small evidence and profile files must be unchanged at HEAD. Scene/library
files may be site-owned; their bytes must match the committed profile's SHA-256.
Host checks include Python, JetPack, queried power mode and observed CPU
configuration; changed inputs need a new prospectively committed profile.

Every registry skill plus return-to-ready needs a committed qualification record
matching the harness, scene, library and adapter hashes. It names the explicit
reset procedure and object-success rule and pins the small committed raw trial
evidence. The qualification schema is metadata binding existing integration
evidence; it does not implement or validate a new physical scoring algorithm.
Referenced traces and declarations still require operator review. Existing
LiftScorer and PlaceScorer are candidates for pick_cup and cup_pick_and_place.
No other task is silently assigned those cup criteria.

The launch-only --confirm-5090-released flag records an operator release for the
current batch. It must only be used after the operator actually frees the GPU.
Neither a merge instruction nor an expired earlier batch implies that release.
The actual preflight report retains the missing release as a blocker. The flag
does not execute a batch and never enables physical motion.

Next deployment work:
- Supply and qualify the actual Isaac integration, site scene and skill library,
  including task-specific reset and object-success evidence for all six skills.
- Freeze the real run profile while preserving the registered suite and scoring.
- After explicit 5090 release, verify the live twin and run the registered trials,
  ledger every outcome and report the measured metrics with device provenance.

The default remains blocked: there is no qualified command-agent twin binding,
no frozen Isaac version/ROS domain, missing task qualification evidence and no
operator 5090 release. ROS client availability is checked without starting it.
Phase C remains parked; this preparation contains no physical execution path.

Change Report:
1. Added typed campaign/qualification contracts, static preflight, source/input
   integrity checks and schedule expansion to existing command-agent modules;
   extended the existing CLI and its dedicated test workflow. Added this
   operational guide, a scaffold profile and observed Thor configuration.
2. Hermetic gate and preflight tests, repository lint and type checks run on
   Thor; exact counts and source hashes are retained in phase_b_validation.json.
3. Live Isaac, 5090, motion and benchmark tests are not run because release and
   qualified bindings remain unavailable. Static fixture tests are not measurements.
4. A full suite executor and live binding qualification remain future work; a
   ready preflight does not itself run or score the registered campaign.
5. A hardware safety review and the existing Phase 0B/operator requirements
   remain mandatory before any physical Phase C execution.

Hosted validation: all 132 command-agent gate/preflight tests, lint and typing
passed on Python 3.10 and 3.12 at 194f083. The small job record is retained in
phase_b_ci.json. This checks static preparation only and adds no Isaac trial or
latency result. Source hashes match the validated snapshot.

Change Report (CI evidence): added the completed hosted job record and updated
validation and integrity hashes. No source changed; no code tests repeated.
Unqualified bindings, unavailable ROS configuration and missing operator release
still prevent the actual campaign. The existing physical safety review remains
required before any robot execution.


## Offline trial reporting

Status: **implemented, unmeasured**. The CLI can summarize an original JSONL
ledger without a planner, ROS, simulator, GPU release or hardware connection:

    physical-ai-lab command-agent --report-profile reports/agents/command_agent/phase_b_profile.template.json --trial-ledger reports/agents/command_agent/phase_b_trials.jsonl

The current input ledger is empty: no trials have run. Missing rows stay in the
80-trial schedule, rates and latency statistics are null until observations
exist, and an incomplete report exits 2. An empty ledger is not a campaign.

Each command_campaign_trial_v1 row binds its registered index/case/seed to the
profile SHA-256 and harness commit. It retains the original evaluate() audit,
explicit infrastructure error or per-step object-success booleans, and hashed
raw-evidence references. The site integration must supply the qualified reset,
camera-text rendering, object observations and frozen scorer; this report utility
does not implement those operations or change any gate after a result is seen.

The reducer checks gate-record structure, original request/context, deterministic
backend identities, timestamp order, profile/host binding and evidence hashes.
It keeps the fixed 45-adversarial/35-benign denominators, missing rows, failed
expectations, infrastructure errors and fixture rows. G3-not-reached and missing
audits remain separate from finite observed timings; p95 uses nearest rank.
Per-step object scores are required for every accepted multi-step proposal.

Aggregation never marks a campaign measured or certifies deployment readiness,
even with a complete Isaac-labelled ledger. The live binding, raw traces, reset,
camera frames and task scoring still require the frozen-profile review in the
pre-registration protocol. Hermetic fixtures cannot become Isaac measurements.
No campaign executor, scene integration or physical executor is added here.
