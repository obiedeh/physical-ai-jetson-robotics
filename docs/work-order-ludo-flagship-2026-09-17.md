# Work order: deliver the Ludo flagship on real hardware

Issued 2026-09-17. Owned through to completion; report to the operator.
Everything else in the portfolio is parked (Focus Rule).

> **Scope correction, 2026-09-18:** D1 through D5 apply exclusively to the
> physical Synria/Alicia-D arm. Yahboom ROSMASTER M3 Pro artifacts are useful
> historical references, but they cannot satisfy, advance, or substitute for
> any Ludo flagship delivery stage. Recovered Synria real serial/state/MoveIt
> bring-up and real-to-simulation mirroring exist in an external working tree;
> they are **implemented, unmeasured** until formalized and evidenced here.

## 1. The goal

A robot arm plays Ludo on a physical board: it picks up its own token,
moves it to the square the game engine chose, with the roll coming from a
physically rolled and read die, turn after turn, with every turn scored
and recorded. The manipulation policy is learned from recorded human
teleoperation demonstrations, fine-tuned, and improved on pick-and-place
and other contact-rich motions until it is reliable enough to play.

Delivery is defined by artifacts, never by demos or videos alone:

| Stage | Done when this artifact exists |
|---|---|
| D1 Real demonstrations | A committed dataset summary for 100 or more recorded teleoperation episodes on the physical Synria/Alicia-D arm, with provenance per session (device, date, operator, scene, camera, power source), episode-level operator success labels, and quality-gate results. |
| D2 Learned pick-and-place on the real arm | A pre-registered, closed-loop evaluation of a trained policy on the physical Synria/Alicia-D arm with at least **14 operator-confirmed, camera-backed physical-object successes in 20 trials**. Commit the protocol before the run, then the EvalLog, camera-linked operator labels, scorer output, and every negative result. |
| D3 Physical Ludo turn | **One SUCCESSFUL** engine-planned turn executed by the policy with the physical Synria/Alicia-D arm on the physical board, independently camera-backed and operator-confirmed, with the roll supplied by the operator and `turns.jsonl` committed. |
| D4 Physical roll in the loop | **Three consecutive SUCCESSFUL physical-roll turns**: Synria physically rolls the die, the physical result is read and fed to `game.plan_turn(roll=value)`, and each policy-executed board move is camera-backed and operator-confirmed. Commit all turn records. |
| D5 A game | **One successful full game to a winner** on the physical board, with physical rolls and Synria policy-executed moves, every turn scored, misses ledgered, and a `ludo_stats.py` aggregate committed. |

Each stage moves its README status-table row only when its artifact is
committed, with date and hardware. Nothing is upgraded on a video.
For D1 only, quality-valid failed demonstrations may be retained, labeled and
counted when the pre-registered dataset protocol permits it. Failed D2
evaluations, D3 turns, D4 physical-roll turns and D5 games are mandatory
ledger entries, but do not advance those stages.

## 2. Where it actually stands (read before planning)

Labels follow the README: measured, implemented unmeasured, scaffold,
planned. Every claim below points at its record.

Measured, simulation:
- Ludo expert executor (scripted grasp, PPO reach, kinematic attach) 35/36
  turns, seeds 10 to 12, Isaac Sim on RTX 5090, 2026-08-19/20
  (`reports/ludo_stats_frozen/`). Physical dice roll and multi-turn
  physical-roll game exist as simulation videos, not scored runs
  (`docs/notes/ludo_game_milestones_2026-08-21.md`).
- GR00T N1.7 Ludo lane: eval01 0/20 corrected (target-blind corpus),
  eval02 on the target-observable corpus 0/20, 17 of 20 never grasped
  (`reports/ludo_groot17_eval02_tobs/session_summary.json`,
  `reports/training/ludo_groot17_v2_tobs/`). No learned policy has
  completed a Ludo turn, in sim or real.
- GR00T inference cost on Jetson AGX Thor, 101.6 ms median TensorRT
  (`reports/thor_trt_benchmark/`).

Measured, other real hardware (non-qualifying background):
- Yahboom ROSMASTER M3 Pro on Jetson Orin NX: day-one probes 2026-08-20;
  board contract and three operator-confirmed motion tests 2026-08-20 (prose
  in `docs/notes/yahboom_strategy_2026-08-20.md`, sections 12 and 16);
  vendor SLAM stack resource cost, stationary, 2026-09-16
  (`reports/slam/sessions/2026-09-16_stationary/`). These artifacts do not
  satisfy or advance D1 through D5 because they are not from the physical
  Synria/Alicia-D arm.

Implemented, unmeasured:
- The M3 Pro recording path: Orin ZMQ host (`m3pro_host.py`, launcher
  `scripts/jetson/m3pro_host.sh`) driven from the RTX 5090 by the keyboard
  recorder (`record_kbd.py`, venv `~/.venv/lerobot17`, lerobot 0.6.2,
  Python 3.12), writing a LeRobotDataset with `observation.images.front`,
  `observation.state` and `action`. Verified end to end 2026-08-20 (strategy
  note section 27), but is not the Synria recording path and cannot count
  toward D1. Zero qualifying Synria episodes are recorded.
- Safety floor in `hardware.py`: joint and relative clamps, base speed
  clamps, 500 ms deadman, zeros on exit and exception. The base firmware has
  no `/cmd_vel` timeout (section 12); this is M3-specific prior art, not a
  proven Synria safety control.
- Recovered physical Synria/Alicia-D serial connection, state publication,
  MoveIt integration, and real-arm-to-Isaac mirroring exist outside this
  repository in an uncommitted working tree. They are implemented,
  unmeasured and must be formalized with reproducible commands, provenance,
  safety checks and committed evidence before use as a stage dependency.
  The durable recovery audit is
  `reports/synria/recovered_real_arm_state_2026-09-17.md`.
- Recovered real-to-sim object interaction uses **GraspGlue**, a kinematic
  attach/release mechanism. It is not contact-rich grasp evidence, a physical
  object pickup, an autonomous policy result, or a D1 dataset.
- The Inspect Robots embodiment `rosmaster_m3pro` and scorer
  `m3pro_pickplace` (section 19) are M3-specific and cannot score Synria D2.
  A Synria-specific scorer and operator protocol remain to be formalized.
- Game logic: `ludo_engine`, `game_core` (commands, dice, adapters),
  `isaac/scripts/ludo_turn_executor.py`, `ludo_stats.py`.

Scaffold or planned:
- Synria 6DOF arm: this repository has simulation scaffolding but no committed
  physical bring-up evidence. Externally recovered physical bring-up exists
  as described above, but remains implemented, unmeasured until it is
  formalized here (`docs/ROADMAP.md`; `docs/SYNRIA_UPSTREAM_TODO.md`). Synria
  simulation policy training is blocked by asset defects: arm links without
  colliders, dead finger colliders, grasp gate 0/12
  (`docs/SYNRIA_GR00T_LEDGER.md`, "GRIPPER REPAIR CORRECTED FINDINGS";
  `isaac/scripts/synria_grasp_feasibility.py`).
- Physical board perception: none. No board-state recognition, token
  detection or physical die reader exists for a real camera. The physical
  roll pipeline exists only in Isaac Sim.
- Sim-to-real loop (Leg D): planned. Recovered Synria real-to-sim joint
  mirroring is implemented, unmeasured and does not establish sim-to-real
  transfer. Kinematic GraspGlue demonstrations do not validate contact
  physics. No Synria sim/real gap table exists.

Non-flagship device state, 2026-09-16 (retained as background only):
- Orin NX reflashed to the OEM image. `jetson` is the only account. Nothing
  is to be removed, stopped or cleaned up on it; additions under `jetson`
  are fine. The `m3pro-host.service` from August is gone; the host must be
  started with `scripts/jetson/m3pro_host.sh` or re-created as a service.
  The micro-ROS agent autostarts only on a desktop login. Repo clone at
  `/home/jetson/github/physical-ai-jetson-robotics`. 8.7 GB disk free.
  Details: `docs/rosmaster_m3pro/orin_discovery_2026-09-10.md`.
- RTX 5090: Isaac Sim venv `~/.venv/isaacsim5`, GR00T at `~/Isaac-GR00T`,
  lerobot client venv `~/.venv/lerobot17`, 454 GB free. This is a candidate
  Synria simulation/training host, not evidence of a physical result.
- Thor: GR00T and ACT policy servers from August may still be resident;
  not needed until D2 serving. Not a training box for this order.

## 3. Standing rules

1. Evidence commit rule: artifacts are committed when
   produced, with provenance. Raw datasets, videos and checkpoints stay out
   of git; summaries, manifests, hashes and EvalLogs go in.
2. No claim is upgraded in the README without an artifact carrying date and
   hardware. Simulation stays labelled simulation. A policy that completes
   a turn in sim has not played Ludo.
3. Safety on the physical Synria/Alicia-D arm: robot secured on its stand for
   any new motion primitive; e-stop within reach for every session; use only
   verified Synria joint/gripper limits and command units; require a measured
   timeout/deadman or equivalent stop path. The default exit and fault action
   is a controller-supported controlled stop/hold. Use torque-off only after
   the arm is in a verified support-safe pose, or when the manufacturer or
   controller emergency procedure explicitly requires it; an unsupported arm
   may fall when torque is removed. M3 `hardware.py` clamps are reference
   code, not Synria safety evidence. Record every incident in the session log.
4. Use the actual Synria controller and policy-serving hosts recorded during
   preflight. The Orin state above is M3 background and is not a default
   Synria deployment assumption. Thor and the 5090 remain candidate compute
   hosts. Do not run unrelated benchmarks on any device.
5. Lessons already paid for: before recording that an asset or a robot is
   defective, prove the tool can see it and the rig can move it (GR00T
   ledger, standing lesson). A single negative eval does not prove data
   scale is the cause (handoff 2026-09-07, item 3).
6. Which arm: D1 through D5 use only the physical Synria/Alicia-D arm. M3 Pro
   sessions and artifacts are non-qualifying background. Recovered Synria
   bring-up must be formalized under `docs/SYNRIA_UPSTREAM_TODO.md` before
   motion or recording; do not infer a safe or measured path from its having
   run once.

## 4. Documentation and recording, mandatory from the first session

The operator wants every activity and every policy decision recorded so
progress can be replayed as a timeline and used for research write-ups,
in the style of the existing ledgers (`docs/SYNRIA_GR00T_LEDGER.md`,
`docs/notes/ludo_game_milestones_2026-08-21.md`, per-run `provenance.json`).

- `docs/ludo_flagship/ACTIVITY_LOG.md`: append-only, one entry per working
  session and per notable event, UTC timestamp first, what was done, what
  was observed, what changed, links to the artifacts. Never rewritten.
- `docs/ludo_flagship/MILESTONES.md`: the D1 to D5 table with a dated entry
  when each is reached, the artifact path, and the numbers. Also
  intermediate milestones with dates (first episode recorded, first 10,
  first 50, first policy served, first real grasp, first scored turn).
- `docs/ludo_flagship/POLICY_LEDGER.md`: one row per trained policy
  version: date, data (dataset id, episode count, hash), recipe and
  hyperparameters, checkpoint hash, evaluation artifact, result, and the
  decision taken because of it. Negative results are rows, not omissions.
- `docs/ludo_flagship/DECISIONS.md`: dated decisions with the reason and the
  evidence they rested on, including decisions later reversed.
- Every recording session directory carries `provenance.json` (physical
  Synria/Alicia-D device identity, controller and host, git SHA, date,
  operator, power state at start and end, scene description, camera settings)
  and a `session_notes.md`; every evaluation writes an
  EvalLog and a frozen stats file through `ludo_stats.py` or the Inspect
  Robots scorer.
- Playback: keep a `docs/ludo_flagship/timeline.jsonl` with one JSON line
  per milestone and per session (`t_utc`, `kind`, `title`, `artifact`,
  `numbers`) so a page or chart can be generated from it without reading
  prose. Small media that illustrates a milestone (one still, one short
  clip under 10 MB) may be committed under `reports/ludo_flagship/media/`;
  raw video stays out.

Commit these with the work they describe, in the same commit.

## 5. Plan

### Phase 0A, read-only Synria recovery and safety artifact (no motion)

- Confirm the physical Synria/Alicia-D arm identity, controller/firmware,
  serial device by stable identifier, power and e-stop arrangement, that a
  physical Ludo board, tokens, die and cup are on hand and their dimensions,
  and what surface the arm works on. Do not assume M3/Orin wiring, battery
  telemetry, joint limits, command units or safety behavior applies.
- Formalize the recovered Synria serial/state/MoveIt bring-up in this
  repository, beginning with
  `reports/synria/recovered_real_arm_state_2026-09-17.md`. Inspect code,
  configuration and existing bags; identify the stable serial device without
  opening it; document controller-supported stop/hold, manufacturer emergency
  behavior, joint ordering, units, limits and a support-safe pose. Record
  device and host provenance. This phase is read-only and produces a committed
  safety/recovery artifact before any command-capable process is started.

### Phase 0B, operator-authorized reduced-speed motion evidence

- Proceed only with the operator physically present and explicitly authorizing
  motion after Phase 0A is accepted. With the arm secured and e-stop in reach,
  verify read-only state first, then reduced-speed joint limits, gripper
  mapping, timeout/deadman behavior, controlled stop/hold on every exit path,
  and a repeatable safe home. Torque-off is permitted only in a verified
  support-safe pose or under the manufacturer/controller emergency procedure.
  Commit `reports/synria/first_safe_motion.md` with commands, observations and
  operator authorization; history-reported motion is not this evidence.
- Formalize the Synria recording bridge. Run one disposable 20 s smoke
  episode and check synchronized timestamps, non-NaN joint state, physical
  camera frames, action ranges within verified Synria clamps, and correct
  gripper/finger dimensionality. Define and version the physical dataset's
  state/action contract from observed controller and recorder interfaces; do
  not assert a seven-to-eight mapping from history. Derive a passive second
  finger only if the selected model requires it and the derivation is
  documented and validated against the physical mechanism. Do not keep the
  smoke as D1 data.
- Measure the workspace: which board squares the arm reaches from its
  mount without the base moving. Record the reachable set; it defines
  which Ludo turns are executable in D3.
- Write the first `ACTIVITY_LOG.md` entry and the preflight `DECISIONS.md`
  entries. Commit.

### Phase 1, data collection to D1

- Task definition on the real board, written before recording: token from
  square A to square B, place upright within the square, gripper released,
  arm retracts. Success label rules the operator will apply per episode.
- Recording protocol: use the formalized Synria/Alicia-D teleoperation and
  recording bridge, episodes of 20 to 30 s, reset window, scene randomisation
  across the reachable squares, token colours, and lighting where possible.
  Record from a physical camera whose placement, calibration, resolution,
  rate and timestamps are captured in provenance; add a verified C10 wrist
  view if available. Do not assume the M3 `record_kbd.py`/Orbbec path is
  compatible.
- Quality gates per session, automated: frame count, NaN-free state, camera
  present, action within clamps, episode length; a session summary JSON per
  session committed under `reports/ludo_flagship/data/<session>/`.
- Milestones: 10 episodes (protocol shakedown), 50, 100. Report the operator
  success rate of the demonstrations themselves; demonstrations that fail
  are kept and labelled, not deleted.
- Object success ground truth: define a Synria physical-board observation
  path before collection. Unless an independent sensor is added, ground
  truth is the operator's label per episode plus a timestamp-linked still of
  the final physical board state from the verified camera, saved with the
  episode index. Robot joint state and GraspGlue state are not object-success
  ground truth. State this limitation in every artifact.

### Phase 2, training and real-arm evaluation to D2

- Recipes, in order of cost: ACT (use existing `serve_act_policy.py` and
  `eval_act_synria.py` only as patterns after their physical Synria interfaces
  are verified), SmolVLA fine-tune, then GR00T N1.7 LoRA with the
  target-observable lesson applied. One policy row in the POLICY_LEDGER per
  run. No recovered mirroring or GraspGlue run is a trained-policy result.
- Serving host is selected only after measuring the physical Synria control
  loop's compute, latency, network and stop-path requirements. Candidate
  hosts are the Synria controller host, Thor or the RTX 5090; do not assume
  the M3's Orin topology. Record inference latency, end-to-end command
  latency and power on the actual serving device.
- Evaluation: implement and validate a Synria/Alicia-D embodiment and
  pick-place scorer; do not use `rosmaster_m3pro` or `m3pro_pickplace` as a
  substitute. Pre-register and commit the 20-trial physical protocol before
  executing it, with scene randomisation and camera-backed operator success
  rules. D2 requires at least 14/20 operator-confirmed physical-object
  successes. Report first-try rate, validated funnel counts and independent
  operator success. Commit the EvalLog, scorer output, linked camera evidence
  and frozen stats file. Ledger all failures; they do not advance D2.
- The >=14/20 threshold is a pre-registered intermediate gate for repeatable
  majority performance before Ludo integration, not a final-production
  reliability claim. It may be changed only prospectively by an operator
  decision recorded before the affected evaluation begins; never re-gate an
  evaluation after seeing its results.
- Iterate on data and recipe with the ledger as the record. Stop conditions
  for a recipe: three consecutive evaluations without improvement, written
  up with a hypothesis before moving on.

### Phase 3, Ludo turns to D3 and D4

- Port `ludo_turn_executor.py`'s scoring and `turns.jsonl` to the physical
  Synria/Alicia-D arm:
  the engine plans the move, the executor converts squares to the recorded
  task frame, the policy executes, the operator grades, the scorer writes
  the turn record. Rolls come from the operator first. D3 advances only after
  one SUCCESSFUL camera-backed, operator-confirmed physical turn; an attempted
  or failed turn remains ledgered without stage advancement.
- Board and die perception for D4: choose the simplest thing that can be
  measured, for example printed fiducials on the board corners and
  colour-thresholded tokens from the front camera, and a die reader on a
  fixed dump pad. Each perception component gets its own accuracy
  measurement against operator truth before it is trusted in a turn.
- Physical roll: pick die, drop into cup, shake, invert, read. The sim
  routine (`docs/notes/ludo_game_milestones_2026-08-21.md`, B2b) is the
  reference for the sequence, not for the parameters.
- D4 advances only after three consecutive SUCCESSFUL camera-backed,
  operator-confirmed physical-roll turns. Failed attempts reset the
  consecutive count and remain in the ledger.

### Phase 4, the game to D5, then the loop

- Complete one successful full game to a winner on the physical board with the Synria/Alicia-D arm and physical
  rolls, every turn scored, misses ledgered,
  retries counted, `ludo_stats.py` aggregate committed. Incomplete or failed
  games remain ledgered and do not advance D5.
- Only after D5: the sim-to-real gap table (Leg D), comparing the same
  policy on the same turns in the Isaac twin and on the physical Synria arm, causes
  ledgered per miss. This is the research output; do not start it before
  real results exist.

## 6. Reporting

At the end of every session, append to `ACTIVITY_LOG.md` and post to the
operator: what ran, on which device, the artifact paths, the numbers, what
failed, and the next step. Use the labels measured, implemented unmeasured,
scaffold, planned exactly as the README does. When a result is negative,
say so in the first line.

## 7. Out of scope until D5

Rover SLAM runs, the portfolio site, the security, AI-RAN and
safety-observability repositories, Thor thread-pool branches, chess and
checkers, the two-embodiment headline game, Isaac ROS, any new
architecture. If a parked item blocks a stage, say which stage and why,
and do the minimum.
