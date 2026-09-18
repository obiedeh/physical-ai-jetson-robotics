# Codex work order: deliver the Ludo flagship on real hardware

Issued 2026-09-17. Codex owns this to completion. Do not wait for a Claude
review; report to the operator. Everything else in the portfolio is parked
(`AGENTS.md`, Focus Rule).

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
| D1 Real demonstrations | A committed dataset summary for 100 or more recorded teleoperation episodes on the real arm, with provenance per session (device, date, operator, scene, camera, battery), episode-level operator success labels, and quality-gate results. |
| D2 Learned pick-and-place on the real arm | A closed-loop evaluation of a trained policy on the real arm, 20 or more scored trials, success judged by the operator grader plus the trajectory funnel, committed as an EvalLog with the scorer output, including the negative results. |
| D3 Physical Ludo turn | One engine-planned turn executed by the policy on the physical board, scored, `turns.jsonl` committed, with the roll supplied by the operator. |
| D4 Physical roll in the loop | The die rolled and read physically, feeding `game.plan_turn(roll=value)`; three consecutive turns recorded. |
| D5 A game | A full game to a winner with physical rolls and policy-executed moves, every turn scored, misses ledgered, `ludo_stats.py` aggregate committed. |

Each stage moves its README status-table row only when its artifact is
committed, with date and hardware. Nothing is upgraded on a video.

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

Measured, real hardware:
- Yahboom ROSMASTER M3 Pro on Jetson Orin NX: day-one probes 2026-08-20;
  board contract and three operator-confirmed motion tests 2026-08-20 (prose
  in `docs/notes/yahboom_strategy_2026-08-20.md`, sections 12 and 16);
  vendor SLAM stack resource cost, stationary, 2026-09-16
  (`reports/slam/sessions/2026-09-16_stationary/`).

Implemented, unmeasured (code exists and ran once, no data collected):
- The real-arm recording path: Orin ZMQ host (`m3pro_host.py`, launcher
  `scripts/jetson/m3pro_host.sh`) driven from the RTX 5090 by the keyboard
  recorder (`record_kbd.py`, venv `~/.venv/lerobot17`, lerobot 0.6.2,
  Python 3.12), writing a LeRobotDataset with `observation.images.front`,
  `observation.state` and `action`. Verified end to end 2026-08-20 (strategy
  note section 27). Zero episodes recorded to date.
- Safety floor in `hardware.py`: joint and relative clamps, base speed
  clamps, 500 ms deadman, zeros on exit and exception. The base firmware has
  no `/cmd_vel` timeout (section 12); the deadman is the only protection.
- Leg A evaluation: Inspect Robots embodiment `rosmaster_m3pro` and scorer
  `m3pro_pickplace` (section 19). Success is the operator grader; the
  trajectory funnel is a labelled proxy, never ground truth.
- Game logic: `ludo_engine`, `game_core` (commands, dice, adapters),
  `isaac/scripts/ludo_turn_executor.py`, `ludo_stats.py`.

Scaffold or planned:
- Synria 6DOF arm: simulation only. The record has no physical Synria arm
  bring-up (`docs/ROADMAP.md` lists "Real Synria arm bring-up" as planned;
  `docs/SYNRIA_UPSTREAM_TODO.md` names `reports/synria/real_serial_bringup.md`
  as an output that does not exist). Synria simulation policy training is
  blocked by asset defects: arm links without colliders, dead finger
  colliders, grasp gate 0/12 (`docs/SYNRIA_GR00T_LEDGER.md`, "GRIPPER REPAIR
  CORRECTED FINDINGS"; `isaac/scripts/synria_grasp_feasibility.py`).
- Physical board perception: none. No board-state recognition, token
  detection or physical die reader exists for a real camera. The physical
  roll pipeline exists only in Isaac Sim.
- Sim-to-real loop (Leg D): planned. The M3 Pro Isaac twin is primitive
  geometry with dimensions listed as measurement debt; no gap table exists.
  What is in place is one direction only: real teleoperation to dataset.
  Treat "sim-to-real and real-to-sim wiring in place" as: recording path
  live, sim corpora and GR00T fine-tune and serve pipeline live, twin and
  gap ledger not started.

Device state, 2026-09-16:
- Orin NX reflashed to the OEM image. `jetson` is the only account. Nothing
  is to be removed, stopped or cleaned up on it; additions under `jetson`
  are fine. The `m3pro-host.service` from August is gone; the host must be
  started with `scripts/jetson/m3pro_host.sh` or re-created as a service.
  The micro-ROS agent autostarts only on a desktop login. Repo clone at
  `/home/jetson/github/physical-ai-jetson-robotics`. 8.7 GB disk free.
  Details: `docs/rosmaster_m3pro/orin_discovery_2026-09-10.md`.
- RTX 5090: Isaac Sim venv `~/.venv/isaacsim5`, GR00T at `~/Isaac-GR00T`,
  lerobot client venv `~/.venv/lerobot17`, 454 GB free.
- Thor: GR00T and ACT policy servers from August may still be resident;
  not needed until D2 serving. Not a training box for this order.

## 3. Standing rules

1. Evidence commit rule (`AGENTS.md`): artifacts are committed when
   produced, with provenance. Raw datasets, videos and checkpoints stay out
   of git; summaries, manifests, hashes and EvalLogs go in.
2. No claim is upgraded in the README without an artifact carrying date and
   hardware. Simulation stays labelled simulation. A policy that completes
   a turn in sim has not played Ludo.
3. Safety on the real arm: robot on the stand for any new motion primitive;
   e-stop within reach for every session; the host deadman on; zeros sent
   on every exit path; never bypass `hardware.py` clamps to "make a grasp
   reach". Record every incident in the session log.
4. Orin: OEM image, no removals. Thor and the 5090 as above. Do not run
   unrelated benchmarks on any device.
5. Lessons already paid for: before recording that an asset or a robot is
   defective, prove the tool can see it and the rig can move it (GR00T
   ledger, standing lesson). A single negative eval does not prove data
   scale is the cause (handoff 2026-09-07, item 3).
6. Which arm: the only arm with a verified real path is the M3 Pro. Start
   there. If the operator confirms a physical Synria arm is present and
   powered, open a separate bring-up lane for it under
   `docs/SYNRIA_UPSTREAM_TODO.md` item "real serial bring-up" and do not let
   it delay D1 on the M3 Pro.

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
- Every recording session directory carries `provenance.json` (device,
  git SHA, date, operator, battery at start and end, scene description,
  camera settings) and a `session_notes.md`; every evaluation writes an
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

### Phase 0, preflight (one session, no policy work)

- Confirm with the operator: which arm (M3 Pro assumed), whether a physical
  Synria arm is present and powered, that a physical Ludo board, tokens, die
  and cup are on hand and their dimensions, and what surface the arm works on.
- Orin: start the host with `scripts/jetson/m3pro_host.sh`; if it must
  survive reboots, add a systemd user or system service under `jetson`
  (an addition, allowed). Verify from the 5090: connect, camera frame,
  battery, home lands, deadman fires when the client stops.
- Re-run the recording smoke: one 20 s episode, check non-NaN state, camera
  frames present, action ranges within clamps. Do not keep it as data.
- Measure the workspace: which board squares the arm reaches from its
  mount without the base moving. Record the reachable set; it defines
  which Ludo turns are executable in D3.
- Write the first `ACTIVITY_LOG.md` entry and the preflight `DECISIONS.md`
  entries. Commit.

### Phase 1, data collection to D1

- Task definition on the real board, written before recording: token from
  square A to square B, place upright within the square, gripper released,
  arm retracts. Success label rules the operator will apply per episode.
- Recording protocol: `record_kbd.py` from the 5090, episodes of 20 to 30 s,
  reset window, scene randomisation across the reachable squares, token
  colours, and lighting where possible. Camera: the front Orbbec RGB;
  add the wrist view if a second camera is available.
- Quality gates per session, automated: frame count, NaN-free state, camera
  present, action within clamps, episode length; a session summary JSON per
  session committed under `reports/ludo_flagship/data/<session>/`.
- Milestones: 10 episodes (protocol shakedown), 50, 100. Report the operator
  success rate of the demonstrations themselves; demonstrations that fail
  are kept and labelled, not deleted.
- Object success ground truth: the board has no joint feedback or object
  sensor. Ground truth is the operator's label per episode plus a still of
  the final board state from the front camera, saved with the episode index.
  State this in every artifact.

### Phase 2, training and real-arm evaluation to D2

- Recipes, in order of cost: ACT (existing `serve_act_policy.py` and
  `eval_act_synria.py` patterns), SmolVLA fine-tune on the 5090, GR00T N1.7
  LoRA with the target-observable lesson applied. One policy row in the
  POLICY_LEDGER per run.
- Serving: on the Orin only if it fits 8 GB alongside the host; otherwise
  serve from Thor or the 5090 over the network and record the added
  latency. Measure and record inference latency and power on whichever
  device serves, using the existing harness patterns.
- Evaluation: Inspect Robots run with `--embodiment rosmaster_m3pro
  --scorer m3pro_pickplace --grader operator`, 20 or more trials, fixed
  protocol, scene randomised as in training. Report first-try rate, funnel
  counts and operator success. Commit the EvalLog and a frozen stats file.
- Iterate on data and recipe with the ledger as the record. Stop conditions
  for a recipe: three consecutive evaluations without improvement, written
  up with a hypothesis before moving on.

### Phase 3, Ludo turns to D3 and D4

- Port `ludo_turn_executor.py`'s scoring and `turns.jsonl` to the real arm:
  the engine plans the move, the executor converts squares to the recorded
  task frame, the policy executes, the operator grades, the scorer writes
  the turn record. Rolls come from the operator first (D3).
- Board and die perception for D4: choose the simplest thing that can be
  measured, for example printed fiducials on the board corners and
  colour-thresholded tokens from the front camera, and a die reader on a
  fixed dump pad. Each perception component gets its own accuracy
  measurement against operator truth before it is trusted in a turn.
- Physical roll: pick die, drop into cup, shake, invert, read. The sim
  routine (`docs/notes/ludo_game_milestones_2026-08-21.md`, B2b) is the
  reference for the sequence, not for the parameters.

### Phase 4, the game to D5, then the loop

- Full game with physical rolls, every turn scored, misses ledgered,
  retries counted, `ludo_stats.py` aggregate committed.
- Only after D5: the sim-to-real gap table (Leg D), comparing the same
  policy on the same turns in the Isaac twin and on the real arm, causes
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
