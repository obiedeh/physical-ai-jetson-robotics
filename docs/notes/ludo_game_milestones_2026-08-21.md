# Ludo — "robot plays a real game" milestones (2026-08-21)

Pivot from the failing learned-policy thread (GR00T eval 0-5%) to building the
game on the parts that WORK: `ludo_engine` (rules) + `game_core.LudoGameAdapter`
(move→arm commands, accepts external `roll`) + the 97.2% scripted expert.

## Ladder
- **A. Scripted game** ✅ REAL. `ludo_soak_s9`: 12-turn 4-player game, 91.7%
  first-try. Video: `reports/ludo_soak_s9_gameplay.mp4`.
- **A+. Complete game to a winner** ✅ DONE. `--endgame --seed 0` (red@39 vs
  green@40): 15 turns, red CAPTURES green (T3, both cmds ok) then races home to
  WIN (T15). 7 commands, 5/7 first-try (T1,T7 missed; game still completes).
  Rendered at 640px via `SYNRIA_CAMERA_RES=640`. Video:
  `reports/ludo_endgame_demo_640.mp4`. All squares reach-verified
  (0.23<hypot(x+0.22,y)<0.52) incl. capture-return to yard.
  - Repro: `SYNRIA_CAMERA_RES=640 ~/.venv/isaacsim5/bin/python
    isaac/scripts/ludo_turn_executor.py --headless --endgame --seed 0
    --commands 40 --out reports/ludo_endgame_demo`
  - The endgame setup + winner report + cam-res override committed (e58ddaa).
- **B. Physical dice** 🔨 NEXT. Scene already has a die + cup; adapter already
  takes `plan_turn(roll=value)`. Build scripted skills: pick die → drop in cup →
  roll (shake cup) → dump on table → READ value by overhead camera → feed to the
  engine. This makes rolls physical (goal #2) and is the novel core.
- **C. Opponent.** Sim self-play works (engine plays all colors). Human = a
  turn-input interface.
- **D. Learned GR00T policy** ⏸ parked (documented negative result:
  groot17_eval01_findings). The hard upgrade, not the road to results.

## Notes
- Physical model: ONE cup teleported per move represents the current token; the
  arm pick-carry-places it pick_xy→place_xy. Token/player count is engine-side
  only (no scene change) — that's why a 2-player short game needed no new assets.
- Reach dead-zone: ~6 of 52 track squares (near base edge, e.g. track 11) are
  out of reach; blue's home-entry sits there → avoid blue for reach-critical
  demos. red/green endgame stays fully reachable.

## B progress (2026-08-21)
- **B1 DONE — physical die you can roll & read (fairly):**
  - `game_core/dice.py` — die value from rest orientation (top face->pip,
    opposite faces sum 7, settled-gating). Invariant tests 4/4 (`test_dice.py`).
  - `recorder_env_cfg.add_die` — opt-in 18mm cube RigidObject (training env
    untouched). Tuned physics: angular_damping 0.03, restitution 0.35.
  - `ludo_turn_executor --roll-test N` — throw die N times, read each rest value.
    Verified FAIR: 36 throws {1:10,2:6,3:3,4:4,5:8,6:5}, chi2=5.7 (<11.07,
    uniform); 35/36 settle by ~61 steps. Reads all 1-6.
- **B2 NEXT — the manipulation:** pick die -> drop in a CUP (need an open-cup
  container asset) -> shake/roll in cup -> invert to dump on table -> read
  (B1) -> feed `plan_turn(roll=value)`. Hardest part: cup container asset +
  grasping the 18mm die + in-cup agitation + dump motion.

## B2 progress (2026-08-21)
- **B2a DONE — cup container:** `isaac/usd/props/dice_cup.usda` (trimesh->USD,
  convexDecomposition collision), `add_cup` helper, `--cup-test`. Die drops in
  and STAYS: 12/12 contained. Sized graspable: 42mm outer (gripper opens ~50mm),
  32mm inner (holds the 18mm die). Regenerate: `make_dice_cup.py`.
- **B2b NEXT — arm manipulation (the hard frontier):**
  1. arm grasps the 18mm die -> moves over cup -> releases (drop-in). Reuses the
     top-down scripted-expert grasp, retargeted to die + cup xy.
  2. arm grasps the cup (42mm) -> shake trajectory (die tumbles inside) ->
     INVERT ~180deg over the table to dump. **The inversion is a NEW motion the
     pick-place expert doesn't do** (it only does top-down grasp/lift/place) —
     this is a real manipulation-controller build, not a quick adapt.
  3. read dumped die (B1) -> `plan_turn(roll=value)`.
- Constraint found: gripper max span ~50mm (fingers +-25mm) -> cup outer must be
  <=50mm; die 18mm graspable. Narrower cup = slower die settle (give more steps
  or a settling shake before reading).

## B2b resume plan (arm manipulation) — primitives identified
Build the die->cup->shake->dump routine on the executor's existing primitives
(do NOT reinvent IK):
- `dls_step(target_pos, hq, phase, ori_w)` — DLS IK to a grasp-center target
  `target_pos` with target orientation quat `hq`; returns clamped joint dq.
- `dls_pos_step(target_pos)` — position-only IK.
- `mdp._grasp_center_local(env)` — current grasp center; `robot.data.body_quat_w`.
- action: `action[:,:6]=arm_tgt-default_arm`, `action[:,6]=grip_cmd`
  (`OPEN_CMD=1.0`, `CLOSE_CMD=-1.0`), applied via `wrapper.step(action)`.
- proven grasp geometry (execute_command ~L570): SIDE-pick — stand off 60mm on
  the base-side of the object, descend open, slide in horizontally, close.

Step sequence (verify a demo at each):
1. **die pick + drop-in-cup:** side-pick the die at its xy (may need a tighter
   close for the 18mm die vs the 40mm cup); lift; move grasp-center above cup
   centre + ~5cm; OPEN -> die falls in (containment already proven 12/12).
2. **cup grasp + shake:** side-pick the cup (42mm, fits ~50mm gripper); lift a
   few cm; oscillate wrist/xy a few cycles (die tumbles inside — randomises).
3. **invert-dump (NEW motion):** rotate wrist ~180deg over an empty table spot so
   the cup mouth points down; die pours out; set cup back down.
4. **read + feed:** `die_value_up(die.data.root_quat_w[0])` when `settled(...)`
   -> `game.plan_turn(roll=value)` -> the A+ game loop plays with PHYSICAL rolls.

Status 2026-08-21: B1 (fair die) + B2a (cup contains die) DONE & verified.
B2b is an iterative manipulation build (each step = a few Isaac tuning cycles);
the novel piece is the inversion (step 3) — the pick-place expert never rotates
the wrist past top-down. Next session: implement step 1 as a `--dice-in` mode
and tune the die grasp.

## B2b step 1 — first crack (2026-08-21): built & runs, grasp needs tuning
`--dice-in N` mode added (arm picks die -> drops in cup). Runs clean; die
CONTAINMENT check reused. Result so far: **0/3 — the 18mm die is not grasped.**
Diagnosed failure modes (grasp-center vs die, from `[dicein] at-die` prints):
- **ori-hold servo (dls_step, ori_w=0.8):** stalls ~68mm short of the die — the
  orientation-hold fights the reach (the documented DLS stalemate).
- **position-only (dls_pos_step):** overshoots ~118mm, wrist uncontrolled.
=> A hand-rolled top-down servo is the wrong approach. **Next iteration: reuse
the executor's PROVEN side-pick** (execute_command ~L570: standoff 60mm base-side
-> descend open -> slide in horizontally -> close), retargeted to the die at its
xy and a tighter close for 18mm. That grasp is already tuned for the cup; adapt
its geometry/close for the die rather than re-deriving IK. Then lift -> over cup
-> open (drop-in). This is iterative manipulation tuning (several Isaac cycles).

## B2b step1 — iteration on level_dq (2026-08-21): 28mm reach, close but not capturing
Switched the die grasp to the expert's own `level_dq(target, up=[0,0,1], phase)`
(servo position + keep gripper LEVEL, yaw free). Frame capture added
(`dump_frame`). Video: `reports/dice_in_attempt.mp4`.
- Result: reach err 68-118mm (hand-rolled servo) -> **28mm** (level_dq); gripper
  closes 40->22mm. Die NOT lifted (die_z unchanged) -> gripper closes ~beside the
  18mm die (28mm > the ~9mm needed to straddle it).
- Next lever(s) to close the 28mm: (a) final position-only refinement step after
  the level_dq descend (dls_pos_step for the last cm); (b) verify GZ (die grasp
  height = table+12mm) vs where the fingers actually meet; (c) confirm
  `_grasp_center_local` offset; (d) if still short, a genuine side-slide (standoff
  60mm base-side -> horizontal slide-in like the cup expert). Iterative; each
  cycle ~2min Isaac.

## B2b COMPLETE (2026-08-24) — full physical dice roll works end-to-end
The die→cup→shake→invert-dump→read sequence runs and reads a valid settled value.
Video: `reports/ludo_dice_roll.mp4` (rolled=1, settled=True, align=1.00).

Key implementation decisions (in `--dice-in N [--dump]`):
- **Die pickup via kinematic glue** (not friction grip). The level-IK (`level_dq`)
  stalls ~28-41mm short of any target (DLS stalemate — confirmed: relocating the
  die to an easier radius made gc_err *worse*, 28→41mm, so it's not reach). So the
  die is pinned to the grasp center on close and carried glued — the SAME
  convention the accepted A/A+ token/cup carry already uses.
- **Cup grab + shake + invert-dump:** the cup is glued to the gripper's actual
  pose (position + orientation-since-grab), so a real ~180° wrist roll
  (`servo_ori` → `dls_step` to an inverted target quat) pours the die out. The
  die stays a free rigid body and tumbles/pours via physics.
- **Settled read:** die has low angular damping (B1 fairness), so it spins a long
  time and can land edge-balanced. Fix: bleed excess spin during settle + re-drop
  from 4cm on a cocked landing (honest "re-roll a cocked die"). Reader
  `die_value_up`/`settled`/`face_alignment` from `game_core/dice.py`.

Infra notes (5090): Isaac boot intermittently crashes with a `carb::tasking`
recursion assertion when stale `/dev/shm/carb-*` segments accumulate from killed
runs — clear them before each launch. Long runs (>120s, auto-backgrounded) are
sometimes reaped; `SYNRIA_NO_FRAMES=1` makes validation runs finish in-foreground,
and `SYNRIA_FRAME_STRIDE=N` decimates frames to shorten the video run.

Repro: `SYNRIA_FRAME_STRIDE=2 SYNRIA_CAMERA_RES=640 PYTHONPATH=<repo>
~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py --headless
--dice-in 1 --dump --out reports/dice_roll_final`

## Remaining: C — physical rolls feeding a real game
Wire the dump-read value into `game.plan_turn(roll=value)` so a full game is
played with physically-rolled dice (adds die+cup to the game scene, rolls each
turn). The primitive is done; this is the integration step.

## C DONE (2026-08-24) — physical rolls drive a real game
`--physical-rolls N` adds die+cup to the game scene and, each turn, PHYSICALLY
rolls (die->cup->shake->invert-dump->read via the B2b routine) and feeds the read
value to the engine as `game.plan_turn(roll=value)`; the planned move is then
executed by the same A/A+ pick-place machinery. Proven end-to-end:

    [ludo] PHYSICAL ROLL = 1
    [ludo] TURN 1: red rolls 1: token 0 ('track',39)->('track',40), captures [('green',0)]

i.e. the physically-rolled die (value 1) made the engine advance red and CAPTURE
green — a real, consequential game move produced by a physical roll. The full
pipeline is: pick/carry/place tokens (A/A+) + physical dice roll (B2b) + engine
plays to a winner (A+), now joined so rolls are physical.

Repro (frames-off validation, fast): `SYNRIA_NO_FRAMES=1 SYNRIA_CAMERA_RES=224
PYTHONPATH=<repo> ~/.venv/isaacsim5/bin/python isaac/scripts/ludo_turn_executor.py
--headless --endgame --physical-rolls 1 --commands 3 --seed 0 --out reports/pg`

Note: a full multi-turn physical-roll game is a long run; on the 5090 the Isaac
process is intermittently reaped past ~120s, so multi-turn frames videos need
retries. Each component video is delivered separately (ludo_dice_roll.mp4 for the
physical roll; ludo_endgame_demo_640.mp4 for the full game to a winner).

## Ludo project status: COMPLETE
- A  scripted game (video) ✅
- A+ full game to a winner (video) ✅
- B1 fair physical die + reader ✅
- B2a cup contains die ✅
- B2b full physical roll: pick die->cup->shake->invert-dump->read (video) ✅
- C  physical rolls feed the engine -> real game move (capture) ✅

## C+ (2026-08-27) — integrated multi-turn physical-roll game VIDEO
`reports/ludo_physical_game.mp4` (4m29s @30fps, 640px, stride 2, 8057 frames):
three physical rolls (1, 1, 2) drive a real game — TURN 1 red 39->40 captures
green (first attempt MISS 45mm, built-in retry OK 20mm), TURN 2 green rolls 1
with no legal move (skipped), TURN 3 red 40->42. `game session done: 3 turns,
3 commands`, all commands OK in `reports/ludo_live_02/turns.jsonl`.

Root cause of the earlier "Isaac reaped past ~120s" note: the GUI-mode run
(`ludo_live_01`) died at 306s with `[omni.physx.plugin] Subscription cannot be
changed during the event call` -> carb `Mutex ... Recursion not allowed`. This
is a UI/timeline re-entrancy, not a process reap. Headless runs the full 27 min
without incident. Always run integrated games with `--headless`.

Launcher (retries on crash, clears stale carb shm): `scripts/run_ludo_live.sh
reports/<out>` (env: ROLLS=3 CMDS=6 SEED=0 MAX=3).
