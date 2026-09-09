# Product roadmap — board-game robot (ludo -> chess -> checkers)

## Now: P2 sim MVP (in bring-up)
Game engine -> TurnPlan commands -> E33 scripted expert executes moves
physically in sim. First successful move: 2026-08-17 (red, 4 squares,
18 mm placement error).

## Lodged: THE DICE RITUAL (signature feature — operator spec 2026-08-17)
Full physical dice roll, human-style:
1. Gripper picks the die, puts it in the cup.
2. Gripper grasps the cup and SHAKES it: two simultaneous motions —
   up/down and left/right — composed into a kinematic circular motion
   about a point axis (orbital stir).
3. Arm carries the cup to the edge of the table and FLIPS it to throw
   the die onto the table.
4. Perception reads the pips (Gemini ER-2 phase-gate or FoundationPose
   local); the read roll feeds plan_turn(roll=...) — the plumbing
   already accepts an external roll.
Why: no one demos this; it turns a utility motion into personality.
Build notes: shake = sinusoidal joint-space composition (z + y at 90°
phase offset -> circular orbit of the wrist point); flip = wrist-roll
joint past the cup's pour angle at the table edge; die must settle
inside the overhead camera's readable zone — add a low-walled tray
area if scatter is unreliable. Sim first (die physics exists in the
ludo scene), hardware later.

## Then
- Multi-token physical board (16 tokens in scene, per-token prims)
- Dice-read perception benchmark (pip counting, G-series methodology)
- TRT-served GR00T as the executor's policy option (engines built)
- Chess MVP on the same executor (adapter is done and tested)
- P4: real hardware per readiness gates (docs/HARDWARE.md)
