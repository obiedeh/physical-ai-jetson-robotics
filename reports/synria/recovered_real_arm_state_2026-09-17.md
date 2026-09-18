# Recovered physical Synria/Alicia-D state

Audit date: 2026-09-18 UTC. This is a read-only recovery record, not a motion
test, dataset artifact, safety acceptance or delivery-stage result.

## Source checkout and provenance

- External checkout inspected: `/home/oedeh/github/Alicia-D-Isaac-Teleop`.
- Checked-out HEAD: `ad203a91e05081c7e2a11d6a2b83a221d7538d07`.
- `/home/oedeh/github/Alicia-D-ROS2` is a symbolic link to
  `Alicia-D-Isaac-Teleop`; it is not a separate checkout.
- The external checkout is dirty. Tracked modifications are
  `debug_isaac/alicia_d_live_scene.usd`,
  `scripts/isaac_sim/alicia_d_live_scene.py`, and
  `scripts/run_alicia_stack.sh`. The untracked file is
  `debug_isaac/pick_scene_state.json`.
- Nothing in that dirty tree was copied, edited, committed or treated as
  reproducible evidence by this audit.

## Facts established by code and file inspection

- The checkout contains driver/state paths that publish `/joint_states`,
  scripts defaulting to `/dev/ttyACM1`, MoveIt/IK utilities, and a live Isaac
  scene that subscribes to `/joint_states`.
- `scripts/isaac_sim/alicia_d_live_scene.py` contains `GraspGlue` and a
  `/pick_objects_offset` subscription. GraspGlue is a kinematic simulation
  attach/release aid; it does not prove contact-rich or physical grasping.
- Existing ROS bag metadata was inspected for two MoveIt range tests and one
  hand-guided range session. The 18.80 s and 144.29 s MoveIt bags report 447
  and 3,606 `/joint_states` messages; the 241.34 s hand-guided bag reports
  11,794 `/joint_states` messages. This confirms recorded state traffic in
  those historical sessions, not current calibration, safe motion, object
  success, autonomy or a D1 dataset. Bag payload semantics were not replayed.
- `debug_isaac/pick_scene_state.json` stores the token at approximately
  `[-40.134, -8.365, 0.012]` metres. That implausible position makes this
  saved scene state invalid as a trustworthy recovery input.
- At audit time no `/dev/ttyACM*` device and no Alicia/Synria driver, Isaac,
  or micro-ROS process were observed. Current connectivity and runtime health
  are therefore unestablished.

## Runtime claims reported by operator/history, not re-proven here

Prior history reports that the physical arm connected on `/dev/ttyACM1`,
passed self-checks, published live joint states, mirrored motion into Isaac,
and drove simulated GraspGlue cycles for a token, die and cup, including a
sustained simulated cup carry. Those claims are useful recovery leads. They
are not promoted to measured facts here because this audit did not reproduce
the runtime, and the relevant working tree is dirty and uncommitted.

## Explicit non-claims

- No current device connection or safe command path is established.
- No physical object was shown picked or placed.
- No camera-backed physical-object success evidence exists here.
- No autonomous policy execution is established.
- No qualifying D1 recording or physical dataset state/action contract exists.
- No D1-D5 delivery stage is reached.

## Required next evidence

First, commit a read-only Phase-0 recovery/safety artifact that documents the
controller's supported controlled stop/hold, manufacturer emergency procedure,
units, ordering, limits and support-safe pose. Only after operator presence and
explicit motion authorization may reduced-speed motion be attempted and logged
in `reports/synria/first_safe_motion.md`. Controlled stop/hold is the default;
torque-off is used only in a verified support-safe pose or when the documented
manufacturer/controller emergency procedure requires it.
