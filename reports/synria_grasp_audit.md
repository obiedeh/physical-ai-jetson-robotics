# Synria gripper vs cup — grasp feasibility audit (2026-07-31)

Mission: verify whether the simulated cup can be physically grasped by
the Synria gripper. Measure everything from assets; assume nothing.
Reproduce with `isaac/scripts/synria_grasp_feasibility.py` (modes:
production / g30 / nopad / padfix; --free_air, --no_self_collision,
--hover_test). Debug visualization:
`reports/synria_gripper_clearance.png`. Trial JSONs:
`reports/synria_grasp_trials_*.json`, `reports/synria_grasp_hover_*.json`.

## Current cup

- Asset: primitive `sim_utils.CylinderCfg` in
  `isaac/isaaclab_tasks/synria_pickplace/env_cfg.py` ("piece").
- Dimensions: radius 0.02 m -> **max outside diameter 40.0 mm**,
  height 50 mm (`task_geometry.py: PIECE_RADIUS_M/PIECE_HEIGHT_M`).
- Taper: none (straight cylinder) — diameter at any grasp height 40.0 mm.
- Collision geometry: the primitive itself (PhysX convex of the
  cylinder); visual == collision; **no scale applied** anywhere.
- Mass 50 g; material static/dynamic friction 1.3/1.1, combine "max".

## Synria gripper (measured, not quoted)

- Sources: `isaac/usd/robots/synria_6dof_arm.urdf` (joints),
  `isaac/usd/robots/meshes/Alicia_D_v5_6/follower_standard/
  {left,right}_gripper_50mm.STL` (geometry),
  `isaac/usd/robots/synria_6dof_arm_v2/synria_6dof_arm.usd` (production).
- Prismatic fingers: left 0..+25 mm, right -25..0 mm, effort 5 N,
  velocity 12 m/s. **Confirmed in-sim**: joint_pos_limits report
  [0, 25] / [-25, 0] mm. No mimic joints, no transmissions, no scale.
- Finger frames sit ±25.5 mm from the centerline; STL inner pad faces
  are 0.5 mm inside the frames -> **max physical fingertip separation /
  usable inside clearance = 50.0 mm at joint 0** (matches the vendor's
  "50mm" file naming), closing to 0.0 mm at full travel.
- Finger: 43.3 mm wide x 60 mm long x 28.5 mm deep; flat inner pad face,
  contact area ~1800 mm^2. Convex hull of the finger contains 36%
  phantom volume (concavities filled) — relevant to any hull-approximated
  variant.
- Commanded grasp width used by control code:
  `GRIPPER_GRASP_WIDTH_M = 0.045` (approach at 45 mm).
- Sim vs hardware: **no vendor datasheet exists** (established G30 —
  even the vendor MJCF carries placeholder values). The "50mm" product
  naming matches the measured 50.0 mm opening; hardware verification of
  the real jaw opening remains an open item.

## Deterministic scripted trials (fixed pose, no policy)

| configuration | trials | result | finger stall (travel) |
|---|---|---|---|
| production asset | 10 | 0 lifted | 0.4 / 0.3 mm |
| g30 (convex-decomposition) | 4 | 0 lifted | 0.4 / 0.3 mm |
| production, self-collision off | 5 | 0 lifted | 0.4 / 0.3 mm |
| production, FREE AIR (cup 1 m away) | 2 | — | **0.4 / 0.3 mm** |
| nopad (static pads deactivated) | 7 | 0 lifted | **25 / 25 mm — closes through the cup, zero contact** |
| padfix (re-authored pads, 1st attempt) | 10+10 hover | 0 genuinely held | 0.5–5.7 mm (jam persists; see defect 5) |

Hover-test note: the recorded 10/10 "held" is a threshold artifact — the
constant 47.3 mm drop equals the fall to the ground; the cup was NOT
held. Corrected criterion goes into the next script revision.

## Root cause chain (each step measured)

1. **The arm has NO collision geometry at all.** Only 4 collision prims
   exist in the entire robot USD, all on the two fingers. (The elbow can
   pass through the table unimpeded; the v2 import comment claiming
   "convex-hull colliders on all 9 links" is wrong.)
2. **The finger MESH colliders are dead** — corrupt `extent` attributes;
   with the static pads removed the fingers close 25 mm straight
   through the cup with zero contact.
3. **The only live colliders were two hand-authored `pad_collider`
   boxes — and they are static**: inner faces at -24.6/+25.4 mm never
   move with the joints. The fingers' travel stops at 0.4/0.3 mm — the
   exact distance to those faces (matched to 0.1 mm) — in every
   configuration including free air.
   **This immovable phantom cage is the months-long "grip wall": no
   training run ever closed the gripper onto anything.**
4. Harness findings while testing (documented for reproducibility): the
   60 mm finger length means gripper-frame z must be ~61 mm for
   fingertips at ground level — the RL env's grasp-height constants
   should be audited for the same frame-vs-fingertip offset.
5. First re-authoring attempt (`synria_6dof_arm_padfix.usd`:
   deactivate static pads, add 4 mm boxes parented to the finger
   bodies) produces moving colliders that DO contact — but still jams
   at ~1 mm: the box orientation in the link's local frame is wrong
   (authored from URDF-frame assumptions; must be derived from the
   USD's composed link transforms). Fix incomplete; direction proven.

## Feasibility verdict

- **Production asset: IMPOSSIBLE — 0/10 scripted grasps; the gripper
  cannot close.** Not a size problem; an asset-integrity problem.
- Nominal geometry once fixed: 50.0 mm opening vs 40.0 mm cup =
  **10 mm total clearance, 5 mm per side** — the Marginal/Acceptable
  boundary; well under the 20 mm comfort margin once perception
  (±4 mm measured), approach drift, spawn variation (±80 mm), and
  contact compression are budgeted.
- As-controlled (45 mm approach width): 5 mm total / 2.5 mm per side —
  **Marginal**.

## Maximum recommended cup diameter

| criterion | diameter |
|---|---|
| absolute maximum (fits at nominal max opening) | < 50 mm |
| with 5 mm clearance per side | 40 mm |
| with 10 mm clearance per side | 30 mm |
| **recommended (opening − 20 mm)** | **30 mm** |
| practical target range | **25–30 mm** |

The 20 mm margin is appropriate here (not conservative): measured
perception error ±4 mm, spawn range ±80 mm, no closed-loop recovery
behavior, and sim-to-real gap with an unverified hardware opening.
Suggested next-run object: **28 mm diameter x 60 mm tall** cylinder
(extra height keeps the grasp band away from the ground and the
fingertip-depth issue).

## Required changes before Synria policy training continues

1. Re-author the finger pad colliders correctly parented to
   `left_gripper`/`right_gripper` bodies using USD-composed transforms
   (complete the `padfix` work; generator:
   `isaac/scripts/make_padfix_usd.py`).
2. Fix or replace the corrupt finger mesh collision prims.
3. Add collision geometry to the arm links (base..link6).
4. Wire the FIXED asset into `env_cfg.py` (currently loads the broken
   `synria_6dof_arm.usd`; the g30 file was never adopted either).
5. Replace the cup with a 25–30 mm object (suggest 28 x 60 mm).
6. Audit RL grasp-height constants for the 60 mm frame-vs-fingertip
   offset.
7. Gate: `synria_grasp_feasibility.py` must pass >= 9/10 scripted
   grasps (with a corrected held-criterion) before any policy training.

## Decision rule status

**Synria policy training: BLOCKED.** None of the three unblock
conditions is met yet; condition 3 (gripper geometry changed and
validated) is in progress via the padfix asset.

---

# AMENDMENT (2026-07-31 evening) — earlier root-cause claim CORRECTED

The repair mission (task #16) produced measurements that **refute the
"static phantom pad colliders" conclusion above**. Recorded here rather
than edited away, per the no-hidden-failures rule.

## What the repair work measured

Phase 1 (composition map, `reports/gripper_phase1_composition.json`):
- Both `pad_collider` cubes are authored **under the moving finger
  links** (`left_gripper` / `right_gripper`), in link-local coordinates
  `(0, ∓30, ∓13.75) mm` — i.e. **correctly parented**, not static in the
  world. Their composed world AABB equals the visual finger AABB.
- Finger prismatic joints: axis Z, body0 `link6`, body1 the finger link,
  limits [0, 25] / [-25, 0] mm. No mimic, no scale, no transmission.
- Only 4 collision prims exist in the whole robot (2 mesh + 2 pad boxes).
  **The arm links genuinely have no collision geometry** (unchanged).
- The finger MESH collision prims report degenerate/infinite extents
  (±3.4e38) — still considered non-functional (unchanged).

Phase 4 (no-object travel, 10 cycles, 0/25/50/75/100%/0):

| asset | base at floor | base raised 0.5 m |
|---|---|---|
| production (vendor pads + meshes) | FAIL (erratic 15–25 mm) | **PASS (25.00 mm both fingers, all cycles)** |
| nopad (all finger colliders off) | PASS | — |
| padfix / padfix_l / padfix2 (re-authored pads) | FAIL | **PASS** |
| padfix_l with NO ground and NO cup | **PASS** | — |

**Corrected root cause of every "jam" in the audit above:** the test
harness drove the arm with the URDF's real gains (5 Nm, stiffness 80),
which cannot hold the commanded pose against gravity. The arm sagged to
the floor and the fingers closed against the **ground plane**. Remove
the ground (or raise the base, or stiffen the harness gains) and travel
is textbook. The 0.4/0.3 mm "matched to 0.1 mm" stall was floor contact,
not a phantom cage — a coincidence I over-read.

## The defect that remains (new, unresolved)

Phase 5 (known-width blocks, elevated base, blocks free-standing on a
kinematic pedestal, verified positioned between the pads —
`block_off_from_mid_mm = [0.9, 0.0, -29.8]`, i.e. x/y aligned within
1 mm at the intended pad-center height):

| asset | 20 mm | 28 mm | 30 mm | 40 mm |
|---|---|---|---|---|
| production | 0/3 | 0/3 | 0/3 | 0/3 |
| g30 | 0/3 | 0/3 | 0/3 | 0/3 |
| padfix2 | 0/3 | 0/3 | 0/3 | 0/3 |

In every trial the fingers travelled the **full 25 mm per side** and the
measured jaw width closed to **0.01–0.03 mm** — the pads pass straight
through a correctly-placed block of any width, with no contact and no
block displacement.

**Current best hypothesis:** none of the finger collision prims
participate in contacts with dynamic rigid bodies at all — the mesh
colliders because their extents are degenerate, and the pad cubes for a
reason not yet identified (candidate causes: scaled-`Cube`-under-
articulation-link cooking, missing/ineffective collision approximation
attributes, or Isaac Lab's spawn path not re-applying the collision API
on referenced layers). This is consistent with every historical
observation — including "the gripper closes through the cup" and the
project-long grip wall — but it is a hypothesis, not yet a measurement.

## Consequence for the earlier verdicts

- "Cup too large" remains **not** the problem: geometry (50.0 mm opening
  vs 40 mm cup) is marginal-acceptable and the 25–30 mm recommendation
  stands as good practice, not as the fix.
- The gripper's **kinematics are healthy**; its **contact behaviour is
  not**.
- Synria policy training remains **BLOCKED** and the acceptance gate is
  unchanged.

---

# SECOND AMENDMENT (2026-07-31, late) — "no contact" was also a test bug

A runtime probe of the **spawned** articulation (not the USD on disk)
settled the remaining question:

```
SIM collider: /World/Robot/left_gripper/collisions/.../node_STL_BINARY_  enabled=False
SIM collider: /World/Robot/left_gripper/pad_collider                     enabled=True
SIM collider: /World/Robot/right_gripper/collisions/.../node_STL_BINARY_ enabled=False
SIM collider: /World/Robot/right_gripper/pad_collider                    enabled=True
```

- The finger **mesh** colliders are `enabled=False` in the live scene —
  genuinely dead, consistent with their degenerate extents. They are
  redundant, not fatal.
- The **pad box colliders are enabled and present**.

Why every test still reported "closes straight through": the pad
collider's world AABB spans roughly **−15 mm to +58 mm in z relative to
the finger frame** — the finger extends *diagonally* from its mount, not
straight down. Every block/hover/grasp test placed its object at
"frame − 30 mm", i.e. **below the collider entirely**. The objects were
never between the pads.

With placement corrected to the true pad centre (computed from the body
pose and the collider's link-local offset, `pad_center_world()` in the
gate script), the fingers **stall on contact** instead of closing
through: travel drops from the full 25 mm to 0–6 mm for a block at pad
height.

## Corrected status of the Synria gripper

| claim | status |
|---|---|
| "static phantom pad colliders block travel" | **refuted** (harness arm-sag onto the floor) |
| "no finger collider contacts dynamic bodies" | **refuted** (test placement bug; pads do contact) |
| finger mesh colliders dead | **confirmed** (`enabled=False`), redundant |
| arm links have no collision geometry | **confirmed** — still a real defect |
| 50.0 mm opening, 40 mm cup = 5 mm/side | **confirmed**; 25–30 mm object still recommended |

**Phase 5 is not yet passed**: with contact established, the measured
jaw widths are still not trustworthy because the arm drifts under
contact load at these gains and the pedestal fixture re-places the block
between trials. The next step is a rigid fixture — object pose fixed
relative to the *pads* and re-measured immediately before the close —
not further asset edits.

## What this means for the project history

Two of this session's own test-harness bugs (an under-powered arm that
sagged onto the floor, then objects placed outside the collider)
produced two confident-but-wrong "the asset is broken" conclusions. The
Synria gripper's kinematics and pad collision are in better shape than
either earlier conclusion claimed. The still-open items are: arm-link
colliders, a trustworthy width fixture, and — most likely the real
control-side bug — **grasp poses that target the finger frame instead of
the pad centre**, which should be audited in both the RL and GR00T
environments.

---

# THIRD AMENDMENT (2026-08-04) — Phase 5 PASSES; the pedestal was the jam

The previous section closed with "Phase 5 is not yet passed ... the
measured jaw widths are still not trustworthy because the arm drifts
under contact load". That diagnosis named the right remedy (a rigid
fixture) but the wrong dominant cause. Rebuilt, the gauge passes
**12/12** on the unmodified production asset.

## The actual cause of the 0/12

The old harness stood each block **free** on a 120 x 120 x 20 mm
kinematic pedestal whose top was placed at `pad_center_z - 25 mm`. But
the pad collider is a 43.3 x 60 x 28.5 mm box mounted **diagonally**
(local x and y both map into world x/z at 45 deg), so it spans roughly
**73 mm in world z** — from `pad_center_z - 36.5 mm` to
`pad_center_z + 36.5 mm`.

The pedestal top therefore sat about **11 mm inside the pad colliders**,
and at 120 mm square it could not miss them in xy. The fingers were
jamming on *the pedestal*. This is visible in the old data and was
misread at the time: with a 20 mm block the fingers reported **0.06 mm**
of travel and a 49.86 mm "measured width" — they never moved at all, so
there was nothing for arm drift to explain.

Arm drift was real but secondary: the block world-x varied 103.6 →
127.5 mm across trials because the arm settled differently each time.

## The fixture that replaced it

1. **No pedestal.** Blocks are kinematic, so they need no support and
   cannot be knocked off-centre. A kinematic body is still a fully
   contactful collider — exactly what a width gauge needs.
2. **The arm is locked**: `Joint1-6` state is written every physics
   step, so contact load cannot move the pads.
3. **The pad centre is re-measured immediately before every close** and
   the block re-pinned to it; the offset is recorded again at stall, so
   rigidity is *proven per trial* rather than assumed.
4. **A no-block control close runs first** and must reach full travel.
   If the rig cannot close on empty air it exits 5 rather than emitting
   any width. This directly encodes the standing lesson.

## Results (`reports/synria_blocks_production.json`)

Rig self-test, no block, arm locked: **L 24.99 / R 24.99 mm** — full
travel, so every stall below is caused by the block.

| block | expected travel/side | measured L / R | width err | closing-axis centring |
|---|---|---|---|---|
| 20 mm | 15.00 mm | 14.97 / 14.95 | +0.08 mm | 0.00 mm |
| 28 mm | 11.00 mm | 10.97 / 10.94 | +0.09 mm | +0.08 mm |
| 30 mm | 10.00 mm | 9.97 / 9.94 | +0.09 mm | +0.07 mm |
| 40 mm | 5.00 mm | 4.99 / 4.99 | +0.02 mm | -0.08 mm |

Worst block-vs-pad drift across all 12 trials: **0.08 mm** (was ~21 mm).

**Why this is not a fourth false positive.** Four different widths
produce four *distinct* stall values, each matching the geometric
prediction to within 0.09 mm. A jam, a phantom collider or a pinned-object
artefact would all yield width-*independent* numbers. The measurement is
differential, and the empty-air control bounds it from the other side.

## Scope limit — state this honestly

The fixture blocks are **kinematic**, so this validates **contact
geometry and finger kinematics only**. It measures nothing about
friction, holding force, or whether a *free* object survives a lift.
That is Phase 9's job (the 10-trial grasp gate), which remains unrun.

## Corrected phase status

Phase 5 (known-width block validation): **PASSED, 12/12** on
`synria_6dof_arm.usd`, including the 40 mm cup stress case, which now
also passes as a contact-geometry matter.

The Synria gripper's opening, travel, symmetry and pad contact geometry
are all now measured-correct on the vendor asset. **Every "the asset is
broken" verdict this project produced has now been traced to the test
harness** — three for three. The remaining known asset defect is the
missing arm-link collision geometry (Phase 6).

---

# FOURTH AMENDMENT (2026-08-04) — the 30 mm grasp-pose bug, and Phase 9 PASSES

The Third Amendment closed by naming the remaining suspect: "grasp poses
that target the finger frame instead of the pad centre". That suspect is
now **confirmed, measured, and fixed**, and with it the Phase 9 gate
passes 10/10 on the unmodified production asset.

## The control-side bug (confirmed, 30.0 mm)

`mdp._grasp_center_local` carried the docstring *"midpoint of the two
finger pads"*. It computed the midpoint of the two finger **link
frames**:

```python
pads = robot.data.body_pos_w[:, [ids["left_body"], ids["right_body"]], :]
return pads.mean(dim=1) - env.scene.env_origins
```

Those are not the same point. Measured from the USD:

| | world x (mm) | world y | world z |
|---|---|---|---|
| finger-frame midpoint (what the code returned) | 313.37 | 0.35 | 107.33 |
| true pad-collider midpoint | 292.16 | 0.35 | 128.55 |
| **error** | **-21.21** | 0.00 | **+21.22** |

**30.0 mm** in the approach plane. The y component cancels, so the frame
midpoint is a correct *closing-axis* centre — which is why this survived
so long — but it is wrong in x and z by more than the entire clearance
budget of the 40 mm cup (5 mm per side).

Consequence: driving the reported grasp centre onto the piece parks the
real pads 21.2 mm above and 21.2 mm behind it. **The policy was being
paid to reach a pose from which the piece is not between the pads.**
This was a *residual* of an earlier fix — the flange-to-frame correction
(10-13 cm) was applied and the remaining 30 mm was never noticed.

Fixed by transforming the measured pad offsets by each finger's world
rotation (`task_geometry.PAD_LOCAL_OFFSET_M`), the same computation the
Phase 5 gauge uses. The identical bug was fixed in the gate script's own
trial loop, which had been placing the cup at the frame midpoint and
descending to a frame height — leaving the pads ~57 mm above the cup.
That was the whole of the earlier 0/10.

## Phase 9 gate: 10/10 (`reports/synria_grasp_trials_production.json`)

Cup 40 mm x 50 mm on a 50 mm work surface, pad centre descending to
surface+45 mm, +/-3 mm per-trial placement jitter, scripted only:

| trial | jitter (mm) | stall L/R (mm) | closed width | lifted | slipped |
|---|---|---|---|---|---|
| 0 | +2.03, -2.21 | 2.8 / 7.2 | 40.0 mm | yes | no |
| 1 | -2.99, +0.26 | 5.3 / 4.7 | 40.0 mm | yes | no |
| 2 | +2.38, +1.83 | 6.8 / 3.2 | 40.0 mm | yes | no |
| 3 | -0.52, -2.95 | 2.1 / 7.9 | 40.0 mm | yes | no |
| 4 | -1.61, +2.53 | 7.5 / 2.5 | 40.0 mm | yes | no |
| 5 | +2.90, -0.78 | 4.2 / 5.8 | 40.0 mm | yes | no |
| 6 | -2.66, -1.38 | 3.6 / 6.4 | 40.0 mm | yes | no |
| 7 | +1.03, +2.82 | 7.8 / 2.2 | 40.0 mm | yes | no |
| 8 | +1.14, -2.77 | 2.2 / 7.8 | 40.0 mm | yes | no |
| 9 | -2.72, +1.27 | 6.3 / 3.7 | 40.0 mm | yes | no |

The jitter direction decides which finger travels further, and the
**sum is 10.0 mm in every trial** — closed width exactly the 40 mm cup
diameter. That internal consistency is the evidence the grasp is real:
a jam or a phantom contact would not track the offset.

Against the brief's acceptance criterion: >= 9/10 (**10/10**), pad
clearance above the work surface 22.0 mm (no phantom collision, no
penetration), no unexplained jam (width = cup diameter every trial), and
no false success from the object resting on a surface (the cup rises
75 -> 205 mm and is still gripped at ~5 mm travel after transport).

## Three more rig bugs found on the way — the pattern held

Getting here required fixing three further *harness* faults, none of them
asset defects. The count of false "the asset is broken" verdicts caused
by this project's own test rigs is now **six for six**.

1. **The pedestal inside the pads** (Phase 5, Third Amendment).
2. **A false POSITIVE**: the first corrected gate reported 10/10, but
   trial 0 had closed to 49.8 mm — fingers effectively open — scooped the
   cup on the way up and passed on `lifted` alone. `success` never checked
   that the fingers closed on anything. Now gated on closed width.
3. **A false NEGATIVE**: driving the arm kinematically (the Phase 5 fix)
   writes joint *velocity* as zero every step, so the solver saw
   stationary pads, generated no tangential friction impulse, and left a
   correctly-gripped cup behind — `lifted=False` on 10/10 trials whose
   grip was textbook. The rig now teleports only to *place* the hand and
   hands over to real PD dynamics for hold/lift/transport, where friction
   must do the work.

Also measured and now asserted rather than guessed: **the pad collider
extends 36.1 mm below its own centre**. Any work surface higher than
`pad_centre - 36.1 mm` is inside the pads. That single number explains
the pedestal jam, a 100 mm table that intersected the gripper's own rest
pose, and the floor contact at cup mid-height.

## Standing lesson, restated

Every false verdict in this project has come from the test rig. Two of
this session's three were *tuning artefacts in the pass criterion
itself* — one too weak (scoop scored as success), one too strong (a
correct grasp scored as failure). Verify the rig, then verify the
criterion, and only then believe the number.

## Phase status after this session

| # | Phase | State |
|---|---|---|
| 1-4 | inspect / colliders / travel | Done or moot (unchanged) |
| 5 | Known-width block validation | **PASSED 12/12** |
| 6 | Arm-link collision geometry | **Not started** — still the one real asset defect |
| 7 | Wire into production env + startup assert | Not started |
| 8 | 28 mm validation object + grasp-frame audit | **Grasp-frame audit DONE** (the 30 mm bug); 28 mm object still to wire in |
| 9 | Deterministic grasp gate | **PASSED 10/10** |

---

# FIFTH AMENDMENT (2026-08-04) — Phase 6 is MOOT; the arm colliders were always live

The last standing asset defect was: *"Arm links still have no collision
geometry — a real, unfixed defect (only 4 collision prims exist in the
whole robot, all on the fingers)."* That is **wrong**, and it was wrong
for the same reason as every other false verdict in this project: the
tool, not the asset.

## Why the audit saw only 4 colliders

The production wrapper (`synria_6dof_arm.usd`) sets `instanceable = false`
on **only** the two gripper subtrees, to author the pad cubes and disable
the redundant gripper mesh colliders. Every arm link therefore remains
`instanceable = True`, so its children live in a USD *prototype*. A plain
`Usd.Stage.Traverse()` does not descend into instance proxies, so the
audit's probe saw the gripper colliders (de-instanced, hence visible) and
nothing else — and concluded the arm had none.

Traversing with `Usd.TraverseInstanceProxies()` finds **11** colliders,
not 4:

| body | collider | enabled |
|---|---|---|
| base_link, link1 … link6 | `<link>/collisions/<link>/node_STL_BINARY_` | **ON** (7 of them) |
| left/right_gripper | `collisions/..._50mm/node_STL_BINARY_` | off (redundant, as documented) |
| left/right_gripper | `pad_collider` | **ON** |

## Runtime proof, not just USD

USD metadata claiming `collisionEnabled=True` is not evidence that PhysX
cooked the shapes, so `isaac/scripts/probe_arm_colliders.py` proves it
physically: command Joint2 through +0.9 rad with a 1000 kg kinematic slab
in the arm's path, against a control run with the slab parked 5 m away.

| run | Joint2 commanded | reached |
|---|---|---|
| control (slab 5 m away) | +0.900 rad | **+0.840 rad** |
| obstacle (slab in the swept volume) | +0.900 rad | **+0.045 rad** |

The arm stalls dead on the obstacle. The colliders are live.

**The first version of this probe reported the opposite** — "PASSED
THROUGH", with the blocked run travelling *further* than the control,
which is incoherent. The slab had been placed at a guessed x, and link2
turns out not to translate at all during a Joint2 sweep (it is on the
rotation axis: 0.0 mm of travel). The probe now *measures* each link's
swept volume first and places the obstacle at the mid-sweep point of the
link that actually moves (link3, 195 mm). Guessing where the geometry is,
instead of measuring it, is the identical mistake that put the Phase 5
pedestal inside the pads and the Phase 9 table inside the gripper — three
times in one session.

## Consequence

**Phase 6 is MOOT — no work required.** With Phases 4, 5 and 9 passing
and the arm colliders verified live, **the Synria asset has no known
remaining defect.** Every one of the seven "the asset is broken" findings
this project produced has now been traced to its own tooling:

1. static phantom pad colliders (harness arm-sag onto the floor)
2. no finger collider contacts dynamic bodies (object placement bug)
3. Phase 5 block jam (pedestal inside the pads)
4. Phase 9 scoop scored as success (criterion too weak)
5. Phase 9 correct grasps scored as failure (zeroed joint velocity)
6. a 100 mm table intersecting the gripper's own rest pose
7. arm links have no collision geometry (traversal missed instance proxies)

**Standing lesson, final form:** before recording that an asset is
defective, prove the tool can see it and prove the rig can move it. In
this project that check would have caught seven findings out of seven.
