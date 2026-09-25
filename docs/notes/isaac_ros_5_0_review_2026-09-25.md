# Isaac ROS 5.0 — Review Handoff Note

Date: 2026-09-25. Author: Claude (session review). Reviewer: Codex.
Context: Operator asked Claude for a first-pass read on the Isaac ROS 5.0
release announcement (ROS 2 Lyrical + Ubuntu 24.04 + FoundationStereo
fine-tuning + FoundationPose 5.5× speedup + Thor/Orin support). Codex
returned a stricter counter-review. This note documents both, records
Claude's acknowledged misses, and lists what Codex should implement (if
anything) versus what stays parked under the AGENTS focus rule.

Nothing in this note has been committed to code. It is a review artifact.

---

## Sources

- [Isaac ROS 5.0 announcement](https://nvidianews.nvidia.com/) (post shared by operator)
- [Isaac ROS getting-started guide](https://nvidia-isaac-ros.github.io/getting_started/index.html) — supported environment matrix
- [Isaac ROS 5.0 release notes](https://nvidia-isaac-ros.github.io/) — NITROS-package removal
- [`docs/STACK_WATCHLIST.md`](../STACK_WATCHLIST.md) — pre-existing gating for FoundationPose / cuMotion adoption
- [`AGENTS.md`](../../AGENTS.md) focus rule (parked until autonomous real-arm pick-and-place artifact exists)
- Operator-supplied Codex review (2026-09-25, 11:07)

---

## Isaac ROS 5.0 — what actually changed

- ROS 2 Lyrical support (Ubuntu 24.04).
- JetPack 7.2 target on Jetson.
- FoundationPose inference ~5.5× faster.
- FoundationStereo fine-tuning workflow.
- Old direct NITROS packages removed; replaced by standard ROS messages
  backed by `rosidl::Buffer` for zero-copy GPU transport between
  colocated nodes.
- GPU partitioning (compute, not memory or safety domains).
- Named agent-assisted setup/manipulation/perception workflows.

---

## Claude's first-pass read

Position: "lands right on top of what we've been building" — FoundationPose
into a new `physical_ai_lab/perception/` module, FoundationStereo as a
bonus for Yahboom's binocular depth cam, three doc additions proposed
(CLAUDE_SKILLS index, HARDWARE Jetson pointer, VENDOR_INTEGRATION_MAP
CAD-to-SimReady note, GR00T-FINETUNE Phase 3 LeRobot-viz callout).

Underlying assumption: Isaac ROS 5.0 is additive and can be wired in
opportunistically. No migration timing risk called out.

---

## Codex's counter-review

Codex agreed on the high-level "helps but do not migrate now" position
but pushed back on specifics.

**Where Codex tightened the analysis:**

- FoundationPose requires **RGB-D + detection box + CAD mesh**. The Synria
  documented setup is monocular RGB (C10). FoundationPose does not plug
  in as-is.
- The NITROS-package removal is a real API break for downstream repos,
  even though it does not affect ours today (no direct NITROS integration
  present).
- FoundationStereo has no immediate benefit — the documented Synria
  setup does not currently include a stereo camera. Yahboom's binocular
  unit is not guaranteed to satisfy FoundationStereo's baseline /
  calibration assumptions.
- Isaac ROS 5.0 officially targets ROS 2 Lyrical, Ubuntu 24.04, JetPack 7.2,
  with 128+ GB NVMe recommended. Current repo state:
  - RTX workstation: ROS 2 Jazzy
  - Orin NX: ROS 2 Humble + JetPack 6.2, limited free disk
  - Synria vendor packages: Humble-oriented
  - In-place upgrade of the Orin NX is not a safe target.
- Concrete 6-step isolated Thor spike plan: separate container/workspace,
  FoundationPose benchmark vs recorded + synthetic frames, pose error and
  latency for cup/die/tokens, then a standard-ROS interface into the
  existing arm-control path, cuMotion only after physical robot model
  and controlled-stop are proven, FoundationStereo skipped.
- `docs/STACK_WATCHLIST.md` already gates FoundationPose and cuMotion
  behind real-hardware adoption. Codex's recommendation preserves that.

---

## Claude's acknowledged misses

Three specific corrections Codex was right about:

1. **FoundationPose RGB-D requirement.** Claude claimed FoundationPose
   slots into board-state perception. It does not without depth. Adoption
   requires either adding a depth sensor to the arm, an intermediate
   RGB-only depth estimator, or a different perception path that uses
   Isaac Sim ground-truth depth during training and something else on
   real hardware.
2. **NITROS API removal in 5.0.** Not called out in Claude's first pass.
   No effect today; important for any future Thor-side node written
   against the wrong interface.
3. **`docs/STACK_WATCHLIST.md` already exists.** Claude's proposed
   "Isaac ROS 5.0 tracking note in `docs/JETSON_DEPLOYMENT.md`" duplicates
   pre-existing gating.

---

## Agreed recommendation

Both reviewers agree on the following:

- Do not migrate the main robot-control stack now.
- Complete D1 (Synria bring-up) and D2 (real demonstrations + first
  learned real-arm pick-and-place result) on the current
  Humble/Jazzy/JetPack 6.2 paths.
- After D1/D2, run an isolated Isaac ROS 5.0 spike on Thor:
  1. Separate container or workspace, not an in-place upgrade.
  2. Benchmark FoundationPose against recorded Ludo-board frames and
     synthetic frames from the Isaac scene. Only meaningful if / after
     the arm has a depth path.
  3. Measure pose error and latency for cup, die, and tokens.
  4. Feed validated perception across a standard ROS interface into the
     existing arm-control path.
  5. Evaluate cuMotion only after the physical robot model, collision
     geometry, limits, and controlled-stop behavior are proven.
  6. Skip FoundationStereo until calibrated stereo hardware is deliberately
     adopted.
- Keep the existing `docs/STACK_WATCHLIST.md` gating as the source of
  truth for perception/motion adoption.

---

## Focus-rule verdict on the four docs edits Claude proposed

Under the `AGENTS.md` "everything parked until D1/D2 artifact exists"
rule, three of the four fail the "serves the deliverable" test.

| Proposed edit | Serves D1/D2? | Verdict |
|---|---|---|
| `docs/CLAUDE_SKILLS.md` (new file — skill index) | No — reference material | Do not commit now |
| `docs/HARDWARE.md` Jetson Setup Checklist → skill pointer | No — cosmetic | Do not commit now |
| `docs/VENDOR_INTEGRATION_MAP.md` `omniverse-cad-to-simready` note | No — vendor material was removed 2026-09-07; speculative until re-import happens | Do not commit now |
| `docs/SYNRIA_GR00T_FINETUNE.md` Phase 3 `i4h-lerobot-viz` callout | Marginal yes — dataset sanity before GR00T-Mimic amplification is on the D2 critical path | Optional; small; single-file |

Operator has not authorized any of the four yet. Default is "stop entirely".
The one edit that survives on merit is the Phase 3 LeRobot-viz callout.

---

## What Codex should consider adding

None of the below are asks; they are candidates Codex can review against
the focus rule and take or reject.

1. **Verify `docs/STACK_WATCHLIST.md` covers the FoundationPose RGB-D
   requirement** as a prerequisite bullet, not just a "wait for real
   hardware" gate. If it does not, one line spelling out the depth
   dependency avoids a future re-litigation.
2. **Record the NITROS API removal** in `STACK_WATCHLIST.md` so any
   future Thor-side ROS work targets `rosidl::Buffer`-backed messages
   rather than the deprecated interface. Even a one-line "adopt Isaac
   ROS 5.0 `rosidl::Buffer` transport for zero-copy" note is enough.
3. **Phase 3 LeRobot-viz callout** — if Codex agrees the sanity check
   is on-critical-path for D2, land it as a minimal edit against the
   current `docs/SYNRIA_GR00T_FINETUNE.md`. Note that the current file
   has `SYNRIA_EMBODIMENT_TAG = "new_embodiment"` and camera keys
   `wrist`/`overhead` (G3 migration), which Claude's earlier draft did
   not have. Any re-application needs to match the current text.
4. **Isaac ROS 5.0 environment matrix** captured somewhere in
   `docs/JETSON_DEPLOYMENT.md` or `docs/STACK_WATCHLIST.md` so the
   "Orin NX cannot in-place upgrade" fact is written down before someone
   attempts it. Codex's counter-review already contains the text.

Everything else in Claude's first pass — the CLAUDE_SKILLS index, the
HARDWARE.md pointer, the CAD-to-SimReady note in
VENDOR_INTEGRATION_MAP — should be treated as parked side quests until
the D1/D2 artifact exists.

---

## Handoff

Operator will pass this file to Codex. Codex owns any resulting commit.
Claude is not planning follow-up edits without further instruction.
