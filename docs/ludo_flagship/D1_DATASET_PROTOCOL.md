# D1 physical demonstration dataset protocol

Status: **implemented, unmeasured** recording and quality-gate path. D1 remains
**planned** until at least 100 qualifying physical Synria/Alicia-D episodes and
their committed summaries exist.

## Task

Each episode begins with one Ludo token upright in square A. The operator uses
the existing Alicia-D leader/follower teleoperation to pick up the token, move
it to square B, place it upright within the square boundary, release it, and
retract the follower arm clear of the board. The recording process observes
the existing teleoperation; it does not mediate or replace it.

In the operator-confirmed wiring, the leader drives the follower through a
hardware sync cable and is not connected to the PC. The PC observes only the
follower's `/joint_states`. The default `next_state` action is the following
sampled follower position/gripper state, a proxy rather than a directly read
leader command. This matches the operator-reported vendor recording convention
for this wiring; no physical recording or vendor runtime was tested here.

Direct `leader` actions are optional when the leader is additionally connected
to the PC by USB. Use the isolated read-only launch described in the
[runbook](D1_OPERATOR_RUNBOOK.md), explicitly select `--action-source leader`,
and keep its dataset separate. Existing hardware-sync teleoperation is unchanged.

Episodes target 20–30 seconds and stop at a hard 30-second cap. The C10 wrist
camera records `observation.images.wrist`; one fixed USB webcam records
`observation.images.front`. Every episode stores a timestamp-linked final
front-camera still.

## Operator labels

Mark `success` only when all of these are visually true in the final state:

1. The intended token moved from square A to square B.
2. The token is upright and entirely within square B.
3. The gripper released the token.
4. The follower arm retracted clear of the token and board.

Mark `failure` when any condition is false or uncertain. Use `discard` only
for a setup interruption, accidental key press, missing consent, or a recorder
fault that makes the episode unusable. Do not discard a genuine task failure.

Quality-valid failed demonstrations remain labeled and count toward the D1
episode total. Training selection may exclude them later, but it must not erase
them from the session record.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Scene randomisation

Pre-register square A and square B for every episode. Across a session, vary
reachable start and target squares, token colour, token orientation within the
start square, and lighting within safe operating bounds. Keep the board, fixed
camera, arm mount, and safety perimeter stationary. Record every reset or
deviation in `session_notes.md`.

## Dataset contract and gates

The contract is `synria_physical_v1`. Each dataset declares the gripper type
(`50mm` or `100mm`) and action source (default `next_state`, optional `leader`).
Limits in
[`config/synria_limits.yaml`](../../config/synria_limits.yaml) are candidate
values until `verified_by` and `verified_on` are filled by the operator. A
session may be quality-valid while limits are unverified, but it contributes
zero qualifying D1 episodes until that verification exists.

Every episode must pass frame-count, finite-value, dual-camera presence,
state/action limit, timestamp-skew, duration, gripper-dimensionality, and
visual-sanity gates. The visual gate rejects black, frozen, and duplicated
camera streams. Smoke episodes are disposable and never count.
