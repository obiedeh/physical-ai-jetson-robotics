"""The one implementation of "where are the finger pads".

Why this module exists
----------------------
The pad-centre computation is the single most error-prone quantity in this
project, and it has been got wrong three times:

1. The approach reward measured from ``tool0``, the wrist flange, which sits
   10-13 cm past the grasp pose. The policy learned to drive the flange onto
   the piece and stall at contact.
2. The fix moved it to the midpoint of the two finger LINK FRAMES and called
   that "the midpoint of the pads". It is not: the fingers extend diagonally,
   so each pad collider centre sits 21.2 mm behind and 21.2 mm above its link
   frame -- a 30.0 mm residual, which is larger than the entire 5 mm-per-side
   clearance budget for the 40 mm cup. Every experiment up to and including
   E22 trained against a reward that paid for a pose the arm cannot grasp
   from.
3. A round of block/hover tests assumed "frame minus 30 mm in world z", which
   lands outside the collider entirely, and had to be thrown out.

Each of those cost days. The through-line is not carelessness -- it is that
the same six-line ``quat_apply`` loop was copy-pasted into four call sites
across two files, so "fixing" it meant finding all four, and the constant it
reads was duplicated into a second private copy that the shared one could not
keep in sync.

So this module owns both the constant and the math, and :func:`verify_against_stage`
checks them against the USD at runtime rather than trusting that the numbers
someone typed still describe the asset. Import from here; do not re-derive.

The offsets are link-LOCAL and must be rotated by the finger's world
orientation before use -- that rotation is exactly what mistake (3) skipped.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from isaac.isaaclab_tasks.synria_pickplace.task_geometry import PAD_LOCAL_OFFSET_M

if TYPE_CHECKING:
    import torch

#: Body names of the two finger links in the articulation.
FINGER_BODY_NAMES: dict[str, str] = {"left": "left_gripper", "right": "right_gripper"}

#: USD prim sub-path of the pad collider under each finger link.
PAD_COLLIDER_SUBPATH: str = "pad_collider"

#: Full extents of a pad collider box (m), from the USD ``xformOp:scale`` on a
#: unit Cube. The pads are mounted DIAGONALLY, so these do not align with world
#: axes and the vertical span they sweep is larger than any single extent --
#: which is why :func:`pad_lowest_z_world` exists rather than callers reaching
#: for ``PAD_HALF_EXTENTS_M[2]``.
PAD_EXTENTS_M: tuple[float, float, float] = (0.043314368, 0.060000002, 0.0285)
PAD_HALF_EXTENTS_M: tuple[float, float, float] = tuple(  # type: ignore[assignment]
    e / 2.0 for e in PAD_EXTENTS_M
)

#: How far the USD may drift from PAD_LOCAL_OFFSET_M before we refuse to run.
#: 0.5 mm: tight enough to catch a real asset edit, loose enough to absorb
#: float32 round-tripping through USD.
PAD_OFFSET_TOLERANCE_M: float = 5e-4


#: Distance between the finger-frame midpoint and the true pad midpoint, in
#: the rest pose: the exact error mistake (2) above baked into every experiment
#: through E22. NOT derivable from PAD_LOCAL_OFFSET_M alone -- the offsets are
#: link-local mirror images that would average to zero, and the residual only
#: appears once each is rotated by its own finger's world orientation. The
#: fingers are mounted such that both local +y components map to the SAME world
#: direction, so world x and z reinforce (-21.21, +21.22 mm) while world y
#: cancels. Pinned against the measured USD in tests/test_gripper_geometry.py.
FRAME_TO_PAD_MIDPOINT_RESIDUAL_M: float = 0.030003

#: Same quantity per finger, before the world-y cancellation.
FRAME_TO_PAD_PER_SIDE_M: float = 0.033003


def pad_centers_world(
    robot: Any,
    body_index: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(left, right)`` world-frame pad collider centres, each ``(num_envs, 3)``.

    ``robot`` is an Isaac Lab ``Articulation``. ``body_index`` optionally
    supplies pre-resolved body indices as ``{"left": i, "right": j}`` -- pass it
    on hot paths (the MDP resolves them once and caches them on the env) and
    omit it in scripts, where the name lookup is not worth threading through.

    Points are in WORLD coordinates. Callers comparing against env-local task
    geometry must subtract ``env.scene.env_origins`` themselves; that is
    deliberately not done here, because the feasibility harness runs a single
    env at the world origin and would be silently wrong either way round if
    this module guessed which convention the caller wanted.

    Use this when you need the two pads separately -- the grasp axis, the
    per-pad clearance checks. Use :func:`pad_center_world` for the midpoint.
    """
    import torch as _torch
    from isaaclab.utils.math import quat_apply  # type: ignore[import-not-found]

    centers = []
    for side in ("left", "right"):
        if body_index is not None:
            bidx = body_index[side]
        else:
            bidx = robot.body_names.index(FINGER_BODY_NAMES[side])
        pos = robot.data.body_pos_w[:, bidx, :]      # (N, 3)
        quat = robot.data.body_quat_w[:, bidx, :]    # (N, 4) w-first
        local = _torch.tensor(
            PAD_LOCAL_OFFSET_M[side], device=pos.device, dtype=pos.dtype
        ).expand(pos.shape[0], 3)
        centers.append(pos + quat_apply(quat, local))
    return centers[0], centers[1]


def pad_center_world(
    robot: Any,
    body_index: dict[str, int] | None = None,
) -> torch.Tensor:
    """``(num_envs, 3)`` world-frame midpoint of the two pad collider centres.

    This is the grasp reference point: the place the piece must end up for the
    fingers to close around it. See :func:`pad_centers_world` for the frame
    convention and the ``body_index`` fast path.
    """
    left, right = pad_centers_world(robot, body_index)
    return 0.5 * (left + right)


def pad_lowest_z_world(
    robot: Any,
    body_index: dict[str, int] | None = None,
) -> torch.Tensor:
    """``(num_envs,)`` world Z of the lowest point of either pad collider.

    This — not the finger frame, and not the pad centre — is what decides how
    close the hand can approach a work surface. The pads are 43.3 x 60 x
    28.5 mm mounted diagonally, so they sweep a vertical span larger than any
    single extent, and the bottom sits well below the pad centre at a typical
    grasp pose. Guessing it is how the Phase 5 pedestal and the table both
    ended up geometrically inside the pads, producing block tests that scored
    0/12 for reasons that had nothing to do with the gripper.

    Computed by projecting each half-extent axis through the finger's world
    rotation and summing the absolute Z contributions, which bounds the rotated
    box exactly at its lowest corner.
    """
    import torch as _torch
    from isaaclab.utils.math import quat_apply  # type: ignore[import-not-found]

    centers = pad_centers_world(robot, body_index)
    lows = []
    for side_idx, side in enumerate(("left", "right")):
        if body_index is not None:
            bidx = body_index[side]
        else:
            bidx = robot.body_names.index(FINGER_BODY_NAMES[side])
        quat = robot.data.body_quat_w[:, bidx, :]          # (N, 4)
        center = centers[side_idx]                          # (N, 3)
        z_span = _torch.zeros(center.shape[0], device=center.device, dtype=center.dtype)
        for axis in range(3):
            vec = _torch.zeros(3, device=center.device, dtype=center.dtype)
            vec[axis] = PAD_HALF_EXTENTS_M[axis]
            z_span = z_span + quat_apply(quat, vec.expand(center.shape[0], 3))[:, 2].abs()
        lows.append(center[:, 2] - z_span)
    return _torch.minimum(lows[0], lows[1])


def verify_env_geometry(env: Any, *, strict: bool = True) -> bool:
    """Verify a live Isaac Lab env's pad colliders against the constants.

    Returns True when the check actually ran and passed. Returns False, after
    printing why, when the scene could not be inspected -- a missing stage, an
    unexpected prim layout, a build of Isaac without USD introspection. It
    never reports success it did not earn, because a check that quietly
    verifies nothing is worse than no check: it makes the next person trust
    numbers nobody validated.

    Raises ``ValueError`` on a genuine mismatch when ``strict``.
    """
    try:
        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
    except Exception as exc:  # noqa: BLE001 - any import/stage failure is the same outcome
        print(
            f"[gripper_geometry] pad geometry UNVERIFIED (no USD stage: {exc}). "
            "Reward and grasp gates are running on unchecked constants.",
            flush=True,
        )
        return False

    # Env 0 is representative: all envs clone the same articulation asset.
    # By the time the scene is built, Isaac Lab has already rewritten
    # "{ENV_REGEX_NS}" into the regex form "/World/envs/env_.*", so both
    # spellings have to be collapsed to a concrete path.
    robot_prim_path = env.scene["robot"].cfg.prim_path
    robot_prim_path = robot_prim_path.replace("{ENV_REGEX_NS}", f"{env.scene.env_ns}/env_0")
    robot_prim_path = re.sub(r"/env_\.\*(?=/|$)", "/env_0", robot_prim_path)
    try:
        found = verify_against_stage(stage, robot_prim_path, strict=strict)
    except LookupError as exc:
        print(
            f"[gripper_geometry] pad geometry UNVERIFIED ({exc}). "
            "Reward and grasp gates are running on unchecked constants.",
            flush=True,
        )
        return False

    print(
        f"[gripper_geometry] pad colliders verified against USD "
        f"({len(found)}/2 sides, tolerance {PAD_OFFSET_TOLERANCE_M * 1000:.1f} mm) "
        f"at {robot_prim_path}",
        flush=True,
    )
    return True


def verify_against_stage(
    stage: Any,
    robot_prim_path: str,
    *,
    strict: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Check :data:`PAD_LOCAL_OFFSET_M` against the pad colliders in the USD.

    Returns the offsets actually found on the stage, keyed by side.

    Raises ``ValueError`` when the stage disagrees by more than
    :data:`PAD_OFFSET_TOLERANCE_M` and ``strict`` is set. Failing the run is the
    point: a silent mismatch here does not crash, it trains for two hours
    against geometry that does not describe the robot, and the result is
    indistinguishable from a bad reward design. That is precisely how the 30 mm
    error survived E17 through E22 while five separate diagnoses were written
    for its symptoms.

    Raises ``LookupError`` when no pad prims are found at all. That is NOT
    treated as "nothing to check": a wrong prim path would silently verify
    nothing while reporting success, which is the same failure mode this
    function exists to remove. The caller decides whether an unverifiable
    scene is fatal.
    """
    from pxr import Usd  # type: ignore[import-not-found]

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise LookupError(f"no prim at {robot_prim_path}")

    # The finger links are not necessarily direct children: the arm USD may
    # nest them under the referenced asset's own root. Walk instead of
    # assuming a depth, and include instance proxies -- with thousands of
    # cloned envs the descendants are instanced, and the default traversal
    # skips straight past them.
    descendants = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
    }

    found: dict[str, tuple[float, float, float]] = {}
    mismatches: list[str] = []

    for side, body_name in FINGER_BODY_NAMES.items():
        finger = descendants.get(body_name)
        if finger is None:
            continue
        prim = stage.GetPrimAtPath(
            finger.GetPath().AppendChild(PAD_COLLIDER_SUBPATH)
        )
        if not prim or not prim.IsValid():
            continue
        attr = prim.GetAttribute("xformOp:translate")
        if not attr or not attr.HasAuthoredValue():
            continue
        actual = tuple(float(v) for v in attr.Get())
        found[side] = actual  # type: ignore[assignment]

        expected = PAD_LOCAL_OFFSET_M[side]
        delta = math.dist(actual, expected)
        if delta > PAD_OFFSET_TOLERANCE_M:
            mismatches.append(
                f"  {side}: USD {actual} vs PAD_LOCAL_OFFSET_M {expected} "
                f"(off by {delta * 1000:.2f} mm)"
            )

    if not found:
        raise LookupError(
            f"no pad collider prims under {robot_prim_path}/<finger>/"
            f"{PAD_COLLIDER_SUBPATH}"
        )

    if mismatches:
        message = (
            "Finger pad colliders in the USD no longer match "
            "task_geometry.PAD_LOCAL_OFFSET_M:\n"
            + "\n".join(mismatches)
            + "\n\nThe approach reward and every grasp gate are computed from that "
            "constant. Running now trains against geometry that does not describe "
            "the robot. Re-measure with isaac/scripts/probe_gripper_runtime.py and "
            "update task_geometry.PAD_LOCAL_OFFSET_M, or revert the asset edit."
        )
        if strict:
            raise ValueError(message)
        print(f"[gripper_geometry] WARNING: {message}", flush=True)

    return found
