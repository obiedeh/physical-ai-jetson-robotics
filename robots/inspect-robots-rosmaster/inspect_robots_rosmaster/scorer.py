"""Scorers for the ROSMASTER M3 Pro tabletop pick-place task (Inspect Robots).

Real-robot honesty (the eval01 lesson): the board has NO object-pose sensor and
NO joint feedback, so there is no privileged success oracle. Success therefore
comes from the operator/VLM grader (``record.operator_judgement``); this scorer
adds the *measurable* pick-place funnel from the recorded ARM trajectory —
grasp actuation, grasp->release cycle, and arm travel — so the EvalLog carries
both the human's verdict and reproducible diagnostics.

Registered via the ``inspect_robots.scorers`` entry point as ``m3pro_pickplace``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from inspect_robots.registry import scorer
from inspect_robots.scorer import Score

if TYPE_CHECKING:  # avoid import cycles / heavy deps at registration time
    from inspect_robots.rollout import TrialRecord
    from inspect_robots.scene import Target


def _action_traj(record) -> list[list[float]]:
    """The commanded 6-vectors (joint degrees) over the episode. Falls back to
    observation.state['joint_pos'] when an action is missing."""
    traj = []
    for s in record.steps:
        a = getattr(s.action, "data", None)
        if a is not None:
            traj.append([float(x) for x in list(a)[:6]])
        else:
            jp = (s.observation.state or {}).get("joint_pos")
            if jp is not None:
                traj.append([float(x) for x in list(jp)[:6]])
    return traj


@dataclass(frozen=True)
class _M3ProPickPlace:
    name: str = "m3pro_pickplace"
    grip_idx: int = 5              # joint6 = gripper
    grip_min_travel_deg: float = 15.0   # actuation counts as a grasp attempt
    grip_return_tol_deg: float = 12.0   # returned near open => release
    min_arm_travel_deg: float = 30.0    # the arm actually did work (not idle)

    def __call__(self, record: "TrialRecord", target: "Target | None") -> Score:
        traj = _action_traj(record)
        n = len(traj)
        if n < 2:
            return Score(value=False, explanation="no usable trajectory",
                         metadata={"n_steps": n})
        grip = [row[self.grip_idx] for row in traj]
        g0 = grip[0]
        grip_range = max(grip) - min(grip)
        grasped = any(abs(g - g0) > self.grip_min_travel_deg for g in grip)
        released = grasped and abs(grip[-1] - g0) <= self.grip_return_tol_deg
        arm_travel = 0.0
        for a, b in zip(traj, traj[1:]):
            arm_travel += sum(abs(a[i] - b[i]) for i in range(6) if i != self.grip_idx)
        active = arm_travel > self.min_arm_travel_deg

        funnel = {
            "n_steps": n,
            "arm_travel_deg": round(arm_travel, 1),
            "arm_active": active,
            "grip_range_deg": round(grip_range, 1),
            "grasp_actuated": grasped,
            "grasp_release_cycle": released,
        }

        # authoritative success = the operator/VLM judgement (no oracle exists)
        judged = getattr(record, "operator_judgement", None)
        if judged is not None:
            value = bool(judged)
            funnel["decided_via"] = "operator_judgement"
            expl = (f"operator judged {'SUCCESS' if value else 'failure'}; "
                    f"funnel: active={active} grasp={grasped} cycle={released} "
                    f"travel={arm_travel:.0f}deg")
            note = getattr(record, "operator_note", None)
            if note:
                expl += f" | note: {note}"
            return Score(value=value, explanation=expl, metadata=funnel)

        # unattended: report the measurable proxy, clearly labelled as NOT success
        proxy = bool(active and released)
        funnel["decided_via"] = "trajectory_proxy (NOT ground-truth success)"
        return Score(
            value=proxy,
            explanation=("no operator judgement — trajectory proxy "
                         f"(active grasp->release): {proxy}. This is a motion "
                         "signature, not verified task success; add --grader "
                         "operator or vlm for real success."),
            metadata=funnel,
        )


@scorer("m3pro_pickplace")
def m3pro_pickplace(grip_idx: int = 5, grip_min_travel_deg: float = 15.0,
                    grip_return_tol_deg: float = 12.0,
                    min_arm_travel_deg: float = 30.0):
    """Tabletop pick-place funnel scorer for the M3 Pro (see module docstring)."""
    return _M3ProPickPlace(
        grip_idx=int(grip_idx),
        grip_min_travel_deg=float(grip_min_travel_deg),
        grip_return_tol_deg=float(grip_return_tol_deg),
        min_arm_travel_deg=float(min_arm_travel_deg))
