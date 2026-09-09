"""Read a physical die's value from its pose (Milestone B, physical dice).

In sim the die is a small cube RigidObject; its top-face value is decided by
its orientation, exactly like a real die at rest. This module maps an
orientation quaternion (world) to the pip value now showing up (+Z world).

Face convention (right-handed, opposite faces sum to 7, standard Western die):
    local +Z -> 1   local -Z -> 6
    local +Y -> 2   local -Y -> 5
    local +X -> 3   local -X -> 4

`die_value_up(quat_wxyz)` returns the value on the face whose local normal is
most aligned with world +Z after rotation. `settled(...)` guards against
reading a die that is still tumbling.
"""
from __future__ import annotations

import numpy as np

# local face normal -> pip value
_FACES = (
    (np.array([0.0, 0.0, 1.0]), 1),
    (np.array([0.0, 0.0, -1.0]), 6),
    (np.array([0.0, 1.0, 0.0]), 2),
    (np.array([0.0, -1.0, 0.0]), 5),
    (np.array([1.0, 0.0, 0.0]), 3),
    (np.array([-1.0, 0.0, 0.0]), 4),
)


def quat_to_rotmat(quat_wxyz) -> np.ndarray:
    """(w,x,y,z) unit quaternion -> 3x3 rotation matrix (Isaac uses wxyz)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def die_value_up(quat_wxyz) -> int:
    """Pip value on the upward (+Z world) face for this die orientation."""
    R = quat_to_rotmat(quat_wxyz)
    up = np.array([0.0, 0.0, 1.0])
    best_val, best_dot = 1, -2.0
    for normal, val in _FACES:
        d = float(np.dot(R @ normal, up))
        if d > best_dot:
            best_dot, best_val = d, val
    return best_val


def face_alignment(quat_wxyz) -> float:
    """How flat the die sits: 1.0 = a face dead-flat up, lower = tilted/on edge.
    Use to reject reads while the die is cocked or mid-tumble."""
    R = quat_to_rotmat(quat_wxyz)
    up = np.array([0.0, 0.0, 1.0])
    return max(abs(float(np.dot(R @ n, up))) for n, _ in _FACES)


def settled(lin_vel, ang_vel, quat_wxyz, *, v_tol=0.02, w_tol=0.2,
            align_tol=0.97) -> bool:
    """True when the die has stopped (low speed) AND rests on a face."""
    v = float(np.linalg.norm(np.asarray(lin_vel, dtype=float)))
    w = float(np.linalg.norm(np.asarray(ang_vel, dtype=float)))
    return v < v_tol and w < w_tol and face_alignment(quat_wxyz) >= align_tol
