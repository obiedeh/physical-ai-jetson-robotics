"""Invariant tests for the die-value reader (no sim needed)."""
import numpy as np
from game_core.dice import die_value_up, face_alignment, settled


def _q(axis, deg):
    a = np.array(axis, float); a /= np.linalg.norm(a); r = np.radians(deg)
    return np.array([np.cos(r / 2), *(a * np.sin(r / 2))])


def test_axis_orientations_cover_all_six():
    vals = {
        die_value_up([1, 0, 0, 0]),
        die_value_up(_q([1, 0, 0], 180)),
        die_value_up(_q([1, 0, 0], 90)),
        die_value_up(_q([1, 0, 0], -90)),
        die_value_up(_q([0, 1, 0], 90)),
        die_value_up(_q([0, 1, 0], -90)),
    }
    assert vals == {1, 2, 3, 4, 5, 6}


def test_opposite_faces_sum_to_seven():
    # rotating the read orientation 180 deg about X swaps up<->down face
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        up = die_value_up(q)
        # compose q with a 180-about-world-X flip -> the down face comes up
        f = _q([1, 0, 0], 180)
        w0, x0, y0, z0 = f; w1, x1, y1, z1 = q
        flipped = np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ])
        assert up + die_value_up(flipped) == 7


def test_always_valid():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        assert die_value_up(q) in (1, 2, 3, 4, 5, 6)


def test_settled_gating():
    assert settled([0, 0, 0], [0, 0, 0], [1, 0, 0, 0]) is True
    assert settled([0, 0, 0], [0, 0, 5], [1, 0, 0, 0]) is False
    assert settled([0, 0, 0], [0, 0, 0], _q([1, 0, 0], 45)) is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"  {name} PASS")
    print("all dice tests pass")
