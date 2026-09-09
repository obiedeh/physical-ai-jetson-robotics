"""Locks the finger-pad geometry to the USD asset it claims to describe.

Why these tests exist
---------------------
The pad-centre computation has been got wrong three times, each costing days:
the reward measured from the wrist flange; then from the finger LINK FRAMES
mislabelled as "the pads" (a 30 mm error that poisoned every experiment
through E22); then a block test that assumed "frame minus 30 mm in world z",
which lands outside the collider entirely.

None of those were caught by a test, because there was no test that compared
the numbers in the source against the asset. Every one of them was found by a
human re-measuring the USD by hand, weeks later, after a training run had
already been read as evidence about the task.

So these tests do exactly that comparison, automatically, against
``reports/gripper_phase1_composition.json`` -- the measured dump of the actual
USD. If the asset is edited and the constants are not, they fail. If a future
session reintroduces a private copy of the offsets, :func:`test_no_duplicate_pad_constants`
fails.

No Isaac Lab, no torch, no GPU required.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from isaac.isaaclab_tasks.synria_pickplace.gripper_geometry import (
    FRAME_TO_PAD_MIDPOINT_RESIDUAL_M,
    FRAME_TO_PAD_PER_SIDE_M,
    PAD_EXTENTS_M,
    PAD_HALF_EXTENTS_M,
    PAD_OFFSET_TOLERANCE_M,
)
from isaac.isaaclab_tasks.synria_pickplace.task_geometry import (
    GRIPPER_OPEN_M,
    PAD_LOCAL_OFFSET_M,
    PIECE_RADIUS_M,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMPOSITION = _REPO_ROOT / "reports" / "gripper_phase1_composition.json"

#: Prismatic travel of each finger, per the URDF (left 0..0.025, right
#: -0.025..0). The pair sets the true fully-open width.
_URDF_FINGER_TRAVEL_M = 0.025


@pytest.fixture(scope="module")
def composition() -> dict:
    """The measured USD dump that every constant here is derived from."""
    if not _COMPOSITION.exists():
        pytest.skip(f"composition report not present: {_COMPOSITION}")
    return json.loads(_COMPOSITION.read_text())


# ---------------------------------------------------------------------------
# The constants must match the asset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_pad_offset_matches_usd(composition: dict, side: str) -> None:
    """PAD_LOCAL_OFFSET_M is the pad collider's authored translate, not a guess."""
    authored = composition[f"{side}_pad_old"]["attrs"]["xformOp:translate"]
    actual = tuple(float(v) for v in re.findall(r"-?[\d.eE+-]+", str(authored)))
    expected = PAD_LOCAL_OFFSET_M[side]

    assert len(actual) == 3, f"could not parse translate for {side}: {authored!r}"
    delta = math.dist(actual, expected)
    assert delta <= PAD_OFFSET_TOLERANCE_M, (
        f"{side} pad offset drifted from the USD by {delta * 1000:.3f} mm: "
        f"constant {expected} vs asset {actual}. The approach reward and every "
        f"grasp gate are computed from this constant."
    )


def test_pad_extents_match_usd(composition: dict) -> None:
    """PAD_EXTENTS_M is the collider's authored scale (unit Cube, so scale == size)."""
    authored = composition["left_pad_old"]["attrs"]["xformOp:scale"]
    actual = tuple(float(v) for v in re.findall(r"-?[\d.eE+-]+", str(authored)))
    assert actual == pytest.approx(PAD_EXTENTS_M, abs=1e-9), (
        f"pad collider extents drifted: constant {PAD_EXTENTS_M} vs asset {actual}"
    )
    assert PAD_HALF_EXTENTS_M == pytest.approx(
        tuple(e / 2 for e in PAD_EXTENTS_M), abs=1e-12
    )


def test_pads_are_mirror_symmetric() -> None:
    """The two offsets must mirror, or the grasp centre is off-axis.

    An asymmetric pair would put the midpoint away from the line between the
    pads, so the piece would be pulled toward one finger and squeezed out.
    """
    left = PAD_LOCAL_OFFSET_M["left"]
    right = PAD_LOCAL_OFFSET_M["right"]
    for axis, (lv, rv) in enumerate(zip(left, right, strict=True)):
        assert lv == pytest.approx(-rv, abs=1e-9), (
            f"axis {axis} is not mirrored: left {lv} vs right {rv}"
        )


# ---------------------------------------------------------------------------
# The residual that poisoned E1-E22
# ---------------------------------------------------------------------------


def _measured_residuals(composition: dict) -> tuple[float, float]:
    """(midpoint, per-side) frame-to-pad distances in metres, from the USD dump."""
    def world(key: str) -> list[float]:
        return [v / 1000.0 for v in composition[key]["world_translate_mm"]]

    lf, rf = world("left_link"), world("right_link")
    lp, rp = world("left_pad_old"), world("right_pad_old")
    frame_mid = [(a + b) / 2 for a, b in zip(lf, rf, strict=True)]
    pad_mid = [(a + b) / 2 for a, b in zip(lp, rp, strict=True)]
    return math.dist(pad_mid, frame_mid), math.dist(lp, lf)


def test_frame_to_pad_residual_matches_measurement(composition: dict) -> None:
    """The recorded residual is the real one, so the docs cannot lie about it.

    This is the number that made the difference between "the policy cannot
    learn to grasp" and "the reward was aiming 30 mm off the contact surface".
    """
    midpoint, per_side = _measured_residuals(composition)
    assert midpoint == pytest.approx(FRAME_TO_PAD_MIDPOINT_RESIDUAL_M, abs=1e-5)
    assert per_side == pytest.approx(FRAME_TO_PAD_PER_SIDE_M, abs=1e-5)


def test_residual_exceeds_clearance_budget() -> None:
    """Regression guard: records WHY the link-frame midpoint was unusable.

    The clearance between the piece and each pad at grasp width is what the
    approach error has to fit inside. The 30 mm residual does not fit -- it is
    larger than the whole budget -- which is precisely why targeting the link
    frames rewarded a pose the arm cannot grasp from. If someone ever proposes
    reverting to frame midpoints, this test states the cost.
    """
    piece_diameter = 2 * PIECE_RADIUS_M
    clearance_per_side = (GRIPPER_OPEN_M - piece_diameter) / 2
    assert FRAME_TO_PAD_MIDPOINT_RESIDUAL_M > clearance_per_side, (
        "the recorded residual no longer exceeds the clearance budget; if the "
        "asset really changed, re-derive this test's premise rather than "
        "loosening it"
    )


def test_gripper_open_width_matches_urdf_travel() -> None:
    """GRIPPER_OPEN_M tracks the URDF, not a servo datasheet.

    This constant was 0.085 for a long time, taken from a servo-gripper spec
    that does not describe this arm; the URDF fingers travel 25 mm each.
    """
    assert GRIPPER_OPEN_M == pytest.approx(2 * _URDF_FINGER_TRAVEL_M, abs=1e-9)


# ---------------------------------------------------------------------------
# Structural guard: one source of truth, enforced
# ---------------------------------------------------------------------------


class _FakeAttr:
    def __init__(self, value: tuple[float, float, float] | None) -> None:
        self._value = value

    def __bool__(self) -> bool:
        return self._value is not None

    def HasAuthoredValue(self) -> bool:  # noqa: N802 - mirrors the USD API
        return self._value is not None

    def Get(self) -> tuple[float, float, float] | None:  # noqa: N802
        return self._value


class _FakePrim:
    def __init__(self, name: str, translate=None, children=()) -> None:
        self._name = name
        self._translate = translate
        self.children = {c._name: c for c in children}

    def __bool__(self) -> bool:
        return True

    def IsValid(self) -> bool:  # noqa: N802
        return True

    def GetName(self) -> str:  # noqa: N802
        return self._name

    def GetPath(self):  # noqa: N802
        return self

    def AppendChild(self, name: str):  # noqa: N802
        return self.children.get(name, _MISSING)

    def GetAttribute(self, _name: str) -> _FakeAttr:  # noqa: N802
        return _FakeAttr(self._translate)

    def walk(self):
        yield self
        for child in self.children.values():
            yield from child.walk()


class _MissingPrim:
    def __bool__(self) -> bool:
        return False

    def IsValid(self) -> bool:  # noqa: N802
        return False


_MISSING = _MissingPrim()


class _FakeStage:
    def __init__(self, root: _FakePrim) -> None:
        self._root = root

    def GetPrimAtPath(self, path):  # noqa: N802
        if path == "/robot":
            return self._root
        return path if isinstance(path, (_FakePrim, _MissingPrim)) else _MISSING


def _stage_with(left, right) -> _FakeStage:
    """A stage whose two finger links carry the given pad translates."""
    return _FakeStage(
        _FakePrim(
            "robot",
            children=(
                _FakePrim("left_gripper", children=(_FakePrim("pad_collider", left),)),
                _FakePrim("right_gripper", children=(_FakePrim("pad_collider", right),)),
            ),
        )
    )


@pytest.fixture
def fake_pxr(monkeypatch: pytest.MonkeyPatch):
    """Stand in for ``pxr`` so the USD check is testable without Isaac Sim."""
    import sys
    import types

    usd = types.SimpleNamespace(
        PrimRange=lambda root, _flags=None: root.walk(),
        TraverseInstanceProxies=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "pxr", types.SimpleNamespace(Usd=usd))
    yield


def test_verify_against_stage_accepts_matching_asset(fake_pxr) -> None:
    from isaac.isaaclab_tasks.synria_pickplace.gripper_geometry import (
        verify_against_stage,
    )

    stage = _stage_with(PAD_LOCAL_OFFSET_M["left"], PAD_LOCAL_OFFSET_M["right"])
    found = verify_against_stage(stage, "/robot")
    assert set(found) == {"left", "right"}


def test_verify_against_stage_rejects_drifted_asset(fake_pxr) -> None:
    """A 5 mm asset edit must stop the run, not train against stale geometry."""
    from isaac.isaaclab_tasks.synria_pickplace.gripper_geometry import (
        verify_against_stage,
    )

    drifted = tuple(v + 0.005 for v in PAD_LOCAL_OFFSET_M["left"])
    stage = _stage_with(drifted, PAD_LOCAL_OFFSET_M["right"])
    with pytest.raises(ValueError, match="no longer match"):
        verify_against_stage(stage, "/robot")


def test_verify_against_stage_reports_when_it_verified_nothing(fake_pxr) -> None:
    """Absent pads raise rather than returning an empty 'all clear'.

    A wrong prim path must not read as a successful check — that is the same
    silent-success failure this whole module exists to remove.
    """
    from isaac.isaaclab_tasks.synria_pickplace.gripper_geometry import (
        verify_against_stage,
    )

    stage = _FakeStage(_FakePrim("robot"))
    with pytest.raises(LookupError):
        verify_against_stage(stage, "/robot")


def test_readme_constants_match_source() -> None:
    """The README's constants table must agree with task_geometry.py.

    Every value in that table had drifted: GRIPPER_OPEN_M was documented as
    0.085 (a servo datasheet that does not describe this arm) long after the
    code moved to the URDF's 0.05, the grasp width said 0.015 against a real
    0.045, and the approach radius said 0.020 against a real 0.15. Docs that
    disagree with the code are worse than absent ones -- they were read, and
    believed, during the diagnoses that chased this geometry for weeks.
    """
    from isaac.isaaclab_tasks.synria_pickplace import task_geometry

    readme = (
        _REPO_ROOT / "isaac/isaaclab_tasks/synria_pickplace/README.md"
    ).read_text()

    documented = {
        "GRIPPER_OPEN_M": task_geometry.GRIPPER_OPEN_M,
        "GRIPPER_GRASP_WIDTH_M": task_geometry.GRIPPER_GRASP_WIDTH_M,
        "GRASP_APPROACH_RADIUS": task_geometry.GRASP_APPROACH_RADIUS_M,
        "GRASP_CONFIRM_STEPS": task_geometry.GRASP_CONFIRM_STEPS,
    }

    for name, source_value in documented.items():
        match = re.search(rf"^{name}\s*=\s*([\d.]+)", readme, re.MULTILINE)
        assert match, f"README no longer documents {name}"
        assert float(match.group(1)) == pytest.approx(float(source_value)), (
            f"README says {name} = {match.group(1)} but task_geometry.py has "
            f"{source_value}"
        )


def test_no_duplicate_pad_constants() -> None:
    """Fails if any file re-declares the pad offsets instead of importing them.

    This is the test that would have prevented the whole class of bug. The
    f933e73 commit message stated the offsets were "recorded once ... so the
    two call sites cannot drift" -- while ``synria_grasp_feasibility.py`` still
    carried its own private ``_PAD_LOCAL`` copy of the same numbers. The claim
    was true of the intent and false of the code, and nothing checked.

    Scans for the literal pad magnitudes outside the module that owns them.
    """
    owner = "isaac/isaaclab_tasks/synria_pickplace/task_geometry.py"
    # The two magnitudes that together identify a pad-offset triple.
    signature = re.compile(r"0\.030?\b[^\n]{0,40}0\.01375|0\.01375[^\n]{0,40}0\.030?\b")

    offenders: list[str] = []
    for path in sorted(_REPO_ROOT.glob("isaac/**/*.py")):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == owner:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            # Prose in docstrings and comments is how these numbers get
            # explained; only executable re-declarations are the problem.
            if stripped.startswith(("#", "*", '"""', "'''")):
                continue
            if signature.search(line):
                offenders.append(f"{rel}:{lineno}: {stripped}")

    assert not offenders, (
        "pad offset literals re-declared outside task_geometry.py — import "
        "PAD_LOCAL_OFFSET_M instead so the two copies cannot drift:\n  "
        + "\n  ".join(offenders)
    )
