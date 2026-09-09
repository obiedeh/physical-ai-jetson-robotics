"""Tests for the eval harness's uncertainty reporting.

The regime that matters here is small counts near zero — the ledger's decisive
stage, ``place``, has come in at 1/259 and 4/263. Anything that behaves well on
average but badly at the boundary would be useless for exactly the decisions
this is meant to inform.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac" / "scripts"))

from eval_stats import (  # noqa: E402
    Z_95,
    decision_is_supported,
    spread,
    wilson_interval,
)


def test_interval_brackets_the_rate() -> None:
    iv = wilson_interval(33, 262)          # E23's place stage
    assert iv.low < iv.rate < iv.high
    assert iv.rate == pytest.approx(33 / 262)


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(0, 259), (1, 259), (4, 263), (259, 259), (1, 1)],
)
def test_interval_stays_inside_zero_one(successes: int, trials: int) -> None:
    """The normal approximation goes negative at 1/259; Wilson must not.

    This is the whole reason for choosing Wilson — these are real observed
    counts from seeds 42/43, not hypotheticals.
    """
    iv = wilson_interval(successes, trials)
    assert 0.0 <= iv.low <= iv.rate <= iv.high <= 1.0


def test_normal_approximation_would_have_gone_negative() -> None:
    """Pins the motivation: at 1/259 the textbook interval is unusable."""
    p = 1 / 259
    normal_low = p - Z_95 * math.sqrt(p * (1 - p) / 259)
    assert normal_low < 0.0
    assert wilson_interval(1, 259).low >= 0.0


def test_more_episodes_narrows_the_interval() -> None:
    narrow = wilson_interval(100, 1000)
    wide = wilson_interval(10, 100)
    assert narrow.width < wide.width


def test_zero_trials_does_not_raise() -> None:
    """An eval that completed no episodes is a real outcome, not an exception."""
    assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)


def test_impossible_counts_raise() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


# ---------------------------------------------------------------------------
# Across-seed spread — the thing E25 actually measured
# ---------------------------------------------------------------------------


def test_spread_reproduces_the_e25_grasp_numbers() -> None:
    """Seeds 42/43/44 grasp: 0.186 / 0.205 / 0.522."""
    s = spread([0.186, 0.205, 0.522])
    assert s.lo == 0.186
    assert s.hi == 0.522
    assert s.mean == pytest.approx(0.304, abs=1e-3)
    assert s.ratio == pytest.approx(2.8, abs=0.05)


def test_spread_reproduces_the_e25_full_numbers() -> None:
    """Seeds 42/43/44 full: 0.015 / 0.004 / 0.071 — an ~18x spread."""
    s = spread([0.015, 0.004, 0.071])
    assert s.ratio == pytest.approx(17.75, abs=0.1)


def test_zero_seed_gives_infinite_ratio_not_a_crash() -> None:
    """A seed that scored zero is not a bounded factor from one that did not."""
    assert spread([0.0, 0.05, 0.08]).ratio == math.inf


def test_single_seed_has_no_spread() -> None:
    """One seed reports a ratio of 1.0 — which is the lie E1-E23 told.

    The value is correct in isolation and meaningless as evidence, which is why
    the harness reports the seed COUNT alongside it.
    """
    assert spread([0.366]).ratio == 1.0


def test_spread_needs_a_value() -> None:
    with pytest.raises(ValueError):
        spread([])


# ---------------------------------------------------------------------------
# The guard rule
# ---------------------------------------------------------------------------


def test_delta_smaller_than_seed_noise_is_not_supported() -> None:
    """E22 0.366 vs the E25 mean 0.304 — a 1.2x delta against a 2.8x spread."""
    assert decision_is_supported(0.366, 0.304, seed_spread_ratio=2.8) is False


def test_delta_larger_than_seed_noise_is_supported() -> None:
    """E22 0.366 vs E23 0.706 on grasp is 1.9x — still inside 2.8x seed noise.

    Recorded deliberately: this is the comparison the ledger called "roughly
    doubled", and by this rule one run each does not support it.
    """
    assert decision_is_supported(0.366, 0.706, seed_spread_ratio=2.8) is False

    # How demanding that bar is, stated plainly: going from E22's 0.366 to a
    # PERFECT 1.000 grasp rate is 2.73x, which still does not clear a 2.8x seed
    # spread on one run each. At this variance, single-seed evidence cannot
    # establish any effect the grasp stage is physically capable of showing —
    # the only fix is more seeds, not a bigger effect.
    assert decision_is_supported(0.366, 1.0, seed_spread_ratio=2.8) is False

    # With three seeds the spread on the MEAN is far tighter, and normal
    # effects become decidable again.
    assert decision_is_supported(0.366, 0.706, seed_spread_ratio=1.5) is True


def test_zero_rates_are_never_supported() -> None:
    assert decision_is_supported(0.0, 0.08, seed_spread_ratio=2.8) is False
