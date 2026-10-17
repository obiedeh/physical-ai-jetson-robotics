"""The requested-rate rule on a measurement: one callback of slack, five percent at most."""

from __future__ import annotations

import pytest

from synria_lerobot.physical_contract import StateRateMeasurement


def measurement(count: int, duration_s: float = 2.0) -> StateRateMeasurement:
    """A measurement of ``count`` callbacks over ``duration_s`` with a nominal gap."""
    return StateRateMeasurement(count / duration_s, count, duration_s, 0.0, duration_s,
                                duration_s / count)


def test_a_source_paced_at_exactly_the_dataset_rate_passes_on_either_side_of_the_count() -> None:
    """30 callbacks in 2.00007 s read as 14.999 Hz and 29 as 14.499 Hz; both keep up with 15 fps."""
    assert measurement(30, 2.00007).meets_requested_rate(15)
    assert measurement(29, 2.00007).meets_requested_rate(15)
    assert measurement(31, 2.00007).meets_requested_rate(15)
    assert measurement(100).meets_requested_rate(15)


def test_two_callbacks_short_or_more_than_five_percent_slow_fails() -> None:
    """28 in 2 s is two short of 30; 1 in 2 s is one short of 2 but half the requested rate."""
    assert not measurement(28, 2.00007).meets_requested_rate(15)
    assert not measurement(30).meets_requested_rate(30)
    assert not measurement(1).meets_requested_rate(1)
    assert not measurement(50).meets_requested_rate(50 / 0.9)


def test_the_requested_rate_must_be_a_positive_finite_number() -> None:
    """Zero, negative, infinite and boolean requests are rejected, not compared."""
    for bad in (0, -15, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            measurement(30).meets_requested_rate(bad)
