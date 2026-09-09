"""Uncertainty reporting for the Synria eval harness.

Why this exists
---------------
E25 (2026-08-05) ran three seeds at an identical budget on identical code and
measured a 2.8x spread on grasp and ~18x on completed cycles. Everything in the
experiment ledger before it was a single-seed point estimate, so every decision
resting on a delta smaller than that spread was indistinguishable from seed
luck — and at least one wrong conclusion was drawn and nearly acted on before
the third seed arrived.

There are two independent sources of uncertainty and they need separate
treatment, because conflating them is what makes a point estimate look solid:

* **Sampling error** — this checkpoint, this run, only ~262 episodes. Handled
  by :func:`wilson_interval`. Matters most exactly where the ledger's targets
  live: at 4/263 the normal approximation is useless and would report a lower
  bound below zero.
* **Seed variance** — a different RNG seed, same everything else. Handled by
  :func:`spread`. This one is far larger, and no amount of extra episodes in a
  single run will reveal it.

Pure Python: no torch, no Isaac Sim, no numpy. Importable and testable anywhere.
"""

from __future__ import annotations

import math
from typing import NamedTuple

#: z for a two-sided 95% interval.
Z_95 = 1.959963984540054


class Interval(NamedTuple):
    """A rate with a confidence interval."""

    rate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.rate:.3f} [{self.low:.3f}, {self.high:.3f}]"

    @property
    def width(self) -> float:
        return self.high - self.low


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial rate.

    Wilson rather than the normal approximation because the numbers that decide
    things here are small and near zero — ``place`` has come in at 1/259 and
    4/263. The normal interval at 1/259 spans a negative lower bound and
    understates the uncertainty badly; Wilson stays inside [0, 1] and behaves at
    the boundary, which is the regime this project actually operates in.

    Returns a zero-width interval at the origin for ``trials == 0`` rather than
    raising: an eval that completed no episodes is a real (bad) outcome and the
    caller should be able to print it.
    """
    if trials <= 0:
        return Interval(0.0, 0.0, 0.0)
    if successes < 0 or successes > trials:
        raise ValueError(f"successes={successes} outside [0, {trials}]")

    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    ) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin))


class Spread(NamedTuple):
    """Across-seed dispersion for one stage."""

    values: tuple[float, ...]
    mean: float
    lo: float
    hi: float

    @property
    def ratio(self) -> float:
        """max/min, or inf when the minimum is zero.

        Reported as a ratio because that is the shape of the problem: the
        ledger's deltas are multiplicative ("3.3x the prior record"), and a
        ratio is directly comparable against them. inf is honest — a seed that
        scored zero and a seed that did not are not a bounded factor apart.
        """
        return float("inf") if self.lo == 0.0 else self.hi / self.lo

    @property
    def range(self) -> float:
        return self.hi - self.lo


def spread(values: list[float]) -> Spread:
    """Summarise the same stage measured across several seeds."""
    if not values:
        raise ValueError("spread() needs at least one value")
    vals = tuple(float(v) for v in values)
    return Spread(vals, sum(vals) / len(vals), min(vals), max(vals))


def decision_is_supported(
    baseline: Spread | float, candidate: Spread | float, seed_spread_ratio: float
) -> bool:
    """Is a baseline-vs-candidate difference bigger than seed noise?

    The rule E25 forced: a delta smaller than the observed seed spread is not
    evidence. Pass the ratio measured on the stage in question (E25: ~2.8x on
    grasp, ~18x on place/full) and this reports whether the observed difference
    clears it.

    Deliberately crude — a real test would need many more seeds than 3h of GPU
    buys. It exists to stop the specific failure it is named for: reading a
    1.5x change as a result when seeds alone move things by 2.8x.
    """
    b = baseline.mean if isinstance(baseline, Spread) else float(baseline)
    c = candidate.mean if isinstance(candidate, Spread) else float(candidate)
    if b <= 0.0 or c <= 0.0:
        return False
    observed = max(b, c) / min(b, c)
    return observed > seed_spread_ratio
