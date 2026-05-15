"""Simple attack-count forecasting utilities."""

from __future__ import annotations

from collections.abc import Sequence


def moving_average_forecast(values: Sequence[int], *, horizon: int = 1, lookback: int = 5) -> list[float]:
    """Forecast attack counts with a conservative moving average baseline."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not values:
        return [0.0] * horizon
    tail = list(values[-lookback:])
    prediction = sum(tail) / len(tail)
    return [prediction] * horizon

