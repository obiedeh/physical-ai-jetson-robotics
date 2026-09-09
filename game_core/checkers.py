"""Checkers — game #3, queued behind ludo and chess by product priority.

Rules core + adapter land when its turn comes; the interface will match
the others (plan_turn() -> TurnPlan)."""

from __future__ import annotations


class CheckersGame:
    def __init__(self, *a, **k):
        raise NotImplementedError(
            "checkers is game #3 on the product roadmap (ludo -> chess -> "
            "checkers); rules core not yet implemented")


class CheckersAdapter(CheckersGame):
    pass
