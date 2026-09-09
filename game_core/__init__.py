"""game_core — the product's game-agnostic task layer (L4).

Every supported game (ludo, chess, checkers) reduces to one robot
primitive stream: PickPlaceCommand{pick_xy, place_xy}. Game rules live
in per-game engines; geometry adapters map logical squares to world XY;
captured pieces route to staging-zone slots. The executor (L1) consumes
commands and never imports a game; the games never import the robot.

Imports are LAZY per game so a consumer environment only needs the
dependencies of the game it actually plays (e.g. the Isaac venv runs
ludo without python-chess installed).
"""

from .commands import PickPlaceCommand, TurnPlan  # noqa: F401


def __getattr__(name):
    if name == "LudoGameAdapter":
        from .ludo_adapter import LudoGameAdapter
        return LudoGameAdapter
    if name == "ChessGameAdapter":
        from .chess_adapter import ChessGameAdapter
        return ChessGameAdapter
    if name in ("CheckersGame", "CheckersAdapter"):
        from . import checkers
        return getattr(checkers, name)
    raise AttributeError(name)
