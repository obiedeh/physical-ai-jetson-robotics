"""game_core tests: every game brain must emit physically-executable
TurnPlans through the same schema. Run with the lerobot venv (has
python-chess): ~/.venv/lerobot/bin/python -m pytest game_core -q"""

import chess
import pytest

from game_core import ChessGameAdapter, LudoGameAdapter
from game_core.commands import StagingAllocator


def test_ludo_full_game_produces_executable_plans():
    g = LudoGameAdapter(seed=3)
    plans = 0
    for _ in range(20000):
        if g.game_over:
            break
        p = g.plan_turn()
        if p is None:
            continue
        plans += 1
        assert p.commands, "every plan must contain at least one command"
        for c in p.commands:
            assert len(c.pick_xy) == 2 and len(c.place_xy) == 2
        # capture-returns must precede the mover
        reasons = [c.reason for c in p.commands]
        assert reasons[-1] == "move"
    assert g.game_over and plans > 50


def test_ludo_capture_plan_shape():
    g = LudoGameAdapter()
    g.game.positions[("red", 0)] = ("track", 10)
    g.game.positions[("blue", 0)] = ("track", 12)
    p = g.plan_turn(roll=2)
    assert p is not None
    assert [c.reason for c in p.commands] == ["capture-return", "move"]


def test_chess_scholars_mate_capture_and_result():
    g = ChessGameAdapter()
    for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6"):
        g.plan_turn(chess.Move.from_uci(uci))
    p = g.plan_turn(chess.Move.from_uci("h5f7"))  # Qxf7#
    assert [c.reason for c in p.commands] == ["capture-removal", "move"]
    assert g.game_over and g.result() == "1-0"


def test_chess_castling_is_two_commands():
    g = ChessGameAdapter()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"):
        g.plan_turn(chess.Move.from_uci(uci))
    p = g.plan_turn(chess.Move.from_uci("e1g1"))
    assert [c.reason for c in p.commands] == ["move", "castle-rook"]


def test_chess_squares_map_onto_board():
    g = ChessGameAdapter()
    half = g.geom_size / 2 + 1e-9
    for sq in range(64):
        x, y = g.square_xy(sq)
        assert abs(x - g.center[0]) <= half and abs(y - g.center[1]) <= half


def test_staging_slots_unique_and_off_board():
    s = StagingAllocator()
    seen = set()
    for _ in range(32):
        xy = s.next_slot()
        assert xy not in seen
        seen.add(xy)
        assert abs(xy[0] - 0.10) > 0.2 or abs(xy[1]) > 0.2


def test_checkers_is_explicitly_queued():
    from game_core.checkers import CheckersGame
    with pytest.raises(NotImplementedError):
        CheckersGame()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
