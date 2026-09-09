"""Rules-core tests: the product's game logic must be right by proof,
not by demo. Run: pytest ludo_engine/test_ludo.py -q"""

import random

import pytest

from ludo_engine.game import (COLORS, HOME_LEN, START, TOKENS, TRACK_LEN,
                              GamePhase, LudoGame)
from ludo_engine.board import BoardGeometry


def test_needs_six_to_leave_yard():
    g = LudoGame()
    for roll in (1, 2, 3, 4, 5):
        assert g.legal_moves(roll) == []
    moves = g.legal_moves(6)
    assert len(moves) == TOKENS
    assert all(m.dst == ("track", START["red"]) for m in moves)


def test_six_grants_extra_turn_and_three_sixes_forfeit():
    g = LudoGame()
    g.take_turn(6)
    assert g.current == "red"          # extra turn after a 6
    g.take_turn(6)
    assert g.current == "red"
    g.take_turn(6)                     # third 6: forfeit, turn passes
    assert g.current == "blue"


def test_capture_returns_token_to_yard():
    g = LudoGame()
    g.positions[("red", 0)] = ("track", 10)
    g.positions[("blue", 0)] = ("track", 12)
    mv = [m for m in g.legal_moves(2) if m.token == 0][0]
    assert mv.captures == (("blue", 0),)
    g.apply(mv)
    assert g.positions[("blue", 0)][0] == "yard"


def test_no_capture_on_safe_star_square():
    g = LudoGame()
    g.positions[("red", 0)] = ("track", 6)
    g.positions[("blue", 0)] = ("track", 8)   # star square
    mv = [m for m in g.legal_moves(2) if m.token == 0][0]
    assert mv.captures == ()


def test_exact_roll_required_to_finish():
    g = LudoGame()
    g.positions[("red", 0)] = ("home", 3)
    assert g.legal_moves(2) != []             # 3+2=5 -> final cell
    assert all(m.dst == ("home", 5) or m.token != 0
               for m in g.legal_moves(2))
    assert [m for m in g.legal_moves(4) if m.token == 0] == []  # overshoot


def test_home_entry_from_track():
    g = LudoGame()
    # red token 49 squares along its lap: two more track squares allowed,
    # then the home column
    g.positions[("red", 0)] = ("track", (START["red"] + 49) % TRACK_LEN)
    mv = [m for m in g.legal_moves(4) if m.token == 0][0]
    assert mv.dst == ("home", 2)              # 49+4 = 53 -> home cell 2


def test_full_random_game_terminates():
    rng = random.Random(7)
    g = LudoGame()
    for _ in range(20000):
        if g.phase == GamePhase.FINISHED:
            break
        g.take_turn(rng.randint(1, 6))
    assert g.phase == GamePhase.FINISHED
    assert len(g.finished_order) >= len(COLORS) - 1


def test_board_geometry_all_positions_on_board():
    b = BoardGeometry()
    half = b.board_size / 2 + 1e-6
    for color in COLORS:
        for s in range(TRACK_LEN):
            x, y = b.world_xy(color, ("track", s))
            assert abs(x - b.center_x) <= half and abs(y - b.center_y) <= half
        for t in range(TOKENS):
            b.world_xy(color, ("yard", t))
        for h in range(HOME_LEN):
            b.world_xy(color, ("home", h))


def test_move_to_command_shape():
    g = LudoGame()
    b = BoardGeometry()
    g.positions[("red", 0)] = ("track", 10)
    g.positions[("blue", 0)] = ("track", 12)
    mv = [m for m in g.legal_moves(2) if m.token == 0][0]
    cmd = b.move_to_command(mv)
    assert set(cmd) == {"color", "token", "pick_xy", "place_xy", "captures"}
    assert cmd["captures"][0]["color"] == "blue"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
