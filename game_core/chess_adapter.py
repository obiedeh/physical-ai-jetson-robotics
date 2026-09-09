"""Chess brain: python-chess rules + a material-greedy move chooser,
mapped to arm commands. Handles the physically-interesting cases:
captures (remove first), castling (two commands), en passant (removal
square differs from destination), promotion (flagged for human piece
swap in the MVP)."""

from __future__ import annotations

import random

import chess

from .commands import PickPlaceCommand, StagingAllocator, TurnPlan

_VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class ChessGameAdapter:
    def __init__(self, board_size: float = 0.40, center=(0.10, 0.0),
                 seed: int = 0):
        self.board = chess.Board()
        self.geom_size = board_size
        self.center = center
        self.staging = StagingAllocator(board_size, center)
        self.rng = random.Random(seed)

    # ---- geometry ------------------------------------------------------
    def square_xy(self, square: int) -> tuple:
        f, r = chess.square_file(square), chess.square_rank(square)
        cell = self.geom_size / 8
        x = self.center[0] + (r - 3.5) * cell
        y = self.center[1] + (f - 3.5) * cell
        return (round(x, 4), round(y, 4))

    # ---- brain ---------------------------------------------------------
    def choose_move(self) -> chess.Move | None:
        moves = list(self.board.legal_moves)
        if not moves:
            return None

        def score(m: chess.Move) -> float:
            s = self.rng.random() * 0.1
            if self.board.is_capture(m):
                victim = self.board.piece_at(m.to_square)
                s += _VALS.get(victim.piece_type, 1) if victim else 1  # ep
            if self.board.gives_check(m):
                s += 0.5
            return s
        return max(moves, key=score)

    # ---- move -> physical plan ----------------------------------------
    def plan_turn(self, move: chess.Move | None = None) -> TurnPlan | None:
        move = move or self.choose_move()
        if move is None:
            return None
        b = self.board
        plan = TurnPlan(game="chess", description=b.san(move))
        mover = b.piece_at(move.from_square)

        if b.is_en_passant(move):
            cap_sq = move.to_square + (-8 if b.turn == chess.WHITE else 8)
            plan.commands.append(PickPlaceCommand(
                self.square_xy(cap_sq), self.staging.next_slot(),
                piece=str(b.piece_at(cap_sq)), reason="capture-removal"))
        elif b.is_capture(move):
            plan.commands.append(PickPlaceCommand(
                self.square_xy(move.to_square), self.staging.next_slot(),
                piece=str(b.piece_at(move.to_square)),
                reason="capture-removal"))

        plan.commands.append(PickPlaceCommand(
            self.square_xy(move.from_square), self.square_xy(move.to_square),
            piece=str(mover), reason="move"))

        if b.is_castling(move):
            if b.is_kingside_castling(move):
                r_from = chess.H1 if b.turn == chess.WHITE else chess.H8
                r_to = chess.F1 if b.turn == chess.WHITE else chess.F8
            else:
                r_from = chess.A1 if b.turn == chess.WHITE else chess.A8
                r_to = chess.D1 if b.turn == chess.WHITE else chess.D8
            plan.commands.append(PickPlaceCommand(
                self.square_xy(r_from), self.square_xy(r_to),
                piece="rook", reason="castle-rook"))

        if move.promotion:
            plan.needs_human = (f"promotion: replace pawn with "
                                f"{chess.piece_name(move.promotion)}")

        self.board.push(move)
        return plan

    @property
    def game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        return self.board.result(claim_draw=True)
