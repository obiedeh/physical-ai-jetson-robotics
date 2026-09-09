"""The command schema between game brains (L4) and the executor (L1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PickPlaceCommand:
    """One physical arm operation. XY in world meters on the table."""
    pick_xy: tuple
    place_xy: tuple
    piece: str = ""            # human-readable ("black knight", "red token 2")
    reason: str = ""           # "move", "capture-removal", "castle-rook", ...


@dataclass
class TurnPlan:
    """Everything the arm must do to realize one game turn, in order.
    Removals always precede the mover (clear the square first)."""
    game: str
    description: str
    commands: list = field(default_factory=list)
    needs_human: str | None = None   # e.g. chess promotion piece swap


class StagingAllocator:
    """Assigns captured pieces to staging-pad slots, round-robin over the
    four pads, row-filling each. Pads match _synria_scene_common's
    add_staging_zones layout."""

    def __init__(self, board_size: float = 0.40, center=(0.10, 0.0),
                 offset: float = 0.03, depth: float = 0.10,
                 slots_per_pad: int = 8):
        self.n = 0
        cx, cy = center
        half = board_size / 2
        d = half + offset + depth / 2
        self.pads = {
            # pads are LONG parallel to their board edge: left/right pads
            # run along X, top/bottom pads along Y (add_staging_zones).
            "left":   (cx, cy - d, 1.0, 0.0),   # (x, y, step_dx, step_dy)
            "right":  (cx, cy + d, -1.0, 0.0),
            "bottom": (cx + d, cy, 0.0, -1.0),
            "top":    (cx - d, cy, 0.0, 1.0),
        }
        self.names = list(self.pads)
        self.slots_per_pad = slots_per_pad
        self.pitch = board_size / slots_per_pad

    def next_slot(self) -> tuple:
        pad = self.names[self.n % 4]
        k = self.n // 4
        x, y, dx, dy = self.pads[pad]
        off = (k - (self.slots_per_pad - 1) / 2) * self.pitch
        self.n += 1
        return (round(x + dx * off, 4), round(y + dy * off, 4))
