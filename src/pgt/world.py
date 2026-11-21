"""Deterministic 7x7 warehouse: dynamics, first-pad termination, rendering, D4, BFS oracle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

N = 7
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # action index -> (d_row, d_col)
GLYPHS = ("circle", "triangle")
TRIMS = ("striped", "dotted")
TIMEOUT = "timeout"

Cell = tuple[int, int]


def other_glyph(glyph: str) -> str:
    return GLYPHS[1 - GLYPHS.index(glyph)]


def other_trim(trim: str) -> str:
    return TRIMS[1 - TRIMS.index(trim)]


@dataclass(frozen=True)
class Layout:
    """Walls, start and pad cells. ``pads[i]`` holds the pad drawn with ``GLYPHS[i]``."""

    walls: frozenset[Cell]
    start: Cell
    pads: tuple[Cell, Cell]

    def pad(self, glyph: str) -> Cell:
        return self.pads[GLYPHS.index(glyph)]

    def pad_at(self, cell: Cell) -> str | None:
        for glyph, pad in zip(GLYPHS, self.pads, strict=True):
            if pad == cell:
                return glyph
        return None

    def to_json(self) -> dict:
        return {
            "walls": sorted(list(w) for w in self.walls),
            "start": list(self.start),
            "pads": {g: list(self.pad(g)) for g in GLYPHS},
        }


def step(layout: Layout, pos: Cell, action: int) -> Cell:
    """Move one cell; bumping into a wall or the border leaves the agent in place.

    ``action`` must be an integer index in ``range(len(ACTIONS))``; anything else (including
    negative indices, which Python would silently wrap) raises ``ValueError``.
    """
    if (
        isinstance(action, bool)
        or not isinstance(action, int | np.integer)
        or not 0 <= action < len(ACTIONS)
    ):
        raise ValueError(f"invalid action index {action!r}; expected 0..{len(ACTIONS) - 1}")
    dr, dc = ACTIONS[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not (0 <= nxt[0] < N and 0 <= nxt[1] < N) or nxt in layout.walls:
        return pos
    return nxt


def run_episode(layout: Layout, actions: list[int], horizon: int) -> tuple[str, int]:
    """Execute actions from the start. Terminal = first pad entered (either glyph).

    Returns (outcome, steps) where outcome is a glyph or ``TIMEOUT``.
    """
    pos = layout.start
    for t, action in enumerate(actions[:horizon], start=1):
        pos = step(layout, pos, action)
        hit = layout.pad_at(pos)
        if hit is not None:
            return hit, t
    return TIMEOUT, min(len(actions), horizon)


def oracle_path(layout: Layout, goal: str, barrier: bool = True) -> list[int] | None:
    """Shortest action list from start to the ``goal`` pad, or None if unreachable.

    With ``barrier`` the opposite pad is impassable, because entering it ends the
    episode. ``barrier=False`` is only used to report detours, never for certification.
    """
    target = layout.pad(goal)
    blocked = set(layout.walls)
    if barrier:
        blocked.add(layout.pad(other_glyph(goal)))
    parent: dict[Cell, tuple[Cell, int] | None] = {layout.start: None}
    queue = deque([layout.start])
    while queue:
        pos = queue.popleft()
        if pos == target:
            actions: list[int] = []
            while parent[pos] is not None:
                pos, action = parent[pos]
                actions.append(action)
            return actions[::-1]
        for action in range(len(ACTIONS)):
            nxt = step(layout, pos, action)
            if nxt not in parent and nxt not in blocked:
                parent[nxt] = (pos, action)
                queue.append(nxt)
    return None


def d4(cell: Cell, k: int) -> Cell:
    """Element k (0..7) of the square's symmetry group: k%4 quarter turns, then a flip if k>=4."""
    r, c = cell
    for _ in range(k % 4):
        r, c = c, N - 1 - r
    if k >= 4:
        c = N - 1 - c
    return r, c


def canonical_topology(walls: frozenset[Cell] | set[Cell]) -> tuple[Cell, ...]:
    """Orientation-free wall key: the smallest sorted wall tuple over all 8 D4 images."""
    return min(tuple(sorted(d4(w, k) for w in walls)) for k in range(8))


# ---------------------------------------------------------------- rendering
CELL_PX = 5
COLORS = {
    "floor": (235, 235, 235),
    "wall": (40, 40, 40),
    "trim": (150, 150, 150),
    "pad": (40, 90, 200),
    "agent": (220, 40, 40),
}


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


GLYPH_MASKS = {
    "circle": _mask([".###.", "#####", "#####", "#####", ".###."]),
    "triangle": _mask(["..#..", ".###.", ".###.", "#####", "#####"]),
}
TRIM_MASKS = {  # cosmetic only: drawn on plain floor, never read by the dynamics
    "striped": _mask([".....", ".....", "#####", ".....", "....."]),
    "dotted": _mask([".....", ".#.#.", ".....", ".#.#.", "....."]),
}
AGENT_MASK = _mask([".....", ".###.", ".###.", ".###.", "....."])


def render(layout: Layout, agent: Cell, trim: str) -> np.ndarray:
    """RGB uint8 image (35, 35, 3). Goal images use ``agent=layout.pad(goal)``."""
    img = np.empty((N * CELL_PX, N * CELL_PX, 3), dtype=np.uint8)
    img[:] = COLORS["floor"]
    for r in range(N):
        for c in range(N):
            tile = img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX]
            glyph = layout.pad_at((r, c))
            if (r, c) in layout.walls:
                tile[:] = COLORS["wall"]
            elif glyph is not None:
                tile[GLYPH_MASKS[glyph]] = COLORS["pad"]
            else:
                tile[TRIM_MASKS[trim]] = COLORS["trim"]
            if (r, c) == agent:
                tile[AGENT_MASK] = COLORS["agent"]
    return img
