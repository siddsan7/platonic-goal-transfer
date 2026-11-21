"""Two-pad boards with 4 glyphs x 4 trims. Dynamics, D4 key and BFS reuse ``pgt.world``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from pgt.followup.design import GLYPHS4, TRIMS4
from pgt.world import ACTIONS, AGENT_MASK, CELL_PX, COLORS, TIMEOUT, N, _mask, step

Cell = tuple[int, int]

# Every glyph differs on the outer ring of its tile, which stays visible under the agent.
GLYPH4_MASKS = {
    "circle": _mask([".###.", "#####", "#####", "#####", ".###."]),
    "triangle": _mask(["..#..", ".###.", ".###.", "#####", "#####"]),
    "square": _mask(["#####", "#####", "#####", "#####", "#####"]),
    "cross": _mask(["..#..", "..#..", "#####", "..#..", "..#.."]),
}
# Floor trims: cosmetic, never read by the dynamics; each has a distinct outer-ring texture.
TRIM4_MASKS = {
    "striped": _mask([".....", ".....", "#####", ".....", "....."]),
    "dotted": _mask([".....", ".#.#.", ".....", ".#.#.", "....."]),
    "checked": _mask(["#...#", ".....", ".....", ".....", "#...#"]),
    "ringed": _mask(["#####", "#...#", "#...#", "#...#", "#####"]),
}


@dataclass(frozen=True)
class Board:
    """Walls, start and two pads; ``glyphs[i]`` is drawn on ``pads[i]``."""

    walls: frozenset[Cell]
    start: Cell
    pads: tuple[Cell, Cell]
    glyphs: tuple[str, str]

    def pad(self, glyph: str) -> Cell:
        return self.pads[self.glyphs.index(glyph)]

    def pad_at(self, cell: Cell) -> str | None:
        for g, p in zip(self.glyphs, self.pads, strict=True):
            if p == cell:
                return g
        return None

    def other(self, glyph: str) -> str:
        return self.glyphs[1 - self.glyphs.index(glyph)]

    def swapped(self) -> Board:
        return Board(self.walls, self.start, self.pads, self.glyphs[::-1])


def oracle_path(board: Board, goal: str) -> list[int] | None:
    """Shortest route to ``goal`` with the other pad impassable (entering it ends the episode)."""
    target = board.pad(goal)
    blocked = set(board.walls) | {board.pad(board.other(goal))}
    parent: dict[Cell, tuple[Cell, int] | None] = {board.start: None}
    queue = deque([board.start])
    while queue:
        pos = queue.popleft()
        if pos == target:
            out: list[int] = []
            while parent[pos] is not None:
                pos, a = parent[pos]
                out.append(a)
            return out[::-1]
        for a in range(len(ACTIONS)):
            nxt = step(board, pos, a)  # step only reads ``walls``
            if nxt not in parent and nxt not in blocked:
                parent[nxt] = (pos, a)
                queue.append(nxt)
    return None


def certified(board: Board, min_route: int, horizon: int) -> bool:
    for g in board.glyphs:
        p = oracle_path(board, g)
        if p is None or not min_route <= len(p) <= horizon:
            return False
    return True


def execute(board: Board, pos: Cell, action: int, t: int, horizon: int) -> tuple[Cell, str | None]:
    """Simulator side: apply the action, report the first terminal (a glyph) or timeout."""
    nxt = step(board, pos, action)
    hit = board.pad_at(nxt)
    if hit is None and t + 1 >= horizon:
        return nxt, TIMEOUT
    return nxt, hit


def render4(board: Board, agent: Cell, trim: str) -> np.ndarray:
    """RGB uint8 (35, 35, 3), same geometry and palette as ``pgt.world.render``."""
    if trim not in TRIMS4 or any(g not in GLYPHS4 for g in board.glyphs):
        raise ValueError("unknown glyph or trim")
    img = np.empty((N * CELL_PX, N * CELL_PX, 3), dtype=np.uint8)
    img[:] = COLORS["floor"]
    for r in range(N):
        for c in range(N):
            tile = img[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX]
            glyph = board.pad_at((r, c))
            if (r, c) in board.walls:
                tile[:] = COLORS["wall"]
            elif glyph is not None:
                tile[GLYPH4_MASKS[glyph]] = COLORS["pad"]
            else:
                tile[TRIM4_MASKS[trim]] = COLORS["trim"]
            if (r, c) == agent:
                tile[AGENT_MASK] = COLORS["agent"]
    return img
