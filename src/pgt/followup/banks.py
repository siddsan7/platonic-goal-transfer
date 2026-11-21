"""Fresh validation/test boards; topologies of every previously evaluated bank are excluded."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

from pgt.followup.design import FollowupDesign
from pgt.followup.world4 import Board, certified
from pgt.world import N, canonical_topology

PLACEHOLDER = ("circle", "triangle")  # glyphs are re-assigned per panel at query time


class BankExhausted(RuntimeError):
    pass


def bank_rng(design: FollowupDesign, name: str) -> np.random.Generator:
    tag = int.from_bytes(hashlib.sha256(f"followup/{name}".encode()).digest()[:8], "little")
    return np.random.default_rng([design.data_seed, tag])


def generate_bank(
    design: FollowupDesign,
    name: str,
    n: int,
    excluded: set[tuple],
    rng: np.random.Generator | None = None,
    max_attempts_per_board: int = 200,
    accept_topology: Callable[[frozenset], bool] | None = None,
) -> list[Board]:
    """``n`` certified boards with distinct canonical topologies not in ``excluded``.

    ``excluded`` is updated in place so a later bank cannot reuse this bank's topologies.
    Raises BankExhausted instead of returning a partial bank.
    """
    rng = bank_rng(design, name) if rng is None else rng
    out: list[Board] = []
    for _ in range(n * max_attempts_per_board):
        if len(out) == n:
            break
        k = int(rng.integers(design.min_walls, design.max_walls + 1))
        cells = [divmod(int(i), N) for i in rng.permutation(N * N)[: k + 3]]
        walls = frozenset(cells[:k])
        key = canonical_topology(walls)
        if key in excluded or (accept_topology is not None and not accept_topology(walls)):
            continue
        board = Board(walls, cells[k], (cells[k + 1], cells[k + 2]), PLACEHOLDER)
        if not certified(board, design.min_route, design.horizon):
            continue
        excluded.add(key)
        out.append(board)
    if len(out) < n:
        raise BankExhausted(f"{name}: {len(out)}/{n} boards")
    return out


def with_glyphs(board: Board, glyphs: tuple[str, str]) -> Board:
    """Certification depends only on cells, so it holds for every glyph assignment."""
    return Board(board.walls, board.start, board.pads, glyphs)


def prior_topologies(layout_jsons: list[dict]) -> set[tuple]:
    """Canonical keys from earlier evaluated banks (e.g. original validation/test layouts)."""
    return {canonical_topology(frozenset(tuple(w) for w in lay["walls"])) for lay in layout_jsons}
