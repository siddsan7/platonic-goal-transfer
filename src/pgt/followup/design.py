"""Frozen design: 4 glyphs x 4 trims, cyclic seen/withheld split, budgets and health gate."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

GLYPHS4 = ("circle", "triangle", "square", "cross")
TRIMS4 = ("striped", "dotted", "checked", "ringed")
COMBOS16 = tuple((g, t) for g in GLYPHS4 for t in TRIMS4)


def is_seen(glyph: str, trim: str) -> bool:
    """Seen iff (glyph - trim) mod 4 in {0, 1}: a cyclic, connected 8 seen / 8 withheld split."""
    return (GLYPHS4.index(glyph) - TRIMS4.index(trim)) % 4 in (0, 1)


SEEN = tuple(c for c in COMBOS16 if is_seen(*c))
WITHHELD = tuple(c for c in COMBOS16 if not is_seen(*c))


def panel_glyphs(trim: str, seen: bool) -> tuple[str, str]:
    """The two glyphs whose (glyph, trim) combo is seen (``seen=True``) or withheld."""
    out = tuple(g for g in GLYPHS4 if is_seen(g, trim) == seen)
    assert len(out) == 2
    return out  # type: ignore[return-value]


ARMS = ("clean", "goal_swap", "trim_swap")
PRIMARY = ("trim_swap", "goal_swap")  # primary contrast: trim_swap - goal_swap, pp, withheld


@dataclass(frozen=True)
class FollowupDesign:
    """Every knob that defines the study. Frozen before any data is generated."""

    data_seed: int = 20261122
    horizon: int = 12
    min_route: int = 2
    min_walls: int = 6
    max_walls: int = 12
    corpus_seeds: tuple[int, ...] = (1, 2, 3, 4)
    anchor_banks: tuple[int, ...] = (1, 2, 3)
    visual_seed: int = 11  # one fixed visual fit; every result is conditional on it
    n_validation_topologies: int = 16
    n_test_topologies: int = 48
    n_anchors_per_seen_combo: int = 8  # anchors use SEEN combos only
    n_blocks: int = 64  # balanced pair blocks per corpus; swaps hit exactly 25% of blocks
    swap_fraction: float = 0.25
    cell_dims: int = 4
    floor_dims: int = 3
    text_hidden: int = 16
    text_out: int = 16
    text_epochs: int = 300
    text_lr: float = 0.05
    ridge_penalty: float = 1.0  # one fixed penalty for every bridge (no tuning)
    tie_tol: float = 1e-9
    # health gate (pre-declared; failing -> INCONCLUSIVE, never rerun with other seeds)
    gate_oracle_min: float = 1.0
    gate_blind_max: float = 0.5
    gate_image_goal_min: float = 0.8
    gate_seen_clean_min: float = 0.8
    # hard caps, seconds
    cap_build_s: int = 600
    cap_pilot_s: int = 900
    cap_test_s: int = 2400
    frame_cache_max: int = 4096

    def to_json(self) -> dict:
        return dataclasses.asdict(self) | {
            "glyphs": GLYPHS4,
            "trims": TRIMS4,
            "seen": SEEN,
            "withheld": WITHHELD,
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()
