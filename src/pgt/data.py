"""Layouts, topology splits, captions, matched pair arms, calibration anchors, eval twins."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from pgt.world import (
    GLYPHS,
    TRIMS,
    Layout,
    N,
    canonical_topology,
    oracle_path,
    other_glyph,
    other_trim,
)

SPLITS = ("train", "calibration", "validation", "test")
ARMS = ("clean", "goal_swap", "trim_swap")
SPLIT_SALT = "pgt-topology-split-v1"  # fixed, so every run seed sees the same splits
COMBOS = tuple((g, t) for g in GLYPHS for t in TRIMS)  # one balanced quartet

# Exactly one glyph word and one trim word per template, so a swap is one token edit.
TRAIN_TEMPLATES = (
    "go to the {glyph} pad in the {trim} warehouse",
    "the goal is the {glyph} marker on {trim} flooring",
)
HELDOUT_TEMPLATES = (  # validation/test phrasing only; same factor vocabulary
    "head for the {glyph} tile where the floor is {trim}",
    "finish on the {glyph} spot of this {trim} room",
)


@dataclass(frozen=True)
class Config:
    seed: int = 20261121
    horizon: int = 12
    min_route: int = 2
    min_walls: int = 6
    max_walls: int = 12
    split_fractions: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1)
    n_train: int = 2048
    n_calibration: int = 256
    n_validation: int = 512
    n_test: int = 512
    swap_fraction: float = 0.25
    n_anchors: int = 128

    def n_quartets(self, split: str) -> int:
        return getattr(self, f"n_{split}")


class SplitLocked(RuntimeError):
    """Raised when test layouts are requested without an explicit unlock."""


class IntegrityError(RuntimeError):
    pass


def rng_for(cfg: Config, *names: str) -> np.random.Generator:
    """Independent, stable random stream per (seed, purpose)."""
    tag = int.from_bytes(hashlib.sha256("/".join(names).encode()).digest()[:8], "little")
    return np.random.default_rng([cfg.seed, tag])


def split_of(walls: frozenset, cfg: Config) -> str:
    key = repr(canonical_topology(walls))
    u = int.from_bytes(hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).digest()[:8], "big") / 2**64
    edge = 0.0
    for name, frac in zip(SPLITS, cfg.split_fractions, strict=True):
        edge += frac
        if u < edge:
            return name
    return SPLITS[-1]


def route_lengths(layout: Layout) -> dict[str, int | None]:
    paths = {g: oracle_path(layout, g) for g in GLYPHS}
    return {g: None if p is None else len(p) for g, p in paths.items()}


def eligible(layout: Layout, cfg: Config) -> bool:
    """Both twins certified: each pad reachable within the horizon, opposite pad as barrier."""
    lengths = route_lengths(layout).values()
    return all(n is not None and cfg.min_route <= n <= cfg.horizon for n in lengths)


def generate_split(cfg: Config, split: str, unlock_test: bool = False) -> list[Layout]:
    """Quartet layouts for one split, in label-swapped twins (2i, 2i+1).

    Pad cells are drawn as an unordered pair and both glyph assignments are kept,
    so glyph is independent of pad position and the position marginals match exactly.
    Each twin pair owns one canonical wall topology not reused anywhere in the split.
    """
    if split == "test" and not unlock_test:
        raise SplitLocked("test split is locked until the protocol is frozen")
    n = cfg.n_quartets(split)
    if n % 4:
        raise ValueError("quartet counts must be multiples of 4 for an exact 25% swap")
    rng = rng_for(cfg, "layouts", split)
    seen: set[tuple] = set()
    out: list[Layout] = []
    while len(out) < n:
        k = int(rng.integers(cfg.min_walls, cfg.max_walls + 1))
        cells = [divmod(int(i), N) for i in rng.permutation(N * N)[: k + 3]]
        walls = frozenset(cells[:k])
        start, a, b = cells[k:]
        key = canonical_topology(walls)
        if key in seen or split_of(walls, cfg) != split:
            continue
        first = Layout(walls, start, (a, b))
        if not eligible(first, cfg):  # eligibility is symmetric in the glyph assignment
            continue
        seen.add(key)
        out += [first, Layout(walls, start, (b, a))]
    return out


# ---------------------------------------------------------------- captions and arms
def caption(combo: tuple[str, str], template: int, heldout: bool = False) -> str:
    templates = HELDOUT_TEMPLATES if heldout else TRAIN_TEMPLATES
    glyph, trim = combo
    return templates[template].format(glyph=glyph, trim=trim)


@dataclass(frozen=True)
class Pair:
    """Left endpoint = goal image of (quartet, image combo); right endpoint = caption text."""

    quartet: int
    image: tuple[str, str]
    template: int
    text_combo: tuple[str, str]

    @property
    def text(self) -> str:
        return caption(self.text_combo, self.template)


def swap_quartets(cfg: Config, n_quartets: int) -> list[int]:
    """The exact quartet indices whose right endpoints are swapped; shared by both swap arms."""
    count = n_quartets * cfg.swap_fraction
    if count != int(count):
        raise ValueError("swap fraction must select a whole number of quartets")
    picked = rng_for(cfg, "swap-quartets").permutation(n_quartets)[: int(count)]
    return sorted(int(i) for i in picked)


def make_pairs(cfg: Config, n_quartets: int, arm: str) -> list[Pair]:
    """Two templates x four combos per quartet. Swaps are bijections inside a quartet that
    flip exactly one factor of the caption, so every arm has the same caption multiset."""
    if arm not in ARMS:
        raise ValueError(arm)
    selected = set(swap_quartets(cfg, n_quartets)) if arm != "clean" else set()
    pairs = []
    for q in range(n_quartets):
        for template in range(len(TRAIN_TEMPLATES)):
            for glyph, trim in COMBOS:
                text = (glyph, trim)
                if q in selected:
                    text = (
                        (other_glyph(glyph), trim)
                        if arm == "goal_swap"
                        else (glyph, other_trim(trim))
                    )
                pairs.append(Pair(q, (glyph, trim), template, text))
    return pairs


def calibration_anchors(cfg: Config, layouts: list[Layout]) -> list[Pair]:
    """Clean anchors, one per calibration topology, balanced over glyph, trim and template."""
    twins = len(layouts) // 2
    per_cell = cfg.n_anchors // (len(COMBOS) * len(TRAIN_TEMPLATES))
    if per_cell * len(COMBOS) * len(TRAIN_TEMPLATES) != cfg.n_anchors or cfg.n_anchors > twins:
        raise ValueError("anchor count must be a multiple of 8 and <= calibration topologies")
    rng = rng_for(cfg, "anchors")
    cells = [(c, t) for c in COMBOS for t in range(len(TRAIN_TEMPLATES))] * per_cell
    order = rng.permutation(len(cells))
    twin_ids = rng.permutation(twins)[: cfg.n_anchors]
    sides = rng.integers(0, 2, size=cfg.n_anchors)
    return [
        Pair(2 * int(twin) + int(side), cells[i][0], cells[i][1], cells[i][0])
        for twin, side, i in zip(twin_ids, sides, order, strict=True)
    ]


@dataclass(frozen=True)
class TwinEpisode:
    """Same layout, start and start image; only the goal (glyph) differs between twins."""

    quartet: int
    trim: str
    goals: tuple[str, str] = GLYPHS


def eval_twins(layouts: list[Layout]) -> list[TwinEpisode]:
    return [TwinEpisode(q, trim) for q in range(len(layouts)) for trim in TRIMS]


# ---------------------------------------------------------------- reporting
def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def route_report(layouts: list[Layout]) -> dict[str, dict[str, int]]:
    """Episodes per certified route length; a detour means the route exceeds Manhattan distance."""
    report: dict[str, dict[str, int]] = {}
    for layout in layouts:
        for glyph, length in route_lengths(layout).items():
            pad = layout.pad(glyph)
            manhattan = abs(pad[0] - layout.start[0]) + abs(pad[1] - layout.start[1])
            row = report.setdefault(f"{length:02d}", {"episodes": 0, "detours": 0})
            row["episodes"] += 1
            row["detours"] += int(length > manhattan)
    return dict(sorted(report.items()))


def config_json(cfg: Config) -> dict:
    return asdict(cfg) | {
        "split_salt": SPLIT_SALT,
        "train_templates": TRAIN_TEMPLATES,
        "heldout_templates": HELDOUT_TEMPLATES,
    }
