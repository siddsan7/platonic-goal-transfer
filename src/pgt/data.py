"""Layouts, topology splits, captions, text-only pair arms, calibration anchors, eval twins."""

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
    render,
)

SPLITS = ("train", "calibration", "validation", "test")
ARMS = ("clean", "goal_swap", "trim_swap")
SPLIT_SALT = "pgt-topology-split-v1"  # fixed, so every run seed sees the same splits
COMBOS = tuple((g, t) for g in GLYPHS for t in TRIMS)  # one balanced quartet

# Model-initialisation seeds for milestone 2; the data seed (Config.seed) never changes.
# PILOT_SEED runs the validation-only pilot. The test split stays locked until the config is
# frozen and the pilot is done; only then is it unlocked explicitly (unlock_test=True).
MODEL_SEEDS = (11, 23, 47, 71, 101)
PILOT_SEED = MODEL_SEEDS[0]

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
    max_candidates_per_quartet: int = 100  # default config needs at most ~6.5 (measured)

    def n_quartets(self, split: str) -> int:
        return getattr(self, f"n_{split}")

    def validate(self, split: str) -> None:
        """Reject parameter combinations that can never produce ``split``."""
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        fractions = self.split_fractions
        problems = [
            (self.min_route < 1, "min_route must be >= 1"),
            (self.min_route > self.horizon, "min_route exceeds horizon"),
            (self.min_walls < 0, "min_walls must be >= 0"),
            (self.max_walls < self.min_walls, "max_walls < min_walls"),
            (self.max_walls + 3 > N * N, "max_walls leaves no room for start and two pads"),
            (len(fractions) != len(SPLITS), "need one split fraction per split"),
            (any(f < 0 for f in fractions), "split fractions must be >= 0"),
            (abs(sum(fractions) - 1) > 1e-9, "split fractions must sum to 1"),
        ]
        budget = self.max_candidates_per_quartet
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            problems.append((True, "max_candidates_per_quartet must be a positive integer"))
        bad = [msg for failed, msg in problems if failed]
        if len(fractions) == len(SPLITS) and fractions[SPLITS.index(split)] <= 0:
            bad.append(f"split {split!r} has fraction 0")
        n = self.n_quartets(split)
        if n < 4 or n % 4:
            bad.append(f"{split}: quartet count {n} must be a positive multiple of 4")
        if bad:
            raise ValueError("impossible Config: " + "; ".join(bad))


class SplitLocked(RuntimeError):
    """Raised when test layouts are requested without an explicit unlock."""


class IntegrityError(RuntimeError):
    pass


class GenerationExhausted(RuntimeError):
    """Raised when a split cannot be filled within the bounded candidate budget."""


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
    Raises ``ValueError`` for an impossible Config and ``GenerationExhausted`` after
    ``n * cfg.max_candidates_per_quartet`` candidates without filling the split.
    """
    if split == "test" and not unlock_test:
        raise SplitLocked("test split is locked until the config is frozen and the pilot is done")
    cfg.validate(split)
    n = cfg.n_quartets(split)
    budget = n * cfg.max_candidates_per_quartet
    rng = rng_for(cfg, "layouts", split)
    seen: set[tuple] = set()
    out: list[Layout] = []
    attempts = 0
    while len(out) < n:
        if attempts >= budget:
            raise GenerationExhausted(
                f"{split}: found {len(out)}/{n} quartets after {budget} candidate attempts"
            )
        attempts += 1
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
class TextPair:
    """Text-only training pair: two captions written with distinct TRAIN_TEMPLATES.

    In a clean pair both sides describe the same complete (glyph, trim) state. There is no
    image or layout here; ``quartet`` only indexes a balanced block of the four combos.
    """

    quartet: int
    left_template: int
    left_combo: tuple[str, str]
    right_template: int
    right_combo: tuple[str, str]

    @property
    def left(self) -> str:
        return caption(self.left_combo, self.left_template)

    @property
    def right(self) -> str:
        return caption(self.right_combo, self.right_template)


# (left template, right template); both directions so each template is equally often per side.
DIRECTIONS = ((0, 1), (1, 0))


def swap_quartets(cfg: Config, n_quartets: int) -> list[int]:
    """The exact quartet indices whose right endpoints are rewired; shared by both swap arms."""
    count = n_quartets * cfg.swap_fraction
    if count != int(count):
        raise ValueError("swap fraction must select a whole number of quartets")
    picked = rng_for(cfg, "swap-quartets").permutation(n_quartets)[: int(count)]
    return sorted(int(i) for i in picked)


def make_pairs(cfg: Config, n_quartets: int, arm: str) -> list[TextPair]:
    """Four combos x two template directions = 8 text pairs per quartet.

    In the selected quartets the right endpoint is rewired to the combo with one factor
    flipped (glyph for ``goal_swap``, trim for ``trim_swap``), keeping its template. The flip
    is a bijection on the quartet's four combos, so both caption multisets and all word
    frequencies match across arms.

    Only 8 distinct clean pairs exist (8 more distinct rewired pairs per swap arm); repeating
    them across quartets sets their weights, it does not add independent text data.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    selected = set(swap_quartets(cfg, n_quartets)) if arm != "clean" else set()
    pairs = []
    for q in range(n_quartets):
        for left_t, right_t in DIRECTIONS:
            for glyph, trim in COMBOS:
                right = (glyph, trim)
                if q in selected:
                    right = (
                        (other_glyph(glyph), trim)
                        if arm == "goal_swap"
                        else (glyph, other_trim(trim))
                    )
                pairs.append(TextPair(q, left_t, (glyph, trim), right_t, right))
    return pairs


@dataclass(frozen=True)
class CalibrationAnchor:
    """Explicit image-to-text record: the goal image of calibration layout ``layout`` for
    ``combo`` with its own caption. Fits the text-to-image map only; never text training."""

    layout: int
    combo: tuple[str, str]
    template: int

    @property
    def text(self) -> str:
        return caption(self.combo, self.template)

    def image(self, layouts: list[Layout]) -> np.ndarray:
        lay = layouts[self.layout]
        glyph, trim = self.combo
        return render(lay, lay.pad(glyph), trim)


def calibration_anchors(cfg: Config, layouts: list[Layout]) -> list[CalibrationAnchor]:
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
        CalibrationAnchor(2 * int(twin) + int(side), cells[i][0], cells[i][1])
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
    """Counts per certified route length, from twin-ordered layouts (2i, 2i+1).

    ``routes``: distinct geometric start-to-pad routes, counted once per twin pair; the twins
    share walls, start and pad cells, so every route is shared by both glyph labels.
    ``episodes``: evaluation episodes that use those routes = routes x 2 glyph labels x 2 trims
    (trim is cosmetic and never changes a route). ``detours``: routes longer than Manhattan.
    """
    report: dict[str, dict[str, int]] = {}
    for layout in layouts[::2]:
        for glyph, length in route_lengths(layout).items():
            pad = layout.pad(glyph)
            manhattan = abs(pad[0] - layout.start[0]) + abs(pad[1] - layout.start[1])
            row = report.setdefault(f"{length:02d}", {"routes": 0, "episodes": 0, "detours": 0})
            row["routes"] += 1
            row["episodes"] += len(GLYPHS) * len(TRIMS)
            row["detours"] += int(length > manhattan)
    return dict(sorted(report.items()))


def config_json(cfg: Config) -> dict:
    return asdict(cfg) | {
        "split_salt": SPLIT_SALT,
        "train_templates": TRAIN_TEMPLATES,
        "heldout_templates": HELDOUT_TEMPLATES,
        "model_seeds": MODEL_SEEDS,
        "pilot_seed": PILOT_SEED,
    }
