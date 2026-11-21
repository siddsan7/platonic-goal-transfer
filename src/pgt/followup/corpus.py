"""Text corpora for the follow-up: seen combos only, matched one-factor in-support flips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgt.followup.design import ARMS, GLYPHS4, SEEN, TRIMS4, FollowupDesign, is_seen

# Real phrasings with different word co-occurrences; each corpus seed draws two of them.
TEMPLATE_POOL = (
    "go to the {glyph} pad in the {trim} warehouse",
    "the goal is the {glyph} marker on {trim} flooring",
    "walk over to the {glyph} sign beside {trim} tiles",
    "stop at the {glyph} plate in a {trim} hall",
    "reach the {glyph} slab on the {trim} deck",
    "your target is the {glyph} pad with {trim} trim",
)
HELDOUT_TEMPLATES = (  # validation/test phrasing only, never in any training corpus
    "head for the {glyph} tile where the floor is {trim}",
    "finish on the {glyph} spot of this {trim} room",
)


def caption(combo: tuple[str, str], template: str) -> str:
    return template.format(glyph=combo[0], trim=combo[1])


def glyph_flip(combo: tuple[str, str]) -> tuple[str, str]:
    """The other glyph seen with this trim (a bijection on SEEN that keeps the trim)."""
    g, t = combo
    return next(x for x in GLYPHS4 if x != g and is_seen(x, t)), t


def trim_flip(combo: tuple[str, str]) -> tuple[str, str]:
    """The other trim seen with this glyph (a bijection on SEEN that keeps the glyph)."""
    g, t = combo
    return g, next(x for x in TRIMS4 if x != t and is_seen(g, x))


@dataclass(frozen=True)
class Pair:
    block: int
    left: str
    right: str
    left_combo: tuple[str, str]
    right_combo: tuple[str, str]


def corpus_templates(corpus_seed: int) -> tuple[str, str]:
    """Two distinct training templates; they vary independently with the corpus seed."""
    idx = np.random.default_rng([corpus_seed, 7001]).permutation(len(TEMPLATE_POOL))[:2]
    return TEMPLATE_POOL[int(idx[0])], TEMPLATE_POOL[int(idx[1])]


def swap_blocks(design: FollowupDesign, corpus_seed: int) -> list[int]:
    """Exactly ``swap_fraction`` of blocks; identical for both swap arms of one corpus."""
    count = design.n_blocks * design.swap_fraction
    if count != int(count) or count <= 0:
        raise ValueError("swap fraction must select a whole, positive number of blocks")
    rng = np.random.default_rng([corpus_seed, 7002])
    return sorted(int(i) for i in rng.permutation(design.n_blocks)[: int(count)])


def make_corpus(design: FollowupDesign, corpus_seed: int, arm: str) -> list[Pair]:
    """Each block holds the 8 seen combos x 2 template directions (16 pairs).

    In swapped blocks the right endpoint is rewired by one in-support factor flip. The flip is
    a bijection on SEEN, so the left and right caption pools are identical multisets in every
    arm. Withheld combos never appear in any corpus.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    ta, tb = corpus_templates(corpus_seed)
    flip = {"goal_swap": glyph_flip, "trim_swap": trim_flip}.get(arm)
    chosen = set(swap_blocks(design, corpus_seed)) if flip else set()
    pairs = []
    for b in range(design.n_blocks):
        for lt, rt in ((ta, tb), (tb, ta)):
            for combo in SEEN:
                right = flip(combo) if b in chosen else combo
                pairs.append(Pair(b, caption(combo, lt), caption(right, rt), combo, right))
    return pairs


def vocabulary() -> tuple[str, ...]:
    """Fixed vocabulary over the training pool and all 16 combos (arm independent)."""
    words = {
        w
        for t in TEMPLATE_POOL
        for g in GLYPHS4
        for tr in TRIMS4
        for w in caption((g, tr), t).split()
    }
    return tuple(sorted(words))
