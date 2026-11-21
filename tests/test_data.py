from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pgt import integrity
from pgt.data import (
    ARMS,
    SPLITS,
    Config,
    IntegrityError,
    SplitLocked,
    calibration_anchors,
    eval_twins,
    generate_split,
    make_pairs,
    split_of,
    swap_quartets,
)
from pgt.world import GLYPHS, canonical_topology, d4, oracle_path, render

SMALL = Config(n_train=64, n_calibration=32, n_validation=32, n_test=32, n_anchors=16)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small_layouts():
    return {s: generate_split(SMALL, s, unlock_test=True) for s in SPLITS}


def test_generation_is_deterministic(small_layouts):
    again = generate_split(SMALL, "validation")
    assert again == small_layouts["validation"]
    other_seed = generate_split(Config(**{**SMALL.__dict__, "seed": 1}), "validation")
    assert other_seed != again
    assert all(split_of(lay.walls, SMALL) == "validation" for lay in other_seed)


def test_test_split_is_locked():
    with pytest.raises(SplitLocked):
        generate_split(SMALL, "test")


def test_splits_pass_structural_checks_and_are_topology_disjoint(small_layouts):
    for split, layouts in small_layouts.items():
        integrity.check_split(SMALL, split, layouts)
    integrity.check_disjoint(small_layouts)
    keys = {s: {canonical_topology(lay.walls) for lay in lays} for s, lays in small_layouts.items()}
    for a in SPLITS:
        for b in SPLITS:
            assert a == b or not keys[a] & keys[b]


def test_every_orientation_lands_in_the_same_split(small_layouts):
    for split, layouts in small_layouts.items():
        for lay in layouts[::2]:
            for k in range(8):
                assert split_of(frozenset(d4(w, k) for w in lay.walls), SMALL) == split


def test_twins_are_equally_reachable_with_identical_start_view(small_layouts):
    layouts = small_layouts["validation"]
    for twin in eval_twins(layouts):
        lay = layouts[twin.quartet]
        lengths = [len(oracle_path(lay, g)) for g in twin.goals]
        assert all(SMALL.min_route <= n <= SMALL.horizon for n in lengths)
        start_view = render(lay, lay.start, twin.trim)  # one observation serves both goals
        goal_views = [render(lay, lay.pad(g), twin.trim) for g in twin.goals]
        assert not np.array_equal(*goal_views)
        assert all(v.shape == start_view.shape for v in goal_views)


def test_pad_positions_are_independent_of_glyph(small_layouts):
    for layouts in small_layouts.values():
        assert Counter(lay.pad("circle") for lay in layouts) == Counter(
            lay.pad("triangle") for lay in layouts
        )


def test_arms_swap_exactly_a_quarter_with_matched_marginals():
    stats = integrity.check_arms(SMALL)
    assert stats["selected_quartets"] == SMALL.n_train // 4
    arms = {arm: make_pairs(SMALL, SMALL.n_train, arm) for arm in ARMS}
    goal_wrong = sum(p.image[0] != p.text_combo[0] for p in arms["goal_swap"])
    trim_wrong = sum(p.image[1] != p.text_combo[1] for p in arms["trim_swap"])
    assert goal_wrong == trim_wrong == len(arms["clean"]) // 4
    assert all(p.image[1] == p.text_combo[1] for p in arms["goal_swap"])
    assert all(p.image[0] == p.text_combo[0] for p in arms["trim_swap"])


def test_swap_fraction_must_be_exact():
    with pytest.raises(ValueError):
        swap_quartets(SMALL, 66)


def test_captions_carry_no_ids_coordinates_or_actions():
    integrity.check_captions()
    assert integrity.FORBIDDEN.search("turn left at row 3")
    for pair in make_pairs(SMALL, 4, "clean"):
        assert set(pair.text.split()) & set(GLYPHS)


def test_calibration_anchors_balanced_and_from_distinct_layouts(small_layouts):
    integrity.check_anchors(SMALL, small_layouts["calibration"])
    anchors = calibration_anchors(SMALL, small_layouts["calibration"])
    assert Counter(a.image[0] for a in anchors) == {"circle": 8, "triangle": 8}


def test_integrity_errors_are_raised_not_asserted():
    with pytest.raises(IntegrityError):
        integrity.check(False, "boom")


def test_committed_manifest_reproduces():
    assert integrity.main(["--check", str(ROOT / "manifests" / "default.json")]) == 0
