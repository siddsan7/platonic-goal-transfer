import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pgt import integrity
from pgt.data import (
    ARMS,
    MODEL_SEEDS,
    PILOT_SEED,
    SPLITS,
    TRIMS,
    CalibrationAnchor,
    Config,
    GenerationExhausted,
    IntegrityError,
    SplitLocked,
    TextPair,
    calibration_anchors,
    caption,
    eval_twins,
    generate_split,
    make_pairs,
    route_report,
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


def test_text_pairs_are_text_only_distinct_paraphrases():
    names = {f.name for f in dataclasses.fields(TextPair)}
    assert not names & integrity.IMAGE_FIELDS
    for pair in make_pairs(SMALL, 4, "clean"):
        assert isinstance(pair.left, str) and isinstance(pair.right, str)
        assert pair.left_template != pair.right_template and pair.left != pair.right
        assert integrity.factors(pair.left) == integrity.factors(pair.right) == pair.left_combo
    directions = Counter((p.left_template, p.right_template) for p in make_pairs(SMALL, 4, "clean"))
    assert directions == {(0, 1): 16, (1, 0): 16}


def test_arms_rewire_exactly_a_quarter_with_matched_marginals():
    stats = integrity.check_arms(SMALL)
    assert stats["selected_quartets"] == SMALL.n_train // 4
    assert stats["unique_pairs"] == {"clean": 8, "goal_swap": 16, "trim_swap": 16}
    arms = {arm: make_pairs(SMALL, SMALL.n_train, arm) for arm in ARMS}
    for arm, factor in (("goal_swap", 0), ("trim_swap", 1)):
        wrong = [p for p in arms[arm] if p.left_combo != p.right_combo]
        assert len(wrong) == len(arms["clean"]) // 4
        assert all(p.left_combo[1 - factor] == p.right_combo[1 - factor] for p in arms[arm])
        assert all(
            integrity.factors(p.right)[factor] != integrity.factors(p.left)[factor] for p in wrong
        )
        assert Counter(p.right for p in arms[arm]) == Counter(p.right for p in arms["clean"])
        assert [p.left for p in arms[arm]] == [p.left for p in arms["clean"]]


def test_arm_check_rejects_a_wrong_factor_flip(monkeypatch):
    def flip_trim_for_every_arm(cfg, n, arm):
        pairs = make_pairs(cfg, n, "clean")
        if arm == "clean":
            return pairs
        selected = set(swap_quartets(cfg, n))
        return [
            dataclasses.replace(
                p, right_combo=(p.left_combo[0], TRIMS[1 - TRIMS.index(p.left_combo[1])])
            )
            if p.quartet in selected
            else p
            for p in pairs
        ]

    monkeypatch.setattr(integrity, "make_pairs", flip_trim_for_every_arm)
    with pytest.raises(IntegrityError, match="goal_swap: rewiring flips the wrong factor"):
        integrity.check_arms(SMALL)


def test_swap_fraction_must_be_exact():
    with pytest.raises(ValueError):
        swap_quartets(SMALL, 66)


def test_captions_name_requested_factors_and_keep_heldout_separate():
    assert integrity.check_captions() == {"train_captions": 8, "heldout_captions": 8}
    assert integrity.FORBIDDEN.search("turn left at row 3")
    for heldout in (False, True):
        for t in range(2):
            for combo in [(g, tr) for g in GLYPHS for tr in TRIMS]:
                assert integrity.factors(caption(combo, t, heldout)) == combo
    with pytest.raises(IntegrityError):
        integrity.factors("go to the circle pad near the triangle in the dotted warehouse")


def test_caption_check_rejects_a_template_naming_the_wrong_factor(monkeypatch):
    monkeypatch.setattr(
        integrity, "caption", lambda combo, t, heldout=False: f"go to the circle pad {combo[1]}"
    )
    with pytest.raises(IntegrityError, match="wrong factors"):
        integrity.check_captions()


def test_calibration_anchors_balanced_and_from_distinct_layouts(small_layouts):
    layouts = small_layouts["calibration"]
    integrity.check_anchors(SMALL, layouts)
    anchors = calibration_anchors(SMALL, layouts)
    assert all(isinstance(a, CalibrationAnchor) for a in anchors)
    assert Counter(a.combo[0] for a in anchors) == {"circle": 8, "triangle": 8}
    a = anchors[0]
    lay = layouts[a.layout]
    assert np.array_equal(a.image(layouts), render(lay, lay.pad(a.combo[0]), a.combo[1]))
    assert integrity.factors(a.text) == a.combo


def test_model_seeds_fixed_and_pilot_is_first():
    assert MODEL_SEEDS == (11, 23, 47, 71, 101) and PILOT_SEED == 11
    assert Config().seed == 20261121


@pytest.mark.parametrize(
    "change",
    [
        {"min_route": 13},
        {"min_route": 0},
        {"max_walls": 5},
        {"max_walls": 47},
        {"split_fractions": (0.7, 0.1, 0.2, 0.1)},
        {"split_fractions": (0.8, 0.1, 0.0, 0.1)},
        {"n_validation": 30},
        {"n_validation": 0},
        {"max_candidates_per_quartet": 0},
        {"max_candidates_per_quartet": 1.1},
        {"max_candidates_per_quartet": 5.0},
        {"max_candidates_per_quartet": True},
        {"max_candidates_per_quartet": "5"},
    ],
)
def test_impossible_config_is_rejected_before_generation(change):
    with pytest.raises(ValueError, match="impossible Config"):
        generate_split(dataclasses.replace(SMALL, **change), "validation")


def test_unfillable_split_fails_after_bounded_attempts():
    # 46 walls fill the grid: start touches both pads, so no route of length >= 2 exists.
    full = dataclasses.replace(SMALL, min_walls=46, max_walls=46, max_candidates_per_quartet=5)
    with pytest.raises(GenerationExhausted, match=r"validation: found 0/32 .* 160 candidate"):
        generate_split(full, "validation")


def test_route_report_counts_routes_shared_by_twins_and_trims(small_layouts):
    layouts = small_layouts["validation"]
    report = route_report(layouts)
    assert sum(r["routes"] for r in report.values()) == len(layouts)
    episodes = sum(len(t.goals) for t in eval_twins(layouts))
    assert sum(r["episodes"] for r in report.values()) == episodes == 4 * len(layouts)
    for first, second in zip(layouts[::2], layouts[1::2], strict=True):
        for glyph in GLYPHS:  # twins share each route under the other glyph label
            other = GLYPHS[1 - GLYPHS.index(glyph)]
            assert oracle_path(first, glyph) == oracle_path(second, other)


def test_integrity_errors_are_raised_not_asserted():
    with pytest.raises(IntegrityError):
        integrity.check(False, "boom")


def test_committed_manifest_reproduces():
    assert integrity.main(["--check", str(ROOT / "manifests" / "default.json")]) == 0
