import itertools

import numpy as np
import pytest

from pgt.data import Config, generate_split
from pgt.world import (
    CELL_PX,
    GLYPHS,
    TIMEOUT,
    Layout,
    canonical_topology,
    d4,
    oracle_path,
    render,
    run_episode,
    step,
)

UP, DOWN, LEFT, RIGHT = range(4)


def corridor(walls):
    """Start top-left, triangle two cells right, circle four cells right."""
    return Layout(frozenset(walls), (0, 0), ((0, 4), (0, 2)))


def test_walls_and_border_block_movement():
    lay = corridor({(1, 0)})
    assert step(lay, (0, 0), UP) == (0, 0)
    assert step(lay, (0, 0), LEFT) == (0, 0)
    assert step(lay, (0, 0), DOWN) == (0, 0)
    assert step(lay, (0, 0), RIGHT) == (0, 1)


@pytest.mark.parametrize("bad", [-1, 4, True, 1.0, "0"])
def test_invalid_action_index_is_rejected(bad):
    lay = corridor(set())
    with pytest.raises(ValueError, match="invalid action"):
        step(lay, (0, 0), bad)
    with pytest.raises(ValueError, match="invalid action"):
        run_episode(lay, [RIGHT, bad], horizon=12)
    assert step(lay, (0, 0), np.int64(RIGHT)) == (0, 1)


def test_first_pad_entered_is_terminal_even_if_wrong():
    lay = corridor(set())
    assert run_episode(lay, [RIGHT] * 6, horizon=12) == ("triangle", 2)
    assert run_episode(lay, [DOWN, UP], horizon=12) == (TIMEOUT, 2)
    assert run_episode(lay, [DOWN] * 20, horizon=12) == (TIMEOUT, 12)


def test_opposite_pad_is_a_barrier_for_certification():
    sealed = corridor({(1, c) for c in range(6)})  # row 0 only exits through the triangle
    assert oracle_path(sealed, "circle") is None
    assert len(oracle_path(sealed, "circle", barrier=False)) == 4
    detour = corridor({(1, 1), (1, 2), (1, 3)})
    path = oracle_path(detour, "circle")
    assert len(path) == 8
    assert run_episode(detour, path, horizon=12) == ("circle", 8)


def test_oracle_is_optimal_against_exhaustive_search():
    layouts = [lay for lay in generate_split(Config(n_validation=32), "validation")]
    checked = 0
    for lay in layouts:
        for glyph in GLYPHS:
            path = oracle_path(lay, glyph)
            if len(path) > 5:
                continue
            checked += 1
            assert run_episode(lay, path, 12) == (glyph, len(path))
            for seq in itertools.product(range(4), repeat=len(path) - 1):
                assert run_episode(lay, list(seq), 12)[0] != glyph
    assert checked >= 10


def test_trim_is_cosmetic_and_visible():
    lay = corridor({(1, 1), (3, 3)})
    striped, dotted = render(lay, lay.start, "striped"), render(lay, lay.start, "dotted")
    assert striped.shape == (35, 35, 3) and striped.dtype == np.uint8
    assert not np.array_equal(striped, dotted)
    diff = np.any(striped != dotted, axis=2)
    for cell in [*lay.walls, *lay.pads]:  # trim never touches walls or pads
        r, c = cell
        assert not diff[r * CELL_PX : (r + 1) * CELL_PX, c * CELL_PX : (c + 1) * CELL_PX].any()


def test_goal_image_marks_target_on_same_map():
    lay = corridor({(1, 1)})
    start_img = render(lay, lay.start, "dotted")
    for glyph in GLYPHS:
        goal_img = render(lay, lay.pad(glyph), "dotted")
        changed = {
            (r // CELL_PX, c // CELL_PX)
            for r, c in zip(*np.nonzero(np.any(goal_img != start_img, axis=2)), strict=True)
        }
        assert changed == {lay.start, lay.pad(glyph)}
    assert not np.array_equal(
        render(lay, lay.pads[0], "dotted"), render(lay, lay.pads[1], "dotted")
    )


@pytest.mark.parametrize("k", range(8))
def test_canonical_topology_is_d4_invariant(k):
    walls = {(0, 1), (2, 5), (3, 3), (6, 0)}
    assert canonical_topology({d4(w, k) for w in walls}) == canonical_topology(walls)


def test_d4_elements_are_distinct():
    cell = (0, 1)
    assert len({d4(cell, k) for k in range(8)}) == 8
