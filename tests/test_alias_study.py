"""Alias-study tests on hand-coded fixtures and mocked models only: no fitting, no real boards."""

import json
import math
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from pgt.alias_study import study as S
from pgt.followup import engine
from pgt.followup.corpus import HELDOUT_TEMPLATES, TEMPLATE_POOL, corpus_templates, vocabulary
from pgt.followup.design import GLYPHS4, SEEN, TRIMS4, WITHHELD, FollowupDesign
from pgt.followup.fitting import BagOfWords
from pgt.followup.world4 import Board

DESIGN = FollowupDesign()
BOARD = Board(frozenset({(3, 3)}), (0, 0), ((0, 3), (3, 0)), ("circle", "triangle"))


# ---------------------------------------------------------------- lexicon and pairs
def test_nonces_are_novel_and_vocab_is_shared():
    words = set(vocabulary()) | set(GLYPHS4) | set(TRIMS4)
    words |= {w for t in (*TEMPLATE_POOL, *HELDOUT_TEMPLATES) for w in t.split()}
    assert len(set(S.NONCES)) == 4 and not set(S.NONCES) & words
    assert set(S.vocab()) == set(vocabulary()) | set(S.NONCES)


def test_mappings_are_latin_rotations_in_two_blocks():
    maps = S.mappings()
    assert len(maps) == 8 and maps == S.mappings()  # fixed seeds -> deterministic
    for m in maps:
        assert sorted(m) == sorted(S.NONCES) and sorted(m.values()) == sorted(GLYPHS4)
    for block in (maps[:4], maps[4:]):
        for a in S.NONCES:
            assert sorted(m[a] for m in block) == sorted(GLYPHS4)  # each alias -> each glyph once


def _marginals(c: Counter):
    left, right = Counter(), Counter()
    for (lt, rt), n in c.items():
        left[lt] += n
        right[rt] += n
    return left, right


def test_link_counts_matched_marginals_and_semantics():
    m = S.mappings()[0]
    ta, tb = corpus_templates(S.CORPUS_SEED)
    aligned, indep = S.link_counts(m, "aligned"), S.link_counts(m, "independent")
    assert sum(aligned.values()) == sum(indep.values()) == 4 * 4 * 2 * 4
    assert _marginals(aligned) == _marginals(indep)
    for alias in S.NONCES:
        for trim in TRIMS4:
            a_cap = ta.format(glyph=alias, trim=trim)
            for g in GLYPHS4:
                c_cap = tb.format(glyph=g, trim=trim)
                assert aligned[(a_cap, c_cap)] == (4 if g == m[alias] else 0)
                assert indep[(a_cap, c_cap)] == 1
                # reverse direction, other template: canonical (a) -> alias (b)
                rev = (ta.format(glyph=g, trim=trim), tb.format(glyph=alias, trim=trim))
                assert aligned[rev] == (4 if g == m[alias] else 0) and indep[rev] == 1
    for c in (aligned, indep):  # trim kept, alias on exactly one side, templates differ
        for lt, rt in c:
            assert sum(w in S.NONCES for w in (lt + " " + rt).split()) == 1
            assert {w for w in lt.split() if w in TRIMS4} == {w for w in rt.split() if w in TRIMS4}
    with pytest.raises(ValueError):
        S.link_counts(m, "untrained")


def test_training_pairs_share_canonical_part_and_weights():
    m = S.mappings()[3]
    canon = S.canonical_counts(DESIGN)
    assert all(not set(lt.split() + rt.split()) & set(S.NONCES) for lt, rt in canon)
    ka, wa = S.training_pairs(DESIGN, m, "aligned")
    ki, wi = S.training_pairs(DESIGN, m, "independent")
    assert math.isclose(wa.sum(), 1) and math.isclose(wi.sum(), 1)
    da, di = dict(zip(ka, wa, strict=True)), dict(zip(ki, wi, strict=True))
    for k in canon:  # identical canonical pairs and identical weight on them in both arms
        assert da[k] == di[k]
    left_a, left_i = Counter(), Counter()
    for (lt, _), w in da.items():
        left_a[lt] += w
    for (lt, _), w in di.items():
        left_i[lt] += w
    assert all(math.isclose(left_a[k], left_i[k]) for k in left_a | left_i)


# ---------------------------------------------------------------- mocked components
class Vision:
    """Per-cell mean colour: differs between the two pads, a stand-in for the visual model."""

    def outcome(self, images):
        x = np.asarray(images, dtype=float).reshape(len(images), 7, 5, 7, 5, 3)
        return x.mean((2, 4)).reshape(len(images), -1)


def anchors():
    out = []
    for i, combo in enumerate(SEEN * 8):
        other = next(g for g in GLYPHS4 if g != combo[0])
        glyphs = (combo[0], other) if i % 2 else (other, combo[0])
        out.append((combo, Board(BOARD.walls, BOARD.start, BOARD.pads, glyphs)))
    return out


def mock_components(planner=None):
    bow = BagOfWords(S.vocab())
    proj = {
        (m, arm): np.random.default_rng([m, k]).normal(size=(len(bow.index), 6))
        for m in range(8)
        for k, arm in enumerate(S.LEARNED)
    }
    return S.Components(
        vision=Vision(),
        planner=planner or (lambda img, goal, steps: SimpleNamespace(action=0)),
        anchors=anchors(),
        encode=lambda m, arm, texts: np.tanh(bow(texts) @ proj[(m, arm)]),
        raw=bow,
    )


def test_anchor_texts_canonical_only():
    comp = mock_components()
    texts = S.anchor_texts(comp)
    assert len(texts) == 64 and not {w for t in texts for w in t.split()} & set(S.NONCES)
    for combo in (WITHHELD[0], (S.NONCES[0], TRIMS4[0])):  # withheld or alias labels refused
        comp.anchors = [(combo, BOARD)]
        with pytest.raises(ValueError):
            S.anchor_texts(comp)


def test_raw_probe_is_exactly_blind_to_aliases():
    comp = mock_components()
    probes = S.fit_probes(DESIGN, comp, 8)
    check = S.raw_alias_check(comp, probes)
    assert check == {"coef_zero": True, "goals_equal": True, "ok": True}
    vecs = S.goal_vectors(comp, probes, S.mappings())
    v = {vecs[(m, "raw", "nonce", "circle", "dotted", 0)].tobytes() for m in range(8)}
    assert len(v) == 1  # same raw goal whichever alias names the circle
    learned = {vecs[(m, "aligned", "nonce", "circle", "dotted", 0)].tobytes() for m in range(8)}
    assert len(learned) > 1  # the check is not vacuous: learned spaces do see the alias


# ---------------------------------------------------------------- synthetic rows
def synthetic_rows(board_ids, margins=None, outcome="correct"):
    margins = margins or {"aligned": 2.0, "independent": 0.5, "untrained": 0.0}
    rows = []
    for bid, q, m, arm, kind in sorted(S.expected_keys(board_ids, 8), key=str):
        d_t = 1.0
        mg = margins.get(arm, 1.0) if kind == "nonce" else 1.0
        rows.append(
            {
                "board": bid,
                "query": q,
                "mapping": m,
                "arm": arm,
                "kind": kind,
                "d_target": d_t,
                "d_foil": d_t + mg,
                "margin": (d_t + mg) - d_t,
                "outcome": outcome,
            }
        )
    return rows


RAW_OK = {"coef_zero": True, "goals_equal": True, "ok": True}
IDS = ["test/0", "test/1", "test/2"]


def test_primary_sign_pairing_and_bootstrap():
    rows = synthetic_rows(IDS)
    s = S.summarize(rows, IDS, 8, RAW_OK, DESIGN)
    assert s["status"] == "VALID" and s["coverage"]["ok"]
    assert s["coverage"]["expected"] == 3 * 64 * (8 * 4 * 2 + 1)
    assert s["primary"]["mean"] == pytest.approx(1.5)
    assert s["secondary"]["aligned_minus_untrained_nonce"]["mean"] == pytest.approx(2.0)
    assert s["bootstrap"]["ci95"] == pytest.approx([1.5, 1.5])
    flipped = S.paired_summary(rows, "independent", "aligned")
    assert flipped["mean"] == pytest.approx(-1.5)
    assert s["secondary"]["executed_aligned_minus_independent_nonce"]["pp"] == 0


def test_lost_or_duplicate_keys_block_the_contrast():
    rows = synthetic_rows(IDS)
    s = S.summarize(rows[1:], IDS, 8, RAW_OK, DESIGN)
    assert s["status"] == "INCOMPLETE" and s["coverage"]["missing"] == 1
    assert s["primary"] is None and s["bootstrap"] is None
    s = S.summarize(rows + rows[:1], IDS, 8, RAW_OK, DESIGN)
    assert s["coverage"]["duplicates"] == 1 and s["primary"] is None
    with pytest.raises(ValueError):
        S.paired_board_means(
            [r for r in rows if r["arm"] != "independent" or r["board"] != "test/1"],
            "aligned",
            "independent",
        )


@pytest.mark.parametrize("breaker", ["canonical", "image", "nan", "raw_rows", "raw_vectors"])
def test_health_failures_make_the_run_invalid(breaker):
    rows = synthetic_rows(IDS)
    raw = dict(RAW_OK)
    if breaker == "canonical":
        for r in rows:
            if r["arm"] == "independent" and r["kind"] == "canonical":
                r["outcome"] = "wrong_pad"
    elif breaker == "image":
        for r in rows:
            if r["arm"] == "image":
                r["outcome"] = "timeout"
    elif breaker == "nan":
        rows[5]["margin"] = float("nan")
    elif breaker == "raw_rows":
        r = next(r for r in rows if r["arm"] == "raw" and r["kind"] == "nonce")
        r["d_target"] += 0.25
        r["margin"] = r["d_foil"] - r["d_target"]
    else:
        raw["ok"] = False
    s = S.summarize(rows, IDS, 8, raw, DESIGN)
    assert s["status"] == "INVALID" and not s["diagnostics"]["ok"]
    assert s["primary"] is None and s["secondary"] is None and s["bootstrap"] is None


# ---------------------------------------------------------------- end to end (mocked)
def test_run_once_rows_verify_and_planner_inputs(tmp_path):
    seen_args = []

    def planner(img, goal, steps):
        seen_args.append((type(img), getattr(img, "shape", None), type(goal), type(steps)))
        return SimpleNamespace(action=0)

    comp = mock_components(planner)
    out = tmp_path / "alias-run"
    s = S.run_study(
        out,
        DESIGN,
        comp_factory=lambda d, b, maps: comp,
        boards_factory=lambda d, n: [("test/0", BOARD)],
        n_boards=1,
        code={"commit": "fixture", "dirty": False},
    )
    assert s["status"] in ("VALID", "INVALID") and s["coverage"]["ok"]
    assert s["coverage"]["rows"] == 64 * 65
    assert {a[1:] for a in seen_args} == {((35, 35, 3), np.ndarray, int)}
    rows, _ = S.read_rows(out / "rows.jsonl.gz")
    keys = {
        "board",
        "query",
        "mapping",
        "arm",
        "kind",
        "target",
        "foil",
        "word",
        "panel",
        "trim",
        "order",
        "template",
        "split",
        "d_target",
        "d_foil",
        "margin",
        "outcome",
        "steps",
    }
    assert all(set(r) == keys for r in rows)
    assert {(r["panel"], r["trim"], r["target"]) for r in rows if r["arm"] == "image"} == {
        ("seen" if (GLYPHS4.index(g) - TRIMS4.index(t)) % 4 in (0, 1) else "withheld", t, g)
        for g in GLYPHS4
        for t in TRIMS4
    }  # all 16 glyph/trim combos are queried
    for r in rows:
        if r["kind"] == "nonce":
            assert S.mappings()[r["mapping"]][r["word"]] == r["target"]
    assert S.verify(out, DESIGN) == []
    plan = json.loads((out / "plan.json").read_text())
    assert plan["boards"][0]["id"] == "test/0" and len(plan["mappings"]) == 8

    S.write_rows(out / "rows.jsonl.gz", rows[1:])  # tamper: drop one row
    assert "rows digest" in S.verify(out, DESIGN)
    with pytest.raises(FileExistsError):
        S.run_study(
            out,
            DESIGN,
            comp_factory=lambda d, b, m: comp,
            boards_factory=lambda d, n: [("test/0", BOARD)],
            n_boards=1,
            code={"commit": "fixture", "dirty": False},
        )


def test_cap_leaves_incomplete_without_contrast(tmp_path):
    comp = mock_components()

    def slow(d, budget, maps):
        budget.start -= 1e6  # pretend the budget is spent
        return comp

    s = S.run_study(
        tmp_path / "r",
        DESIGN,
        comp_factory=slow,
        boards_factory=lambda d, n: [("test/0", BOARD)],
        n_boards=1,
        code={"commit": "fixture", "dirty": False},
    )
    assert s["status"] == "INCOMPLETE" and "primary" not in s
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["status"] == "INCOMPLETE"


def test_cli_refuses_existing_out(tmp_path, capsys):
    assert S.main(["run", "--out", str(tmp_path)]) == 2
    assert "refused" in capsys.readouterr().out


def test_queries_match_published_layout():
    assert len(engine.queries([BOARD], "test")) == 64
