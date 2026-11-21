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
    assert len({tuple(m[a] for a in S.NONCES) for m in maps}) == 8  # eight distinct mappings
    first, second = S.base_permutations()
    assert second not in S._rotations(first)  # disjoint rotation classes
    assert not set(S._rotations(first)) & set(S._rotations(second))
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
    """Full-layout rows. Raw rows score a target-independent goal (distance depends only on
    the glyph), so competing targets on one scene are exact reverses, as in a real run."""
    margins = margins or {"aligned": 2.0, "independent": 0.5, "untrained": 0.0}
    maps = S.mappings()
    rows = []
    for q, board in engine.queries([BOARD] * len(board_ids), "test"):
        bid = board_ids[int(q.board_id.rsplit("/", 1)[1])]
        base = {"split": "test", "board": bid, "panel": q.panel, "trim": q.trim,
                "order": q.order, "template": q.template, "query": q.key,
                "target": q.goal, "foil": board.other(q.goal)}  # fmt: skip
        cells = [(None, "image", "image", None)]
        for m, mapping in enumerate(maps):
            for arm in S.ARMS:
                for kind in S.KINDS:
                    word = S.inverse(mapping)[q.goal] if kind == "nonce" else q.goal
                    cells.append((m, arm, kind, word))
        for m, arm, kind, word in cells:
            if arm == "raw":
                d_t = 1.0 + GLYPHS4.index(q.goal)
                d_f = 1.0 + GLYPHS4.index(base["foil"])
            else:
                d_t = 1.0
                d_f = d_t + (margins.get(arm, 1.0) if kind == "nonce" else 1.0)
            rows.append(
                base
                | {
                    "mapping": m,
                    "arm": arm,
                    "kind": kind,
                    "word": word,
                    "d_target": d_t,
                    "d_foil": d_f,
                    "margin": d_f - d_t,
                    "outcome": outcome,
                    "steps": 3 if outcome != "timeout" else DESIGN.horizon,
                }  # fmt: skip
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


@pytest.mark.parametrize(
    "breaker", ["canonical", "image", "nan", "null", "raw_rows", "raw_reverse", "raw_vectors"]
)
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
    elif breaker == "null":
        rows[5]["d_target"] = rows[5]["margin"] = None
    elif breaker == "raw_rows":
        r = next(r for r in rows if r["arm"] == "raw" and r["kind"] == "nonce")
        r["d_target"] += 0.25
        r["margin"] = r["d_foil"] - r["d_target"]
    elif breaker == "raw_reverse":  # equal across mappings, but not reversed across targets
        for r in rows:
            if r["arm"] == "raw" and r["kind"] == "nonce":
                r["d_target"], r["d_foil"], r["margin"] = 1.0, 2.0, 1.0
    else:
        raw["ok"] = False
    s = S.summarize(rows, IDS, 8, raw, DESIGN)
    assert s["status"] == "INVALID" and not s["diagnostics"]["ok"]
    assert s["primary"] is None and s["secondary"] is None and s["bootstrap"] is None


# ---------------------------------------------------------------- end to end (mocked)
FIXTURE = {"boards_factory": lambda d, n: [("test/0", BOARD)], "n_boards": 1}


def _fixture_run(path, comp=None, vision=None):
    comp = comp or mock_components()
    if vision is not None:
        comp.vision = vision
    return S.run_study(
        path,
        DESIGN,
        comp_factory=lambda d, b, maps: comp,
        code={"commit": "fixture", "dirty": False},
        **FIXTURE,
    )


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("alias") / "run"
    _fixture_run(out)
    rows, _ = S.read_rows(out / "rows.jsonl.gz")
    return out, rows


def _tampered(src, dst, rows=None, plan=None, summary=None):
    import shutil

    shutil.copytree(src, dst)
    if rows is not None:
        s = json.loads((dst / "summary.json").read_text())
        s["rows_sha256"] = S.write_rows(dst / "rows.jsonl.gz", rows)  # digest made consistent
        (dst / "summary.json").write_text(json.dumps(s))
    if plan is not None:
        (dst / "plan.json").write_text(json.dumps(plan))
    if summary is not None:
        (dst / "summary.json").write_text(json.dumps(summary))
    return dst


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", "crashed"),  # unknown enum
        ("arm", "oracle"),  # unknown enum
        ("panel", "withheld?"),  # altered label
        ("foil", "square"),  # altered glyph
        ("word", "vexlo"),  # wrong expected word for this mapping/target
        ("steps", 0),  # step bound
        ("d_target", -1.0),  # negative squared distance
    ],
)
def test_verify_rejects_altered_rows(fixture_run, tmp_path, field, value):
    out, rows = fixture_run
    rows = [dict(r) for r in rows]
    i = next(i for i, r in enumerate(rows) if r["kind"] == "nonce" and r[field] != value)
    rows[i][field] = value
    fails = S.verify(_tampered(out, tmp_path / "t", rows=rows), DESIGN, **FIXTURE)
    assert any(f.startswith("rows:") for f in fails), fails


def test_verify_rejects_altered_plan_settings_and_status(fixture_run, tmp_path):
    out, _ = fixture_run
    plan = json.loads((out / "plan.json").read_text())
    summary = json.loads((out / "summary.json").read_text())

    moved = json.loads(json.dumps(plan))
    moved["boards"][0]["start"] = [1, 1]  # geometry
    assert "plan boards" in S.verify(_tampered(out, tmp_path / "a", plan=moved), DESIGN, **FIXTURE)
    remap = json.loads(json.dumps(plan))
    remap["mappings"][0], remap["mappings"][1] = remap["mappings"][1], remap["mappings"][0]
    remapped = _tampered(out, tmp_path / "b", plan=remap)
    assert "plan mappings" in S.verify(remapped, DESIGN, **FIXTURE)
    for key, val, fail in (
        ("status", "PASS", "status enum"),
        ("study", "other", "study"),
    ):
        bad = summary | {key: val}
        assert fail in S.verify(_tampered(out, tmp_path / key, summary=bad), DESIGN, **FIXTURE)
    bad = json.loads(json.dumps(summary))
    bad["settings"]["canonical_min"] = 0.5
    assert "settings" in S.verify(_tampered(out, tmp_path / "s", summary=bad), DESIGN, **FIXTURE)
    # the standard (published-bank) selection would not accept a one-board fixture plan
    moved_board = Board(frozenset(), (1, 1), BOARD.pads, BOARD.glyphs)
    other = {"boards_factory": lambda d, n: [("test/0", moved_board)], "n_boards": 1}
    assert "plan boards" in S.verify(out, DESIGN, **other)
    renamed = {"boards_factory": lambda d, n: [("test/7", BOARD)], "n_boards": 1}
    assert "plan boards" in S.verify(out, DESIGN, **renamed)


class HugeVision(Vision):
    """Finite image vectors whose squared distances overflow float64 when scored."""

    def outcome(self, images):
        return super().outcome(images) * 1e160


def test_scoring_overflow_writes_strict_json_nulls_and_invalid(tmp_path):
    out = tmp_path / "huge"
    s = _fixture_run(out, vision=HugeVision())
    assert s["status"] == "INVALID" and not s["diagnostics"]["finite_margins"]
    assert s["primary"] is None and s["secondary"] is None and s["bootstrap"] is None
    rows, _ = S.read_rows(out / "rows.jsonl.gz")
    assert any(r["margin"] is None for r in rows)
    for r in rows:
        if r["d_target"] is None or r["d_foil"] is None:
            assert r["margin"] is None
    for name in ("summary.json", "plan.json"):  # strict JSON: no NaN/Infinity anywhere
        json.loads((out / name).read_text(), parse_constant=lambda c: pytest.fail(c))
    assert all(x["mean_margin"] is None or math.isfinite(x["mean_margin"]) for x in s["rates"])
    assert S.verify(out, DESIGN, **FIXTURE) == []  # round-trips and recomputes


def test_cap_after_analysis_voids_contrasts(tmp_path, monkeypatch):
    real = S.summarize

    def late(*a, **k):
        out = real(*a, **k)
        budget_holder["b"].start -= 1e6  # the budget runs out during analysis
        return out

    budget_holder = {}
    real_budget = engine.Budget

    def make_budget(*a, **k):
        budget_holder["b"] = real_budget(*a, **k)
        return budget_holder["b"]

    monkeypatch.setattr(S, "summarize", late)
    monkeypatch.setattr(S.engine, "Budget", make_budget)
    s = _fixture_run(tmp_path / "late")
    assert s["status"] == "INCOMPLETE" and "report" in s["incomplete_reason"]
    assert s["primary"] is None and s["secondary"] is None and s["bootstrap"] is None
    assert S.verify(tmp_path / "late", DESIGN, **FIXTURE)[0].startswith("run stopped")


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
    assert S.verify(out, DESIGN, **FIXTURE) == []
    plan = json.loads((out / "plan.json").read_text())
    assert plan["boards"][0]["id"] == "test/0" and len(plan["mappings"]) == 8

    S.write_rows(out / "rows.jsonl.gz", rows[1:])  # tamper: drop one row
    assert "rows digest" in S.verify(out, DESIGN, **FIXTURE)
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
    assert s["status"] == "INCOMPLETE" and s["primary"] is None and s["bootstrap"] is None
    assert json.loads((tmp_path / "r" / "summary.json").read_text())["status"] == "INCOMPLETE"


def test_cli_refuses_existing_out(tmp_path, capsys):
    assert S.main(["run", "--out", str(tmp_path)]) == 2
    assert "refused" in capsys.readouterr().out


def test_queries_match_published_layout():
    assert len(engine.queries([BOARD], "test")) == 64


# ---------------------------------------------------------------- overflow-safe aggregation
def test_huge_finite_values_aggregate_without_overflow():
    assert S._safe_mean([1e308, 1e308]) == 1e308
    assert S._mean_diff([(1e308, -1e308), (-1e308, 1e308)]) == 0.0
    boot = S.board_bootstrap({"a": 1e308, "b": 1e308, "c": 1e308})
    assert boot["ci95"] == [1e308, 1e308]
    rows = synthetic_rows(IDS, margins={"aligned": 1e308, "independent": 0.0, "untrained": 0.0})
    for r in rows:  # huge finite margins whose sums overflow but whose means fit
        if r["arm"] == "aligned" and r["kind"] == "nonce":
            r["d_target"], r["d_foil"] = 0.0, 1e308
            r["margin"] = 1e308
    s = S.summarize(rows, IDS, 8, RAW_OK, DESIGN)
    assert s["status"] == "VALID", s["diagnostics"]
    assert s["primary"]["mean"] == 1e308 and s["bootstrap"]["ci95"] == [1e308, 1e308]
    assert json.loads(json.dumps(s, allow_nan=False)) == s


def test_unrepresentable_paired_contrast_is_invalid():
    rows = synthetic_rows(IDS)
    for r in rows:  # aligned - independent = 1.5e308 - (-1.5e308): mean does not fit
        if r["kind"] == "nonce" and r["arm"] in ("aligned", "independent"):
            r["d_target"], r["d_foil"] = (0.0, 1.5e308) if r["arm"] == "aligned" else (1.5e308, 0.0)
            r["margin"] = r["d_foil"] - r["d_target"]
    with pytest.raises(S.Unrepresentable):
        S.paired_summary(rows, "aligned", "independent")
    s = S.summarize(rows, IDS, 8, RAW_OK, DESIGN)
    assert s["status"] == "INVALID"
    assert s["diagnostics"]["paired_contrasts_representable"] is False
    assert s["primary"] is None and s["secondary"] is None and s["bootstrap"] is None


# ---------------------------------------------------------------- malformed artifacts
def test_verify_reports_corrupt_gzip(fixture_run, tmp_path):
    out, _ = fixture_run
    good = (out / "rows.jsonl.gz").read_bytes()
    for name, blob in [("trunc", good[: len(good) // 2]), ("junk", b"\x1f\x8b" + b"\0" * 40)]:
        dst = _tampered(out, tmp_path / name)
        (dst / "rows.jsonl.gz").write_bytes(blob)
        fails = S.verify(dst, DESIGN, **FIXTURE)
        assert len(fails) == 1 and fails[0].startswith("rows or plan unreadable"), fails


@pytest.mark.parametrize("target", ["summary", "plan", "row", "raw_alias_vectors"])
def test_verify_reports_non_object_json(fixture_run, tmp_path, target):
    out, rows = fixture_run
    s = json.loads((out / "summary.json").read_text())
    if target == "summary":
        dst = _tampered(out, tmp_path / "t", summary=[s])
    elif target == "plan":
        dst = _tampered(out, tmp_path / "t", plan=[1, 2])
    elif target == "row":
        dst = _tampered(out, tmp_path / "t", rows=[*rows[:1], [1], *rows[2:]])
    else:
        dst = _tampered(out, tmp_path / "t", summary=s | {"raw_alias_vectors": "ok"})
    fails = S.verify(dst, DESIGN, **FIXTURE)
    assert len(fails) == 1 and "not" in fails[0] and "object" in fails[0], fails


@pytest.mark.parametrize(
    "field,value",
    [("board", ["test/0"]), ("query", {"k": 1}), ("arm", ["aligned"]), ("outcome", ["x"]),
     ("template", "0"), ("mapping", "0"), ("word", 3), ("steps", "3")],
)  # fmt: skip
def test_verify_reports_malformed_row_types(fixture_run, tmp_path, field, value):
    out, rows = fixture_run
    rows = [dict(r) for r in rows]
    i = next(i for i, r in enumerate(rows) if r["kind"] == "nonce")
    rows[i][field] = value
    fails = S.verify(_tampered(out, tmp_path / "t", rows=rows), DESIGN, **FIXTURE)
    assert any(f.startswith("rows:") for f in fails), fails
