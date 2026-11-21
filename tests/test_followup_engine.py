"""Engine tests on hand-coded fixtures with mocked models. No real data, fitting or heldout."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pgt.followup import engine as E
from pgt.followup.design import SEEN, FollowupDesign
from pgt.followup.world4 import Board

BOARD = Board(frozenset({(3, 3)}), (0, 0), ((0, 3), (3, 0)), ("circle", "triangle"))


def test_queries_balanced_on_same_map():
    qs = E.queries([BOARD], "validation")
    assert len(qs) == 4 * 2 * 2 * 2 * 2  # trims x panels x orders x templates x goals
    assert {b.walls for _, b in qs} == {BOARD.walls}
    for panel in ("seen", "withheld"):
        goals = [q.goal for q, _ in qs if q.panel == panel]
        assert len(set(goals)) == 4 and len(goals) == 32


def test_oracle_and_grade():
    out, steps = E.run_oracle(BOARD, "circle", 12)
    assert out == "circle" and steps == 3
    assert E.grade("timeout", "circle") == "timeout"
    assert E.grade("triangle", "circle") == "wrong_pad"


def test_run_planned_first_terminal_and_cache_bound():
    seq = iter([1, 1, 1, 1])  # mock planner: always "down" toward pad (3,0)
    planner = lambda img, g, s: SimpleNamespace(action=next(seq))  # noqa: E731
    from pgt.world import ACTIONS

    down = (
        next(i for i, a in enumerate(ACTIONS) if tuple(a) in ((1, 0),))
        if not isinstance(ACTIONS[0], str)
        else None
    )
    if down is not None:
        seq = iter([down] * 4)
        planner = lambda img, g, s: SimpleNamespace(action=next(seq))  # noqa: E731
        cache = E.FrameCache(2)
        out, steps = E.run_planned(planner, BOARD, "striped", None, 12, cache)
        assert out == "triangle" and steps == 3 and len(cache.data) <= 2


def test_key_coverage_detects_mismatch():
    r = {
        "condition": "a",
        "corpus": None,
        "bank": None,
        "panel": "seen",
        "board": "b",
        "query": "q",
    }
    assert E.key_coverage([r, r | {"condition": "b"}])["identical"]
    assert not E.key_coverage([r, r | {"condition": "b", "query": "z"}])["identical"]


def test_test_phase_locked_and_out_never_overwritten(tmp_path):
    d = FollowupDesign()
    with pytest.raises(PermissionError):
        E.run_phase("test", d, {"x": 1}, tmp_path / "o", None, [], pilot=None)
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        E.run_phase("pilot", d, {}, tmp_path / "exists", None, [])


def test_cap_marks_incomplete(tmp_path, monkeypatch):
    d = replace(FollowupDesign(), cap_pilot_s=0, cap_build_s=0)
    monkeypatch.setattr(
        E, "build_banks", lambda *a: {"banks": {"validation": [BOARD]}, "n_prior_excluded": 0}
    )

    def factory(design, budget):
        budget.check("fit")

    m = E.run_phase("pilot", d, {"f": 1}, tmp_path / "followup-pilot-x", factory, [])
    assert m["status"] == "INCOMPLETE" and "cap" in m["incomplete_reason"]
    saved = json.loads((tmp_path / "followup-pilot-x" / "manifest.json").read_text())
    assert saved["freeze"] == {"f": 1}


def test_evaluation_cap_is_separate_from_build_cap(tmp_path, monkeypatch):
    d = replace(SMALL, cap_build_s=10_000, cap_pilot_s=0)
    monkeypatch.setattr(
        E, "build_banks", lambda *a: {"banks": {"validation": [BOARD]}, "n_prior_excluded": 0}
    )
    seen = []

    def factory(design, budget):
        seen.append(budget.seconds)
        return _fake_components()

    m = E.run_phase("pilot", d, {"f": 1}, tmp_path / "followup-pilot-e", factory, lambda: [])
    assert seen == [10_000]  # fitting runs against the build cap only
    assert m["status"] == "INCOMPLETE" and m["incomplete_reason"].startswith("evaluation/report")


def test_bridges_reject_withheld_anchors():
    d = FollowupDesign()
    vision = SimpleNamespace(outcome=lambda imgs: np.zeros((len(imgs), 7)))
    withheld = next(
        c
        for c in [(g, t) for g in ("circle",) for t in ("striped", "dotted", "checked", "ringed")]
        if c not in SEEN
    )
    comp = E.Components(
        vision,
        None,
        lambda c, a, t: np.zeros((len(t), 3)),
        lambda t: np.zeros((len(t), 3)),
        lambda bank: [
            (withheld, Board(BOARD.walls, BOARD.start, BOARD.pads, (withheld[0], "triangle")))
        ],
    )
    with pytest.raises(ValueError):
        E.fit_bridges(d, comp)


# ---------------------------------------------------------------- small fake panels
SMALL = replace(FollowupDesign(), corpus_seeds=(1,), anchor_banks=(1,), n_validation_topologies=1)


def _fake_components():
    from pgt.followup.corpus import vocabulary

    vocab = vocabulary()

    def bow(texts):
        return np.array([[t.split().count(w) for w in vocab] for t in texts], dtype=float)

    vision = SimpleNamespace(outcome=lambda imgs: imgs.reshape(len(imgs), -1)[:, :6].astype(float))
    anchors = [
        (
            c,
            Board(
                BOARD.walls,
                BOARD.start,
                BOARD.pads,
                (c[0], "square" if c[0] != "square" else "cross"),
            ),
        )
        for c in SEEN
    ]
    return E.Components(
        vision,
        lambda img, g, s: SimpleNamespace(action=0),
        lambda c, a, t: bow(t) * (1 + ARMS_I[a]),
        bow,
        lambda bank: anchors,
    )


ARMS_I = {"clean": 0, "goal_swap": 1, "trim_swap": 2}


def test_anchor_captions_train_only_and_balanced():
    from pgt.followup.corpus import HELDOUT_TEMPLATES, TEMPLATE_POOL

    texts = E.anchor_captions([SEEN[0]] * len(TEMPLATE_POOL))
    assert len(set(texts)) == len(TEMPLATE_POOL)
    heldout = {h.format(glyph=SEEN[0][0], trim=SEEN[0][1]) for h in HELDOUT_TEMPLATES}
    assert not heldout & set(texts)


def test_build_banks_passes_topology_predicate(monkeypatch):
    seen = {}

    def fake(design, name, n, excluded, accept_topology=None):
        seen[name] = accept_topology
        return [BOARD] * n

    monkeypatch.setattr(E, "generate_bank", fake)
    E.build_banks(SMALL, "test", [], accept=lambda w: True)
    assert set(seen) == {"validation", "test"} and all(callable(f) for f in seen.values())


def test_run_phase_fake_pilot_full_coverage(tmp_path, monkeypatch):
    from pgt.followup import analysis as A

    monkeypatch.setattr(
        E, "build_banks", lambda *a: {"banks": {"validation": [BOARD]}, "n_prior_excluded": 0}
    )
    m = E.run_phase(
        "pilot",
        SMALL,
        {"f": 1},
        tmp_path / "followup-pilot-a",
        lambda d, b: _fake_components(),
        lambda: [],
    )
    assert m["coverage"]["ok"] and m["coverage"]["cells"] == 2 + 3 + 3
    assert m["coverage"]["expected"] == 8 * 64
    assert m["status"] in ("PASS", "INCONCLUSIVE")
    assert any(k.startswith("clean/1/1/") for k in m["discrimination"])
    assert any(k.startswith("raw/None/1/") for k in m["discrimination"])
    rows = [
        json.loads(x)
        for x in (tmp_path / "followup-pilot-a" / "rows.jsonl").read_text().splitlines()
    ]
    assert A.rows_digest(rows) == m["rows_digest"]


def _pass_rows(design):
    from pgt.followup import analysis as A

    rows = []
    for cell in A.expected_cells(design):
        for panel, q in A.expected_queries():
            ok = cell[0] != "blind"
            rows.append(
                {
                    "split": "validation",
                    "condition": cell[0],
                    "corpus": cell[1],
                    "bank": cell[2],
                    "panel": panel,
                    "board": "validation/0",
                    "query": q,
                    "outcome": "correct" if ok else "wrong_pad",
                    "steps": 3,
                }
            )
    return rows


def test_strict_coverage_rejects_duplicates_missing_and_bad_outcomes():
    from pgt.followup import analysis as A

    rows = _pass_rows(SMALL)
    assert A.strict_coverage(rows, SMALL, "validation", 1)["ok"]
    assert not A.strict_coverage(rows + rows[:1], SMALL, "validation", 1)["ok"]
    assert not A.strict_coverage(rows[1:], SMALL, "validation", 1)["ok"]
    assert not A.strict_coverage([rows[0] | {"outcome": "x"}] + rows[1:], SMALL, "validation", 1)[
        "ok"
    ]
    assert not A.strict_coverage(rows, SMALL, "test", 1)["ok"]
    assert not A.strict_coverage(rows, FollowupDesign(), "validation", 1)["ok"]


def test_verify_pilot_and_bootstrap(tmp_path):
    from pgt.followup import analysis as A

    d = replace(SMALL, n_validation_topologies=1)
    rows = _pass_rows(d)
    s = A.summarize(rows, [], d, "validation", 1)
    assert s["bootstrap"]["n_resamples"] == 2000 and s["bootstrap"]["ci95_pp"] == [0.0, 0.0]
    m = {"phase": "pilot", "freeze": {"f": 1}, "design_digest": d.digest(), "status": "PASS"} | s
    out = tmp_path / "p"
    out.mkdir()
    E._write(out, json.loads(json.dumps(m, default=str)), rows)
    assert E.verify_pilot(out / "manifest.json", {"f": 1}, d)["status"] == "PASS"
    with pytest.raises(ValueError):
        E.verify_pilot(out / "manifest.json", {"f": 2}, d)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows[1:]))
    with pytest.raises(ValueError):
        E.verify_pilot(out / "manifest.json", {"f": 1}, d)
