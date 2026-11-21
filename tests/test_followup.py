"""Pure fixture tests for pgt.followup: hand-coded arrays, boards and rows only.

No study seed, bank, corpus, training run or rollout is produced here. Toy corpora use a
non-study corpus seed (999) and a 4-block design.
"""

import dataclasses
import json
import subprocess
import sys
import time
import types
import warnings
from collections import Counter

import numpy as np
import pytest

import pgt.followup
from pgt.followup import analysis, bridge, cli, corpus, encoder, prereg
from pgt.followup.design import COMBOS16, SEEN, TRIMS4, WITHHELD, FollowupDesign, panel_glyphs
from pgt.followup.world4 import GLYPH4_MASKS, TRIM4_MASKS, Board, certified, oracle_path, render4
from pgt.world import AGENT_MASK

TOY = dataclasses.replace(FollowupDesign(), n_blocks=4)


def test_seen_split_is_cyclic_8_8_and_panels_have_two_each():
    assert len(SEEN) == len(WITHHELD) == 8 and set(SEEN) | set(WITHHELD) == set(COMBOS16)
    for t in TRIMS4:
        assert len(panel_glyphs(t, True)) == len(panel_glyphs(t, False)) == 2


def test_masks_distinguishable_on_outer_ring():
    ring = ~AGENT_MASK
    for masks in (GLYPH4_MASKS, TRIM4_MASKS):
        rings = {m[ring].tobytes() for m in masks.values()}
        assert len(rings) == 4


def test_board_render_and_barrier_oracle():
    walls = frozenset({(1, 1), (1, 2), (2, 1)})
    b = Board(walls, (0, 0), ((0, 3), (3, 0)), ("square", "cross"))
    assert len(oracle_path(b, "square")) == 3 and certified(b, 2, 12)
    img = render4(b, b.pad("cross"), "ringed")
    assert img.shape == (35, 35, 3) and img.dtype == np.uint8
    blocked = Board(walls, (0, 0), ((0, 1), (1, 0)), ("square", "cross"))
    assert oracle_path(blocked, "square") == [3] and not certified(blocked, 2, 12)


def test_flips_are_in_support_one_factor_bijections():
    for flip, keep in ((corpus.glyph_flip, 1), (corpus.trim_flip, 0)):
        image = [flip(c) for c in SEEN]
        assert sorted(image) == sorted(SEEN)
        assert all(a[keep] == b[keep] and a != b for a, b in zip(SEEN, image, strict=True))


def test_toy_corpus_pools_match_and_exactly_quarter_blocks_swap():
    pools = {}
    for arm in ("clean", "goal_swap", "trim_swap"):
        pairs = corpus.make_corpus(TOY, 999, arm)
        pools[arm] = (Counter(p.left for p in pairs), Counter(p.right for p in pairs))
        changed = {p.block for p in pairs if p.left_combo != p.right_combo}
        assert len(changed) == (0 if arm == "clean" else 1)
        assert all(c in SEEN for p in pairs for c in (p.left_combo, p.right_combo))
    assert pools["clean"] == pools["goal_swap"] == pools["trim_swap"]
    assert not set(corpus.HELDOUT_TEMPLATES) & set(corpus.TEMPLATE_POOL)


def test_pair_loss_and_network_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    xl, xr = rng.normal(size=(5, 6)), rng.normal(size=(5, 6))
    w = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    p = encoder.init_params(rng, 6, 4, 3)
    _, grads = encoder.loss_and_grad(p, xl, xr, w)
    eps = 1e-6
    for a, g in zip(p.flat(), grads, strict=True):
        for idx in np.ndindex(a.shape):
            old = a[idx]
            a[idx] = old + eps
            up = encoder.loss_and_grad(p, xl, xr, w)[0]
            a[idx] = old - eps
            dn = encoder.loss_and_grad(p, xl, xr, w)[0]
            a[idx] = old
            assert abs((up - dn) / (2 * eps) - g[idx]) < 1e-6


def test_anti_collapse_prefers_spread_aligned_outputs():
    w = np.ones(4)
    const = np.ones((4, 2))
    spread = np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]])
    assert encoder.pair_loss(const, const, w)[0] == pytest.approx(0)
    assert encoder.pair_loss(spread, spread, w)[0] < 0
    assert encoder.effective_rank(const) == 0


def test_ridge_and_additive_keyword_bridge_cover_withheld_combos():
    x = np.array([[0.0], [1], [2], [3]])
    r = bridge.fit_ridge(x, 2 * x + 1, penalty=1e-9)
    assert np.allclose(r(np.array([[5.0]])), [[11.0]])
    xs = bridge.factor_onehot(list(SEEN))
    assert np.linalg.matrix_rank(xs) == 7  # all glyph+trim effects identified up to the sum
    assert bridge.factor_onehot([WITHHELD[0]]).sum() == 2
    assert bridge.keyword_combo("finish on the cross spot of this ringed room") == (
        "cross",
        "ringed",
    )


def _rows():
    rows = []
    for cond, outs in (
        ("trim_swap", ["correct", "correct"]),
        ("goal_swap", ["correct", "wrong_pad"]),
    ):
        for q, o in enumerate(outs):
            rows.append(
                dict(
                    corpus=1,
                    bank=1,
                    condition=cond,
                    panel="withheld",
                    board="b",
                    query=q,
                    outcome=o,
                )
            )
    return rows


def test_primary_is_paired_and_refuses_unmatched_sets():
    d = dataclasses.replace(FollowupDesign(), corpus_seeds=(1,), anchor_banks=(1,))
    assert analysis.primary(_rows(), d)["mean_pp"] == 50.0
    with pytest.raises(ValueError):
        analysis.primary(_rows()[:-1], d)


def test_health_gate_and_ties_and_digest():
    d = FollowupDesign()
    base = dict(corpus=None, bank=None, panel="seen", board="b", query=0)
    rows = [
        base | {"condition": "oracle", "outcome": "correct"},
        base | {"condition": "blind", "outcome": "wrong_pad"},
        base | {"condition": "image", "outcome": "correct"},
        base | {"condition": "clean", "outcome": "timeout"},
    ]
    gate = analysis.health_gate(rows, d)
    assert gate["status"] == "INCONCLUSIVE" and not gate["checks"]["seen_clean"]
    disc = analysis.goal_discrimination([{"correct": 1, "foil": 1}, {"correct": 2, "foil": 1}])
    assert disc == {"n": 2, "wins": 1, "ties": 1, "losses": 0}
    assert analysis.counts_digest(rows) == analysis.counts_digest(list(reversed(rows)))


# ---------------------------------------------------------------- prereg gate and CLI
# A throwaway Git repo with hand-written fixture files; nothing here is a real prediction, and
# the engine is a mock module, so no data, training, pilot or test is ever produced.


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "clone dir"
    (r / "docs").mkdir(parents=True)
    _git(r, "init", "-q")
    _git(r, "config", "user.name", "Fixture Writer")
    _git(r, "config", "user.email", "fixture@example.invalid")
    _git(r, "config", "core.autocrlf", "false")
    (r / "docs" / "followup.md").write_bytes(b"fixture plan\r\nline two\r\n")
    (r / "docs" / "protocol.md").write_text("fixture protocol\n", encoding="utf-8")
    (r / "pred.json").write_text(
        json.dumps({"author": "Fixture Writer", "prediction": "fixture words only"}),
        encoding="utf-8",
    )
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "fixture")
    monkeypatch.setattr(prereg, "REPO", r)
    monkeypatch.chdir(r)
    return r


def test_freeze_resolves_commit_from_git_and_is_written_once(repo):
    out = repo / "freeze.json"
    assert cli.main(["freeze", "--prediction", "pred.json", "--write", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert rec["prediction"]["commit"] == head and rec["prediction"]["git_author"]
    assert rec["plan"]["path"] == "docs/followup.md" and "docs/protocol.md" in rec["originals"]
    assert cli.main(["check", "--prediction", "pred.json", "--freeze", str(out)]) == 0
    before = out.read_bytes()
    assert cli.main(["freeze", "--prediction", "pred.json", "--write", str(out)]) == 2
    assert out.read_bytes() == before and not list(repo.glob(".freeze.json.*"))


def test_crlf_and_lf_checkouts_give_same_plan_digest(repo):
    a = prereg.make_freeze(FollowupDesign(), "pred.json", repo)
    plan = repo / "docs" / "followup.md"
    plan.write_bytes(plan.read_bytes().replace(b"\r\n", b"\n"))
    _git(repo, "commit", "-qam", "lf")
    assert prereg.make_freeze(FollowupDesign(), "pred.json", repo)["plan"] == a["plan"]


@pytest.mark.parametrize("state", ["modified", "staged", "untracked"])
def test_uncommitted_prediction_or_plan_is_refused(repo, state):
    if state == "untracked":
        (repo / "new.json").write_text('{"author": "A B", "prediction": "own"}', encoding="utf-8")
        with pytest.raises(prereg.PreregError, match="uncommitted"):
            prereg.make_freeze(FollowupDesign(), "new.json", repo)
        return
    target = repo / ("docs/followup.md" if state == "modified" else "pred.json")
    target.write_text(target.read_text(encoding="utf-8") + " edit", encoding="utf-8")
    if state == "staged":
        _git(repo, "add", "pred.json")
    with pytest.raises(prereg.PreregError, match="uncommitted"):
        prereg.make_freeze(FollowupDesign(), "pred.json", repo)


@pytest.mark.parametrize(
    "data",
    [
        {"author": "", "prediction": "own words"},
        {"author": "A B", "prediction": "TBD"},
        {"author": "A B", "prediction": "   "},
        {"author": "<your name>", "prediction": "own"},
        {"author": "A B", "prediction": "[Your prediction...]"},
        {"author": None, "prediction": "own words"},
        {"author": "A B", "prediction": ["own", "words"]},
        {"author": "A B"},
        {"author": "A B", "prediction": 42},
    ],
)
def test_placeholder_prediction_is_refused(repo, data):
    (repo / "pred.json").write_text(json.dumps(data), encoding="utf-8")
    _git(repo, "commit", "-qam", "edit")
    with pytest.raises(prereg.PreregError):
        prereg.check_prediction("pred.json", repo)


@pytest.mark.parametrize(
    "text",
    [
        "Accuracy on withheld pairs < seen pairs, but > blind; the gap shrinks...",
        "I expect a <5 pp gap... todo-list items aside, transfer holds.",
    ],
)
def test_real_prediction_text_with_angles_and_ellipsis_is_accepted(repo, text):
    (repo / "pred.json").write_text(json.dumps({"author": "A B", "prediction": text}), "utf-8")
    _git(repo, "commit", "-qam", "edit")
    assert prereg.check_prediction("pred.json", repo)["author"] == "A B"


def test_freeze_drift_and_out_dir_rules(repo, tmp_path):
    rec = prereg.make_freeze(FollowupDesign(), "pred.json", repo)
    with pytest.raises(prereg.PreregError, match="design_digest"):
        prereg.verify_freeze(rec | {"design_digest": "x"}, FollowupDesign(), "pred.json", repo)
    for bad in ("runs", "followup-pilot-"):
        with pytest.raises(prereg.PreregError):
            cli._out_dir(tmp_path / bad, "pilot")
    (tmp_path / "followup-pilot-a").mkdir()
    with pytest.raises(prereg.PreregError, match="exists"):
        cli._out_dir(tmp_path / "followup-pilot-a", "pilot")


def _pilot_dir(path, freeze, mutate=None):
    """Write the two files the engine writes (manifest.json, rows.jsonl) from hand-coded rows."""
    d = FollowupDesign()
    d = dataclasses.replace(d, n_validation_topologies=1, corpus_seeds=(1,), anchor_banks=(1,))
    rows = [
        dict(
            split="validation",
            condition=c,
            corpus=corpus_id,
            bank=bank,
            panel=panel,
            board="validation/0",
            query=query,
            outcome="wrong_pad" if c == "blind" else "correct",
            steps=3,
        )
        for c, corpus_id, bank in analysis.expected_cells(d)
        for panel, query in analysis.expected_queries()
    ]
    manifest = {
        "phase": "pilot",
        "freeze": freeze,
        "design_digest": d.digest(),
        "coverage": analysis.strict_coverage(rows, d, "validation", 1),
        "rows_digest": analysis.rows_digest(rows),
        "health": analysis.health_gate(rows, d),
        "counts": analysis.count_table(rows),
        "counts_digest": analysis.counts_digest(rows),
    }
    manifest["status"] = manifest["health"]["status"]
    if mutate:
        mutate(manifest, rows)
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps(manifest, default=str), encoding="utf-8")
    (path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_pilot_validation_recomputes_from_engine_files(tmp_path):
    d = dataclasses.replace(
        FollowupDesign(), n_validation_topologies=1, corpus_seeds=(1,), anchor_banks=(1,)
    )
    fz = {"k": "v"}
    assert cli.validate_pilot(_pilot_dir(tmp_path / "ok", fz), fz, d)["status"] == "PASS"
    bad = {
        "freeze": lambda m, r: m.update(freeze={"k": "other"}),
        "phase": lambda m, r: m.update(phase="test"),
        "status_only": lambda m, r: r[3].update(outcome="timeout"),
        "digest": lambda m, r: m.update(counts_digest="0" * 64),
        "coverage": lambda m, r: r[0].update(query="other"),
    }
    for name, mutate in bad.items():
        with pytest.raises(prereg.PreregError):
            cli.validate_pilot(_pilot_dir(tmp_path / name, fz, mutate), fz, d)


def test_pilot_and_test_run_engine_only_through_deadline_child(repo, monkeypatch, tmp_path):
    """The public chain main -> gates -> run_with_deadline(_child_entry) -> _child -> engine,
    with the spawn replaced by an in-process call so the mock engine is visible."""
    fz = repo / "freeze.json"
    assert cli.main(["freeze", "--prediction", "pred.json", "--write", str(fz)]) == 0
    freeze = json.loads(fz.read_text(encoding="utf-8"))
    calls, spawns = [], []
    fake = types.ModuleType("pgt.followup.engine")

    def run_phase(phase, design, freeze, out, pilot=None):
        calls.append((phase, freeze, out, pilot))
        return {"status": "PASS"}

    fake.run_phase = run_phase
    monkeypatch.setitem(sys.modules, "pgt.followup.engine", fake)
    monkeypatch.setattr(pgt.followup, "engine", fake, raising=False)

    def in_process(target, args, seconds):
        spawns.append((target, args[0], seconds))
        try:
            target(*args)
        except SystemExit as exc:
            return exc.code
        raise AssertionError("child entry must exit")

    monkeypatch.setattr(cli, "run_with_deadline", in_process)
    record = {"status": "PASS", "verified": True}
    monkeypatch.setattr(cli, "validate_pilot", lambda pilot_dir, fz_, d: record)
    base = ["--prediction", "pred.json", "--freeze", str(fz)]

    assert cli.main(["pilot", *base, "--out", str(tmp_path / "bad-name")]) == 2
    assert not calls and not spawns
    out = tmp_path / "followup-pilot-x"
    assert cli.main(["pilot", *base, "--out", str(out)]) == 0
    assert calls == [("pilot", freeze, out, None)]
    assert spawns[-1] == (cli._child_entry, "pilot", cli.deadline_s("pilot", FollowupDesign()))

    test_out = tmp_path / "followup-test-x"
    assert cli.main(["test", *base, "--out", str(test_out), "--pilot", str(tmp_path / "p")]) == 0
    assert calls[-1][:3] == ("test", freeze, test_out) and calls[-1][3] is record
    assert spawns[-1][1:] == ("test", cli.deadline_s("test", FollowupDesign()))


def test_pilot_gate_failure_never_reaches_engine(repo, monkeypatch, tmp_path):
    fz = repo / "freeze.json"
    assert cli.main(["freeze", "--prediction", "pred.json", "--write", str(fz)]) == 0
    monkeypatch.setattr(cli, "run_with_deadline", lambda *a: pytest.fail("child started"))
    args = ["test", "--prediction", "pred.json", "--freeze", str(fz)]
    out = ["--out", str(tmp_path / "followup-test-x"), "--pilot", str(tmp_path / "none")]
    assert cli.main([*args, *out]) == 2


@pytest.mark.parametrize("hidden", ["_run", "run", "_execute"])
def test_no_public_subcommand_bypasses_the_deadline(hidden):
    with pytest.raises(SystemExit):
        cli.main([hidden, "--prediction", "p.json", "--freeze", "f.json", "--out", "o"])


def test_hard_deadline_kills_a_real_spawned_process():
    start = time.monotonic()
    assert cli.run_with_deadline(time.sleep, (60,), 2) == cli.EXIT_DEADLINE
    assert time.monotonic() - start < 30
    d = FollowupDesign()
    assert cli.deadline_s("pilot", d) == d.cap_build_s + d.cap_pilot_s + cli.DEADLINE_GRACE_S
    assert cli.deadline_s("test", d) == d.cap_build_s + d.cap_test_s + cli.DEADLINE_GRACE_S


def test_real_spawned_child_rechecks_gates(tmp_path):
    args = ("pilot", tmp_path / "pred.json", tmp_path / "missing.json", tmp_path / "o", None)
    assert cli.run_with_deadline(cli._child_entry, args, 120) == cli.EXIT_REFUSED
    assert not (tmp_path / "o").exists()


# ---------------------------------------------------------------- amended encoder update
def _overflow_fixture():
    """Synthetic fixture on which the original fixed 0.05 step overflows within a few epochs."""
    rng = np.random.default_rng(7)
    xl, xr = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    p = encoder.init_params(rng, 5, 4, 3)
    p.w2 *= 20
    return p, xl, xr, np.full(6, 1 / 6)


def _fixed_step(p, xl, xr, w, epochs, lr):
    q = encoder.Params(*(a.copy() for a in p.flat()))
    for _ in range(epochs):
        loss, grads = encoder.loss_and_grad(q, xl, xr, w)
        for a, g in zip(q.flat(), grads, strict=True):
            a -= lr * g
    return q, loss


def test_fixed_step_overflows_on_fixture_but_backtracking_stays_finite_and_monotone():
    p, xl, xr, w = _overflow_fixture()
    with np.errstate(all="ignore"):
        _, loss = _fixed_step(p, xl, xr, w, 20, 0.05)
    assert not np.isfinite(loss)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q, trace = encoder.train(p, xl, xr, w, 20, 0.05)
    assert len(trace) == 21 and all(np.isfinite(trace))
    assert all(np.isfinite(a).all() for a in q.flat())
    slack = encoder.DESCENT_RTOL * np.maximum(1, np.abs(trace[:-1]))
    assert np.all(np.diff(trace) <= slack) and trace[-1] < trace[0]
    assert trace[-1] == pytest.approx(encoder.loss_and_grad(q, xl, xr, w)[0])
    again = encoder.train(p, xl, xr, w, 20, 0.05)  # deterministic on the same fixture
    assert again[1] == trace and all(
        np.array_equal(a, b) for a, b in zip(again[0].flat(), q.flat(), strict=True)
    )


def test_backtracking_equals_fixed_step_when_initial_step_descends():
    rng = np.random.default_rng(0)
    xl, xr = rng.normal(size=(5, 6)), rng.normal(size=(5, 6))
    w = np.full(5, 0.2)
    p = encoder.init_params(rng, 6, 4, 3)
    fixed, _ = _fixed_step(p, xl, xr, w, 3, 0.05)
    q, trace = encoder.train(p, xl, xr, w, 3, 0.05)
    assert len(trace) == 4
    assert all(np.array_equal(a, b) for a, b in zip(fixed.flat(), q.flat(), strict=True))


@pytest.mark.parametrize("bad", ["w1", "b2", "xl", "w"])
def test_train_rejects_nonfinite_start(bad):
    p, xl, xr, w = _overflow_fixture()
    arrays = {"w1": p.w1, "b2": p.b2, "xl": xl, "w": w}
    arrays[bad].flat[0] = np.nan
    with pytest.raises(FloatingPointError, match=bad):
        encoder.train(p, xl, xr, w, 2, 0.05)


def test_train_rejects_nonfinite_starting_loss(monkeypatch):
    p, xl, xr, w = _overflow_fixture()
    monkeypatch.setattr(encoder, "loss_and_grad", lambda *a: (np.inf, [np.zeros(1)]))
    with pytest.raises(FloatingPointError, match="starting"):
        encoder.train(p, xl, xr, w, 2, 0.05)


def test_train_stops_at_last_finite_point_when_no_trial_step_descends(monkeypatch):
    p, xl, xr, w = _overflow_fixture()
    calls = []

    def fake(q, *a):
        calls.append(1)
        loss = 1.0 if len(calls) == 1 else np.nan  # every trial step is non-finite
        return loss, [np.ones_like(x) for x in q.flat()]

    monkeypatch.setattr(encoder, "loss_and_grad", fake)
    q, trace = encoder.train(p, xl, xr, w, 5, 0.05)
    assert trace == [1.0] and len(calls) == 1 + encoder.MAX_HALVINGS + 1
    assert all(np.array_equal(a, b) for a, b in zip(q.flat(), p.flat(), strict=True))
