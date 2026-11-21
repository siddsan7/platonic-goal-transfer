import ast
import dataclasses
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pytest

from pgt import data as D
from pgt import experiment as E
from pgt import planner as P
from pgt import stats
from pgt import text as T
from pgt import vision as V
from pgt import world as W
from pgt.data import ARMS, COMBOS, Config, caption, generate_split, make_pairs, rng_for

CFG = Config()
VISION = dict(negatives=8, epochs=3, batch=256, lr=0.5, l2=1e-4, cell_dims=2, floor_dims=1)


@pytest.fixture(scope="module")
def train_layouts():
    return generate_split(CFG, "train")[:96]


@pytest.fixture(scope="module")
def walks(train_layouts):
    return V.random_walks(train_layouts, rng_for(CFG, "vision-walks"), 1, CFG.horizon)


@pytest.fixture(scope="module")
def model(train_layouts, walks):
    goals, _ = V.goal_images(train_layouts[:48])
    return V.train_vision(walks, goals, np.random.default_rng(0), **VISION)


def corridor(walls):
    """Start top-left, triangle two cells right, circle four cells right."""
    return W.Layout(frozenset(walls), (0, 0), ((0, 4), (0, 2)))


# ---------------------------------------------------------------- no oracle in decisions
def test_planner_module_imports_no_simulator_or_data():
    tree = ast.parse(Path(P.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {
                f"{node.module}.{a.name}" if node.module == "pgt" else node.module
                for a in node.names
            }
    assert imported <= {"__future__", "hashlib", "dataclasses", "numpy", "pgt.vision"}
    params = list(inspect.signature(P.Planner.__call__).parameters)
    assert params == ["self", "image", "goal", "steps_left"]


def test_planner_decisions_never_call_simulator_oracle_or_renderer(model, monkeypatch):
    layouts = generate_split(CFG, "validation")[:6]
    frames = [W.render(lay, lay.start, trim) for lay in layouts for trim in W.TRIMS]
    goals = [
        model.outcome(W.render(lay, lay.pad(g), "dotted")[None])[0]
        for lay in layouts
        for g in W.GLYPHS
    ]

    def forbidden(*_, **__):
        raise AssertionError("planner touched the simulator, oracle or renderer")

    for module in (W, V, D, E):
        for name in ("step", "run_episode", "oracle_path", "render", "route_lengths", "eligible"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    planner = P.Planner(model, 1e-3)
    for frame in frames:
        for goal in goals:
            plan = planner(frame, goal, 12)
            assert 0 <= plan.action < 4


def test_imagination_ends_at_first_terminal_and_never_credits_passing_a_wrong_pad(model):
    sealed = corridor({(1, c) for c in range(7)})  # the circle lies behind the triangle
    scene = P.perceive(model, W.render(sealed, sealed.start, "striped"))
    reach = P.imagine(scene, 12)
    circle, triangle = 0 * 7 + 4, 0 * 7 + 2
    assert reach.depth[triangle] == 2 and reach.depth[circle] == -1
    goal = model.outcome(W.render(sealed, sealed.pad("circle"), "striped")[None])[0]
    plan = P.choose(scene, reach, goal, 1e-3)
    # the only predicted-reachable first terminal is chosen; the circle is never credited
    assert plan.cell == triangle and plan.terminal and not plan.fallback
    no_pads = dataclasses.replace(scene, terminal=np.zeros_like(scene.terminal))
    fb = P.choose(no_pads, P.imagine(no_pads, 12), goal, 1e-3)
    assert fb.fallback and not fb.terminal  # off-pad proximity heuristic, labelled as such


def test_negligible_goal_noise_cannot_change_the_plan(model):
    lay = generate_split(CFG, "validation")[0]
    frame = W.render(lay, lay.start, "striped")
    goal = model.outcome(W.render(lay, lay.pad("circle"), "striped")[None])[0]
    planner = P.Planner(model, 1e-3)
    base = planner(frame, goal, 12)
    for eps in (1e-13, -1e-13):
        noisy = planner(frame, goal + eps, 12)
        assert (noisy.action, noisy.cell, noisy.score) == (base.action, base.cell, base.score)


def test_planner_follows_a_detour_using_learned_walls(model):
    detour = corridor({(1, 1), (1, 2), (1, 3)})
    planner = P.Planner(model, 1e-3)
    for glyph, steps in (("circle", 8), ("triangle", 2)):  # circle needs the 8-step detour
        goal = model.outcome(W.render(detour, detour.pad(glyph), "dotted")[None])[0]
        ep = E.run_planned(planner, detour, "dotted", goal, CFG.horizon)
        assert (ep["outcome"], ep["steps"]) == (glyph, steps)
        assert ep["predicted_terminal"] and ep["predicted_steps"] == steps


def test_goal_blind_vector_gives_identical_plans_for_both_twins(model):
    lay = generate_split(CFG, "validation")[3]
    blind = np.zeros(model.embed.dim)
    planner = P.Planner(model, 1e-3)
    eps = [E.run_planned(planner, lay, "striped", blind, CFG.horizon) for _ in W.GLYPHS]
    assert eps[0]["actions"] == eps[1]["actions"]
    assert sum(ep["outcome"] == g for ep, g in zip(eps, W.GLYPHS, strict=True)) <= 1


# ---------------------------------------------------------------- intervention training integrity
class Poison:
    def __getattr__(self, name):
        raise AssertionError("training read true agent positions")

    def __getitem__(self, key):
        raise AssertionError("training read true agent positions")

    def __array__(self, *args, **kwargs):
        raise AssertionError("training read true agent positions")


def test_vision_training_never_reads_true_positions(train_layouts, walks):
    blind = V.Walks(walks.frames, walks.src, walks.dst, walks.actions, walks.done, Poison())
    goals, _ = V.goal_images(train_layouts[:8])
    m = V.train_vision(blind, goals, np.random.default_rng(0), **VISION)
    assert m.displacement.tolist() == [1, 2, 3, 4]  # up, down, left, right learned from pixels
    assert "arm" not in inspect.signature(V.train_vision).parameters


def test_vision_weights_depend_on_seed_only(train_layouts, walks):
    goals, _ = V.goal_images(train_layouts[:8])
    a = V.train_vision(walks, goals, np.random.default_rng(5), **VISION).digest()
    b = V.train_vision(walks, goals, np.random.default_rng(5), **VISION).digest()
    c = V.train_vision(walks, goals, np.random.default_rng(6), **VISION).digest()
    assert a == b
    assert a["detector"] != c["detector"]  # model seed changes init and SGD order
    assert a["displacement"] == c["displacement"]


def test_text_encoder_is_text_only_order_invariant_and_shares_vocabulary():
    assert list(inspect.signature(T.train_text_encoder).parameters) == [
        "pairs",
        "dim",
        "ridge",
        "rank_tol",
    ]
    pairs = make_pairs(CFG, 64, "goal_swap")
    order = np.random.default_rng(0).permutation(len(pairs))
    a = T.train_text_encoder(pairs, 2, 0.01)
    b = T.train_text_encoder([pairs[i] for i in order], 2, 0.01)
    assert np.allclose(a.proj, b.proj)
    vocabs = {T.train_text_encoder(make_pairs(CFG, 64, arm), 2, 0.01).vocab for arm in ARMS}
    assert len(vocabs) == 1


def test_swap_intervention_weakens_exactly_the_flipped_factor():
    texts = [caption(c, t) for t in range(2) for c in COMBOS]
    glyph = np.array([c[0] == "circle" for t in range(2) for c in COMBOS], float)
    trim = np.array([c[1] == "striped" for t in range(2) for c in COMBOS], float)
    for arm, weak in (("goal_swap", glyph), ("trim_swap", trim)):
        enc = T.train_text_encoder(make_pairs(CFG, CFG.n_train, arm), 2, 0.01)
        assert np.allclose(enc.eigenvalues[:2], [0.9772, 0.4886], atol=1e-3)
        z = enc(texts)
        weak_col = z[:, 1]  # the lower-eigenvalue direction
        assert abs(np.corrcoef(weak_col, weak)[0, 1]) > 0.999
    clean = T.train_text_encoder(make_pairs(CFG, CFG.n_train, "clean"), 2, 0.01)
    assert np.allclose(clean.eigenvalues[:2], [0.9772, 0.9772], atol=1e-3)


def test_heldout_phrasing_oov_is_dropped_stably():
    enc = T.train_text_encoder(make_pairs(CFG, 64, "clean"), 2, 0.01)
    for t in range(2):
        train = enc([caption(c, t) for c in COMBOS])
        held = enc([caption(c, t, True) for c in COMBOS])
        assert np.allclose(train, held)
    assert enc.oov([caption(c, 0, True) for c in COMBOS]) > 0
    assert T.unique_states(enc([caption(c, 1, True) for c in COMBOS])) == 4


# ---------------------------------------------------------------- statistics and controls
def test_paired_effect_known_values_and_identical_episodes():
    wins_a, count = np.ones((2, 10)), np.ones((2, 10))
    wins_b = np.tile([1.0, 0.0], (2, 5))
    e = stats.paired_effect((wins_a, count), (wins_b, count), 500, np.random.default_rng(0))
    assert e["effect_pp"] == pytest.approx(50.0) and e["per_seed_pp"] == [50.0, 50.0]
    assert e["ci_pp"][0] < 50 < e["ci_pp"][1]
    same = stats.paired_effect((wins_a, count), (wins_a, count), 200, np.random.default_rng(0))
    assert same["ci_pp"] == [0.0, 0.0]
    with pytest.raises(ValueError):
        stats.paired_effect((wins_a, count), (wins_b, 2 * count), 10, np.random.default_rng(0))


def test_bootstrap_holds_seed_pairing_fixed():
    count = np.ones((2, 2))
    a = np.array([[1.0, 0.0], [0.0, 1.0]])  # seed effects are exact mirror images per cluster
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    e = stats.paired_effect((a, count), (b, count), 300, np.random.default_rng(1))
    assert e["ci_pp"] == [0.0, 0.0]  # same cluster draw for both seeds cancels exactly


def test_cluster_table_counts_by_seed_and_topology():
    recs = [
        {"condition": "x", "seed": 11, "cluster": "a", "success": True},
        {"condition": "x", "seed": 11, "cluster": "a", "success": False},
        {"condition": "x", "seed": 23, "cluster": "b", "success": True},
        {"condition": "y", "seed": 23, "cluster": "b", "success": True},
    ]
    wins, count = stats.cluster_table(recs, "x", [11, 23], ["a", "b"])
    assert wins.tolist() == [[1, 0], [0, 1]] and count.tolist() == [[2, 0], [0, 1]]


def test_procrustes_recovers_rotation_with_padding():
    rng = np.random.default_rng(0)
    text = rng.normal(size=(64, 2))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    image = np.c_[text, np.zeros(64)] @ q + np.array([1.0, -2.0, 0.5])
    m = T.fit_procrustes(text, image)
    assert np.allclose(m(text), image)
    assert np.allclose(m.rot @ m.rot.T, np.eye(3))
    shuffled = T.fit_procrustes(text[rng.permutation(64)], image)
    assert not np.allclose(shuffled(text), image)


def _summary(image=0.9, blind=0.5, oracle=1.0):
    return {
        "oracle": {"success": oracle},
        "image_goal": {"success": image},
        "goal_blind": {"success": blind},
    }


def _info(rel_pred=0.0, rel_persist=1.0, sv=(3.0, 2.0, 0.0)):
    return {
        "one_step": {
            "action_relevant": 10,
            "predictor_error_relevant": rel_pred,
            "persistence_error_relevant": rel_persist,
        },
        "image_latent": {"finite": True, "goal_singular_values": list(sv)},
    }


TEXT_OK = {
    "clean": {
        "finite": True,
        "unique_states_heldout": 4,
        "unique_states_train": 4,
        "effective_rank_heldout": 2.0,
    }
}


def test_gates_fail_on_each_unhealthy_instrument():
    x = E.ExpConfig()
    assert all(E.gates(CFG, x, _summary(), _info(), TEXT_OK).values())
    assert not E.gates(CFG, x, _summary(image=0.79), _info(), TEXT_OK)["image_goal_ge_80pct"]
    assert not E.gates(CFG, x, _summary(blind=0.51), _info(), TEXT_OK)["goal_blind_le_50pct"]
    assert not E.gates(CFG, x, _summary(oracle=0.99), _info(), TEXT_OK)["oracle_100pct"]
    assert not E.gates(CFG, x, _summary(), _info(rel_pred=1.0), TEXT_OK)[
        "one_step_beats_persistence"
    ]
    assert not E.gates(CFG, x, _summary(), _info(sv=(3.0, 0.0, 0.0)), TEXT_OK)[
        "embeddings_finite_noncollapsed"
    ]
    collapsed = {"clean": TEXT_OK["clean"] | {"unique_states_heldout": 1}}
    assert not E.gates(CFG, x, _summary(), _info(), collapsed)["embeddings_finite_noncollapsed"]


class Unlocked(Exception):
    pass


def test_test_split_unlock_requires_healthy_matching_pilot(tmp_path, monkeypatch):
    def refuse_generation(cfg, split, unlock_test=False):
        raise Unlocked(split, unlock_test)

    monkeypatch.setattr(E, "generate_split", refuse_generation)  # the test split is never built
    base = [
        "run",
        "--split",
        "test",
        "--seeds",
        *map(str, D.MODEL_SEEDS),
        "--out",
        str(tmp_path / "o"),
    ]
    with pytest.raises(SystemExit, match="requires --unlock-test"):
        E.main(base)
    pilot = tmp_path / "pilot.json"
    healthy = {
        "split": "validation",
        "seeds": [D.PILOT_SEED],
        "gates_passed": True,
        "status": "complete",
        "eval_layouts_full": True,
        "gates": {str(D.PILOT_SEED): {k: True for k in E.GATE_KEYS}},
        "frozen_digest": E.frozen_digest(CFG, E.ExpConfig()),
    }
    for change, message in [
        ({"gates_passed": False}, "gates did not all pass"),
        ({"frozen_digest": "stale"}, "changed since the pilot"),
        ({"seeds": [23]}, "pilot seeds"),
        ({"split": "test"}, "not on validation"),
        ({"status": "aborted"}, "did not complete"),
        ({"eval_layouts_full": False}, "not a full validation"),
        ({"gates": {str(D.PILOT_SEED): {k: 1 for k in E.GATE_KEYS}}}, "literally true"),
        ({"gates": {str(D.PILOT_SEED): {"oracle_100pct": True}}}, "incomplete"),
    ]:
        pilot.write_text(json.dumps(healthy | change))
        with pytest.raises(SystemExit, match=message):
            E.main([*base, "--unlock-test", "--pilot", str(pilot)])
    pilot.write_text(json.dumps(healthy))
    with pytest.raises(Unlocked) as exc:
        E.main([*base, "--unlock-test", "--pilot", str(pilot)])
    assert exc.value.args == ("test", True)
    with pytest.raises(SystemExit):
        E.main(["run", "--split", "validation", "--seeds", "12", "--out", str(tmp_path)])


@pytest.mark.parametrize(
    "extra",
    [
        ["--split", "test", "--seeds", "11", "23"],  # test needs exactly the five locked seeds
        ["--split", "test", "--seeds", "11", "11", "23", "47", "71", "101"],
        ["--split", "validation", "--seeds", "11", "11"],
        ["--split", "validation", "--seeds", "11", "--max-minutes", "nan"],
        ["--split", "validation", "--seeds", "11", "--max-minutes", "0"],
        ["--split", "validation", "--seeds", "11", "--eval-layouts", "0"],
        ["--split", "validation", "--seeds", "11", "--text-dim", "3"],
    ],
)
def test_cli_rejects_invalid_runs_before_data_generation(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(E, "generate_split", lambda *a, **k: (_ for _ in ()).throw(Unlocked()))
    with pytest.raises(SystemExit):
        E.main(["run", *extra, "--out", str(tmp_path / "o")])


def test_one_direction_health_needs_two_outcomes_not_four_states():
    one_d = {"finite": True, "unique_states_heldout": 2, "unique_states_train": 2,
             "effective_rank_heldout": 1.0, "retained_eigenvalues": [0.5],
             "singular_values_heldout": [1.0]}  # fmt: skip
    assert E.text_healthy(one_d, 1) and not E.text_healthy(one_d, 2)
    assert not E.text_healthy(one_d | {"unique_states_heldout": 1}, 1)
    assert not E.text_healthy(one_d | {"singular_values_heldout": [0.0]}, 1)
    assert not E.text_healthy(one_d | {"finite": False}, 1)


def test_both_goals_correct_requires_exactly_the_two_goals():
    pairs = {("l", "t", 0, None): [True, True], ("m", "t", 0, None): [True, False]}
    assert E.both_correct(pairs) == 0.5
    for bad in ({}, {("l", "t", 0, None): [True]}, {("l", "t", 0, None): [True] * 3}):
        with pytest.raises(ValueError):
            E.both_correct(bad)


def test_paired_effect_rejects_invalid_tables():
    rng = np.random.default_rng(0)
    ok = (np.array([[1.0, 0.0]]), np.array([[2.0, 1.0]]))
    for bad in [
        (np.zeros((0, 0)), np.zeros((0, 0))),  # empty
        (np.array([[np.nan, 0.0]]), ok[1]),  # non-finite
        (np.array([[3.0, 0.0]]), ok[1]),  # more wins than episodes
        (np.array([[0.0, 0.0]]), np.array([[2.0, 0.0]])),  # cluster with no coverage
    ]:
        with pytest.raises(ValueError):
            stats.paired_effect(bad, bad, 10, rng)
    with pytest.raises(ValueError):
        stats.paired_effect(ok, (ok[0], np.array([[2.0, 2.0]])), 10, rng)  # count mismatch
    with pytest.raises(ValueError):
        stats.paired_effect(ok, ok, 0, rng)


def test_metrics_json_is_strict():
    with pytest.raises(ValueError):
        E.dump({"x": float("nan")})
    assert json.loads(E.dump({"detour_success": None})) == {"detour_success": None}


# ---------------------------------------------------------------- bounded end-to-end smoke
def test_smoke_run_writes_reproducible_outputs(tmp_path):
    start = time.perf_counter()
    code = E.main(
        ["run", "--split", "validation", "--seeds", "11", "--eval-layouts", "2",
         "--max-minutes", "3", "--out", str(tmp_path)]
    )  # fmt: skip
    assert time.perf_counter() - start < 120
    assert code == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert metrics["gates_passed"] and metrics["frozen_digest"] == manifest["frozen_digest"]
    assert manifest["model_seeds_locked"] == list(D.MODEL_SEEDS)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    needed = {"seed", "cluster", "trim", "goal", "outcome", "steps", "route_length", "detour",
              "actions", "positions", "condition", "success"}  # fmt: skip
    assert all(needed <= set(r) for r in rows)
    layouts = generate_split(CFG, "validation")
    for r in rows:  # trajectories replay exactly in the simulator
        lay = layouts[r["layout"]]
        assert W.run_episode(lay, r["actions"], CFG.horizon)[1] == r["steps"]
        assert r["outcome"] == W.run_episode(lay, r["actions"], CFG.horizon)[0]
    assert {"trim_swap_minus_goal_swap", "clean_minus_goal_swap"} <= set(metrics["effects"])
    assert (tmp_path / "summary.md").read_text().startswith("# validation run")


def test_paired_effect_rejects_any_empty_seed_cluster_cell():
    count = np.eye(2)  # every row and column covered, but two cells have no episodes
    with pytest.raises(ValueError, match="cell"):
        stats.paired_effect((count, count), (count, count), 10, np.random.default_rng(0))


@pytest.mark.parametrize("content", ["[]", "null", '"x"', "{not json", b"\xff\xfe"])
def test_malformed_pilot_json_is_refused_before_split_generation(tmp_path, monkeypatch, content):
    monkeypatch.setattr(E, "generate_split", lambda *a, **k: (_ for _ in ()).throw(Unlocked()))
    pilot = tmp_path / "pilot.json"
    pilot.write_bytes(content if isinstance(content, bytes) else content.encode())
    with pytest.raises(SystemExit, match="test unlock refused: .*pilot"):
        E.main(["run", "--split", "test", "--seeds", *map(str, D.MODEL_SEEDS),
                "--unlock-test", "--pilot", str(pilot), "--out", str(tmp_path / "o")])  # fmt: skip
