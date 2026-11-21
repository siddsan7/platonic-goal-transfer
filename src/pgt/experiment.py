"""Milestone 2 runner: train frozen vision per seed, text encoders per arm, evaluate, gate.

    python -m pgt.experiment run --split validation --seeds 11 --out runs/pilot-seed11
    python -m pgt.experiment run --split test --seeds 11 23 47 71 101 --unlock-test \
        --pilot runs/pilot-seed11/metrics.json --out runs/final-test

The test split is only generated with ``--unlock-test`` and a pilot metrics file whose gates
all passed on validation seed 11 with the identical frozen configuration and source digest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pgt import planner as P
from pgt import stats
from pgt import vision as V
from pgt.data import (
    ARMS,
    MODEL_SEEDS,
    PILOT_SEED,
    Config,
    calibration_anchors,
    caption,
    config_json,
    digest,
    eligible,
    eval_twins,
    generate_split,
    make_pairs,
    rng_for,
)
from pgt.text import (
    effective_rank,
    fit_procrustes,
    parse_words,
    train_text_encoder,
    unique_states,
)
from pgt.world import GLYPHS, TIMEOUT, TRIMS, canonical_topology, oracle_path, render, step

SRC = Path(__file__).resolve().parent
HELDOUT = (0, 1)  # both held-out templates are evaluated for every text-goal episode


@dataclasses.dataclass(frozen=True)
class ExpConfig:
    walk_layouts: int = 1024  # first n train layouts (twin-complete) for random-action walks
    walks_per_layout_trim: int = 1
    goal_fit_layouts: int = 512  # train layouts whose goal images fit the outcome embedding
    detector_negatives: int = 8
    epochs: int = 3
    batch: int = 256
    lr: float = 0.5
    l2: float = 1e-4
    cell_dims: int = 2
    floor_dims: int = 1
    text_dim: int = 2
    text_ridge: float = 0.01
    length_penalty: float = 1e-3
    probe_layouts: int = 256  # evaluation-split layouts for the one-step prediction probe
    eval_layouts: int | None = None  # None = whole split (smoke tests use fewer)
    shuffles: int = 4  # independent shuffled-anchor permutations (fixed data-seed streams)
    bootstrap: int = 2000
    image_gate: float = 0.80
    blind_gate: float = 0.50


class GateFailure(RuntimeError):
    pass


class TimeLimit(RuntimeError):
    pass


def source_digest() -> str:
    m = hashlib.sha256()
    for path in sorted(SRC.glob("*.py")):
        m.update(path.name.encode())
        m.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return m.hexdigest()


def frozen_digest(cfg: Config, xcfg: ExpConfig) -> str:
    return digest(
        {"data": config_json(cfg), "exp": dataclasses.asdict(xcfg), "src": source_digest()}
    )


def cluster_id(layout) -> str:
    return hashlib.sha256(repr(canonical_topology(layout.walls)).encode()).hexdigest()[:12]


def seeded(seed: int, purpose: str) -> np.random.Generator:
    tag = int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:8], "little")
    return np.random.default_rng([seed, tag])


# ---------------------------------------------------------------- episodes
def run_planned(planner: P.Planner, layout, trim: str, goal: np.ndarray, horizon: int) -> dict:
    """The planner sees rendered frames only; the simulator executes the chosen actions."""
    pos = layout.start
    actions, positions, first = [], [list(pos)], None
    outcome = TIMEOUT
    for t in range(horizon):
        plan = planner(render(layout, pos, trim), goal, horizon - t)
        if first is None:
            first = plan
        pos = step(layout, pos, plan.action)
        actions.append(plan.action)
        positions.append(list(pos))
        hit = layout.pad_at(pos)
        if hit is not None:
            outcome = hit
            break
    final = render(layout, pos, trim)
    actual = planner.model.outcome(final[None])[0]
    return {
        "outcome": outcome,
        "steps": len(actions),
        "actions": actions,
        "positions": positions,
        "predicted_terminal": first.terminal,
        "predicted_steps": first.depth,
        "probe_error": float(np.linalg.norm(actual - first.outcome)),
    }


def run_oracle(layout, goal: str, horizon: int) -> dict:
    """Evaluation-only reference: BFS on the true map, executed in the simulator."""
    pos = layout.start
    actions, positions, outcome = [], [list(layout.start)], TIMEOUT
    for a in oracle_path(layout, goal)[:horizon]:
        pos = step(layout, pos, a)
        actions.append(a)
        positions.append(list(pos))
        hit = layout.pad_at(pos)
        if hit is not None:
            outcome = hit
            break
    return {"outcome": outcome, "steps": len(actions), "actions": actions, "positions": positions}


# ---------------------------------------------------------------- shared components
def build_shared(cfg: Config, xcfg: ExpConfig) -> dict:
    """Seed-independent data and text encoders (the text estimator is closed-form)."""
    t0 = time.perf_counter()
    train = generate_split(cfg, "train")
    calib = generate_split(cfg, "calibration")
    walks = V.random_walks(
        train[: xcfg.walk_layouts],
        rng_for(cfg, "vision-walks"),
        xcfg.walks_per_layout_trim,
        cfg.horizon,
    )
    goal_imgs, _ = V.goal_images(train[: xcfg.goal_fit_layouts])
    t1 = time.perf_counter()
    encoders = {
        arm: train_text_encoder(make_pairs(cfg, cfg.n_train, arm), xcfg.text_dim, xcfg.text_ridge)
        for arm in ARMS
    }
    anchors = calibration_anchors(cfg, calib)
    anchor_imgs = np.stack([a.image(calib) for a in anchors])
    t2 = time.perf_counter()
    return {
        "walks": walks,
        "goal_imgs": goal_imgs,
        "encoders": encoders,
        "anchors": anchors,
        "anchor_imgs": anchor_imgs,
        "shuffles": [
            rng_for(cfg, "shuffled-anchors", str(k)).permutation(len(anchors))
            for k in range(xcfg.shuffles)
        ],
        "timing": {"data_and_walks_s": t1 - t0, "text_and_anchors_s": t2 - t1},
        "walk_stats": {
            "frames": int(len(walks.frames)),
            "transitions": int(len(walks.src)),
            "terminal_transitions": int(walks.done.sum()),
        },
    }


def text_report(shared: dict) -> dict:
    out = {}
    combos = [(g, t) for g in GLYPHS for t in TRIMS]
    for arm, enc in shared["encoders"].items():
        train_z = enc([caption(c, t) for t in range(2) for c in combos])
        held = [caption(c, t, True) for t in HELDOUT for c in combos]
        held_z = enc(held)
        out[arm] = {
            "eigenvalues": [round(float(x), 6) for x in enc.eigenvalues],
            "retained_eigenvalues": [
                round(float(x), 6) for x in enc.eigenvalues[: enc.proj.shape[1]]
            ],
            "singular_values_heldout": [
                round(float(x), 6) for x in np.linalg.svd(held_z - held_z.mean(0), compute_uv=False)
            ],
            "effective_rank_train": round(effective_rank(train_z), 4),
            "effective_rank_heldout": round(effective_rank(held_z), 4),
            "unique_states_train": unique_states(train_z),
            "unique_states_heldout": unique_states(held_z),
            "heldout_equals_train_latent": bool(np.allclose(train_z, held_z)),
            "oov_tokens_heldout": enc.oov(held),
            "finite": bool(np.isfinite(train_z).all() and np.isfinite(held_z).all()),
            "digest": enc.digest(),
        }
    return out


# ---------------------------------------------------------------- one seed
def one_step_probe(model: V.VisionModel, walks: V.Walks) -> dict:
    """Predicted vs persisted next agent cell in the learned factored state.

    Targets are the detector's reading of the actual next frame (the model's own latent);
    accuracy against the true simulator cell is reported as an evaluation-only probe.
    """
    p = V.patches(walks.frames)
    cells = model.agent_cell(p)
    true = walks.positions[:, 0] * 7 + walks.positions[:, 1]
    pred = V.predict_next_cell(model, p[walks.src], cells[walks.src], walks.actions)
    target, persist = cells[walks.dst], cells[walks.src]
    relevant = target != persist  # action changed the observed state
    err = lambda a, mask: float((a[mask] != target[mask]).mean()) if mask.any() else 0.0  # noqa: E731
    everything = np.ones_like(relevant)
    return {
        "transitions": int(len(target)),
        "action_relevant": int(relevant.sum()),
        "predictor_error_relevant": err(pred, relevant),
        "persistence_error_relevant": err(persist, relevant),
        "predictor_error_all": err(pred, everything),
        "persistence_error_all": err(persist, everything),
        # squared error of the one-hot agent-cell latent equals 2 x error rate
        "detector_accuracy_vs_true_cell": float((cells == true).mean()),
        "terminal_detector_accuracy": float(
            (
                (model.terminal.prob(V.ring(p[np.arange(len(p)), true])) > 0.5)[walks.dst]
                == walks.done
            ).mean()
        ),
    }


def evaluate_seed(cfg, xcfg, shared, seed, split, layouts, deadline) -> tuple[list, dict]:
    timing = {}
    t0 = time.perf_counter()
    model = V.train_vision(
        shared["walks"],
        shared["goal_imgs"],
        seeded(seed, "vision"),
        negatives=xcfg.detector_negatives,
        epochs=xcfg.epochs,
        batch=xcfg.batch,
        lr=xcfg.lr,
        l2=xcfg.l2,
        cell_dims=xcfg.cell_dims,
        floor_dims=xcfg.floor_dims,
    )
    timing["vision_train_s"] = time.perf_counter() - t0
    probe_walks = V.random_walks(
        layouts[: xcfg.probe_layouts], rng_for(cfg, "probe-walks", split), 1, cfg.horizon
    )
    probe = one_step_probe(model, probe_walks)

    # calibration: anchor image latents and one Procrustes map per arm (+ shuffled control)
    anchors, anchor_z = shared["anchors"], model.outcome(shared["anchor_imgs"])
    maps, shuffled = {}, {}
    for arm, enc in shared["encoders"].items():
        tz = enc([a.text for a in anchors])
        maps[arm] = fit_procrustes(tz, anchor_z)
        shuffled[arm] = [fit_procrustes(tz[perm], anchor_z) for perm in shared["shuffles"]]
    combo_mean = {
        (g, t): anchor_z[[i for i, a in enumerate(anchors) if a.combo == (g, t)]].mean(0)
        for g in GLYPHS
        for t in TRIMS
    }
    blind = anchor_z.mean(axis=0)

    goal_img_z = {}
    for q, lay in enumerate(layouts):
        imgs = np.stack([render(lay, lay.pad(g), t) for g in GLYPHS for t in TRIMS])
        for z, (g, t) in zip(
            model.outcome(imgs), [(g, t) for g in GLYPHS for t in TRIMS], strict=True
        ):
            goal_img_z[(q, g, t)] = z

    planner = P.Planner(model, xcfg.length_penalty)
    records = []
    t1 = time.perf_counter()
    for twin in eval_twins(layouts):
        q, trim = twin.quartet, twin.trim
        lay = layouts[q]
        base = {"seed": seed, "split": split, "layout": q, "cluster": cluster_id(lay), "trim": trim}
        for goal in twin.goals:
            route = len(oracle_path(lay, goal))  # reporting only; never given to the planner
            manhattan = abs(lay.pad(goal)[0] - lay.start[0]) + abs(lay.pad(goal)[1] - lay.start[1])
            info = base | {"goal": goal, "route_length": route, "detour": route > manhattan}
            conditions = [
                ("oracle", None, None, None),
                ("image_goal", None, None, goal_img_z[(q, goal, trim)]),
                ("goal_blind", None, None, blind),
            ]
            for h in HELDOUT:
                text = caption((goal, trim), h, heldout=True)
                conditions.append(("glyph_word", h, None, combo_mean[parse_words(text)]))
                for arm, enc in shared["encoders"].items():
                    tz = enc([text])[0]
                    conditions.append((f"text:{arm}", h, None, maps[arm](tz)))
                    for k, shuffled_map in enumerate(shuffled[arm]):
                        conditions.append((f"shuffled:{arm}", h, k, shuffled_map(tz)))
            for name, template, variant, vec in conditions:
                if name == "oracle":
                    ep = run_oracle(lay, goal, cfg.horizon)
                else:
                    ep = run_planned(planner, lay, trim, vec, cfg.horizon)
                records.append(
                    info
                    | {"condition": name, "template": template, "variant": variant}
                    | ep
                    | {"success": ep["outcome"] == goal}
                )
        if time.perf_counter() > deadline:
            raise TimeLimit(f"seed {seed}: wall-clock limit reached during evaluation")
    timing["evaluation_s"] = time.perf_counter() - t1

    retrieval = {}
    for arm in ARMS:
        hits, both = [], []
        for q in range(len(layouts)):
            for t in TRIMS:
                for h in HELDOUT:
                    ok = []
                    for g in GLYPHS:
                        v = maps[arm](shared["encoders"][arm]([caption((g, t), h, True)])[0])
                        d = {gg: np.linalg.norm(v - goal_img_z[(q, gg, t)]) for gg in GLYPHS}
                        ok.append(min(d, key=d.get) == g)
                    hits += ok
                    both.append(all(ok))
        retrieval[arm] = {
            "accuracy": float(np.mean(hits)),
            "goal_sensitive_agreement": float(np.mean(both)),
        }

    anchor_sv = np.linalg.svd(anchor_z - anchor_z.mean(0), compute_uv=False)
    goal_z = np.stack(list(goal_img_z.values()))
    goal_sv = np.linalg.svd(goal_z - goal_z.mean(0), compute_uv=False)
    info = {
        "component_digests": model.digest(),
        "displacement": model.displacement.tolist(),
        "one_step": probe,
        "image_latent": {
            "dim": model.embed.dim,
            "cell_spectrum": [round(float(x), 6) for x in model.embed.cell_spectrum[:6]],
            "floor_spectrum": [round(float(x), 6) for x in model.embed.floor_spectrum[:6]],
            "anchor_singular_values": [round(float(x), 6) for x in anchor_sv],
            "goal_singular_values": [round(float(x), 6) for x in goal_sv],
            "finite": bool(np.isfinite(anchor_z).all() and np.isfinite(goal_z).all()),
        },
        "procrustes_residual": {
            arm: float(
                np.linalg.norm(
                    maps[arm](shared["encoders"][arm]([a.text for a in anchors])) - anchor_z
                )
                / np.sqrt(len(anchors))
            )
            for arm in ARMS
        },
        "retrieval": retrieval,
        "planner_cache": {"frames": len(planner._scenes), "imaginations": len(planner._reach)},
        "timing": timing,
    }
    return records, info


# ---------------------------------------------------------------- metrics and gates
def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["seed"], r["condition"])].append(r)
    out = {}
    for (seed, cond), rows in sorted(groups.items()):
        n = len(rows)
        pairs = defaultdict(list)
        lengths = defaultdict(list)
        for r in rows:
            pairs[(r["layout"], r["trim"], r["template"], r["variant"])].append(r["success"])
            lengths[f"{r['route_length']:02d}"].append(r["success"])
        row = {
            "episodes": n,
            "success": sum(r["success"] for r in rows) / n,
            "wrong_pad": sum(r["outcome"] not in (r["goal"], TIMEOUT) for r in rows) / n,
            "timeout": sum(r["outcome"] == TIMEOUT for r in rows) / n,
            "both_goals_correct": float(np.mean([all(v) for v in pairs.values()])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "detour_success": (
                float(np.mean(detours))
                if (detours := [r["success"] for r in rows if r["detour"]])
                else None
            ),
            "by_route_length": {k: [len(v), float(np.mean(v))] for k, v in sorted(lengths.items())},
        }
        if "probe_error" in rows[0]:
            row["probe"] = {
                "mean_outcome_error": float(np.mean([r["probe_error"] for r in rows])),
                "terminal_kind_agreement": float(
                    np.mean([r["predicted_terminal"] == (r["outcome"] != TIMEOUT) for r in rows])
                ),
                "step_count_agreement": float(
                    np.mean([r["predicted_steps"] == r["steps"] for r in rows])
                ),
            }
        variants = sorted({r["variant"] for r in rows if r["variant"] is not None})
        if variants:
            row["per_variant_success"] = [
                float(np.mean([r["success"] for r in rows if r["variant"] == v])) for v in variants
            ]
        out.setdefault(str(seed), {})[cond] = row
    return out


def gates(cfg, xcfg, summary, seed_info, text) -> dict:
    s = summary
    g = {}
    g["oracle_100pct"] = s["oracle"]["success"] == 1.0
    g["image_goal_ge_80pct"] = s["image_goal"]["success"] >= xcfg.image_gate
    one = seed_info["one_step"]
    g["one_step_beats_persistence"] = (
        one["action_relevant"] > 0
        and one["predictor_error_relevant"] < one["persistence_error_relevant"]
    )
    img = seed_info["image_latent"]
    top = img["goal_singular_values"]
    g["embeddings_finite_noncollapsed"] = (
        img["finite"]
        and len(top) >= 2
        and top[1] > 0.05 * top[0]
        and all(
            t["finite"]
            and t["unique_states_heldout"] == 4
            and t["unique_states_train"] == 4
            and t["effective_rank_heldout"] >= 1.5
            for t in text.values()
        )
    )
    g["goal_blind_le_50pct"] = s["goal_blind"]["success"] <= xcfg.blind_gate
    return g


def controls(summary: dict) -> dict:
    """Goal-blind and shuffled-anchor controls side by side, with timeout shares."""
    blind = summary["goal_blind"]
    out = {
        "goal_blind": {k: blind[k] for k in ("success", "wrong_pad", "timeout")},
        "note": (
            "goal-blind feeds one goal vector to both twins of a pair, so their plans are "
            "identical and at most one twin can succeed (<=50% by construction); a high timeout "
            "share means the control vector is nearest an off-pad outcome, not a pad"
        ),
    }
    for cond, row in summary.items():
        if cond.startswith("shuffled:"):
            out[cond] = {
                "success": row["success"],
                "timeout": row["timeout"],
                "per_permutation_success": row["per_variant_success"],
            }
    return out


def render_summary(metrics: dict, manifest: dict) -> str:
    """Human-readable validation/test summary with wall times and gate outcomes."""
    lines = [
        f"# {metrics['split']} run, seeds {metrics['seeds']}",
        "",
        f"Status: {metrics['status']}. Gates passed: {metrics['gates_passed']}. "
        f"Total wall time: {metrics['timing']['total_s']:.1f} s "
        f"(data+walks {metrics['timing']['data_and_walks_s']:.1f} s, "
        f"text+anchors {metrics['timing']['text_and_anchors_s']:.1f} s).",
        f"Frozen digest: `{metrics['frozen_digest']}`. Episodes: {manifest['episodes']}. "
        f"Eval layouts: {manifest['eval_layouts']} (eligible {manifest['eligible_layouts']}).",
        "",
        "## Gates",
        "",
    ]
    for seed, g in metrics["gates"].items():
        t = metrics["per_seed"][seed]["timing"]
        lines.append(
            f"- seed {seed}: "
            + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in g.items())
            + f" (vision train {t['vision_train_s']:.1f} s,"
            + f" evaluation {t.get('evaluation_s', 0):.1f} s)"
        )
    lines += ["", "## Conditions (success / wrong pad / timeout / both-goals-correct)", ""]
    for seed, rows in metrics["summary"].items():
        lines.append(f"Seed {seed}:")
        lines.append("")
        lines.append("| condition | episodes | success | wrong pad | timeout | both correct |")
        lines.append("|---|---|---|---|---|---|")
        for cond, r in rows.items():
            lines.append(
                f"| {cond} | {r['episodes']} | {r['success']:.3f} | {r['wrong_pad']:.3f} | "
                f"{r['timeout']:.3f} | {r['both_goals_correct']:.3f} |"
            )
        lines.append("")
    lines += ["## Paired effects (pp, topology-clustered bootstrap, seed pairing fixed)", ""]
    for name, e in metrics["effects"].items():
        lines.append(
            f"- {name}: {e['effect_pp']:+.2f} pp, "
            f"95% CI [{e['ci_pp'][0]:+.2f}, {e['ci_pp'][1]:+.2f}], "
            f"per seed {[round(x, 2) for x in e['per_seed_pp']]}, {e['clusters']} clusters"
        )
    lines += ["", "## Text encoders", ""]
    for arm, t in metrics["text"].items():
        lines.append(
            f"- {arm}: retained eigenvalues {t['retained_eigenvalues']}, effective rank "
            f"{t['effective_rank_heldout']}, unique states {t['unique_states_heldout']}, "
            f"held-out latent equals train latent: {t['heldout_equals_train_latent']}, "
            f"OOV tokens {t['oov_tokens_heldout']}"
        )
    lines += ["", "## Per-seed instrument", ""]
    for seed, info in metrics["per_seed"].items():
        one = info["one_step"]
        lines.append(
            f"- seed {seed}: one-step error action-relevant {one['predictor_error_relevant']:.4f} "
            f"vs persistence {one['persistence_error_relevant']:.4f} (all states "
            f"{one['predictor_error_all']:.4f} vs {one['persistence_error_all']:.4f}); "
            f"retrieval {info['retrieval']}; Procrustes residual {info['procrustes_residual']}; "
            f"component digests {info['component_digests']}"
        )
    return "\n".join(lines) + "\n"


def run(args) -> int:
    started = time.perf_counter()
    deadline = started + 60 * args.max_minutes
    cfg, xcfg = Config(), ExpConfig(eval_layouts=args.eval_layouts)
    frozen = frozen_digest(cfg, xcfg)
    if args.split == "test":
        if not args.unlock_test or not args.pilot:
            raise SystemExit("test split requires --unlock-test and --pilot <validation metrics>")
        pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
        problems = [
            (pilot.get("split") != "validation", "pilot was not on validation"),
            (pilot.get("seeds") != [PILOT_SEED], f"pilot seeds must be [{PILOT_SEED}]"),
            (not pilot.get("gates_passed"), "pilot gates did not all pass"),
            (pilot.get("frozen_digest") != frozen, "config or source changed since the pilot"),
            (args.eval_layouts is not None, "test runs use the whole split"),
        ]
        bad = [m for failed, m in problems if failed]
        if bad:
            raise SystemExit("test unlock refused: " + "; ".join(bad))
    layouts = generate_split(cfg, args.split, unlock_test=args.split == "test" and args.unlock_test)
    if xcfg.eval_layouts is not None:
        layouts = layouts[: xcfg.eval_layouts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shared = build_shared(cfg, xcfg)
    text = text_report(shared)
    records, per_seed, status = [], {}, "complete"
    try:
        for seed in args.seeds:
            recs, info = evaluate_seed(cfg, xcfg, shared, seed, args.split, layouts, deadline)
            records += recs
            per_seed[str(seed)] = info
    except TimeLimit as exc:
        status = f"aborted: {exc}"
    summary = summarize(records)
    seeds_done = [int(s) for s in per_seed]
    clusters = sorted({r["cluster"] for r in records})
    effects = {}
    if seeds_done:
        rng = rng_for(cfg, "bootstrap", args.split)
        tab = {
            c: stats.cluster_table(records, c, seeds_done, clusters)
            for c in {r["condition"] for r in records}
        }
        for name, a, b in [
            ("trim_swap_minus_goal_swap", "text:trim_swap", "text:goal_swap"),
            ("clean_minus_goal_swap", "text:clean", "text:goal_swap"),
            ("clean_minus_trim_swap", "text:clean", "text:trim_swap"),
        ]:
            effects[name] = stats.paired_effect(tab[a], tab[b], xcfg.bootstrap, rng)
    gate = {s: gates(cfg, xcfg, summary[s], per_seed[s], text) for s in per_seed}
    passed = status == "complete" and all(all(v.values()) for v in gate.values())
    wall = time.perf_counter() - started
    metrics = {
        "split": args.split,
        "seeds": seeds_done,
        "status": status,
        "frozen_digest": frozen,
        "gates": gate,
        "gates_passed": passed,
        "summary": summary,
        "effects": effects,
        "controls": {s: controls(summary[s]) for s in per_seed},
        "text": text,
        "per_seed": per_seed,
        "walk_stats": shared["walk_stats"],
        "timing": shared["timing"] | {"total_s": wall},
    }
    manifest = {
        "command": ["python", "-m", "pgt.experiment", *args.argv],
        "frozen_digest": frozen,
        "source_digest": source_digest(),
        "data_config": config_json(cfg),
        "exp_config": dataclasses.asdict(xcfg),
        "split": args.split,
        "eval_layouts": len(layouts),
        "eval_layout_digest": digest([lay.to_json() for lay in layouts]),
        "eligible_layouts": sum(eligible(lay, cfg) for lay in layouts),
        "anchor_digest": digest([[a.layout, a.combo, a.template] for a in shared["anchors"]]),
        "seeds": seeds_done,
        "model_seeds_locked": list(MODEL_SEEDS),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "episodes": len(records),
        "status": status,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    (out / "summary.md").write_text(render_summary(metrics, manifest), encoding="utf-8")
    print(
        json.dumps(
            {"status": status, "gates_passed": passed, "wall_s": round(wall, 1), "out": str(out)}
        )
    )
    return 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Milestone 2 experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=("validation", "test"), required=True)
    r.add_argument("--seeds", type=int, nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-minutes", type=float, default=20.0)
    r.add_argument("--eval-layouts", type=int, default=None, help="smoke runs only")
    r.add_argument("--unlock-test", action="store_true")
    r.add_argument("--pilot", help="validation pilot metrics.json required to unlock test")
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    bad_seeds = [s for s in args.seeds if s not in MODEL_SEEDS]
    if bad_seeds:
        parser.error(f"seeds must be from the locked set {MODEL_SEEDS}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
