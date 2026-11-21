"""Follow-up study runner. Called only after the CLI gates (prediction, freeze, fresh --out).

Fitted models are injected through ``Components`` so the orchestration (banks, balanced
queries, conditions, first-terminal execution, rows, metrics, manifests, caps) is testable with
hand-coded mocks. Action choice for every text/reference condition comes only from the frozen
image planner; the oracle is an evaluation-only BFS reference executed in the simulator.

Budget: the in-process cap is cooperative wall-clock only (checked between scenes and stages);
the CLI child process carries the outer hard OS deadline. Memory is not traced (tracemalloc
slows numpy-heavy loops several-fold); caches are bounded per scene instead.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from pgt.followup import analysis as A
from pgt.followup.banks import generate_bank, prior_topologies, with_glyphs
from pgt.followup.bridge import factor_onehot, fit_ridge, keyword_combo
from pgt.followup.corpus import HELDOUT_TEMPLATES, TEMPLATE_POOL, caption
from pgt.followup.design import ARMS, SEEN, TRIMS4, FollowupDesign, panel_glyphs
from pgt.followup.world4 import Board, execute, oracle_path, render4
from pgt.world import TIMEOUT

PHASES = ("pilot", "test")
SPLIT = {"pilot": "validation", "test": "test"}


class CapExceeded(RuntimeError):
    pass


class Vision(Protocol):
    def outcome(self, images: np.ndarray) -> np.ndarray: ...


@dataclass
class Components:
    """Frozen fitted pieces. ``encode(corpus, arm, texts)`` -> (n, d) text vectors.
    ``raw(texts)`` -> (n, v) bag of words over the SAME fixed vocabulary for every arm."""

    vision: Vision
    planner: Callable  # (image, goal_vec, steps_left) -> object with .action
    encode: Callable[[int, str, list[str]], np.ndarray]
    raw: Callable[[list[str]], np.ndarray]
    anchor_boards: Callable[[int], list[tuple[tuple[str, str], Board]]]  # bank -> seen anchors


@dataclass
class Budget:
    """Cooperative wall-clock budget (see module docstring for what is measured)."""

    seconds: float
    start: float = field(default_factory=time.perf_counter)

    def check(self, where: str) -> None:
        if time.perf_counter() - self.start > self.seconds:
            raise CapExceeded(f"wall-clock cap {self.seconds}s exceeded at {where}")


# ---------------------------------------------------------------- banks and queries
def evaluation_topology(walls) -> bool:
    """Fresh validation/test topologies must have the evaluation role; training/calibration
    topologies use the other role of the same helper, so the two are disjoint."""
    from pgt.followup import fitting  # lazy: avoids a circular import at module load

    return fitting.topology_role(walls) == "eval"


def build_banks(
    design: FollowupDesign, phase: str, prior_layouts: list[dict], accept=evaluation_topology
) -> dict:
    """Validation always; test only for phase 'test'. Each bank has its own named RNG and
    excludes every original evaluated topology and every earlier follow-up bank."""
    if phase not in PHASES:
        raise ValueError(phase)
    excluded = prior_topologies(prior_layouts)
    n_prior = len(excluded)
    sizes = {"validation": design.n_validation_topologies, "test": design.n_test_topologies}
    names = ("validation", "test") if phase == "test" else ("validation",)
    banks = {n: generate_bank(design, n, sizes[n], excluded, accept_topology=accept) for n in names}
    return {"banks": banks, "n_prior_excluded": n_prior}


@dataclass(frozen=True)
class Query:
    board_id: str
    panel: str
    trim: str
    order: int  # 0 or 1: which glyph sits on pads[0]; both orders on the same map
    template: int
    goal: str

    @property
    def key(self) -> str:
        return f"{self.trim}|o{self.order}|t{self.template}|{self.goal}"


def queries(boards: list[Board], split: str) -> list[tuple[Query, Board]]:
    """Per board x trim x panel: the two panel glyphs in both pad positions, both held-out
    templates and both goals, all on the exact same map (balanced by construction)."""
    out = []
    for i, base in enumerate(boards):
        for trim in TRIMS4:
            for panel, seen in (("seen", True), ("withheld", False)):
                g = panel_glyphs(trim, seen)
                for order, glyphs in enumerate((g, g[::-1])):
                    b = with_glyphs(base, glyphs)
                    for t in range(len(HELDOUT_TEMPLATES)):
                        for goal in g:
                            out.append((Query(f"{split}/{i}", panel, trim, order, t, goal), b))
    return out


# ---------------------------------------------------------------- execution
class FrameCache:
    """Bounded cache of current-position renders, shared across goals of one scene."""

    def __init__(self, cap: int):
        self.cap, self.data = cap, {}

    def get(self, board: Board, pos, trim: str) -> np.ndarray:
        k = (board, tuple(pos), trim)
        if k not in self.data:
            if len(self.data) >= self.cap:
                self.data.pop(next(iter(self.data)))
            self.data[k] = render4(board, pos, trim)
        return self.data[k]


def run_planned(planner, board: Board, trim: str, goal_vec, horizon: int, cache: FrameCache):
    pos, outcome, steps = board.start, TIMEOUT, 0
    for t in range(horizon):
        plan = planner(cache.get(board, pos, trim), goal_vec, horizon - t)
        pos, hit = execute(board, pos, int(plan.action), t, horizon)
        steps += 1
        if hit is not None:
            outcome = hit
            break
    return outcome, steps


def run_oracle(board: Board, goal: str, horizon: int):
    pos, outcome = board.start, TIMEOUT
    path = oracle_path(board, goal) or []
    for t, a in enumerate(path[:horizon]):
        pos, hit = execute(board, pos, a, t, horizon)
        if hit is not None:
            return hit, t + 1
    return outcome, len(path[:horizon])


def grade(outcome: str, goal: str) -> str:
    if outcome == goal:
        return "correct"
    return "timeout" if outcome == TIMEOUT else "wrong_pad"


def goal_image(vision: Vision, board: Board, goal: str, trim: str) -> np.ndarray:
    return vision.outcome(render4(board, board.pad(goal), trim)[None])[0]


# ---------------------------------------------------------------- bridges (seen anchors only)
def anchor_captions(combos: list[tuple[str, str]]) -> list[str]:
    """Train-phrasing captions only (never a held-out template). Template frames cycle within
    each combo, so combos and frames are balanced; the texts are shared by every arm/corpus."""
    seen_n: dict = {}
    out = []
    for c in combos:
        i = seen_n[c] = seen_n.get(c, -1) + 1
        out.append(caption(c, TEMPLATE_POOL[(i + SEEN.index(c)) % len(TEMPLATE_POOL)]))
    return out


def fit_bridges(design: FollowupDesign, comp: Components) -> dict:
    """Per anchor bank: image targets from seen anchors; ridge from each text space."""
    out = {}
    for bank in design.anchor_banks:
        anchors = comp.anchor_boards(bank)
        if any(c not in SEEN for c, _ in anchors):
            raise ValueError("anchors must use seen combos only")
        combos = [c for c, _ in anchors]
        y = np.stack([goal_image(comp.vision, b, c[0], c[1]) for c, b in anchors])
        texts = anchor_captions(combos)
        kw = [keyword_combo(t) for t in texts]
        out[(None, bank, "keyword")] = fit_ridge(factor_onehot(kw), y, design.ridge_penalty)
        out[(None, bank, "raw")] = fit_ridge(comp.raw(texts), y, design.ridge_penalty)
        for corpus in design.corpus_seeds:
            for arm in ARMS:
                x = comp.encode(corpus, arm, texts)
                out[(corpus, bank, arm)] = fit_ridge(x, y, design.ridge_penalty)
        out[("blind", bank)] = y.mean(0)
    return out


def text_vectors(design, comp, bridges) -> dict:
    """Board-independent goal vectors, computed once per (combo, held-out template)."""
    out = {}
    for trim in TRIMS4:
        for goal in sorted({g for s in (True, False) for g in panel_glyphs(trim, s)}):
            for t, tpl in enumerate(HELDOUT_TEMPLATES):
                text = caption((goal, trim), tpl)
                vecs = {}
                for bank in design.anchor_banks:
                    vecs[("keyword", None, bank)] = bridges[(None, bank, "keyword")](
                        factor_onehot([keyword_combo(text)])
                    )[0]
                    vecs[("raw", None, bank)] = bridges[(None, bank, "raw")](comp.raw([text]))[0]
                    vecs[("blind", None, bank)] = bridges[("blind", bank)]
                    for corpus in design.corpus_seeds:
                        for arm in ARMS:
                            vecs[(arm, corpus, bank)] = bridges[(corpus, bank, arm)](
                                comp.encode(corpus, arm, [text])
                            )[0]
                out[(goal, trim, t)] = vecs
    return out


def _score(target, v) -> float:
    return -round(float(np.sum((target - v) ** 2)), 9)


def evaluate(design, comp, bridges, split: str, boards, budget: Budget) -> tuple[list, list]:
    """Episodes are deterministic in (scene, goal vector), so equal goal vectors in one scene
    (e.g. blind, or keyword across templates) share one trajectory via a per-scene memo."""
    rows, disc = [], []
    tv = text_vectors(design, comp, bridges)
    scene = memo = images = cache = None
    for q, board in queries(boards, split):
        if (board, q.trim) != scene:
            budget.check(f"{split}:{q.board_id}:{q.trim}")
            scene, memo, cache = (board, q.trim), {}, FrameCache(design.frame_cache_max)
            images = {g: goal_image(comp.vision, board, g, q.trim) for g in board.glyphs}
        base = {"split": split, "panel": q.panel, "board": q.board_id, "query": q.key}
        vecs = {("image", None, None): images[q.goal]} | tv[(q.goal, q.trim, q.template)]
        for (cond, corpus, bank), vec in vecs.items():
            k = np.asarray(vec, dtype=float).tobytes()
            if k not in memo:
                memo[k] = run_planned(comp.planner, board, q.trim, vec, design.horizon, cache)
            outcome, steps = memo[k]
            rows.append(
                base
                | {
                    "condition": cond,
                    "corpus": corpus,
                    "bank": bank,
                    "outcome": grade(outcome, q.goal),
                    "steps": steps,
                }
            )
            if cond not in ("image", "blind"):
                disc.append(
                    {
                        "condition": cond,
                        "corpus": corpus,
                        "bank": bank,
                        "panel": q.panel,
                        "correct": _score(images[q.goal], vec),
                        "foil": _score(images[board.other(q.goal)], vec),
                    }
                )
        outcome, steps = run_oracle(board, q.goal, design.horizon)
        rows.append(
            base
            | {
                "condition": "oracle",
                "corpus": None,
                "bank": None,
                "outcome": grade(outcome, q.goal),
                "steps": steps,
            }
        )
    return rows, disc


def key_coverage(rows: list[dict]) -> dict:
    """Every condition cell must cover exactly the same (panel, board, query) keys."""
    cells: dict = {}
    for r in rows:
        cells.setdefault((r["condition"], r["corpus"], r["bank"]), set()).add(
            (r["panel"], r["board"], r["query"])
        )
    sets = list(cells.values())
    return {
        "cells": len(cells),
        "keys": len(sets[0]) if sets else 0,
        "identical": all(s == sets[0] for s in sets),
    }


# ---------------------------------------------------------------- phase runner
def _write(out: Path, manifest: dict, rows: list[dict] | None = None) -> None:
    if rows is not None:
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _production_hooks(comp_factory, prior_layouts):
    """Real defaults, imported lazily so importing the engine never loads training code."""
    if comp_factory is None or prior_layouts is None:
        from pgt.followup import fitting

        comp_factory = comp_factory or fitting.fit_components
        prior_layouts = fitting.prior_layouts if prior_layouts is None else prior_layouts
    return comp_factory, prior_layouts


def run_phase(
    phase: str,
    design: FollowupDesign,
    freeze: dict,
    out: Path,
    comp_factory=None,
    prior_layouts=None,
    pilot: dict | None = None,
) -> dict:
    """``comp_factory(design, budget)`` fits/loads components (training counts against the cap).
    ``prior_layouts`` is a list or a zero-arg callable, called inside the budget. Pilot evaluates
    validation only; test needs a PASS pilot record under the same freeze and design."""
    if phase not in PHASES:
        raise ValueError(phase)
    if phase == "test" and (
        not pilot
        or pilot.get("freeze") != freeze
        or pilot.get("design_digest") != design.digest()
        or pilot.get("health", {}).get("status") != "PASS"
        or pilot.get("status") != "PASS"
    ):
        raise PermissionError("test requires a PASS pilot from the same freeze and design")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite or resume
    cap = design.cap_pilot_s if phase == "pilot" else design.cap_test_s
    budget = Budget(cap + design.cap_build_s)
    manifest = {
        "phase": phase,
        "freeze": freeze,
        "design": design.to_json(),
        "design_digest": design.digest(),
        "status": "INCOMPLETE",
        "budget_measures": "cooperative wall-clock only; outer OS deadline in CLI",
    }
    _write(out, manifest)  # INCOMPLETE marker until the run finishes and validates
    rows: list = []
    try:
        comp_factory, prior_layouts = _production_hooks(comp_factory, prior_layouts)
        layouts = prior_layouts() if callable(prior_layouts) else prior_layouts
        built = build_banks(design, phase, layouts)
        budget.check("banks")
        comp = comp_factory(design, budget)
        budget.check("components")
        bridges = fit_bridges(design, comp)
        split = SPLIT[phase]
        boards = built["banks"][split]
        rows, disc = evaluate(design, comp, bridges, split, boards, budget)
        manifest["n_prior_excluded"] = built["n_prior_excluded"]
        manifest |= A.summarize(rows, disc, design, split, len(boards))
        ok = manifest["coverage"]["ok"]
        manifest["status"] = manifest["health"]["status"] if ok else "INCOMPLETE"
        if manifest["status"] != "PASS":
            manifest["primary"] = manifest["bootstrap"] = None
    except CapExceeded as exc:
        manifest["incomplete_reason"] = str(exc)
    except Exception as exc:  # recorded, then re-raised: never reported as PASS
        manifest["status"], manifest["incomplete_reason"] = "FAILED", repr(exc)
        raise
    finally:
        manifest["elapsed_s"] = time.perf_counter() - budget.start
        _write(out, manifest, rows)
    return manifest


def verify_pilot(manifest_path: Path, freeze: dict, design: FollowupDesign) -> dict:
    """Re-read a completed pilot's rows and recompute coverage, health gate and digests."""
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = (manifest_path.parent / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    if m.get("phase") != "pilot" or m.get("status") != "PASS":
        raise ValueError("not a completed PASS pilot")
    if m.get("freeze") != freeze or m.get("design_digest") != design.digest():
        raise ValueError("pilot freeze or design differs")
    cov = A.strict_coverage(rows, design, "validation", design.n_validation_topologies)
    if not cov["ok"] or m.get("coverage") != cov:
        raise ValueError(f"pilot coverage invalid: {cov['errors'][:5]}")
    redo = json.loads(json.dumps(A.health_gate(rows, design)))
    if redo != m.get("health") or redo["status"] != "PASS":
        raise ValueError("pilot health gate does not recompute to PASS")
    for k, v in (("counts_digest", A.counts_digest(rows)), ("rows_digest", A.rows_digest(rows))):
        if m.get(k) != v:
            raise ValueError(f"pilot {k} does not match its rows")
    return m
