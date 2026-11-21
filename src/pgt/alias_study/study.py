"""Bounded alias-transfer study (additive; the original and follow-up packages are reused unedited).

Question: when four novel nonce words are linked to ordinary glyph words only through text-only
synthetic caption pairs, does an encoder trained on the *aligned* links place the calibrated goal
for a nonce caption closer to the target pad's goal image than an encoder trained on matched
*independent* links? Protocol and limits: docs/alias-protocol.md.

    python -m pgt.alias_study run --out runs/alias-<label>      (wrap in an outer 300 s timeout)
    python -m pgt.alias_study verify --out runs/alias-<label>

Nothing runs on import. ``run`` refuses an existing ``--out`` and a dirty Git tree (unless
``--allow-dirty``, which is recorded). The only time control inside the process is a cooperative
budget checked between stages and scenes; the hard limit is the orchestrator's outer timeout. A
killed run leaves ``summary.json`` with status INCOMPLETE.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
import math
import subprocess
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pgt.followup import encoder as E
from pgt.followup import engine
from pgt.followup.bridge import fit_ridge
from pgt.followup.corpus import (
    HELDOUT_TEMPLATES,
    TEMPLATE_POOL,
    caption,
    corpus_templates,
    make_corpus,
    vocabulary,
)
from pgt.followup.design import GLYPHS4, SEEN, TRIMS4, FollowupDesign, panel_glyphs
from pgt.followup.world4 import Board

# ---------------------------------------------------------------- fixed settings (pre-run)
NONCES = ("vexlo", "trumbik", "olquen", "sprindt")  # not English words; absent from every
# template, the fitted vocabulary and every calibration caption (checked in tests and at run).
ALIAS_BLOCK_SEEDS = (5101, 5102)  # one base permutation per block, then its 4 rotations; the
# second base is drawn only from permutations outside the first base's rotation class.
CORPUS_SEED = 1  # canonical paraphrase pairs and templates: follow-up clean corpus, seed 1
ANCHOR_BANK = 1  # follow-up calibration bank 1 (64 seen-combo anchors), shared by every probe
INIT_STREAM = 7103  # encoder init rng = [CORPUS_SEED, INIT_STREAM, mapping index]
N_BOARDS = 12  # the first 12 of the 48 published follow-up test topologies (ids test/0..11)
CANONICAL_MIN = 0.8  # ordinary-word executed success, aligned and independent arms each
BUDGET_S = 270.0  # cooperative budget; the orchestrator's outer hard timeout is 300 s
BOOT_N, BOOT_SEED = 2000, 20261121

LEARNED = ("aligned", "independent", "untrained")
ARMS = (*LEARNED, "raw")
KINDS = ("nonce", "canonical")
ROW_ARMS = (*ARMS, "image")
ROW_KINDS = (*KINDS, "image")
OUTCOMES = ("correct", "wrong_pad", "timeout")
STATUSES = ("VALID", "INVALID", "INCOMPLETE", "FAILED")
SPLIT = "test"


# ---------------------------------------------------------------- design
def _rotations(perm: tuple[int, ...]) -> list[tuple[int, ...]]:
    n = len(perm)
    return [tuple(perm[(i + r) % n] for i in range(n)) for r in range(n)]


def base_permutations() -> list[tuple[int, ...]]:
    """Block 1: ``default_rng(seed1).permutation(4)``. Block 2: drawn conditional on a rotation
    class disjoint from block 1's: every permutation of range(4) outside block 1's rotations,
    enumerated in sorted order, one chosen by ``default_rng(seed2).integers(len(available))``.
    Bounded (24 candidates) and deterministic; the eight mappings are therefore distinct."""
    n = len(GLYPHS4)
    first = tuple(int(x) for x in np.random.default_rng(ALIAS_BLOCK_SEEDS[0]).permutation(n))
    taken = set(_rotations(first))
    available = [p for p in sorted(itertools.permutations(range(n))) if p not in taken]
    second = available[int(np.random.default_rng(ALIAS_BLOCK_SEEDS[1]).integers(len(available)))]
    return [first, second]


def mappings() -> list[dict[str, str]]:
    """Two blocks of four Latin rotations (see ``base_permutations``). Within a block every
    alias maps to every glyph once; the rotations are not independent draws."""
    out = []
    for perm in base_permutations():
        for rot in _rotations(perm):
            out.append({a: GLYPHS4[rot[i]] for i, a in enumerate(NONCES)})
    return out


def inverse(mapping: dict[str, str]) -> dict[str, str]:
    return {g: a for a, g in mapping.items()}


def vocab() -> tuple[str, ...]:
    """The follow-up vocabulary plus the four nonces; identical for every arm and the raw count."""
    return tuple(sorted(set(vocabulary()) | set(NONCES)))


def canonical_counts(design: FollowupDesign) -> Counter:
    """The follow-up clean corpus (seen combos, two templates), counted, duplicates as weights."""
    return Counter((p.left, p.right) for p in make_corpus(design, CORPUS_SEED, "clean"))


def link_counts(mapping: dict[str, str], arm: str) -> Counter:
    """Bidirectional cross-template alias<->canonical links over all 4 aliases x 4 trims.

    Direction 0: alias caption (template a) -> canonical caption (template b); direction 1:
    canonical (a) -> alias (b). The trim is kept. Per alias, trim and direction there are four
    links: ``aligned`` repeats the mapped glyph four times, ``independent`` uses each glyph once.
    Both arms therefore have identical left and right caption marginals and total weight.
    """
    ta, tb = corpus_templates(CORPUS_SEED)
    out: Counter = Counter()
    for alias in NONCES:
        for trim in TRIMS4:
            if arm == "aligned":
                glyphs = [mapping[alias]] * len(GLYPHS4)
            elif arm == "independent":
                glyphs = list(GLYPHS4)
            else:
                raise ValueError(arm)
            for g in glyphs:
                out[(caption((alias, trim), ta), caption((g, trim), tb))] += 1
                out[(caption((g, trim), ta), caption((alias, trim), tb))] += 1
    return out


def training_pairs(design: FollowupDesign, mapping: dict[str, str], arm: str):
    """Sorted unique pairs and normalised weights: canonical pairs + this arm's links."""
    counts = canonical_counts(design) + link_counts(mapping, arm)
    keys = sorted(counts)
    w = np.array([counts[k] for k in keys], dtype=float)
    return keys, w / w.sum()


# ---------------------------------------------------------------- fitted components
@dataclass
class Components:
    vision: object  # .outcome(images) -> goal-image vectors
    planner: Callable  # (image, goal_vec, steps_left) -> object with .action
    anchors: list[tuple[tuple[str, str], Board]]  # seen-combo calibration boards
    encode: Callable[[int, str, list[str]], np.ndarray]  # (mapping index, arm, texts)
    raw: Callable[[list[str]], np.ndarray]  # bag of words over vocab()
    # accepted text-training epochs per "mapping/arm"; design.text_epochs is only the maximum
    epochs_run: dict[str, int] = field(default_factory=dict)


def fit_encoders(design: FollowupDesign, bow, maps, budget, epochs_run: dict) -> dict:
    """Per mapping one init, cloned into aligned (trained), independent (trained), untrained.
    ``design.text_epochs`` is the configured maximum; ``encoder.train`` stops early when no
    finite descent step exists, so the accepted epochs per arm are recorded in ``epochs_run``."""
    out = {}
    for m, mapping in enumerate(maps):
        init = E.init_params(
            np.random.default_rng([CORPUS_SEED, INIT_STREAM, m]),
            len(bow.index),
            design.text_hidden,
            design.text_out,
        )
        out[(m, "untrained")] = E.Params(*(a.copy() for a in init.flat()))
        for arm in ("aligned", "independent"):
            keys, w = training_pairs(design, mapping, arm)
            out[(m, arm)], trace = E.train(
                init,
                bow([k[0] for k in keys]),
                bow([k[1] for k in keys]),
                w,
                design.text_epochs,
                design.text_lr,
            )
            epochs_run[f"{m}/{arm}"] = len(trace) - 1
            budget.check(f"fit:text-{m}-{arm}")
    return out


def real_components(design: FollowupDesign, budget, maps) -> Components:
    """Rebuild the follow-up visual model and calibration bank 1 exactly as ``fit_components``
    does (same design, same shared exclude set, vision first), then fit this study's encoders."""
    from pgt.followup import fitting  # lazy: importing this module never loads training code

    excluded: set[tuple] = set()
    model = fitting.fit_vision(design, excluded, budget)
    planner = fitting.BoundedPlanner(model, fitting.LENGTH_PENALTY, fitting.PLANNER_SCENE_CAP)
    anchors = fitting.anchor_bank(design, ANCHOR_BANK, excluded)
    budget.check("fit:anchors")
    bow = fitting.BagOfWords(vocab())
    epochs_run: dict[str, int] = {}
    params = fit_encoders(design, bow, maps, budget, epochs_run)

    def encode(m: int, arm: str, texts: list[str]) -> np.ndarray:
        return E.forward(params[(m, arm)], bow(texts))[0]

    return Components(model, planner, anchors, encode, bow, epochs_run)


def published_boards(design: FollowupDesign, n: int = N_BOARDS) -> list[tuple[str, Board]]:
    """The first ``n`` boards of the published follow-up test bank, regenerated by the frozen
    bank code (validation first, then test, sharing one exclude set), with their original ids."""
    from pgt.followup import fitting

    built = engine.build_banks(design, "test", fitting.prior_layouts())
    return [(f"{SPLIT}/{i}", b) for i, b in enumerate(built["banks"]["test"][:n])]


def board_record(bid: str, b: Board) -> dict:
    return {
        "id": bid,
        "walls": sorted(b.walls),
        "start": b.start,
        "pads": b.pads,
    }


# ---------------------------------------------------------------- probes and goal vectors
def anchor_texts(comp: Components) -> list[str]:
    combos = [c for c, _ in comp.anchors]
    if any(c not in SEEN for c in combos):
        raise ValueError("calibration anchors must use seen canonical combos only")
    texts = engine.anchor_captions(combos)
    if any(w in NONCES for t in texts for w in t.split()):
        raise ValueError("a nonce appears in a calibration caption")
    return texts


def fit_probes(design: FollowupDesign, comp: Components, n_maps: int) -> dict:
    """One centred ridge per text space on the SAME canonical anchors and image targets."""
    texts = anchor_texts(comp)
    y = np.stack([engine.goal_image(comp.vision, b, c[0], c[1]) for c, b in comp.anchors])
    raw = engine.require_finite("raw anchors", comp.raw(texts))
    probes = {"raw": fit_ridge(raw, y, design.ridge_penalty)}
    for m in range(n_maps):
        for arm in LEARNED:
            x = engine.require_finite(f"anchor vectors {m}/{arm}", comp.encode(m, arm, texts))
            probes[(m, arm)] = fit_ridge(x, y, design.ridge_penalty)
    return probes


def text_for(kind: str, goal: str, trim: str, template: int, mapping: dict[str, str]) -> str:
    word = inverse(mapping)[goal] if kind == "nonce" else goal
    return caption((word, trim), HELDOUT_TEMPLATES[template])


def goal_vector(comp, probes, m: int, arm: str, text: str) -> np.ndarray:
    if arm == "raw":
        v = probes["raw"](comp.raw([text]))[0]
    else:
        v = probes[(m, arm)](comp.encode(m, arm, [text]))[0]
    return engine.require_finite(f"goal vector {m}/{arm}/{text}", v)


def goal_vectors(comp, probes, maps) -> dict:
    out = {}
    for m, mapping in enumerate(maps):
        for arm in ARMS:
            for kind in KINDS:
                for trim in TRIMS4:
                    for goal in GLYPHS4:
                        for t in range(len(HELDOUT_TEMPLATES)):
                            text = text_for(kind, goal, trim, t, mapping)
                            out[(m, arm, kind, goal, trim, t)] = goal_vector(
                                comp, probes, m, arm, text
                            )
    return out


def raw_alias_check(comp: Components, probes: dict) -> dict:
    """Raw counts never see a nonce in calibration, so the ridge coefficients of every nonce
    column are zero and competing aliases get bit-identical goals in every trim and template."""
    coef = probes["raw"].coef
    cols = [int(np.flatnonzero(comp.raw([a])[0])[0]) for a in NONCES]
    coef_zero = bool(np.all(coef[cols] == 0))
    equal = True
    for trim in TRIMS4:
        for tpl in HELDOUT_TEMPLATES:
            vs = [probes["raw"](comp.raw([caption((a, trim), tpl)]))[0] for a in NONCES]
            equal &= all(np.array_equal(vs[0], v) for v in vs[1:])
    return {"coef_zero": coef_zero, "goals_equal": bool(equal), "ok": coef_zero and bool(equal)}


# ---------------------------------------------------------------- evaluation
def _sq(a: np.ndarray, b: np.ndarray) -> float | None:
    """Squared distance, or None (JSON null) when finite inputs overflow while scoring. A null
    distance makes the margin null and the run INVALID; NaN/Infinity are never written."""
    with np.errstate(over="ignore", invalid="ignore"):
        d = float(np.sum((np.asarray(a, dtype=float) - b) ** 2))
    return d if math.isfinite(d) else None


def _margin(d_t: float | None, d_f: float | None) -> float | None:
    if d_t is None or d_f is None:
        return None
    m = d_f - d_t
    return m if math.isfinite(m) else None


def evaluate(design, comp: Components, vecs: dict, maps, boards, budget) -> list[dict]:
    """Every (board, query) of the follow-up layout x mapping x arm x kind, plus one image row.
    The planner only ever receives the rendered current frame, the goal vector and steps left;
    episodes are deterministic in (scene, goal vector), so equal vectors share one rollout."""
    rows: list[dict] = []
    scene = memo = images = cache = None
    for q, board in engine.queries([b for _, b in boards], SPLIT):
        bid = boards[int(q.board_id.rsplit("/", 1)[1])][0]  # queries() numbers boards by position
        if (board, q.trim) != scene:
            budget.check(f"eval:{bid}:{q.trim}")
            scene, memo = (board, q.trim), {}
            cache = engine.FrameCache(design.frame_cache_max)
            images = {g: engine.goal_image(comp.vision, board, g, q.trim) for g in board.glyphs}
        foil = board.other(q.goal)
        target_img, foil_img = images[q.goal], images[foil]
        base = {
            "split": SPLIT,
            "board": bid,
            "panel": q.panel,
            "trim": q.trim,
            "order": q.order,
            "template": q.template,
            "query": q.key,
            "target": q.goal,
            "foil": foil,
        }

        entries = [(target_img, {"mapping": None, "arm": "image", "kind": "image", "word": None})]
        for m, mapping in enumerate(maps):
            for arm in ARMS:
                for kind in KINDS:
                    word = inverse(mapping)[q.goal] if kind == "nonce" else q.goal
                    vec = vecs[(m, arm, kind, q.goal, q.trim, q.template)]
                    entries.append((vec, {"mapping": m, "arm": arm, "kind": kind, "word": word}))
        for vec, extra in entries:
            k = np.asarray(vec, dtype=float).tobytes()
            if k not in memo:
                memo[k] = engine.run_planned(
                    comp.planner, board, q.trim, vec, design.horizon, cache
                )
            outcome, steps = memo[k]
            d_t, d_f = _sq(vec, target_img), _sq(vec, foil_img)
            rows.append(
                base
                | extra
                | {
                    "d_target": d_t,
                    "d_foil": d_f,
                    "margin": _margin(d_t, d_f),
                    "outcome": engine.grade(outcome, q.goal),
                    "steps": steps,
                }
            )
    return rows


# ---------------------------------------------------------------- analysis (rows only)
def _row_key(r: dict) -> tuple:
    return (r["board"], r["query"], r["mapping"], r["arm"], r["kind"])


def expected_keys(board_ids: list[str], n_maps: int) -> set[tuple]:
    out = set()
    for bid in board_ids:
        for trim in TRIMS4:
            for panel, seen in (("seen", True), ("withheld", False)):
                g = panel_glyphs(trim, seen)
                for order in range(2):
                    for t in range(len(HELDOUT_TEMPLATES)):
                        for goal in g:
                            qk = engine.Query(bid, panel, trim, order, t, goal).key
                            out.add((bid, qk, None, "image", "image"))
                            for m in range(n_maps):
                                for arm in ARMS:
                                    for kind in KINDS:
                                        out.add((bid, qk, m, arm, kind))
    return out


def coverage(rows: list[dict], board_ids: list[str], n_maps: int) -> dict:
    keys = [_row_key(r) for r in rows]
    have, want = set(keys), expected_keys(board_ids, n_maps)
    dup = len(keys) - len(have)
    missing, extra = len(want - have), len(have - want)
    return {
        "expected": len(want),
        "rows": len(keys),
        "missing": missing,
        "extra": extra,
        "duplicates": dup,
        "ok": missing == 0 and extra == 0 and dup == 0,
    }


def rates(rows: list[dict]) -> list[dict]:
    """Executed first-terminal outcomes and mean margins per arm and text kind."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["kind"])].append(r)
    out = []
    for (arm, kind), sel in sorted(groups.items()):
        c = Counter(r["outcome"] for r in sel)
        out.append(
            {
                "arm": arm,
                "kind": kind,
                "n": len(sel),
                "correct": c["correct"],
                "wrong_pad": c["wrong_pad"],
                "timeout": c["timeout"],
                "success": c["correct"] / len(sel),
                "margin_unavailable": sum(r["margin"] is None for r in sel),
                "mean_margin": _safe_mean([r["margin"] for r in sel]),
            }
        )
    return out


def _number(x) -> bool:
    if type(x) not in (int, float):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:  # JSON integers can exceed the float range
        return False


def _safe_mean(vals: list) -> float | None:
    """Mean of finite numbers; None if any value is unavailable (never NaN/Infinity).
    Exact fsum when the sum fits; otherwise the values are scaled by their largest magnitude
    first, so finite huge values whose sum overflows (1e308 twice) still get their mean."""
    if not vals or not all(_number(v) for v in vals):
        return None
    try:
        m = math.fsum(vals) / len(vals)
    except OverflowError:
        m = math.inf
    if not math.isfinite(m):
        scale = max(abs(v) for v in vals)
        m = math.fsum(v / scale for v in vals) / len(vals) * scale
    return m if math.isfinite(m) else None


class Unrepresentable(ArithmeticError):
    """A paired contrast whose value does not fit in a finite double."""


def _mean_diff(pairs: list[tuple[float, float]]) -> float:
    """Mean of a - b over pairs. Falls back to halved differences when a single a - b
    overflows; raises Unrepresentable if the mean itself does not fit."""
    diffs = [a - b for a, b in pairs]
    if all(math.isfinite(d) for d in diffs):
        m = _safe_mean(diffs)
    else:
        half = _safe_mean([a / 2 - b / 2 for a, b in pairs])
        m = None if half is None else half * 2
    if m is None or not math.isfinite(m):
        raise Unrepresentable("paired mean difference is not a finite double")
    return m


def raw_rows_symmetric(rows: list[dict]) -> bool:
    """Raw nonce rows: identical (d_target, d_foil) across mappings for every query, and the two
    competing targets of one scene (board, panel, trim, order, template) are exact reverses
    (d_target of one is d_foil of the other), because the raw goal ignores which alias it got."""
    by_query: dict = defaultdict(set)
    scenes: dict = defaultdict(dict)
    for r in rows:
        if r["arm"] == "raw" and r["kind"] == "nonce":
            by_query[(r["board"], r["query"])].add((r["d_target"], r["d_foil"]))
            sc = (r["board"], r["panel"], r["trim"], r["order"], r["template"])
            scenes[sc][r["target"]] = (r["foil"], r["d_target"], r["d_foil"])
    if not by_query or any(len(v) != 1 for v in by_query.values()):
        return False
    for targets in scenes.values():
        if len(targets) != 2:
            return False
        for tgt, (foil, d_t, d_f) in targets.items():
            other = targets.get(foil)
            if other is None or other[0] != tgt or (other[1], other[2]) != (d_f, d_t):
                return False
    return True


def _success(rows, **match) -> float | None:
    sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return sum(r["outcome"] == "correct" for r in sel) / len(sel) if sel else None


def diagnostics(rows: list[dict], raw_check: dict, design: FollowupDesign) -> dict:
    """Pre-declared validity checks. Any failure -> INVALID, and no contrast is reported."""
    finite = all(_number(r[k]) for r in rows for k in ("d_target", "d_foil", "margin")) and all(
        r["margin"] == r["d_foil"] - r["d_target"] for r in rows
    )
    image = _success(rows, arm="image")
    canon = {a: _success(rows, arm=a, kind="canonical") for a in ("aligned", "independent")}
    raw_rows_equal = raw_rows_symmetric(rows)
    representable = finite and _contrasts_representable(rows)
    checks = {
        "finite_margins": finite,
        "image_goal_success": image,
        "image_goal_ok": image is not None and image >= design.gate_image_goal_min,
        "canonical_success": canon,
        "canonical_ok": all(v is not None and v >= CANONICAL_MIN for v in canon.values()),
        "raw_alias_vectors": raw_check,
        "raw_alias_rows_equal": raw_rows_equal,
        "paired_contrasts_representable": representable,
    }
    ok = (
        checks["finite_margins"]
        and checks["image_goal_ok"]
        and checks["canonical_ok"]
        and isinstance(raw_check, dict)
        and bool(raw_check.get("ok"))
        and raw_rows_equal
        and representable
    )
    return checks | {
        "thresholds": {
            "image_goal_min": design.gate_image_goal_min,
            "canonical_min": CANONICAL_MIN,
        },
        "ok": ok,
    }


def _contrasts_representable(rows: list[dict]) -> bool:
    try:
        for b in ("independent", "untrained"):
            paired_summary(rows, "aligned", b)
    except Unrepresentable:
        return False
    return True


def paired_board_means(rows: list[dict], a: str, b: str, kind: str = "nonce") -> dict:
    """Per board: mean over mappings of the mean over queries of margin(a) - margin(b), so
    boards, mappings and queries are equally weighted. Raises if any pair is unmatched."""
    by = {
        (r["board"], r["mapping"], r["query"], r["arm"]): r["margin"]
        for r in rows
        if r["kind"] == kind and r["arm"] in (a, b)
    }
    diffs: dict = defaultdict(lambda: defaultdict(list))
    for (bid, m, q, arm), v in by.items():
        if arm == a:
            other = by.get((bid, m, q, b))
            if other is None:
                raise ValueError(f"{a} vs {b}: unmatched key {(bid, m, q)}")
            diffs[bid][m].append((v, other))
    if not diffs:
        raise ValueError("no paired rows")
    n_a = sum(1 for k in by if k[3] == a)
    n_b = sum(1 for k in by if k[3] == b)
    if n_a != n_b:
        raise ValueError(f"{a} vs {b}: unmatched key counts {n_a} != {n_b}")
    per = {}
    for bid, ms in sorted(diffs.items()):
        per[bid] = _safe_mean([_mean_diff(v) for v in ms.values()])
        if per[bid] is None:
            raise Unrepresentable(f"{a} vs {b}: board {bid} mean is not a finite double")
    return per


def paired_summary(rows: list[dict], a: str, b: str, kind: str = "nonce") -> dict:
    per = paired_board_means(rows, a, b, kind)
    mean = _safe_mean(list(per.values()))
    if mean is None:
        raise Unrepresentable(f"{a} vs {b}: mean over boards is not a finite double")
    return {
        "contrast": f"{a} - {b}",
        "kind": kind,
        "unit": "squared foil distance - squared target distance (positive favours target)",
        "mean": mean,
        "per_board": per,
    }


def board_bootstrap(per_board: dict, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Percentile 95% interval resampling the fixed boards; conditional on the fixed lexicons,
    calibration bank and the one visual fit (not a population interval)."""
    vals = np.array([per_board[k] for k in sorted(per_board)], dtype=float)
    # Resample in units of the largest magnitude so sums of finite huge values cannot overflow;
    # every scaled mean and percentile lies in [-1, 1], so rescaling stays finite.
    scale = float(np.max(np.abs(vals))) or 1.0
    idx = np.random.default_rng(seed).integers(0, len(vals), size=(n, len(vals)))
    means = (vals / scale)[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_resamples": n,
        "n_boards": len(vals),
        "ci95": [float(lo) * scale, float(hi) * scale],
        "conditional_on": "fixed boards, lexicon mappings, calibration bank 1, one visual fit",
    }


def executed_paired_pp(rows: list[dict], a: str, b: str, kind: str = "nonce") -> dict:
    by = {
        (r["board"], r["mapping"], r["query"], r["arm"]): r["outcome"] == "correct"
        for r in rows
        if r["kind"] == kind and r["arm"] in (a, b)
    }
    keys = [k[:3] for k in by if k[3] == a]
    if not keys or any((*k, b) not in by for k in keys):
        raise ValueError(f"{a} vs {b}: unmatched executed rows")
    return {"n": len(keys), "pp": 100 * sum(by[(*k, a)] - by[(*k, b)] for k in keys) / len(keys)}


def summarize(
    rows: list[dict], board_ids: list[str], n_maps: int, raw_check: dict, design: FollowupDesign
) -> dict:
    cov = coverage(rows, board_ids, n_maps)
    out = {"coverage": cov, "rates": rates(rows) if rows else []}
    if not cov["ok"]:
        return out | {
            "status": "INCOMPLETE",
            "diagnostics": None,
            "primary": None,
            "secondary": None,
            "bootstrap": None,
        }
    diag = diagnostics(rows, raw_check, design)
    out["diagnostics"] = diag
    if not diag["ok"]:
        return out | {"status": "INVALID", "primary": None, "secondary": None, "bootstrap": None}
    prim = paired_summary(rows, "aligned", "independent")
    return out | {
        "status": "VALID",
        "primary": prim,
        "bootstrap": board_bootstrap(prim["per_board"]),
        "secondary": {
            "aligned_minus_untrained_nonce": paired_summary(rows, "aligned", "untrained"),
            "executed_aligned_minus_independent_nonce": executed_paired_pp(
                rows, "aligned", "independent"
            ),
            "executed_aligned_minus_untrained_nonce": executed_paired_pp(
                rows, "aligned", "untrained"
            ),
        },
    }


# ---------------------------------------------------------------- files
def rows_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode()


def write_rows(path: Path, rows: list[dict]) -> str:
    raw = rows_bytes(rows)
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(raw)
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(raw).hexdigest()


def read_rows(path: Path) -> tuple[list[dict], str]:
    """Raises ValueError for truncated or corrupt gzip and for non-UTF-8 or non-JSON lines."""
    try:
        raw = gzip.decompress(path.read_bytes())
    except (EOFError, zlib.error, gzip.BadGzipFile) as exc:
        raise ValueError(f"corrupt gzip: {exc!r}") from exc
    lines = raw.decode().splitlines()
    rows = [json.loads(x, parse_constant=_reject_constant) for x in lines if x.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _dump(path: Path, obj: dict) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)
    # Each run owns its new output directory. Preserve the last complete checkpoint on timeout.
    pending = path.with_suffix(".pending")
    pending.write_text(text + "\n", encoding="utf-8")
    pending.replace(path)


def git_state() -> dict:
    root = Path(__file__).resolve().parents[3]

    def git(*a: str) -> str:
        r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    commit = git("rev-parse", "HEAD")
    return {"commit": commit or None, "dirty": bool(git("status", "--porcelain")) or not commit}


def settings(design: FollowupDesign, maps, n_boards: int) -> dict:
    return {
        "nonces": NONCES,
        "alias_block_seeds": ALIAS_BLOCK_SEEDS,
        "alias_second_block_rule": "sorted permutations outside block 1's rotation class; "
        "index default_rng(seed2).integers(len(available))",
        "base_permutations": base_permutations(),
        "mappings": maps,
        "text_epochs_max": design.text_epochs,
        "text_epochs_note": "configured maximum; training may stop early (see text_epochs_run)",
        "corpus_seed": CORPUS_SEED,
        "training_templates": corpus_templates(CORPUS_SEED),
        "query_templates": HELDOUT_TEMPLATES,
        "anchor_bank": ANCHOR_BANK,
        "init_stream": INIT_STREAM,
        "n_boards": n_boards,
        "arms": ARMS,
        "kinds": KINDS,
        "canonical_min": CANONICAL_MIN,
        "image_goal_min": design.gate_image_goal_min,
        "budget_s": BUDGET_S,
        "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
        "followup_design_digest": design.digest(),
        "vocab": vocab(),
        "planner_inputs": ["rendered current frame", "goal vector", "steps left"],
        "template_pool_size": len(TEMPLATE_POOL),
    }


# ---------------------------------------------------------------- runner
def run_study(
    out: Path,
    design: FollowupDesign | None = None,
    comp_factory: Callable | None = None,
    boards_factory: Callable | None = None,
    n_boards: int = N_BOARDS,
    budget_s: float = BUDGET_S,
    code: dict | None = None,
) -> dict:
    """Once-only run into a fresh ``out``. Writes plan.json (fixed selection) before fitting,
    rows.jsonl.gz and summary.json after. Status: VALID, INVALID, INCOMPLETE or FAILED."""
    design = design or FollowupDesign()
    comp_factory = comp_factory or real_components
    boards_factory = boards_factory or published_boards
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    maps = mappings()
    budget = engine.Budget(budget_s, "alias-study")
    summary = {
        "study": "alias-transfer",
        "status": "INCOMPLETE",
        "code": code if code is not None else git_state(),
        "settings": settings(design, maps, n_boards),
        "authorship": "Designed by Astra and Fable; implemented by Claude Opus 5.5. "
        "No personal prediction is attached to this study.",
    }
    _dump(out / "summary.json", summary)
    try:
        boards = boards_factory(design, n_boards)
        if len(boards) != n_boards:
            raise ValueError(f"expected {n_boards} boards, got {len(boards)}")
        plan = {"boards": [board_record(i, b) for i, b in boards], "mappings": maps}
        plan_blob = json.dumps(plan, sort_keys=True).encode()
        _dump(out / "plan.json", json.loads(plan_blob))
        summary["plan_sha256"] = hashlib.sha256(plan_blob).hexdigest()
        budget.check("boards")
        comp = comp_factory(design, budget, maps)
        summary["text_epochs_run"] = dict(comp.epochs_run)
        probes = fit_probes(design, comp, len(maps))
        raw_check = raw_alias_check(comp, probes)
        vecs = goal_vectors(comp, probes, maps)
        budget.check("probes")
        rows = evaluate(design, comp, vecs, maps, boards, budget)
        summary["raw_alias_vectors"] = raw_check
        summary["rows_sha256"] = write_rows(out / "rows.jsonl.gz", rows)
        summary |= summarize(rows, [i for i, _ in boards], len(maps), raw_check, design)
        budget.check("report")  # a cap hit after analysis still voids every contrast
    except engine.CapExceeded as exc:
        summary |= {
            "status": "INCOMPLETE",
            "incomplete_reason": str(exc),
            "primary": None,
            "secondary": None,
            "bootstrap": None,
        }
    except Exception as exc:
        summary |= {"status": "FAILED", "incomplete_reason": repr(exc)}
        raise
    finally:
        summary["elapsed_s"] = time.perf_counter() - budget.start
        _dump(out / "summary.json", summary)
    return summary


DERIVED = ("status", "coverage", "rates", "diagnostics", "primary", "secondary", "bootstrap")


ROW_FIELDS = frozenset(
    {
        "split", "board", "panel", "trim", "order", "template", "query", "target", "foil",
        "mapping", "arm", "kind", "word", "d_target", "d_foil", "margin", "outcome", "steps",
    }
)  # fmt: skip


ROW_STR_FIELDS = (
    "split", "board", "panel", "trim", "query", "target", "foil", "arm", "kind", "outcome",
)  # fmt: skip


def _roundtrip(obj):
    return json.loads(json.dumps(obj, allow_nan=False))


def row_errors(rows: list[dict], plan: dict, design: FollowupDesign) -> list[str]:
    """Per-row checks against the fixed layout: fields, enums, scene labels, expected word,
    target/foil glyphs, outcome and step bounds, and null-vs-number consistency."""
    maps = plan["mappings"]
    layout = {}
    base = [
        Board(frozenset(tuple(w) for w in b["walls"]), tuple(b["start"]),
              tuple(tuple(p) for p in b["pads"]), ("circle", "triangle"))
        for b in plan["boards"]
    ]  # fmt: skip
    for q, board in engine.queries(base, SPLIT):
        bid = plan["boards"][int(q.board_id.rsplit("/", 1)[1])]["id"]
        layout[(bid, q.key)] = {
            "split": SPLIT, "panel": q.panel, "trim": q.trim, "order": q.order,
            "template": q.template, "target": q.goal, "foil": board.other(q.goal),
        }  # fmt: skip
    errs: list[str] = []
    for i, r in enumerate(rows):
        bad = []
        if set(r) != ROW_FIELDS:
            errs.append(f"row {i}: fields")
            continue
        if (
            not all(type(r[k]) is str for k in ROW_STR_FIELDS)
            or not all(type(r[k]) is int for k in ("order", "template"))
            or not (r["mapping"] is None or type(r["mapping"]) is int)
            or not (r["word"] is None or type(r["word"]) is str)
        ):
            errs.append(f"row {i}: types")
            continue
        want = layout.get((r["board"], r["query"]))
        if want is None or any(r[k] != v for k, v in want.items()):
            bad.append("layout")
        if r["arm"] not in ROW_ARMS or r["kind"] not in ROW_KINDS:
            bad.append("arm/kind enum")
        elif (r["arm"] == "image") != (r["kind"] == "image"):
            bad.append("image arm/kind")
        elif r["arm"] == "image":
            if r["mapping"] is not None or r["word"] is not None:
                bad.append("image mapping/word")
        elif not (type(r["mapping"]) is int and 0 <= r["mapping"] < len(maps)):
            bad.append("mapping")
        elif want is not None:
            alias = {g: a for a, g in maps[r["mapping"]].items()}
            word = alias.get(r["target"]) if r["kind"] == "nonce" else r["target"]
            if r["word"] != word:
                bad.append("word")
        if r["outcome"] not in OUTCOMES:
            bad.append("outcome enum")
        st = r["steps"]
        if not (type(st) is int and 1 <= st <= design.horizon) or (
            r["outcome"] == "timeout" and st != design.horizon
        ):
            bad.append("steps")
        for k in ("d_target", "d_foil", "margin"):
            v = r[k]
            if v is not None and not (_number(v) and (k == "margin" or v >= 0)):
                bad.append(k)
        dists = _number(r["d_target"]) and _number(r["d_foil"])
        if (r["margin"] is None) == (dists and _number(r["d_foil"] - r["d_target"])):
            bad.append("margin availability")
        if bad:
            errs.append(f"row {i}: {', '.join(bad)}")
    return errs


def verify(
    out: Path,
    design: FollowupDesign | None = None,
    boards_factory: Callable | None = None,
    n_boards: int = N_BOARDS,
) -> list[str]:
    """Recompute every row-derived field and the digests; return a list of failures.

    Also checks the study id, status enum and settings against this code, the plan's mappings
    against the fixed design, the plan's board ids and layout against the regenerated fixed
    selection (``boards_factory`` defaults to the published bank; tests inject fixtures here
    only), and every row against that layout. The vector-level raw check is read from the
    summary (vectors are not stored); its row-level counterpart is recomputed."""
    design = design or FollowupDesign()
    boards_factory = boards_factory or published_boards
    out = Path(out)
    try:
        s = _load(out / "summary.json")
    except (OSError, ValueError) as exc:
        return [f"summary unreadable: {exc!r}"]
    if not isinstance(s, dict):
        return [f"summary is not a JSON object ({type(s).__name__})"]
    if not (out / "rows.jsonl.gz").exists() or not (out / "plan.json").exists():
        return [f"no rows or plan (status {s.get('status')}); nothing to recompute"]
    if s.get("incomplete_reason"):
        return [f"run stopped ({s.get('status')}): {s['incomplete_reason']}"]
    try:
        rows, digest = read_rows(out / "rows.jsonl.gz")
        plan = _load(out / "plan.json")
    except (OSError, ValueError) as exc:
        return [f"rows or plan unreadable: {exc!r}"]
    if not isinstance(plan, dict):
        return [f"plan is not a JSON object ({type(plan).__name__})"]
    bad_rows = [i for i, r in enumerate(rows) if not isinstance(r, dict)]
    if bad_rows:
        return [f"rows: {len(bad_rows)} not JSON objects, first {bad_rows[:3]}"]
    raw_check = s.get("raw_alias_vectors", {})
    if not isinstance(raw_check, dict):
        return [f"raw_alias_vectors is not a JSON object ({type(raw_check).__name__})"]
    flags = ("coef_zero", "goals_equal", "ok")
    if any(type(raw_check.get(k)) is not bool for k in flags) or raw_check["ok"] != (
        raw_check["coef_zero"] and raw_check["goals_equal"]
    ):
        return ["raw_alias_vectors: invalid or inconsistent flags"]
    fails = []
    if s.get("study") != "alias-transfer":
        fails.append("study")
    if s.get("status") not in STATUSES:
        fails.append("status enum")
    maps = mappings()
    if s.get("settings") != _roundtrip(settings(design, maps, n_boards)):
        fails.append("settings")
    if plan.get("mappings") != maps:
        fails.append("plan mappings")
    expected_boards = [board_record(i, b) for i, b in boards_factory(design, n_boards)]
    if plan.get("boards") != _roundtrip(expected_boards):
        fails.append("plan boards")
    if [b["id"] for b in expected_boards] != [f"{SPLIT}/{i}" for i in range(n_boards)]:
        fails.append("board ids")
    if fails:  # never recompute against a plan that is not the fixed design
        return fails
    if digest != s.get("rows_sha256"):
        fails.append("rows digest")
    blob = json.dumps(plan, sort_keys=True).encode()
    if hashlib.sha256(blob).hexdigest() != s.get("plan_sha256"):
        fails.append("plan digest")
    errs = row_errors(rows, plan, design)
    if errs:
        fails.append(f"rows: {len(errs)} invalid, first {errs[:3]}")
        return fails
    ids = [b["id"] for b in plan["boards"]]
    redo = summarize(rows, ids, len(maps), raw_check, design)
    fails += [k for k in DERIVED if _roundtrip(redo).get(k) != s.get(k)]
    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pgt.alias_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--allow-dirty", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        fails = verify(args.out)
        print("verify OK" if not fails else f"verify FAILED: {fails}")
        return 0 if not fails else 1
    if args.out.exists():
        print(f"refused: {args.out} exists; every run needs a fresh directory")
        return 2
    code = git_state()
    if code["dirty"] and not args.allow_dirty:
        print("refused: commit the code first (or pass --allow-dirty, which is recorded)")
        return 2
    s = run_study(args.out, code=code)
    print(json.dumps({"status": s["status"], "out": str(args.out)}))
    return 0 if s["status"] in ("VALID", "INVALID") else 1
