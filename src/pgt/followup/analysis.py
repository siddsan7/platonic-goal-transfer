"""Pre-declared metrics, health gate and counts digest, computed only from supplied rows.

Row schema (one executed episode): corpus, bank, condition, panel ("seen"/"withheld"),
board, query, outcome ("correct"/"wrong_pad"/"timeout"). ``condition`` is a text arm
(clean/goal_swap/trim_swap) or a reference (raw/keyword/image/blind/oracle). References that do
not depend on corpus or anchor bank are stored once with corpus = bank = None (one shared row).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np

from pgt.followup.corpus import HELDOUT_TEMPLATES
from pgt.followup.design import ARMS, PRIMARY, TRIMS4, FollowupDesign, panel_glyphs

OUTCOMES = ("correct", "wrong_pad", "timeout")
STATUS = ("PASS", "INCONCLUSIVE")


def _key(r: dict) -> tuple:
    return (r["panel"], r["board"], r["query"])


def rate(rows: list[dict], **match) -> dict:
    sel = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    n = len(sel)
    counts = {o: sum(r["outcome"] == o for r in sel) for o in OUTCOMES}
    return {"n": n, **counts, "success": counts["correct"] / n if n else None}


def paired_pp(rows: list[dict], a: str, b: str, **match) -> dict:
    """Success(a) - success(b) in pp over episodes present in BOTH conditions; refuses
    unmatched episode sets instead of silently dropping them."""
    ka = {_key(r): r for r in rows if r["condition"] == a and _match(r, match)}
    kb = {_key(r): r for r in rows if r["condition"] == b and _match(r, match)}
    if set(ka) != set(kb) or not ka:
        raise ValueError(f"{a} vs {b}: episode sets differ or are empty")
    diff = sum((ka[k]["outcome"] == "correct") - (kb[k]["outcome"] == "correct") for k in ka)
    return {"n": len(ka), "pp": 100 * diff / len(ka)}


def _match(r: dict, match: dict) -> bool:
    return all(r.get(k) == v for k, v in match.items())


def primary(rows: list[dict], design: FollowupDesign) -> dict:
    """Mean over the corpus x anchor-bank grid (equal weights) of the paired trim_swap -
    goal_swap success difference on withheld combos. Cells share one visual fit and the anchor
    banks are crossed, so the cells are conditional, not independent replications."""
    cells = {}
    for c in design.corpus_seeds:
        for b in design.anchor_banks:
            cells[f"{c}/{b}"] = paired_pp(
                rows, "trim_swap", "goal_swap", corpus=c, bank=b, panel="withheld"
            )
    by_corpus = {
        c: sum(cells[f"{c}/{b}"]["pp"] for b in design.anchor_banks) / len(design.anchor_banks)
        for c in design.corpus_seeds
    }
    mean = sum(x["pp"] for x in cells.values()) / len(cells)
    return {
        "mean_pp": mean,
        "cells": cells,
        "by_corpus": by_corpus,
        "conditional_on": "one visual fit",
    }


def goal_discrimination(scores: list[dict]) -> dict:
    """Within-trim check: score(correct goal image) vs score(foil goal image). Ties are their
    own count, never split or broken."""
    wins = sum(s["correct"] > s["foil"] for s in scores)
    ties = sum(s["correct"] == s["foil"] for s in scores)
    return {"n": len(scores), "wins": wins, "ties": ties, "losses": len(scores) - wins - ties}


def health_gate(rows: list[dict], design: FollowupDesign) -> dict:
    """Pre-declared checks. Failing any one marks the study INCONCLUSIVE (no reruns)."""
    oracle = rate(rows, condition="oracle")["success"]
    blind = rate(rows, condition="blind")["success"]
    image = rate(rows, condition="image")["success"]
    seen_clean = rate(rows, condition="clean", panel="seen")["success"]
    checks = {
        "oracle": oracle is not None and oracle >= design.gate_oracle_min,
        "blind": blind is not None and blind <= design.gate_blind_max,
        "image_goal": image is not None and image >= design.gate_image_goal_min,
        "seen_clean": seen_clean is not None and seen_clean >= design.gate_seen_clean_min,
    }
    values = {"oracle": oracle, "blind": blind, "image_goal": image, "seen_clean": seen_clean}
    return {
        "status": "PASS" if all(checks.values()) else "INCONCLUSIVE",
        "checks": checks,
        "values": values,
    }


def count_table(rows: list[dict]) -> list[dict]:
    """Machine-readable counts per (condition, corpus, bank, panel, outcome)."""
    acc: dict[tuple, int] = defaultdict(int)
    for r in rows:
        acc[(r["condition"], r["corpus"], r["bank"], r["panel"], r["outcome"])] += 1
    keys = ("condition", "corpus", "bank", "panel", "outcome")
    return [dict(zip(keys, k, strict=True)) | {"count": v} for k, v in sorted(acc.items(), key=str)]


def counts_digest(rows: list[dict]) -> str:
    blob = json.dumps(count_table(rows), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- strict coverage
def expected_cells(design: FollowupDesign) -> set[tuple]:
    """(condition, corpus, bank) identities the design defines (47 with the default design)."""
    cells = {("image", None, None), ("oracle", None, None)}
    for b in design.anchor_banks:
        cells |= {(c, None, b) for c in ("keyword", "raw", "blind")}
        cells |= {(a, c, b) for c in design.corpus_seeds for a in ARMS}
    return cells


def expected_queries() -> set[tuple]:
    """(panel, query key) per board; must mirror ``engine.queries`` exactly."""
    out = set()
    for trim in TRIMS4:
        for panel, seen in (("seen", True), ("withheld", False)):
            for order in (0, 1):
                for t in range(len(HELDOUT_TEMPLATES)):
                    out |= {(panel, f"{trim}|o{order}|t{t}|{g}") for g in panel_glyphs(trim, seen)}
    return out


def strict_coverage(rows: list[dict], design: FollowupDesign, split: str, n_boards: int) -> dict:
    """Exact expected keys: no duplicates, missing or extra rows, bad outcomes or splits."""
    boards = [f"{split}/{i}" for i in range(n_boards)]
    want = {
        (cell, p, b, q)
        for cell in expected_cells(design)
        for b in boards
        for p, q in expected_queries()
    }
    errors, got = [], set()
    for r in rows:
        k = (
            (r.get("condition"), r.get("corpus"), r.get("bank")),
            r.get("panel"),
            r.get("board"),
            r.get("query"),
        )
        if k in got:
            errors.append(f"duplicate {k}")
        got.add(k)
        if r.get("outcome") not in OUTCOMES:
            errors.append(f"bad outcome {k}")
        if r.get("split") != split:
            errors.append(f"wrong split {k}")
    errors += [f"missing {k}" for k in sorted(want - got, key=str)[:20]]
    errors += [f"unexpected {k}" for k in sorted(got - want, key=str)[:20]]
    return {
        "ok": not errors,
        "rows": len(rows),
        "expected": len(want),
        "cells": len(expected_cells(design)),
        "errors": errors[:50],
    }


def rows_digest(rows: list[dict]) -> str:
    """Per-query outcome digest, independent of row order."""
    blob = json.dumps(sorted(json.dumps(r, sort_keys=True) for r in rows)).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- secondary reporting
def success_levels(rows: list[dict]) -> list[dict]:
    """Success and outcome counts per (condition, corpus, bank, panel), incl. raw/references."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["corpus"], r["bank"], r["panel"])].append(r)
    keys = ("condition", "corpus", "bank", "panel")
    return [dict(zip(keys, k, strict=True)) | rate(v) for k, v in sorted(groups.items(), key=str)]


def swap_minus_clean(rows: list[dict], design: FollowupDesign) -> dict:
    """Paired swap - clean success (pp) per corpus/bank cell and panel, for both swap arms."""
    return {
        f"{arm}/{c}/{b}/{p}": paired_pp(rows, arm, "clean", corpus=c, bank=b, panel=p)
        for arm in ("goal_swap", "trim_swap")
        for c in design.corpus_seeds
        for b in design.anchor_banks
        for p in ("seen", "withheld")
    }


def discrimination_by_cell(scores: list[dict]) -> dict:
    """Within-trim correct-vs-foil goal-image discrimination per (condition, corpus, bank,
    panel) for every learned arm and the raw/keyword references."""
    groups: dict[str, list] = defaultdict(list)
    for s in scores:
        groups[f"{s['condition']}/{s['corpus']}/{s['bank']}/{s['panel']}"].append(s)
    return {k: goal_discrimination(v) for k, v in sorted(groups.items())}


def topology_bootstrap(
    rows: list[dict], design: FollowupDesign, n: int = 2000, seed: int = 0
) -> dict:
    """Percentile 95% interval for the primary mean, resampling topologies (boards) with
    replacement. Conditional on the fixed corpora, anchor banks and the one visual fit."""
    diff: dict[str, list] = defaultdict(list)
    by = {
        (r["condition"], r["corpus"], r["bank"], r["board"], r["query"]): r["outcome"]
        for r in rows
        if r["panel"] == "withheld" and r["condition"] in PRIMARY
    }
    for (cond, c, b, board, q), o in by.items():
        if cond == PRIMARY[0]:
            other = by[(PRIMARY[1], c, b, board, q)]
            diff[board].append(100 * ((o == "correct") - (other == "correct")))
    boards = sorted(diff)
    per = np.array([np.mean(diff[b]) for b in boards])
    rng = np.random.default_rng([design.data_seed, seed, 2000])
    idx = rng.integers(0, len(per), size=(n, len(per)))
    means = per[idx].mean(1)
    return {
        "n_resamples": n,
        "n_topologies": len(per),
        "mean_pp": float(per.mean()),
        "ci95_pp": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "unit": "topology",
        "conditional_on": "fixed corpora, anchor banks, visual fit",
    }


def summarize(rows, disc, design: FollowupDesign, split: str, n_boards: int) -> dict:
    cov = strict_coverage(rows, design, split, n_boards)
    out = {
        "coverage": cov,
        "counts": count_table(rows),
        "counts_digest": counts_digest(rows),
        "rows_digest": rows_digest(rows),
        "health": health_gate(rows, design),
        "discrimination": discrimination_by_cell(disc),
        "levels": success_levels(rows),
    }
    if cov["ok"]:
        out |= {
            "primary": primary(rows, design),
            "bootstrap": topology_bootstrap(rows, design),
            "swap_minus_clean": swap_minus_clean(rows, design),
        }
    return out
