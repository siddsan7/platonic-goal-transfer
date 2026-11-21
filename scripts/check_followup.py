"""Check the committed follow-up evidence against its manifests and freeze record.

Reads only docs/evidence/followup/{freeze.json, pilot/, test/} and recomputes every
row-derived number with the frozen routines in ``pgt.followup.analysis``. Goal-image
discrimination is computed from scores that are not stored in the rows, so it can only be
read from the manifest; this script says so instead of claiming to check it.

    python scripts/check_followup.py          (prefix PYTHONPATH=src if not installed)
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from pgt.followup import analysis as A
from pgt.followup.design import FollowupDesign

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence" / "followup"
N_BOARDS = {"pilot": 16, "test": 48}
SPLIT = {"pilot": "validation", "test": "test"}
failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


def learned_minus_raw_paired(rows: list[dict], arm: str, panel: str) -> dict:
    """Episode-paired learned-arm minus raw-count success, pooled over the 12 cells."""
    raw = {
        (r["bank"], r["board"], r["query"]): r["outcome"] == "correct"
        for r in rows
        if r["condition"] == "raw" and r["panel"] == panel
    }
    sel = [r for r in rows if r["condition"] == arm and r["panel"] == panel]
    diff = sum((r["outcome"] == "correct") - raw[(r["bank"], r["board"], r["query"])] for r in sel)
    return {"n": len(sel), "pp": 100 * diff / len(sel)}


def phase(name: str, design: FollowupDesign, freeze: dict) -> None:
    d = EVID / name
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    with gzip.open(d / "rows.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    print(f"\n{name}: {len(rows):,} rows")
    check("manifest phase, status PASS", man["phase"] == name and man["status"] == "PASS")
    check("manifest freeze == freeze.json", man["freeze"] == freeze)
    check("design digest == frozen design", man["design_digest"] == design.digest())
    s = A.summarize(rows, [], design, SPLIT[name], N_BOARDS[name])
    check(f"strict coverage ({s['coverage']['rows']:,} expected rows)", s["coverage"]["ok"])
    for key in (
        "counts",
        "counts_digest",
        "rows_digest",
        "health",
        "levels",
        "learned_minus_raw_pp",
        "primary",
        "bootstrap",
        "swap_minus_clean",
    ):
        check(f"{key} recomputed from rows == manifest", json.loads(json.dumps(s[key])) == man[key])
    p, b = s["primary"], s["bootstrap"]
    print(
        f"  primary trim_swap - goal_swap (withheld): {p['mean_pp']:+.2f} pp; "
        f"cells {sorted({c['pp'] for c in p['cells'].values()})}; "
        f"topology bootstrap ({b['n_topologies']} boards) 95% CI {b['ci95_pp']}"
    )
    for cond in ("clean", "goal_swap", "trim_swap", "raw", "keyword", "image", "blind", "oracle"):
        for panel in ("seen", "withheld"):
            r = A.rate(rows, condition=cond, panel=panel)
            print(
                f"  {cond:9s} {panel:8s} {r['correct']:>6,}/{r['n']:<6,} "
                f"wrong_pad {r['wrong_pad']:>5,} timeout {r['timeout']}"
            )
    for arm in ("clean", "goal_swap", "trim_swap"):
        e = learned_minus_raw_paired(rows, arm, "withheld")
        print(f"  paired {arm} - raw (withheld): {e['pp']:+.2f} pp over {e['n']:,} episodes")
    disc = man["discrimination"]
    losses = sum(v["losses"] + v["ties"] for v in disc.values())
    print(
        f"  manifest-only (not recomputable from rows): goal discrimination, "
        f"{len(disc)} cells, ties+losses {losses}; elapsed_s {man['elapsed_s']:.1f}"
    )


def main() -> int:
    design = FollowupDesign()
    freeze = json.loads((EVID / "freeze.json").read_text(encoding="utf-8"))
    print("freeze.json")
    check("design digest matches code", freeze["design_digest"] == design.digest())
    check("prediction commit 8c1cb3d", freeze["prediction"]["commit"].startswith("8c1cb3d"))
    for name in ("pilot", "test"):
        phase(name, design, freeze)
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
