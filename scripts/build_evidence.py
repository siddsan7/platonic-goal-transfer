"""Build and check the compact case-study evidence in ``docs/evidence``.

build    reads the frozen run directories (raw, git-ignored ``episodes.jsonl``), checks every
         count against that run's ``metrics.json`` and ``manifest.json``, then writes the
         compact metrics/manifest copies, ``paired_counts.csv`` and ``replay.json``.
check    uses committed files only: recomputes every paired effect, per-seed rate and bootstrap
         interval from ``paired_counts.csv`` and replays the example episodes in the simulator.
figures  redraws ``effects.png`` and ``replay.png`` from committed files (needs matplotlib).

    python scripts/build_evidence.py build --runs runs/frozen
    python scripts/build_evidence.py check
    python scripts/build_evidence.py figures
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pgt import stats  # noqa: E402
from pgt.data import Config, caption, digest, generate_split, rng_for  # noqa: E402
from pgt.experiment import cluster_id  # noqa: E402
from pgt.world import TIMEOUT, Layout, render, step  # noqa: E402

OUT = ROOT / "docs" / "evidence"
RUNS = ("pilot-d2", "pilot-d1", "final-d2", "final-d1")
ARMS = ("clean", "goal_swap", "trim_swap")
CONTRASTS = (  # same order as pgt.experiment.run, so the bootstrap stream matches
    ("trim_swap_minus_goal_swap", "trim_swap", "goal_swap"),
    ("clean_minus_goal_swap", "clean", "goal_swap"),
    ("clean_minus_trim_swap", "clean", "trim_swap"),
)
FIELDS = (
    "split",
    "text_dim",
    "seed",
    "cluster",
    "arm",
    "episodes",
    "successes",
    "wrong_pad",
    "timeouts",
)
ABSOLUTE = re.compile(r"^([A-Za-z]:[\\/]|/|~)")


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def sanitize_command(command: list[str]) -> list[str]:
    """Strip machine-local absolute prefixes from path arguments; keep every parameter."""
    out = []
    for tok in command:
        if ABSOLUTE.match(tok):
            parts = Path(tok).parts
            tok = (
                "/".join(parts[parts.index("runs") :])
                if "runs" in parts
                else "<local>/" + parts[-1]
            )
        out.append(tok)
    return out


# ---------------------------------------------------------------- build
def build(runs_dir: Path) -> None:
    rows, text_records = [], {}
    for run in RUNS:
        src = runs_dir / run
        metrics, manifest = load_json(src / "metrics.json"), load_json(src / "manifest.json")
        if metrics["status"] != "complete" or not metrics["gates_passed"]:
            fail(f"{run}: not complete and healthy")
        split, dim = metrics["split"], metrics["text_dim"]
        counts = defaultdict(lambda: [0, 0, 0, 0])  # episodes, successes, wrong pad, timeouts
        cond = defaultdict(lambda: [0, 0, 0, 0])
        keyed = {}
        total = 0
        with (src / "episodes.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                total += 1
                wrong = r["outcome"] not in (r["goal"], TIMEOUT)
                tally = (1, int(r["success"]), int(wrong), int(r["outcome"] == TIMEOUT))
                c = cond[(r["seed"], r["condition"])]
                for i, v in enumerate(tally):
                    c[i] += v
                if r["condition"].startswith("text:"):
                    arm = r["condition"][5:]
                    k = counts[(r["seed"], r["cluster"], arm)]
                    for i, v in enumerate(tally):
                        k[i] += v
                    if run.startswith("final"):
                        keyed[
                            (r["seed"], r["layout"], r["trim"], r["goal"], r["template"], arm)
                        ] = r
                if run == "final-d1" and r["condition"] == "image_goal":
                    keyed[(r["seed"], r["layout"], r["trim"], r["goal"], None, "image_goal")] = r
        if total != manifest["episodes"]:
            fail(f"{run}: {total} episode lines, manifest says {manifest['episodes']}")
        for (seed, name), (n, s, w, t) in cond.items():
            ref = metrics["summary"][str(seed)][name]
            got = {"episodes": n, "success": s / n, "wrong_pad": w / n, "timeout": t / n}
            if any(not np.isclose(got[f], ref[f], rtol=0, atol=1e-12) for f in got):
                fail(f"{run} seed {seed} {name}: raw {got} differs from metrics.json")
        for (seed, cluster, arm), v in sorted(counts.items()):
            rows.append(dict(zip(FIELDS, (split, dim, seed, cluster, arm, *v), strict=True)))
        if run.startswith("final"):
            text_records[dim] = keyed

        manifest["command"] = sanitize_command(manifest["command"])
        write_json(OUT / "runs" / run / "metrics.json", metrics)
        write_json(OUT / "runs" / run / "manifest.json", manifest)
        print(f"{run}: {total} episodes match manifest and metrics.json per seed and condition")

    rows.sort(
        key=lambda r: (
            r["split"] != "test",
            -r["text_dim"],
            r["seed"],
            r["cluster"],
            ARMS.index(r["arm"]),
        )
    )
    with (OUT / "paired_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / "replay.json", replay(text_records, runs_dir))
    print(f"wrote {len(rows)} paired-count rows and replay.json")


def replay(text_records: dict, runs_dir: Path) -> dict:
    """First goal_swap/trim_swap divergence in episodes.jsonl line order, per width."""
    divergences, first = {}, {}
    for dim, keyed in text_records.items():
        keys = [k[:5] for k in keyed if k[5] == "goal_swap"]  # dicts keep file line order
        diff = [
            k
            for k in keys
            if keyed[k + ("goal_swap",)]["success"] != keyed[k + ("trim_swap",)]["success"]
        ]
        divergences[str(dim)] = {"paired_text_episodes": len(keys), "divergent": len(diff)}
        if diff:
            first[dim] = diff[0]
    if 1 not in first:
        fail("expected a width-1 divergence")
    seed, layout_ix, trim, goal, template = first[1]
    keyed = text_records[1]
    manifest = load_json(runs_dir / "final-d1" / "manifest.json")
    layouts = generate_split(Config(), "test", unlock_test=True)
    if digest([lay.to_json() for lay in layouts]) != manifest["eval_layout_digest"]:
        fail("regenerated test layouts do not match the manifest digest")
    keep = (
        "condition",
        "seed",
        "layout",
        "cluster",
        "trim",
        "goal",
        "template",
        "outcome",
        "success",
        "steps",
        "actions",
        "positions",
        "route_length",
    )
    episodes = [keyed[(seed, layout_ix, trim, goal, None, "image_goal")]] + [
        keyed[(seed, layout_ix, trim, goal, template, arm)] for arm in ARMS
    ]
    return {
        "source": "final-d1/episodes.jsonl (held-out test split, text width 1)",
        "ordering": "first (seed, layout, trim, goal, template) in episodes.jsonl line order "
        "where text:goal_swap and text:trim_swap differ in success",
        "divergences_by_text_dim": divergences,
        "caption": caption((goal, trim), template, heldout=True),
        "layout": layouts[layout_ix].to_json(),
        "episodes": [{k: e[k] for k in keep} for e in episodes],
    }


# ---------------------------------------------------------------- check
def read_counts() -> list[dict]:
    with (OUT / "paired_counts.csv").open(encoding="utf-8") as fh:
        return [
            {k: (v if k in ("split", "cluster", "arm") else int(v)) for k, v in r.items()}
            for r in csv.DictReader(fh)
        ]


def tables(rows: list[dict], split: str, dim: int):
    sel = [r for r in rows if r["split"] == split and r["text_dim"] == dim]
    seeds = sorted({r["seed"] for r in sel})
    clusters = sorted({r["cluster"] for r in sel})
    si, ci = {s: i for i, s in enumerate(seeds)}, {c: i for i, c in enumerate(clusters)}
    tab = {
        arm: (np.zeros((len(seeds), len(clusters))), np.zeros((len(seeds), len(clusters))))
        for arm in ARMS
    }
    for r in sel:
        wins, count = tab[r["arm"]]
        wins[si[r["seed"]], ci[r["cluster"]]] = r["successes"]
        count[si[r["seed"]], ci[r["cluster"]]] = r["episodes"]
    return seeds, clusters, tab


def check() -> None:
    rows = read_counts()
    for run in RUNS:
        metrics = load_json(OUT / "runs" / run / "metrics.json")
        manifest = load_json(OUT / "runs" / run / "manifest.json")
        if any(ABSOLUTE.match(tok) for tok in manifest["command"]):
            fail(f"{run}: absolute path left in manifest command")
        split, dim = metrics["split"], metrics["text_dim"]
        seeds, clusters, tab = tables(rows, split, dim)
        if seeds != metrics["seeds"]:
            fail(f"{run}: CSV seeds {seeds} differ from metrics {metrics['seeds']}")
        for arm in ARMS:
            wins, count = tab[arm]
            for i, s in enumerate(seeds):
                ref = metrics["summary"][str(s)][f"text:{arm}"]
                if (
                    count[i].sum() != ref["episodes"]
                    or wins[i].sum() / count[i].sum() != ref["success"]
                ):
                    fail(f"{run} seed {s} {arm}: CSV rate differs from metrics.json")
        rng = rng_for(Config(), "bootstrap", split)
        for name, a, b in CONTRASTS:
            got = stats.paired_effect(tab[a], tab[b], metrics["effects"][name]["bootstrap"], rng)
            if got != metrics["effects"][name]:
                fail(f"{run} {name}: CSV gives {got}, metrics.json {metrics['effects'][name]}")
        main = metrics["effects"]["trim_swap_minus_goal_swap"]
        print(
            f"{run}: {split}, width {dim}, {len(seeds)} seed(s) x {len(clusters)} clusters, "
            f"trim_swap - goal_swap = {main['effect_pp']:+.2f} pp, CI {main['ci_pp']} "
            "(reproduced from CSV)"
        )

    rep = load_json(OUT / "replay.json")
    lj = rep["layout"]
    layout = Layout(
        frozenset(tuple(w) for w in lj["walls"]),
        tuple(lj["start"]),
        (tuple(lj["pads"]["circle"]), tuple(lj["pads"]["triangle"])),
    )
    for e in rep["episodes"]:
        pos = layout.start
        if e["cluster"] != cluster_id(layout) or e["positions"][0] != list(pos):
            fail(f"replay {e['condition']}: layout does not match the record")
        for a, nxt in zip(e["actions"], e["positions"][1:], strict=True):
            pos = step(layout, pos, a)
            if list(pos) != nxt:
                fail(f"replay {e['condition']}: simulator disagrees with the logged path")
        outcome = layout.pad_at(pos) or TIMEOUT
        if outcome != e["outcome"] or (outcome == e["goal"]) != e["success"]:
            fail(f"replay {e['condition']}: outcome differs from the record")
    print(f"replay: {len(rep['episodes'])} logged episodes re-executed in the simulator")


# ---------------------------------------------------------------- figures
SERIES = {"clean": "#2a78d6", "goal_swap": "#eb6834", "trim_swap": "#1baf7a"}  # dataviz slots 1-3
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_counts()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, facecolor=SURFACE)
    for ax, dim in zip(axes, (2, 1), strict=True):
        metrics = load_json(OUT / "runs" / f"final-d{dim}" / "metrics.json")
        seeds, _, tab = tables(rows, "test", dim)
        for x, arm in enumerate(ARMS):
            wins, count = tab[arm]
            per_seed = 100 * wins.sum(1) / count.sum(1)
            ax.bar(x, per_seed.mean(), width=0.6, color=SERIES[arm], zorder=2)
            jitter = np.linspace(-0.16, 0.16, len(seeds))
            ax.scatter(x + jitter, per_seed, s=14, color=INK, zorder=3, linewidths=0)
            ax.text(
                x,
                per_seed.mean() + 2,
                f"{per_seed.mean():.0f}%",
                ha="center",
                color=INK,
                fontsize=9,
            )
        blind = 100 * np.mean([metrics["summary"][str(s)]["goal_blind"]["success"] for s in seeds])
        ax.axhline(blind, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(1.5, blind + 2, "goal-blind", ha="center", color=MUTED, fontsize=8)
        eff = metrics["effects"]["trim_swap_minus_goal_swap"]
        ax.set_title(
            f"Text width {dim}: trim_swap - goal_swap = {eff['effect_pp']:+.0f} pp",
            fontsize=10,
            color=INK,
        )
        ax.set_xticks(range(3), ARMS)
        ax.set_ylim(0, 112)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("executed goal success (%)", color=MUTED)
    fig.suptitle(
        "Held-out test split, 5 model seeds, 4,096 text episodes per arm and seed (dots = seeds)",
        fontsize=9,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(OUT / "effects.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    rep = load_json(OUT / "replay.json")
    lj = rep["layout"]
    layout = Layout(
        frozenset(tuple(w) for w in lj["walls"]),
        tuple(lj["start"]),
        (tuple(lj["pads"]["circle"]), tuple(lj["pads"]["triangle"])),
    )
    shown = [e for e in rep["episodes"] if e["condition"] in ("text:goal_swap", "text:trim_swap")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.9), facecolor=SURFACE)
    for ax, e in zip(axes, shown, strict=True):
        ax.imshow(render(layout, layout.start, e["trim"]), interpolation="nearest")
        path = np.array(e["positions"]) * 5 + 2
        arm = e["condition"][5:]
        ax.plot(path[:, 1], path[:, 0], color=SERIES[arm], lw=2.5, marker="o", ms=4)
        verdict = "success" if e["success"] else "wrong pad"
        ax.set_title(f"{arm}: entered {e['outcome']} ({verdict})", fontsize=9, color=INK)
        ax.set_axis_off()
    fig.suptitle(
        f"Seed {shown[0]['seed']}, test layout {shown[0]['layout']}, goal "
        f"{shown[0]['goal']}, caption: “{rep['caption']}”",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(OUT / "replay.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("wrote effects.png and replay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", type=Path, required=True, help="dir with the four frozen runs")
    sub.add_parser("check")
    sub.add_parser("figures")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.runs)
        check()
    elif args.cmd == "check":
        check()
    else:
        figures()


if __name__ == "__main__":
    main()
