"""Reproducible data-integrity check and manifest.

    python -m pgt.integrity                      # run every check, print manifest
    python -m pgt.integrity --check manifests/default.json
    python -m pgt.integrity --write manifests/default.json

The test split is generated here only to verify structure and pin its digest; no model
or tuning code may unlock it before the protocol is frozen.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from pgt.data import (
    ARMS,
    COMBOS,
    HELDOUT_TEMPLATES,
    SPLITS,
    TRAIN_TEMPLATES,
    Config,
    IntegrityError,
    calibration_anchors,
    caption,
    config_json,
    digest,
    generate_split,
    make_pairs,
    route_report,
    split_of,
    swap_quartets,
)
from pgt.world import GLYPHS, TRIMS, Layout, canonical_topology, d4, oracle_path, run_episode

FACTOR_WORDS = set(GLYPHS) | set(TRIMS)
FORBIDDEN = re.compile(
    r"\d|\b(up|down|left|right|north|south|east|west|row|column|col|step|steps|"
    r"action|move|turn|layout|id|grid|cell)\b"
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def check_split(cfg: Config, split: str, layouts: list[Layout]) -> None:
    check(len(layouts) == cfg.n_quartets(split), f"{split}: wrong quartet count")
    keys = [canonical_topology(lay.walls) for lay in layouts]
    check(len(set(keys)) == len(layouts) // 2, f"{split}: topology reused across twin pairs")
    circle_pos, triangle_pos = Counter(), Counter()
    for i, lay in enumerate(layouts):
        check(split_of(lay.walls, cfg) == split, f"{split}: layout {i} hashes to another split")
        check(
            all(canonical_topology({d4(w, k) for w in lay.walls}) == keys[i] for k in range(8)),
            f"{split}: canonical key not D4-invariant",
        )
        check(lay.start not in lay.pads and lay.start not in lay.walls, f"{split}: bad start")
        if i % 2:
            twin = layouts[i - 1]
            check(
                lay.walls == twin.walls and lay.start == twin.start and lay.pads == twin.pads[::-1],
                f"{split}: twin {i} is not a label swap",
            )
        circle_pos[lay.pad("circle")] += 1
        triangle_pos[lay.pad("triangle")] += 1
        for glyph in GLYPHS:  # both goals certified and the oracle actually succeeds
            path = oracle_path(lay, glyph)
            check(
                path is not None and cfg.min_route <= len(path) <= cfg.horizon,
                f"{split}: layout {i} goal {glyph} not reachable in horizon",
            )
            check(
                run_episode(lay, path, cfg.horizon) == (glyph, len(path)),
                f"{split}: oracle failed layout {i} goal {glyph}",
            )
    check(circle_pos == triangle_pos, f"{split}: pad positions depend on glyph label")


def check_disjoint(all_layouts: dict[str, list[Layout]]) -> None:
    owners: dict[tuple, str] = {}
    for split, layouts in all_layouts.items():
        for lay in layouts:
            key = canonical_topology(lay.walls)
            check(owners.setdefault(key, split) == split, f"topology shared by {split}")


def check_captions() -> None:
    for heldout, templates in ((False, TRAIN_TEMPLATES), (True, HELDOUT_TEMPLATES)):
        for t in range(len(templates)):
            for combo in COMBOS:
                text = caption(combo, t, heldout)
                words = text.split()
                check(not FORBIDDEN.search(text), f"forbidden token in caption: {text!r}")
                check(
                    sum(w in GLYPHS for w in words) == 1 and sum(w in TRIMS for w in words) == 1,
                    f"caption must name exactly one glyph and one trim: {text!r}",
                )
    train_frames = {t.replace("{glyph}", "").replace("{trim}", "") for t in TRAIN_TEMPLATES}
    held_frames = {t.replace("{glyph}", "").replace("{trim}", "") for t in HELDOUT_TEMPLATES}
    check(not (train_frames & held_frames), "held-out template repeats a training template")


def check_arms(cfg: Config) -> dict:
    n = cfg.n_train
    selected = swap_quartets(cfg, n)
    check(len(selected) * 4 == n, "swap set is not exactly 25% of quartets")
    arms = {arm: make_pairs(cfg, n, arm) for arm in ARMS}
    clean = arms["clean"]
    check(all(p.image == p.text_combo for p in clean), "clean arm has a mismatched pair")
    sources = Counter((p.quartet, p.image, p.template) for p in clean)
    texts = Counter(p.text for p in clean)
    words = Counter(w for p in clean for w in p.text.split())
    for arm in ("goal_swap", "trim_swap"):
        pairs = arms[arm]
        check(
            Counter((p.quartet, p.image, p.template) for p in pairs) == sources,
            f"{arm}: source multiset differs from clean",
        )
        check(Counter(p.text for p in pairs) == texts, f"{arm}: caption multiset differs")
        check(
            Counter(w for p in pairs for w in p.text.split()) == words,
            f"{arm}: word frequencies differ",
        )
        swapped = [p for p in pairs if p.image != p.text_combo]
        check(sorted({p.quartet for p in swapped}) == selected, f"{arm}: wrong swapped quartets")
        check(len(swapped) * 4 == len(pairs), f"{arm}: swapped pairs are not exactly 25%")
        factor = 0 if arm == "goal_swap" else 1
        for p in swapped:
            diff = [i for i in range(2) if p.image[i] != p.text_combo[i]]
            check(diff == [factor], f"{arm}: swap flips the wrong factor")
            a, b = caption(p.image, p.template).split(), p.text.split()
            check(
                len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) == 1,
                f"{arm}: swap is not a single token edit",
            )
        per_quartet = Counter((p.quartet, p.template, p.text_combo) for p in pairs)
        check(all(v == 1 for v in per_quartet.values()), f"{arm}: swap is not a bijection")
    return {
        "selected_quartets": len(selected),
        "selected_digest": digest(selected),
        **{
            arm: digest([[p.quartet, p.image, p.template, p.text] for p in arms[arm]])
            for arm in ARMS
        },
    }


def check_anchors(cfg: Config, layouts: list[Layout]) -> dict:
    anchors = calibration_anchors(cfg, layouts)
    check(len(anchors) == cfg.n_anchors, "wrong anchor count")
    check(
        len({canonical_topology(layouts[a.quartet].walls) for a in anchors}) == len(anchors),
        "anchors share a calibration layout",
    )
    check(all(a.image == a.text_combo for a in anchors), "anchors must be clean pairs")
    goals = Counter(a.image[0] for a in anchors)
    cells = Counter((a.image, a.template) for a in anchors)
    check(len(set(goals.values())) == 1 and len(set(cells.values())) == 1, "anchors unbalanced")
    return {
        "count": len(anchors),
        "digest": digest([[a.quartet, a.image, a.template] for a in anchors]),
    }


def build_manifest(cfg: Config) -> dict:
    layouts = {s: generate_split(cfg, s, unlock_test=True) for s in SPLITS}
    for split, lays in layouts.items():
        check_split(cfg, split, lays)
    check_disjoint(layouts)
    check_captions()
    return {
        "config": config_json(cfg),
        "splits": {
            s: {
                "quartets": len(lays),
                "topologies": len(lays) // 2,
                "digest": digest([lay.to_json() for lay in lays]),
                "route_lengths": route_report(lays),
            }
            for s, lays in layouts.items()
        },
        "train_pairs": check_arms(cfg),
        "calibration_anchors": check_anchors(cfg, layouts["calibration"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", type=Path, help="write the manifest to this path")
    group.add_argument("--check", type=Path, help="fail unless the manifest equals this file")
    args = parser.parse_args(argv)
    text = json.dumps(build_manifest(Config()), indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    elif args.check:
        if args.check.read_text(encoding="utf-8") != text:
            print(f"manifest mismatch: {args.check}", file=sys.stderr)
            return 1
        print(f"integrity ok, manifest matches {args.check}")
        return 0
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
