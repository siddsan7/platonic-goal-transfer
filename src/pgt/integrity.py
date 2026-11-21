"""Reproducible data-integrity check and manifest.

    python -m pgt.integrity                      # run every check, print manifest
    python -m pgt.integrity --check manifests/default.json
    python -m pgt.integrity --write manifests/default.json

The test split is generated here only to verify structure and pin its digest; no model
or tuning code may unlock it before the config is frozen and the pilot is done.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    CalibrationAnchor,
    Config,
    IntegrityError,
    TextPair,
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
IMAGE_FIELDS = {"image", "layout", "pixels", "render"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def factors(text: str) -> tuple[str, str]:
    """The (glyph, trim) a caption actually names; it must name exactly one of each."""
    words = text.split()
    glyphs = [w for w in words if w in GLYPHS]
    trims = [w for w in words if w in TRIMS]
    check(
        len(glyphs) == 1 and len(trims) == 1, f"caption must name one glyph and one trim: {text!r}"
    )
    return glyphs[0], trims[0]


def frame(template: str) -> str:
    return template.replace("{glyph}", "").replace("{trim}", "")


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


def check_captions() -> dict:
    """Factor identity, distinct paraphrases, and train/held-out template separation."""
    texts: dict[bool, set[str]] = {False: set(), True: set()}
    for heldout, templates in ((False, TRAIN_TEMPLATES), (True, HELDOUT_TEMPLATES)):
        for t in range(len(templates)):
            for combo in COMBOS:
                text = caption(combo, t, heldout)
                check(not FORBIDDEN.search(text), f"forbidden token in caption: {text!r}")
                check(factors(text) == combo, f"caption names the wrong factors: {text!r}")
                texts[heldout].add(text)
        check(len(set(map(frame, templates))) == len(templates), "templates are not distinct")
        check(len(texts[heldout]) == len(templates) * len(COMBOS), "captions are not distinct")
    check(not texts[False] & texts[True], "a held-out caption equals a training caption")
    check(
        not set(map(frame, TRAIN_TEMPLATES)) & set(map(frame, HELDOUT_TEMPLATES)),
        "held-out template repeats a training template",
    )
    return {"train_captions": len(texts[False]), "heldout_captions": len(texts[True])}


def check_arms(cfg: Config) -> dict:
    n = cfg.n_train
    selected = swap_quartets(cfg, n)
    check(len(selected) * 4 == n, "swap set is not exactly 25% of quartets")
    fields = {f.name for f in dataclasses.fields(TextPair)}
    check(not fields & IMAGE_FIELDS, "text-training pairs must not carry images or layouts")
    arms = {arm: make_pairs(cfg, n, arm) for arm in ARMS}
    for pairs in arms.values():
        for p in pairs:
            check(p.left_template != p.right_template, "pair sides must use distinct templates")
            check(p.left != p.right, "pair sides must be distinct paraphrases")
            check(factors(p.left) == p.left_combo, f"left caption factors wrong: {p.left!r}")
            check(factors(p.right) == p.right_combo, f"right caption factors wrong: {p.right!r}")
    clean = arms["clean"]
    check(all(p.left_combo == p.right_combo for p in clean), "clean arm has a mismatched pair")
    per_quartet = Counter((p.quartet, p.left_template, p.left_combo) for p in clean)
    check(
        len(per_quartet) == n * len(TRAIN_TEMPLATES) * len(COMBOS)
        and set(per_quartet.values()) == {1},
        "clean quartets are not balanced over combos and directions",
    )
    lefts = Counter((p.quartet, p.left) for p in clean)
    rights = Counter(p.right for p in clean)
    words = Counter(w for p in clean for w in f"{p.left} {p.right}".split())
    for arm in ("goal_swap", "trim_swap"):
        pairs = arms[arm]
        check(Counter((p.quartet, p.left) for p in pairs) == lefts, f"{arm}: left side differs")
        check(Counter(p.right for p in pairs) == rights, f"{arm}: right caption multiset differs")
        check(
            Counter(w for p in pairs for w in f"{p.left} {p.right}".split()) == words,
            f"{arm}: word frequencies differ",
        )
        swapped = [p for p in pairs if factors(p.left) != factors(p.right)]
        check(sorted({p.quartet for p in swapped}) == selected, f"{arm}: wrong swapped quartets")
        check(len(swapped) * 4 == len(pairs), f"{arm}: swapped pairs are not exactly 25%")
        factor = 0 if arm == "goal_swap" else 1
        for p in swapped:
            left, right = factors(p.left), factors(p.right)
            diff = [i for i in range(2) if left[i] != right[i]]
            check(diff == [factor], f"{arm}: rewiring flips the wrong factor")
            a, b = caption(p.left_combo, p.right_template).split(), p.right.split()
            check(
                len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) == 1,
                f"{arm}: rewired right text is not a one-token edit of the clean right text",
            )
        rewired = Counter((p.quartet, p.right_template, p.right_combo) for p in pairs)
        check(set(rewired.values()) == {1}, f"{arm}: rewiring is not a bijection")
    return {
        "selected_quartets": len(selected),
        "selected_digest": digest(selected),
        "pairs_per_arm": len(clean),
        # Finite support: repeated pairs are weights, not independent text observations.
        "unique_pairs": {arm: len({(p.left, p.right) for p in arms[arm]}) for arm in ARMS},
        **{arm: digest([[p.quartet, p.left, p.right] for p in arms[arm]]) for arm in ARMS},
    }


def check_anchors(cfg: Config, layouts: list[Layout]) -> dict:
    anchors = calibration_anchors(cfg, layouts)
    check(all(isinstance(a, CalibrationAnchor) for a in anchors), "anchors must be explicit")
    check(len(anchors) == cfg.n_anchors, "wrong anchor count")
    check(
        len({canonical_topology(layouts[a.layout].walls) for a in anchors}) == len(anchors),
        "anchors share a calibration topology",
    )
    check(all(factors(a.text) == a.combo for a in anchors), "anchors must be clean pairs")
    check(all(a.image(layouts).shape == (35, 35, 3) for a in anchors), "anchor image shape")
    goals = Counter(a.combo[0] for a in anchors)
    cells = Counter((a.combo, a.template) for a in anchors)
    check(
        len(goals) == len(GLYPHS)
        and len(set(goals.values())) == 1
        and len(cells) == len(COMBOS) * len(TRAIN_TEMPLATES)
        and len(set(cells.values())) == 1,
        "anchors unbalanced",
    )
    return {
        "count": len(anchors),
        "digest": digest([[a.layout, a.combo, a.template] for a in anchors]),
    }


def build_manifest(cfg: Config) -> dict:
    layouts = {s: generate_split(cfg, s, unlock_test=True) for s in SPLITS}
    for split, lays in layouts.items():
        check_split(cfg, split, lays)
    check_disjoint(layouts)
    return {
        "config": config_json(cfg),
        "captions": check_captions(),
        "splits": {
            s: {
                "quartets": len(lays),
                "topologies": len(lays) // 2,
                "digest": digest([lay.to_json() for lay in lays]),
                "route_lengths": route_report(lays),
            }
            for s, lays in layouts.items()
        },
        "train_text_pairs": check_arms(cfg),
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
