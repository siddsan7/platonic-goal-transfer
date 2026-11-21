# Locked protocol

Status: milestone 1 (world, splits, matched pairs) implemented and locked. Milestone 2 (models,
planner, evaluation) is planned below and not implemented. Any change to a locked item must
update `manifests/default.json` and say why.

## Question

With an identical text pool and a fixed learned image-only planner, do text–image training pairs
that change the **intended destination** (`goal_swap`) hurt cross-modal executed first-pad success
more than pairs that change only the **cosmetic floor trim** (`trim_swap`)?

This is a small controlled extension, not a proof of any theorem and not the first alignment
study. A null result is a valid outcome. Context:

- Balestriero & LeCun, <https://arxiv.org/html/2610.39302v1>: the theorem, its existing sweep over
  pair quality q, and the captioning adapter. We vary *which factor* a corrupted pair corrupts
  at a fixed corruption rate, rather than the rate itself.
- Li et al., <https://arxiv.org/html/2608.23720v2>: stitching a visual predictor to other
  modalities and planning in its latent space. We reuse that pattern at toy scale.

## World (`src/pgt/world.py`)

- 7×7 grid. Cell types: floor, wall, start, circle pad, triangle pad. Four actions
  (up, down, left, right); moving into a wall or the border leaves the agent in place.
- **Terminal = first pad entered, either glyph.** Entering the wrong pad ends the episode as a
  failure. Horizon H = 12 steps; reaching it without a pad is a timeout (a failure, counted
  separately).
- Rendering: 35×35 RGB, 5 px per cell. Pads share one colour and differ only in glyph shape.
  The floor carries a trim, `striped` or `dotted`, drawn on plain floor cells only. Trim is
  visible in every image and never enters the dynamics.
- Observation = render with the agent marker at the start. Goal image = the same map and trim
  with the agent marker on the intended pad; nothing else differs (tested).

## Layouts and splits (`src/pgt/data.py`)

- Per candidate: 6–12 wall cells, then start and two pad cells, drawn uniformly without
  replacement. The two pad cells are an unordered pair; **both glyph assignments are kept** as
  a twin pair of layouts, so glyph is independent of pad position and circle/triangle position
  counts match exactly in every split.
- Split by canonical wall topology: the wall set is reduced to the smallest of its 8 D4
  (rotation/reflection) images, hashed with a fixed salt, and assigned train/calibration/
  validation/test at 0.7/0.1/0.1/0.1. All orientations of a topology share a split, and no
  topology appears in two splits or in two twin pairs within a split. Splits do not depend on
  the run seed.
- Certification: a layout is kept only if, for **each** goal, BFS from the start reaches that pad
  in 2..H steps while treating the opposite pad as an impassable barrier (it would end the
  episode). Both twins of an evaluation episode are therefore reachable within the horizon.
- Route lengths and detours (route longer than Manhattan distance) are reported per split in
  the manifest; results in milestone 2 are also broken down by route length.
- Sizes (quartets): train 2048, calibration 256, validation 512, test 512.

## Text

- One **balanced quartet** = one layout × {circle, triangle} goal × {striped, dotted} trim.
  Every quartet item gets a caption from each of two training templates:
  `go to the {glyph} pad in the {trim} warehouse`,
  `the goal is the {glyph} marker on {trim} flooring`.
- Held-out phrasing (validation/test only) uses two different templates with the same factor
  words: `head for the {glyph} tile where the floor is {trim}`,
  `finish on the {glyph} spot of this {trim} room`.
- Captions never contain IDs, digits, coordinates, directions, route text or action labels
  (checked by a forbidden-token rule). Captions name the goal glyph and trim only; the image
  supplies the layout.

## Pair arms

- `clean`: every goal image paired with its own caption (same complete scene).
- `goal_swap` and `trim_swap`: in exactly 25% of train quartets (one seeded index set, shared by
  both arms) the captions are permuted inside the quartet. `goal_swap` exchanges the circle and
  triangle captions at fixed trim; `trim_swap` exchanges striped and dotted at fixed glyph.
- By construction and by check: identical image multiset and caption multiset in all three arms,
  identical word frequencies, every swap flips exactly one factor and is a one-token edit, swaps
  are bijections, and exactly 25% of pairs are mismatched. Goal and trim stay independent.

## Data roles

- **Vision (shared, arm-independent):** image encoder, latent transition model and planner train
  only on train-layout images and transitions (random actions, both trims, both goals in equal
  numbers). Trained once per seed and frozen before any text training.
- **Text (per arm):** the text encoder trains only on that arm's train pairs (balanced over the
  four combos and two templates). It is the only component that differs between arms.
- **Calibration anchors:** 128 clean pairs, one per distinct calibration topology, balanced over
  goal (64/64), trim and template. Used only to fit the text→image map. This is paired
  supervision and is disclosed as such.
- **Validation:** health gates and all tuning. **Test:** locked (`SplitLocked`) until the
  protocol and milestone 2 settings are frozen; the integrity check generates it only to pin its
  digest.

## Seeds, manifest, commands

- Data seed 20261121 (`Config`); separate streams per purpose (layouts per split, swap indices,
  anchors). Milestone 2 run seeds vary model initialisation only.
- `python -m pgt.integrity --check manifests/default.json` regenerates all data, runs every check
  and compares against the committed digests (≈2 s on CPU).

## Milestone 2 plan (not implemented)

- Separate tiny image and text encoders into one compact shared latent dimension; only the text
  encoder retrains across arms. Guard against collapse (variance/covariance term) and report
  latent singular values.
- Fit one centered orthogonal Procrustes map from text to image latents on the 128 anchors.
- Image-only learned action-conditioned latent transition and a frozen planner: short bounded
  rollout candidates scored by latent distance to the goal. No true-environment transitions or
  BFS inside model planning. Plans are then **executed** in the real simulator and scored by
  the actual first pad entered.
- Pilot one seed under a 20-minute wall cap and measure before full runs; then 5 seeds if the
  measured cost fits the compute cap. Lock settings before touching test.
- Health gates (validation, eligible layouts): oracle 100%; image-goal planner ≥80%; predicted
  one-step latent better than persistence on action-relevant states; non-collapsed singular
  values; goal-blind accuracy ≤50% on identical balanced twin episodes; shuffled-calibration
  map compared against the empirical goal-blind control; timeouts counted. Gates check the
  pipeline works; they place no requirement on the research outcome.
- Primary metric: (`trim_swap` − `goal_swap`) executed goal success, percentage points, on the
  same test episodes and seed. Also: clean arm, both-twins-correct rate, same-layout
  image→text goal retrieval (overall and goal-sensitive agreement), per-seed paired effects,
  layout-clustered intervals conditional on seeds, CPU timing, calibration pair counts. No claim
  of generality beyond these tiny networks.
- Report: Evidence at a glance, Architecture, Consequential decision, guided walkthrough
  (deterministic success and failure example), scope, source evidence. Credit the user (study
  design and approval) and the agents (planning and implementation) accurately.
