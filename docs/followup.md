# Follow-up study: withheld glyph–trim combinations

Astra and Fable designed this prospective study. Claude Opus implemented it, and Siddharth
Sanjay supplied [his prediction](followup-prediction.json) before training. The original study
and results are available in the [case study](case-study.md).

## Question

When the text encoder never sees certain glyph–trim combinations, does rewiring 25% of text
pairs by a one-factor glyph flip hurt executed goal success on those withheld combinations more
than a trim flip does?

## Frozen design (`src/pgt/followup/design.py`, digest = `FollowupDesign().digest()`)

- **World**: 7×7 grid with two terminal pads. 4 glyphs × 4 floor trims, real rendered masks; each
  glyph and each trim has a distinct outer-ring pattern (`world4.py`). Dynamics, the D4 topology
  key and the move rule reuse `pgt.world`.
- **Seen rule**: a combination is seen iff `(glyph − trim) mod 4 ∈ {0, 1}`. That gives a cyclic,
  connected 8 seen / 8 withheld split, with two seen and two withheld glyphs in every trim panel.
- **Withholding covers text and calibration only.** Text corpora and calibration anchors use seen
  combinations only. The visual model may see all 16 combinations; every report must say so.
- **Boards**: fresh validation (16) and test (48) banks. They exclude every canonical topology
  from earlier evaluated banks (`banks.prior_topologies`) and from each other. Both pads are
  certified with the other pad as a barrier. Each board is evaluated in every trim panel, for both
  eligible glyphs, in both pad assignments, with a query for each glyph.
- **Text**: each block holds 8 seen combinations × 2 template directions. Exactly 25% of blocks,
  identical across the two swap arms, rewire the right endpoint by a one-factor flip that stays
  inside the seen set. The flip is a bijection, so the left and right caption pools are identical
  in every arm. Each corpus seed draws 2 of 6 real templates. Validation and test use 2
  separate held-out templates.
- **Replication grid**: 4 corpus seeds × 3 anchor banks, fully crossed, with one fixed visual
  fit. Every arm within a corpus starts from the same encoder initialisation, and each encoder
  gets its own bridge. All results are conditional on the single visual fit. The 12 cells are
  not independent replications.
- **Encoder** (`encoder.py`): a one-hidden-layer tanh network with an unconstrained 16-wide
  output. It is trained on weighted pairs with `−2 tr(C_lr) + ‖C_zz‖²_F`, where C_lr is the
  left/right cross-covariance and C_zz is the output covariance. The second term is the
  anti-collapse penalty, and there is no whitening step. The gradient is checked against finite
  differences. Updates use deterministic backtracking from step 0.05 (see the amendment below).
- **Bridges** (`bridge.py`): one centred ridge regression (fixed penalty and rule) fitted on the
  same seen anchors for every condition. Raw word counts get one shared bridge with no text
  training. The keyword baseline fits additive glyph and trim effects on seen anchors, so it is
  defined for withheld combinations.
- **Vision**: cell and floor principal-component dimensions are 4 and 3.

## Protocol amendment (2026-11-21)

Provenance: the original freeze was taken at commit `b24ad17`, with the candidate's prediction
committed at `8c1cb3d`. This plan file was changed on purpose afterwards, so the original freeze
no longer matches; the amended freeze is a new record taken from the amended commit. The
prediction is unchanged.

- **What happened.** The first validation attempt under `b24ad17` raised a NumPy floating-point
  overflow `RuntimeWarning` in `pair_loss` and its gradients (`encoder.py` lines 52–54) during
  fixed-step text-encoder training. That attempt is invalid. No inference is drawn from it, and
  no follow-up test or navigation score from it was read; none informed this change.
- **Encoder update.** `encoder.train` keeps the same objective, data, width, epochs and initial
  step 0.05. Each epoch tries steps 0.05, 0.025, … (at most 30 halvings) and takes the first one
  whose loss, gradients and parameters are finite and whose loss increases by at most
  `1e-12 × max(1, abs(current loss))`. If 0.05 qualifies, the update is identical to the original one. If no trial
  qualifies, training stops at the current finite parameters (treated as converged). Non-finite
  inputs, starting parameters, starting loss or starting gradients raise an error; returned
  parameters are always finite.
- **Non-finite gate.** Goal images, anchor text vectors, goal vectors (every condition) and
  goal-image scores are checked before they reach the planner or any metric. A non-finite value
  fails the run loudly (status FAILED, no primary).
- **Report-time timeout.** If the phase cap is hit after the summary was computed, the run is
  INCOMPLETE and its primary estimate and interval are withheld.
- No model width, objective, data, hyperparameter, seed or health-gate change.

## Pre-declared analysis (`analysis.py`)

- **Primary**: the paired executed-success difference trim_swap − goal_swap on withheld
  combinations, in percentage points, averaged with equal weights over the 12 cells. Each cell
  and each corpus is also reported. Paired comparisons refuse episode sets that do not match.
- **Secondary**: the seen panel; each swap arm against clean; within-trim goal discrimination
  (score of the correct versus the foil goal image, with ties counted separately); wrong-pad and
  timeout counts; and the clean, raw, keyword, image, blind and oracle conditions. The pair
  effect and the raw-count comparison are reported separately.
- **Health gate**: oracle = 100%, blind ≤ 50%, image goal ≥ 80% and seen clean ≥ 80%. If any
  check fails, the study is reported as **INCONCLUSIVE**. No reruns with other seeds.
- **Outputs**: machine-readable count rows (`count_table`) and their digest (`counts_digest`).
  The case-study metrics are limited to the items above.

## Run gating (`python -m pgt.followup.cli`)

The plan is always this file, `docs/followup.md`. The candidate's prediction is a JSON file with
`author` and `prediction` fields in their own words. Empty or placeholder text is refused.

- `freeze --prediction <file> --write <freeze.json>` checks that the plan and the prediction are
  committed at HEAD, with no staged, modified or untracked change, and reads both from the HEAD
  blob. It takes the prediction's commit and Git author from `git log`, never from the file. It
  records the design digest, the plan digest, the prediction digest with its commit and author,
  the original `docs/protocol.md` digest, and a code digest (the follow-up package plus the
  reused original `pgt` modules). Digests use repository-relative paths and treat CRLF as LF, so
  any clone or checkout gives the same IDs. The record is written atomically and never
  overwrites an existing file.
- `check --prediction <file> --freeze <freeze.json>` recomputes every ID and reports any drift.
- `pilot` and `test` (`--prediction`, `--freeze`, `--out`, and `--pilot` for test) refuse to
  start unless every frozen ID still matches and `--out` is a new directory named
  `followup-<phase>-<label>`. For `test`, `--pilot` must be a pilot output directory. Its
  `manifest.json` must name phase `pilot`, the same freeze and the same design digest. The CLI
  then recomputes complete query coverage, outcome and row digests, and the health gate from
  `rows.jsonl`, and all of them must match the manifest and PASS.
- After the gates pass, the phase runs in a private child process (a `multiprocessing` spawn
  target, not a subcommand). The child re-checks every gate and passes the pilot record returned
  by that verification directly to `engine.run_phase(phase, design, freeze, out, pilot=pilot)`.

**Hard budget cap.** Two separate cooperative caps are checked inside the engine between scenes
and stages: the build cap (600 s) covers banks, component fitting and bridges; the phase cap
(900 s pilot, 2400 s test) starts after the build and covers evaluation and the report. Over
either cap the run is recorded as INCOMPLETE. On top of that, the parent kills the child through
the operating system (TerminateProcess on Windows) at build cap + phase cap + 60 s:
600 + 900 + 60 = 1560 s for the pilot and 600 + 2400 + 60 = 3060 s for the test. This kill also
stops a single long training step. `pilot` and `test` are the only public commands that run a
phase, and both always go through this deadline. Each phase can run once per output
directory.
The run uses only local CPU and has no paid services.

## Reproduction

From the repository root, install the development dependencies listed in the README, then run:

```sh
python -m pgt.followup.cli freeze --prediction docs/followup-prediction.json --write freeze.json
python -m pgt.followup.cli pilot --prediction docs/followup-prediction.json --freeze freeze.json --out followup-pilot-01
python -m pgt.followup.cli test --prediction docs/followup-prediction.json --freeze freeze.json --pilot followup-pilot-01 --out followup-test-01
```

The protocol uses one visual fit with 48 training layouts. Each has two random walks per trim.
The outcome embedding sees both pads on all four floor patterns. A salted hash of the wall
topology assigns separate fitting and evaluation pools. The primary interval resamples whole
evaluation layouts 2,000 times, conditional on the four training corpora, three calibration
banks and single visual fit. The original prediction is preserved in
[the author's words](followup-prediction.json).
