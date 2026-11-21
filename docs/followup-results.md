# Follow-up results: a prediction tested on withheld glyph–trim combinations

**Outcome.** On held-out boards, every text arm succeeded on every episode. Rewiring 25% of
text pairs by a glyph (goal) flip hurt executed success on withheld combinations no more than
rewiring by a trim (floor) flip: the primary contrast is **+0.00 pp in all 12 cells**, with a
topology-bootstrap 95% interval of **[0.00, 0.00]**. That is the near-tie that Siddharth named
before training as the result that would change his mind. The plan is
[followup.md](followup.md); the original study is in the [case study](case-study.md).

## Who did what

- **Siddharth Sanjay** wrote the prediction and chose the threshold for reconsidering it.
  Codex committed [his original words](followup-prediction.json) at
  [8c1cb3d](https://github.com/siddsan7/platonic-goal-transfer/commit/8c1cb3dead3f3a30c33deac74e038e6bd74e4c3e)
  before follow-up training.
- **GPT-6.2 Astra and Claude Fable 5.2** planned the study. **Claude Opus 5.5** authored the
  implementation and this report. **GPT-6.2 Sol** reviewed the code. **Codex** ran the phases
  and verified the outputs.

## The prediction, point by point

The prediction is quoted from the committed file; the evidence column comes from the held-out
test rows.

| Siddharth's prediction (his words, abridged) | What the held-out data say |
|---|---|
| 1) "the training pair which disagrees about the floor pattern should function better more often" (trim_swap beats goal_swap) | Not supported. trim_swap and goal_swap both scored 18,432/18,432 on withheld combinations. Primary +0.00 pp in all 12 cells; 95% CI [0.00, 0.00]. |
| 2) "I don't expect a simple word-count baseline to function as well" | Not supported here. Raw word counts scored 4,608/4,608 on withheld combinations, and every learned arm minus raw, paired episode by episode, is +0.00 pp. The additive keyword baseline also scored 4,608/4,608. |
| 3) A tie "(maybe within 5 points)", or a reversal, "would definitely change my mind" | The tie condition was met: 0 pp, well inside 5 pp. No reversal: the trim-flip model did not lose either. |

Success reached the ceiling even with raw word counts. The predicted advantage would have
required at least one wrong turn in the shape-disagreement condition. Across the held-out maps,
there were none. Equal success leaves open whether the encoders learned different internal
representations or would make different errors on harder instructions.

## Held-out evidence (48 fresh topologies)

Success on **withheld** combinations, pooled over 4 corpora × 3 anchor banks. The seen panel
gives the same numbers.

| Condition | Correct / episodes | Wrong pad | Timeout |
|---|---|---|---|
| clean | 18,432 / 18,432 | 0 | 0 |
| goal_swap | 18,432 / 18,432 | 0 | 0 |
| trim_swap | 18,432 / 18,432 | 0 | 0 |
| raw word counts | 4,608 / 4,608 | 0 | 0 |
| keyword (additive) | 4,608 / 4,608 | 0 | 0 |
| image goal | 1,536 / 1,536 | 0 | 0 |
| oracle | 1,536 / 1,536 | 0 | 0 |
| goal-blind | 2,304 / 4,608 (exactly 50%) | 2,304 | 0 |

- **Primary.** trim_swap − goal_swap, paired on withheld combinations: 0.00 pp in each of the 12
  corpus × bank cells (1,536 episodes each) and in each corpus.
- **Interval.** 2,000 resamples of the 48 topologies give [0.00, 0.00]. It is conditional on the
  fixed corpora, anchor banks and the single visual fit, and has zero width because every
  topology shows identical paired outcomes.
- **Swap against clean.** Both swap arms minus clean: 0.00 pp in all 48 cell × panel entries.
- **Health gate.** Oracle 100%, blind 50% (limit ≤ 50%), image goal 100% (≥ 80%), seen clean
  100% (≥ 80%): PASS. Timeouts are 0 in every condition.
- **Goal discrimination (manifest only).** The manifest reports the correct goal image scoring
  above the foil in every episode of all 84 condition cells (no ties, no losses). These scores are
  not in the public rows, so the check script reads this from the manifest and cannot recompute
  it.

**Pilot (16 validation topologies, 48,128 rows).** The same pattern: every learned arm 6,144/6,144
per panel, raw and keyword 1,536/1,536, blind exactly 50%, primary 0.00 pp, CI [0.00, 0.00],
gate PASS. Recorded run times were 10.2 s (pilot) and 26.7 s (test), well inside the 900 s and
2,400 s phase caps.

## What the result does and does not cover

- **Withholding is partial by design.** The eight target combinations were excluded from the text
  pairs and from the calibration anchors. The single shared visual fit saw all 16 combinations,
  and every withheld combination is built from glyph and trim words the text side did see, in
  new sentence templates. A text side that composes known words could therefore succeed here without
  generalising beyond its vocabulary.
- **Ceiling.** With every condition at 100% (blind at its 50% construction ceiling), the test has
  no room to show a difference. A 0 pp result is a tie at the top, not evidence of general
  equivalence.
- **One visual fit.** The 12 cells share it and the anchor banks are crossed, so they are not
  independent replications.
- **Relation to the original study.** The original found +50 pp at width 1 (rank-one text) and
  0 pp at width 2. This follow-up uses a 16-wide encoder and also changes the world (4 glyphs ×
  4 trims), the corpora, the bridge and the board banks. The 0 pp here cannot be attributed to
  width alone.

## Chronology and the numerical amendment

1. `8c1cb3d` (2026-11-21 13:09): Siddharth's prediction committed.
2. `b24ad17`: implementation frozen.
3. The first validation attempt under that freeze raised a NumPy overflow warning in the
   text-encoder loss during fixed-step training. Its outputs were treated as invalid and
   excluded; no navigation score from it was read, and none informed the fix.
4. `0207825` and `45dbd02` (13:21–13:22): the plan was amended in the open
   ([followup.md, "Protocol amendment"](followup.md)). The encoder keeps its objective, data,
   width, epochs and starting step, but halves the step when a trial update is non-finite or
   raises the loss. Non-finite goal vectors or scores now fail a run loudly. No seed,
   hyperparameter, model width or health gate changed, and the prediction was untouched.
5. A new freeze ([freeze.json](evidence/followup/freeze.json)) was taken from the amended
   commit; the pilot and then the test ran once each under it and both passed.

The tradeoff: amending after a failure always risks looking like trial selection. The guard is
that the failure was numerical, it was recorded before any outcome was seen, the fix is
outcome-blind and documented, and there was exactly one valid pilot and one valid test.

## What we learned

- **A pre-registered threshold makes a null result informative.** Siddharth's "within about
  5 points" turned a 0 pp result into a clear answer to his own question rather than a shrug.
- **Ceiling effects hide comparisons.** Once raw word counts also reach 100%, the study cannot
  separate learned text from simple lookup. Revealing a difference would need a harder test,
  such as a smaller encoder or unseen words, which this study did not run.
- **Numerical failures deserve a public paper trail.** The overflow cost a run, but documenting
  it kept the final evidence trustworthy.

## Evidence and how to check it

Files in [docs/evidence/followup/](evidence/followup): `freeze.json`, and for `pilot/` and
`test/` a `manifest.json` plus `rows.jsonl.gz` (one line per executed episode).

```sh
PYTHONPATH=src python scripts/check_followup.py
```

The script reads only those files. It checks that both manifests carry the committed freeze and
the frozen design digest. It then recomputes from the gzip rows, using the frozen
`pgt.followup.analysis` routines: strict coverage (48,128 and 144,384 rows), the counts and row
digests, the health gate, success levels, learned-minus-raw, swap-minus-clean, the primary and
its topology bootstrap. All of these must match the manifest exactly. It also prints the paired
learned − raw effects and lists what it can only read from the manifest (goal discrimination,
run time).
