# Alias-transfer results

**Status: `VALID`.** One run, `runs/alias-transfer-01`, from clean commit
`479afe8b57990b4f0a89713a259869c67af595a0` (`"dirty": false`). The run took 9.7 s. All validity
checks passed and coverage was complete: 49,920 rows, none missing, extra or duplicated.
No other alias run was made. Protocol: [alias-protocol.md](alias-protocol.md). Unchanged
artifacts: [evidence/alias/](evidence/alias/).

## Prediction (recorded before the run)

Siddharth Sanjay wrote the prediction. Codex committed his words in
[PR #9](https://github.com/siddsan7/platonic-goal-transfer/pull/9) before running the merged
revision. The [prediction file](alias-prediction.json) preserves his wording.

> I think consistent pairings will point more clearly to the circle because vexlo always refers to the same shape. I’d expect scrambled and untrained to be close. If either ties or beats consistent pairings, that would change my mind.

In protocol terms, "consistent pairings" is the **aligned** arm and "scrambled" is the
**independent** arm. The prediction names `vexlo` and the circle as an example. The run
averages over all four nonces and eight counterbalanced mappings.

## Primary contrast

The margin is ‖v − foil image‖² − ‖v − target image‖², a squared distance in the frozen visual
embedding space. A positive margin favours the target.

| Contrast (nonce captions) | Paired mean | 95% bootstrap over 12 boards |
| --- | --- | --- |
| aligned − independent | **8.6077923361** | [8.6077923348, 8.6077923374] |

The interval is conditional on the fixed boards, lexicon mappings, calibration bank 1 and the
one visual fit (2,000 resamples, seed 20261121). It is narrow because the 12 per-board means
agree to about the eighth decimal place (8.607792333 to 8.607792340). It does not cover
variation from refitting.

## Secondary contrasts and rates

Each text condition has **6,144 episodes per caption kind** (12 boards × 64 queries × 8 mappings).

- **aligned − untrained** nonce margin: 8.1103314035 (paired mean).
- Executed success, nonce captions: aligned − independent **+46.65 pp**; aligned − untrained
  **+44.92 pp**.

| Arm | Nonce success | Nonce mean margin | Canonical success | Canonical mean margin |
| --- | --- | --- | --- | --- |
| aligned | 98.05% (6,024; 120 wrong pad) | 8.5055 | 99.61% (6,120) | 12.0092 |
| independent | 51.40% (3,158; 2,986 wrong pad) | −0.1022 | 100.00% (6,144) | 12.0556 |
| untrained | 53.13% (3,264; 2,880 wrong pad) | 0.3952 | 91.55% (5,625) | 6.5250 |
| raw | 50.00% (3,072; 3,072 wrong pad) | 0.0000 | 100.00% (6,144) | 12.4212 |

There were no timeouts and no unavailable margins. Image goals succeeded in 768 of 768 episodes.

**Controls.** The raw bag-of-words arm was exactly blind to nonces as designed. Its nonce
coefficients are zero, the four aliases give equal goal vectors, and nonce distances are equal
across mappings. Its 50.00% is the two-pad chance level. The canonical success thresholds (≥ 80%
for aligned and independent) and the image-goal threshold (≥ 80%) were met.

**Untrained minus scrambled (descriptive).** For nonce captions, untrained − independent is
+0.4975 in mean margin and +1.73 pp in executed success (106 more correct episodes out of
6,144). The prediction set no tolerance for "close", so this gap is reported without a threshold.

## The prediction's stated trigger

The prediction named a mind-changing result: the scrambled or untrained arm tying or beating
the consistent arm. Consistent pairings led scrambled pairings by 8.61 margin units and
the untrained encoder by 8.11. The robot also reached the correct pad more often under
consistent pairings. These outcomes support the directional prediction.

## Limitations

- **Reused boards.** The 12 boards are the first 12 published follow-up test topologies, which
  had already been evaluated. This is a conditional paired diagnostic, not a fresh held-out test.
- **One visual fit and one calibration bank.** The interval resamples boards only.
- **Eight counterbalanced lexicons.** These come from two seeded blocks of four Latin rotations,
  so they are not independent draws.

## How to check the packaged artifacts

`docs/evidence/alias/` holds byte-identical copies of the run's `plan.json`, `rows.jsonl.gz` and
`summary.json`. To check them:

```sh
sha256sum docs/evidence/alias/*
# 970a7e821f815879337a0940e14b0c58dde7fe49fee1b024c57cc851a7e03b80  plan.json
# 512ad520bb917fd5e7efed85558e4c4795229c70dfeb7f56b8a61694c95c7bf0  rows.jsonl.gz
# fdde9884ec88529670ac5f60a8f566c5663f62af68cac2e0e1858b002f428733  summary.json
PYTHONPATH=src python -m pgt.alias_study verify --out docs/evidence/alias   # prints "verify OK"
```

`verify` fits no model. It recomputes coverage, rates, diagnostics, the primary and secondary
contrasts, the bootstrap and the rows digest from the rows, and checks the plan against the frozen
design. `summary.json` records `plan_sha256` and `rows_sha256`; the latter is a digest of the row
content, not of the gzip file.
