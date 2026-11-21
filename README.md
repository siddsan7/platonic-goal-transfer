# platonic-goal-transfer

A controlled study of text training pairs and text-goal transfer in a learned visual planner:
when a text encoder learns only from caption-to-caption pairs, does rewiring a pair's positive to
the wrong *goal* hurt a fixed image-only planner more than rewiring it to the wrong *cosmetic
floor trim*? Protocol: [docs/protocol.md](docs/protocol.md).

The experiment uses a deterministic 7×7 warehouse, topology-disjoint splits, and a
shortest-path oracle. It trains a factored visual transition model and independent text encoders.
A separate set of 128 image–text anchors fits the transfer map.

## Outcome

On the held-out split (five seeds, 20,480 text episodes per arm), the locked width-2 condition
gives 100% executed goal success for every text arm, a +0.00 pp trim_swap − goal_swap contrast.
The width-1 capacity stress test, locked before held-out evaluation, gives goal_swap 50% and
trim_swap 100%, a +50.00 pp contrast in every seed. The case study and its evidence are in
[docs/case-study.md](docs/case-study.md).

## Setup (Python 3.12, CPU only, no external data)

```sh
python -m pip install -e ".[dev]"     # numpy 2.4.6, pytest 9.0.3, ruff 0.16.8
python -m pip install matplotlib==3.10.9  # optional, only for build_evidence.py figures
```

The study itself needs only NumPy.

## Commands

```sh
python -m pytest                                          # tests
python -m ruff check . && python -m ruff format --check .  # lint
python -m pgt.integrity --check manifests/default.json     # regenerate data, verify digests
```

Without installing, prefix the integrity command with `PYTHONPATH=src`.

Run the full validation pilots, then the five-seed held-out evaluation for each width. The
runner checks its 20-minute budget between phases and episodes. Use an outer process timeout
when you need a hard limit.

```sh
python -m pgt.experiment run --split validation --seeds 11 --text-dim 2 --out runs/pilot-d2
python -m pgt.experiment run --split validation --seeds 11 --text-dim 1 --out runs/pilot-d1
python -m pgt.experiment run --split test --seeds 11 23 47 71 101 --text-dim 2     --unlock-test --pilot runs/pilot-d2/metrics.json --out runs/final-d2
python -m pgt.experiment run --split test --seeds 11 23 47 71 101 --text-dim 1     --unlock-test --pilot runs/pilot-d1/metrics.json --out runs/final-d1
```

Smoke only: `--eval-layouts N` (positive, not accepted for pilots that unlock test).

## Layout

- `src/pgt/world.py`: grid dynamics, terminal rule, renderer, D4 canonical form, BFS oracle.
- `src/pgt/data.py`: config and seeds, layout generation, splits, captions, text-pair arms
  (`TextPair`), calibration anchors (`CalibrationAnchor`), evaluation twins.
- `src/pgt/integrity.py`: every data check plus the manifest writer.
- `src/pgt/vision.py`, `src/pgt/text.py`, and `src/pgt/planner.py`: encoder training and planning.
- `src/pgt/experiment.py`: validation, test evaluation, and evidence outputs.
- `manifests/default.json`: committed digests and route-length report for the default config.
- `scripts/build_evidence.py` and `docs/evidence/`: compact held-out evidence for the case study
  and the check that reproduces its numbers (`python scripts/build_evidence.py check`).
