# platonic-goal-transfer

A controlled study of text training pairs and text-goal transfer in a learned visual planner:
when a text encoder learns only from caption-to-caption pairs, does rewiring a pair's positive to
the wrong *goal* hurt a fixed image-only planner more than rewiring it to the wrong *cosmetic
floor trim*? Protocol: [docs/protocol.md](docs/protocol.md).

Milestone 1 (this code): a deterministic 7×7 warehouse, D4-canonical topology splits, a BFS
oracle with first-pad termination, three matched text-only pair arms, and separate image-to-text
calibration anchors. No models yet.

## Setup (Python 3.12, CPU only, no external data)

```sh
python -m pip install -e ".[dev]"     # numpy 2.4.6, pytest 9.0.3, ruff 0.16.8
```

## Commands

```sh
python -m pytest                                          # tests
python -m ruff check . && python -m ruff format --check .  # lint
python -m pgt.integrity --check manifests/default.json     # regenerate data, verify digests
```

Without installing, prefix the integrity command with `PYTHONPATH=src`.

## Layout

- `src/pgt/world.py`: grid dynamics, terminal rule, renderer, D4 canonical form, BFS oracle.
- `src/pgt/data.py`: config and seeds, layout generation, splits, captions, text-pair arms
  (`TextPair`), calibration anchors (`CalibrationAnchor`), evaluation twins.
- `src/pgt/integrity.py`: every data check plus the manifest writer.
- `manifests/default.json`: committed digests and route-length report for the default config.
