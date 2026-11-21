"""Paired effects and topology-clustered bootstrap with seed pairing held fixed."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cluster_table(
    records: Sequence[dict], condition: str, seeds: Sequence[int], clusters: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """(seeds, clusters) success sums and episode counts for one condition."""
    s_index = {s: i for i, s in enumerate(seeds)}
    c_index = {c: i for i, c in enumerate(clusters)}
    wins = np.zeros((len(seeds), len(clusters)))
    count = np.zeros((len(seeds), len(clusters)))
    for r in records:
        if r["condition"] != condition:
            continue
        i, j = s_index[r["seed"]], c_index[r["cluster"]]
        wins[i, j] += r["success"]
        count[i, j] += 1
    return wins, count


def paired_effect(
    a: tuple[np.ndarray, np.ndarray],
    b: tuple[np.ndarray, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> dict:
    """Success(a) - success(b) in percentage points, per seed and pooled over seeds.

    The CI resamples topology clusters with replacement; every bootstrap draw applies the same
    cluster weights to every seed (seed pairing fixed) and to both conditions (paired episodes).
    """
    (wa, na), (wb, nb) = a, b
    arrays = (wa, na, wb, nb)
    if any(x.ndim != 2 or x.size == 0 or not np.isfinite(x).all() for x in arrays):
        raise ValueError("tables must be non-empty finite (seeds, clusters) arrays")
    if len({x.shape for x in arrays}) != 1 or not np.array_equal(na, nb):
        raise ValueError("conditions must cover identical episodes")
    if (na < 0).any() or (wa < 0).any() or (wb < 0).any() or (wa > na).any() or (wb > nb).any():
        raise ValueError("success counts must lie in [0, episodes]")
    if (na.sum(axis=1) == 0).any() or (na.sum(axis=0) == 0).any():
        raise ValueError("every seed and every cluster needs at least one episode")
    if n_boot < 1 or not 0 < level < 1:
        raise ValueError("invalid bootstrap size or level")
    per_seed = 100 * (wa.sum(1) / na.sum(1) - wb.sum(1) / nb.sum(1))
    n_clusters = na.shape[1]
    weights = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)
    boot_a = (weights @ wa.T) / (weights @ na.T)  # (boot, seeds)
    boot_b = (weights @ wb.T) / (weights @ nb.T)
    boot = 100 * (boot_a - boot_b).mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "effect_pp": float(per_seed.mean()),
        "per_seed_pp": [float(x) for x in per_seed],
        "ci_pp": [float(lo), float(hi)],
        "clusters": int(n_clusters),
        "bootstrap": int(n_boot),
    }
