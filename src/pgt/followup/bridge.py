"""Text-to-image bridges. Every bridge uses the SAME seen-combo anchors and one fixed penalty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgt.followup.design import GLYPHS4, TRIMS4


@dataclass(frozen=True)
class Ridge:
    x_mean: np.ndarray
    y_mean: np.ndarray
    coef: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) @ self.coef + self.y_mean


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> Ridge:
    """Centred ridge, coef = (Xc^T Xc + penalty I)^-1 Xc^T Yc. Same rule for every bridge."""
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    coef = np.linalg.solve(xc.T @ xc + penalty * np.eye(x.shape[1]), xc.T @ yc)
    return Ridge(xm, ym, coef)


def factor_onehot(combos: list[tuple[str, str]]) -> np.ndarray:
    """Additive glyph + trim indicators: a withheld (glyph, trim) still gets a defined input,
    because every glyph and every trim occurs in some seen anchor."""
    x = np.zeros((len(combos), len(GLYPHS4) + len(TRIMS4)))
    for i, (g, t) in enumerate(combos):
        x[i, GLYPHS4.index(g)] = 1
        x[i, len(GLYPHS4) + TRIMS4.index(t)] = 1
    return x


def keyword_combo(text: str) -> tuple[str, str]:
    """Exact keyword lookup used only by the keyword baseline."""
    words = text.split()
    return next(w for w in words if w in GLYPHS4), next(w for w in words if w in TRIMS4)
