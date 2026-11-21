"""Text-only goal encoder, Procrustes calibration and text-side controls.

Encoder family: symmetric regularised CCA (a spectral paired-view objective) on centred
bag-of-words counts. Both sides of every ``TextPair`` are views of one caption distribution; the
encoder is the projection W maximising tr(W' C_lr W) subject to W' (C_xx + ridge I) W = I, where
C_lr is the symmetrised left/right cross-covariance and C_xx the pooled covariance. The whitening
constraint is the anti-collapse term (outputs are decorrelated with fixed variance). Words are
first reduced to the principal subspace of C_xx (eigenvalues above ``rank_tol`` x the largest),
so directions with no training variance (constant words, unseen words) cannot leak into the
output. The top ``dim`` generalised eigenvectors by eigenvalue are kept; the solution is
closed-form, so it is deterministic, independent of sample order and of the model seed.

OOV: held-out phrasing words absent from the fixed training vocabulary are dropped (counted).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pgt.data import COMBOS, TRAIN_TEMPLATES, TextPair, caption
from pgt.world import GLYPHS, TRIMS


def training_vocabulary() -> tuple[str, ...]:
    """Fixed, arm-independent vocabulary: every word of the 8 training captions."""
    words = {w for t in range(len(TRAIN_TEMPLATES)) for c in COMBOS for w in caption(c, t).split()}
    return tuple(sorted(words))


def bag_of_words(texts: Sequence[str], vocab: Sequence[str]) -> tuple[np.ndarray, int]:
    """Count matrix and the number of out-of-vocabulary tokens dropped."""
    index = {w: i for i, w in enumerate(vocab)}
    x = np.zeros((len(texts), len(vocab)))
    oov = 0
    for row, text in enumerate(texts):
        for word in text.split():
            if word in index:
                x[row, index[word]] += 1
            else:
                oov += 1
    return x, oov


@dataclass(frozen=True)
class TextEncoder:
    vocab: tuple[str, ...]
    mean: np.ndarray
    proj: np.ndarray  # (vocab, dim)
    eigenvalues: np.ndarray  # all generalised eigenvalues in the retained word subspace, desc

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        x, _ = bag_of_words(texts, self.vocab)
        return (x - self.mean) @ self.proj

    def oov(self, texts: Sequence[str]) -> int:
        return bag_of_words(texts, self.vocab)[1]

    def digest(self) -> str:
        return hashlib.sha256(np.round(self.proj, 10).tobytes()).hexdigest()[:16]


def train_text_encoder(
    pairs: Sequence[TextPair], dim: int, ridge: float, rank_tol: float = 1e-9
) -> TextEncoder:
    """Fit the paired-view encoder from text pairs only (no image, layout or pixel input)."""
    vocab = training_vocabulary()
    left, _ = bag_of_words([p.left for p in pairs], vocab)
    right, _ = bag_of_words([p.right for p in pairs], vocab)
    mean = (left.mean(axis=0) + right.mean(axis=0)) / 2
    lc, rc = left - mean, right - mean
    n = len(pairs)
    cxx = (lc.T @ lc + rc.T @ rc) / (2 * n)
    cross = (lc.T @ rc + rc.T @ lc) / (2 * n)
    evals, evecs = np.linalg.eigh(cxx)
    keep = evals > rank_tol * evals.max()
    basis = evecs[:, keep]  # principal word subspace
    s = evals[keep]
    reg = s + ridge * s.mean()  # documented ridge: fraction of the mean retained variance
    whiten = basis / np.sqrt(reg)  # maps words to whitened coordinates
    m = whiten.T @ cross @ whiten
    lam, v = np.linalg.eigh((m + m.T) / 2)
    order = np.argsort(-lam, kind="stable")
    lam, v = lam[order], v[:, order]
    proj = whiten @ v[:, :dim]
    signs = np.sign(proj[np.abs(proj).argmax(axis=0), range(dim)])  # deterministic sign
    return TextEncoder(vocab, mean, proj * signs, lam)


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of the normalised singular values of centred embeddings."""
    s = np.linalg.svd(z - z.mean(axis=0), compute_uv=False)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def unique_states(z: np.ndarray, decimals: int = 6) -> int:
    return len({tuple(row) for row in np.round(z, decimals)})


# ---------------------------------------------------------------- calibration
@dataclass(frozen=True)
class Procrustes:
    """y = (t - text_mean) @ rot + image_mean, rot orthogonal (text zero-padded to image dim)."""

    text_mean: np.ndarray
    image_mean: np.ndarray
    rot: np.ndarray

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return (self._pad(t) - self.text_mean) @ self.rot + self.image_mean

    def _pad(self, t: np.ndarray) -> np.ndarray:
        k = self.rot.shape[0]
        return np.concatenate([t, np.zeros((*t.shape[:-1], k - t.shape[-1]))], axis=-1)


def fit_procrustes(text: np.ndarray, image: np.ndarray) -> Procrustes:
    """Centred orthogonal Procrustes from text latents (n, d) to image latents (n, k), d <= k."""
    k = image.shape[1]
    if text.shape[1] > k:
        raise ValueError("text latent wider than image latent")
    padded = np.concatenate([text, np.zeros((len(text), k - text.shape[1]))], axis=1)
    tm, im = padded.mean(axis=0), image.mean(axis=0)
    u, _, vt = np.linalg.svd((padded - tm).T @ (image - im))
    return Procrustes(tm, im, u @ vt)


def parse_words(text: str) -> tuple[str, str]:
    """Exact keyword lookup used only by the glyph-word baseline (templated language)."""
    words = text.split()
    glyph = next(w for w in words if w in GLYPHS)
    trim = next(w for w in words if w in TRIMS)
    return glyph, trim
