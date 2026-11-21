"""Nonlinear text encoder: one tanh hidden layer, unconstrained 16-wide output.

Objective on weighted pairs (weights normalised to sum to 1), centred by the pooled mean:
    loss = -2 tr(C_lr) + ||C_zz||_F^2
with C_lr = sum_i w_i l_i r_i^T and C_zz = 1/2 sum_i w_i (l_i l_i^T + r_i r_i^T) on centred
outputs. The Frobenius term is the anti-collapse penalty (a constant output scores 0, while
aligned spread outputs score below 0). There is no whitening step, so no gradient flows through
one. Full-batch gradient descent on CPU; deterministic given the initial parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Params:
    w1: np.ndarray  # (vocab, hidden)
    b1: np.ndarray
    w2: np.ndarray  # (hidden, out)
    b2: np.ndarray

    def flat(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]


def init_params(rng: np.random.Generator, vocab: int, hidden: int, out: int) -> Params:
    """Same rng (seeded by corpus seed only) for every arm, so arms share the initialisation."""
    return Params(
        rng.normal(0, 1 / np.sqrt(vocab), (vocab, hidden)),
        np.zeros(hidden),
        rng.normal(0, 1 / np.sqrt(hidden), (hidden, out)),
        np.zeros(out),
    )


def forward(p: Params, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.tanh(x @ p.w1 + p.b1)
    return h @ p.w2 + p.b2, h


def pair_loss(zl: np.ndarray, zr: np.ndarray, w: np.ndarray):
    """Loss and its gradients with respect to the raw (uncentred) outputs."""
    w = w / w.sum()
    wc = w[:, None]
    m = 0.5 * (w @ zl + w @ zr)
    lc, rc = zl - m, zr - m
    c_lr = (lc * wc).T @ rc
    c_zz = 0.5 * ((lc * wc).T @ lc + (rc * wc).T @ rc)
    loss = -2 * np.trace(c_lr) + np.sum(c_zz**2)
    gl = wc * (-2 * rc + 2 * lc @ c_zz)
    gr = wc * (-2 * lc + 2 * rc @ c_zz)
    shift = 0.5 * (gl.sum(0) + gr.sum(0))  # chain rule through the shared mean
    return float(loss), gl - wc * shift, gr - wc * shift


def loss_and_grad(p: Params, xl: np.ndarray, xr: np.ndarray, w: np.ndarray):
    zl, hl = forward(p, xl)
    zr, hr = forward(p, xr)
    loss, gl, gr = pair_loss(zl, zr, w)
    grads = [np.zeros_like(a) for a in p.flat()]
    for x, h, g in ((xl, hl, gl), (xr, hr, gr)):
        grads[2] += h.T @ g
        grads[3] += g.sum(0)
        dh = (g @ p.w2.T) * (1 - h**2)
        grads[0] += x.T @ dh
        grads[1] += dh.sum(0)
    return loss, grads


def train(p: Params, xl, xr, w, epochs: int, lr: float) -> tuple[Params, list[float]]:
    """Return trained copy of ``p`` and the loss trace; ``p`` itself is left untouched."""
    q = Params(*(a.copy() for a in p.flat()))
    trace = []
    for _ in range(epochs):
        loss, grads = loss_and_grad(q, xl, xr, w)
        trace.append(loss)
        for a, g in zip(q.flat(), grads, strict=True):
            a -= lr * g
    return q, trace


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of normalised singular values of centred outputs (collapse -> 0 or 1)."""
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    if s.sum() <= 1e-12:
        return 0.0
    q = s[s > 0] / s.sum()
    return float(np.exp(-(q * np.log(q)).sum()))
