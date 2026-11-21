"""Nonlinear text encoder: one tanh hidden layer, unconstrained 16-wide output.

Objective on weighted pairs (weights normalised to sum to 1), centred by the pooled mean:
    loss = -2 tr(C_lr) + ||C_zz||_F^2
with C_lr = sum_i w_i l_i r_i^T and C_zz = 1/2 sum_i w_i (l_i l_i^T + r_i r_i^T) on centred
outputs. The Frobenius term is the anti-collapse penalty (a constant output scores 0, while
aligned spread outputs score below 0). There is no whitening step, so no gradient flows through
one. Full-batch gradient descent with deterministic step backtracking (see ``train``) on CPU;
deterministic given the initial parameters.
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


# Protocol amendment (2026-11-21): the first validation attempt under b24ad17 overflowed in
# ``pair_loss`` with the unguarded fixed step. Each update now backtracks from ``lr``.
MAX_HALVINGS = 30  # trial steps lr, lr/2, ..., lr/2**30
DESCENT_RTOL = 1e-12  # accepted loss may exceed the current one by at most this relative slack


def _all_finite(loss: float, arrays) -> bool:
    return bool(np.isfinite(loss)) and all(np.isfinite(a).all() for a in arrays)


def train(p: Params, xl, xr, w, epochs: int, lr: float) -> tuple[Params, list[float]]:
    """Full-batch gradient descent with deterministic backtracking on the same objective.

    Each epoch tries steps ``lr, lr/2, ...`` (at most ``MAX_HALVINGS`` halvings) and takes the
    first whose loss, gradients and parameters are all finite and whose loss is at most
    ``loss + DESCENT_RTOL * max(1, |loss|)``. When ``lr`` itself qualifies the update equals the
    original fixed-step update. If no trial qualifies, the current point is treated as converged
    (no finite descent at any trial step) and training stops early with those parameters.

    Returns a trained copy of ``p`` and the loss trace: the starting loss followed by the loss
    after each accepted step, so ``len(trace) == epochs + 1`` unless training stopped early.
    Raises ``FloatingPointError`` if the inputs, starting parameters, starting loss or starting
    gradients are not finite; returned parameters are always finite. ``p`` is left untouched.
    """
    if not (np.isfinite(lr) and lr > 0) or epochs < 0:
        raise ValueError("lr must be finite and positive, epochs non-negative")
    named = zip(("w1", "b1", "w2", "b2"), p.flat(), strict=True)
    for name, a in (("xl", xl), ("xr", xr), ("w", w), *named):
        if not np.isfinite(np.asarray(a, dtype=float)).all():
            raise FloatingPointError(f"non-finite {name} before training")
    q = Params(*(a.copy() for a in p.flat()))
    with np.errstate(over="ignore", invalid="ignore"):
        loss, grads = loss_and_grad(q, xl, xr, w)
    if not _all_finite(loss, grads):
        raise FloatingPointError("non-finite starting loss or gradient")
    trace = [loss]
    for _ in range(epochs):
        for k in range(MAX_HALVINGS + 1):
            step = lr * 0.5**k
            cand = Params(*(a - step * g for a, g in zip(q.flat(), grads, strict=True)))
            with np.errstate(over="ignore", invalid="ignore"):
                c_loss, c_grads = loss_and_grad(cand, xl, xr, w)
            ok = _all_finite(c_loss, [*c_grads, *cand.flat()])
            if ok and c_loss <= loss + DESCENT_RTOL * max(1.0, abs(loss)):
                break
        else:
            return q, trace  # converged: no finite descent at any trial step
        q, loss, grads = cand, c_loss, c_grads
        trace.append(loss)
    return q, trace


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of normalised singular values of centred outputs (collapse -> 0 or 1)."""
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    if s.sum() <= 1e-12:
        return 0.0
    q = s[s > 0] / s.sum()
    return float(np.exp(-(q * np.log(q)).sum()))
