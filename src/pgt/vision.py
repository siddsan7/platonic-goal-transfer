"""Image-only factored visual model: learned agent detector, structured transition predictor,
terminal detector and compact outcome embedding. Trained once per model seed, then frozen.

Engineered (fixed, not learned): the 5 px cell tiling of the 35x35 raster, the fixed central
3x3 "agent mask" that splits each cell patch into a centre and an outer ring, the 4-neighbour
offset vocabulary, a constant "outside the grid" input flag, 0.5 decision thresholds, the
per-descriptor DC removal of the floor statistic, and PCA dimensions. Learned from rendered
frames, actions and the episode-end signal of random-action walks on train layouts only: which
patch holds the agent, which offset each action produces, which target patches block movement,
which entered patches end the episode, and the PCA/whitening of outcome descriptors.

Training consumes frames, actions and done flags only. True agent positions are recorded by the
walk generator for evaluation probes and are never passed to a training function.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pgt.world import (
    AGENT_MASK,
    CELL_PX,
    GLYPHS,
    TRIMS,
    Layout,
    N,
    render,
    step,
)

CELLS = N * N
PATCH = CELL_PX * CELL_PX * 3
# Ring = patch pixels outside the agent mask; identical whether or not the agent is present.
RING_IDX = np.flatnonzero(np.repeat(~AGENT_MASK.reshape(-1), 3))
OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # stay + 4 neighbours (engineered)
ROW, COL = np.divmod(np.arange(CELLS), N)


def patches_u8(images: np.ndarray) -> np.ndarray:
    """(..., 35, 35, 3) uint8 -> (..., 49, 75) uint8, row-major cells."""
    lead = images.shape[:-3]
    x = images.reshape(*lead, N, CELL_PX, N, CELL_PX, 3)
    return np.moveaxis(x, -4, -3).reshape(*lead, CELLS, PATCH)


def patches(images: np.ndarray) -> np.ndarray:
    """(..., 35, 35, 3) uint8 -> (..., 49, 75) float32 in [0, 1]."""
    return patches_u8(images).astype(np.float32) / 255.0


def as_float(p_u8: np.ndarray) -> np.ndarray:
    return p_u8.astype(np.float32) / 255.0


def ring(p: np.ndarray) -> np.ndarray:
    return p[..., RING_IDX]


def floor_stat(p: np.ndarray, exclude: list[int]) -> np.ndarray:
    """Mean full patch over cells not in ``exclude``, minus its own mean (removes the uniform
    shift from the wall count, leaving trim texture and pad pixels)."""
    keep = np.ones(CELLS, dtype=bool)
    keep[list(exclude)] = False
    m = p[keep].mean(axis=0)
    return m - m.mean()


# ---------------------------------------------------------------- random-action walks
@dataclass
class Walks:
    """Frames and actions of random-action episodes. ``positions`` is evaluation-only."""

    frames: np.ndarray  # (F, 35, 35, 3) uint8
    src: np.ndarray  # (T,) frame index before the action
    dst: np.ndarray  # (T,) frame index after the action
    actions: np.ndarray  # (T,)
    done: np.ndarray  # (T,) bool: episode ended at dst (a pad was entered)
    positions: np.ndarray  # (F, 2) true agent cell; probes only, never used in training


def random_walks(
    layouts: list[Layout], rng: np.random.Generator, walks: int, horizon: int
) -> Walks:
    frames, src, dst, acts, done, pos = [], [], [], [], [], []
    for layout in layouts:
        for trim in TRIMS:
            for _ in range(walks):
                cell = layout.start
                frames.append(render(layout, cell, trim))
                pos.append(cell)
                for _ in range(horizon):
                    a = int(rng.integers(4))
                    cell = step(layout, cell, a)
                    frames.append(render(layout, cell, trim))
                    pos.append(cell)
                    src.append(len(frames) - 2)
                    dst.append(len(frames) - 1)
                    acts.append(a)
                    hit = layout.pad_at(cell) is not None
                    done.append(hit)
                    if hit:
                        break
    return Walks(
        np.stack(frames),
        np.array(src),
        np.array(dst),
        np.array(acts),
        np.array(done),
        np.array(pos),
    )


def goal_images(layouts: list[Layout]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Both goals x both trims per layout (balanced), with their (glyph, trim) labels."""
    imgs, labels = [], []
    for layout in layouts:
        for glyph in GLYPHS:
            for trim in TRIMS:
                imgs.append(render(layout, layout.pad(glyph), trim))
                labels.append((glyph, trim))
    return np.stack(imgs), labels


# ---------------------------------------------------------------- learners
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass(frozen=True)
class Logistic:
    w: np.ndarray
    b: float

    def logit(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b

    def prob(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.logit(x))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    epochs: int,
    batch: int,
    lr: float,
    l2: float,
) -> Logistic:
    """Minibatch SGD on balanced-weighted log loss. Init and order come from ``rng``."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    pos = max(y.mean(), 1e-9)
    weight = np.where(y > 0, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))
    w = rng.normal(0.0, 0.01, x.shape[1])
    b = 0.0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            idx = order[i : i + batch]
            err = (sigmoid(x[idx] @ w + b) - y[idx]) * weight[idx]
            w -= lr * (x[idx].T @ err / len(idx) + l2 * w)
            b -= lr * err.mean()
    return Logistic(w, float(b))


@dataclass(frozen=True)
class OutcomeEmbedding:
    """Block PCA with whitening: [cell ring block | floor statistic block]."""

    cell_mean: np.ndarray
    cell_proj: np.ndarray
    floor_mean: np.ndarray
    floor_proj: np.ndarray
    cell_spectrum: np.ndarray
    floor_spectrum: np.ndarray

    @property
    def dim(self) -> int:
        return self.cell_proj.shape[1] + self.floor_proj.shape[1]

    def __call__(self, cell_ring: np.ndarray, floor: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                (cell_ring - self.cell_mean) @ self.cell_proj,
                (floor - self.floor_mean) @ self.floor_proj,
            ],
            axis=-1,
        )


def _pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    var = s**2 / len(x)
    v = vt[:k].T
    v = v * np.where(v[np.abs(v).argmax(axis=0), range(k)] < 0, -1, 1)  # deterministic sign
    return mean, v / np.sqrt(var[:k] + 1e-8), s / np.sqrt(len(x))


# ---------------------------------------------------------------- the frozen model
@dataclass(frozen=True)
class VisionModel:
    detector: Logistic  # full patch -> agent present
    displacement: np.ndarray  # (4,) learned offset index into OFFSETS per action
    displacement_counts: np.ndarray  # (4, 5) evidence for the above
    blocker: Logistic  # [target ring, outside flag] -> move blocked
    terminal: Logistic  # entered ring -> episode ends
    embed: OutcomeEmbedding

    def agent_cell(self, p: np.ndarray) -> np.ndarray:
        """Argmax detector score over cells; ``p`` is (..., 49, 75)."""
        return self.detector.logit(p).argmax(axis=-1)

    def outcome(self, images: np.ndarray) -> np.ndarray:
        """Outcome embedding of real frames: what the agent stands on + floor elsewhere."""
        p = patches(images)
        cells = self.agent_cell(p)
        rings = ring(p[np.arange(len(p)), cells])
        floors = np.stack([floor_stat(pi, [c]) for pi, c in zip(p, cells, strict=True)])
        return self.embed(rings, floors)

    def digest(self) -> dict[str, str]:
        def h(*arrays) -> str:
            m = hashlib.sha256()
            for a in arrays:
                m.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
            return m.hexdigest()[:16]

        e = self.embed
        return {
            "detector": h(self.detector.w, [self.detector.b]),
            "displacement": h(self.displacement),
            "blocker": h(self.blocker.w, [self.blocker.b]),
            "terminal": h(self.terminal.w, [self.terminal.b]),
            "embedding": h(e.cell_mean, e.cell_proj, e.floor_mean, e.floor_proj),
        }


def target_cells(cells: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell index after applying OFFSETS[offset]; second value flags leaving the grid."""
    d = np.array(OFFSETS)[offset]
    r, c = ROW[cells] + d[..., 0], COL[cells] + d[..., 1]
    outside = (r < 0) | (r >= N) | (c < 0) | (c >= N)
    return np.where(outside, cells, np.clip(r, 0, N - 1) * N + np.clip(c, 0, N - 1)), outside


def blocker_input(rings: np.ndarray, outside: np.ndarray) -> np.ndarray:
    return np.concatenate([np.where(outside[..., None], 0.0, rings), outside[..., None]], axis=-1)


def train_vision(
    walks: Walks,
    goal_imgs: np.ndarray,
    rng: np.random.Generator,
    *,
    negatives: int,
    epochs: int,
    batch: int,
    lr: float,
    l2: float,
    cell_dims: int,
    floor_dims: int,
) -> VisionModel:
    """Fit every learned visual component from frames, actions and done flags.

    ``walks.positions`` is deliberately not read here (tested).
    """
    src, dst, acts, done = walks.src, walks.dst, walks.actions, walks.done
    pu = patches_u8(walks.frames)  # (F, 49, 75) uint8; float copies are taken per subset
    changed = (pu[src] != pu[dst]).any(axis=-1)  # (T, 49) pixel change per cell
    moved = changed.any(axis=1)

    # 1) Agent detector: in moved transitions only the agent's old and new cells change, so the
    # agent patch only ever appears in changed cells; floor/pad patches appear unchanged too.
    xs, ys = [], []
    for t in np.flatnonzero(moved):
        ch = np.flatnonzero(changed[t])
        un = np.flatnonzero(~changed[t])
        neg = rng.choice(un, size=min(negatives, len(un)), replace=False)
        xs += [pu[src[t], ch], pu[dst[t], ch], pu[src[t], neg]]
        ys += [np.ones(2 * len(ch)), np.zeros(len(neg))]
    detector = fit_logistic(
        as_float(np.concatenate(xs)), np.concatenate(ys), rng, epochs, batch, lr, l2
    )
    frame_cells = np.concatenate(
        [detector.logit(as_float(pu[i : i + 2048])).argmax(axis=1) for i in range(0, len(pu), 2048)]
    )
    a_src, a_dst = frame_cells[src], frame_cells[dst]

    # 2) Displacement per action: most frequent detected offset among moved transitions.
    counts = np.zeros((4, len(OFFSETS)), dtype=int)
    dr, dc = ROW[a_dst] - ROW[a_src], COL[a_dst] - COL[a_src]
    for k, (orow, ocol) in enumerate(OFFSETS):
        hit = (dr == orow) & (dc == ocol) & moved
        np.add.at(counts[:, k], acts[hit], 1)
    displacement = counts[:, 1:].argmax(axis=1) + 1

    # 3) Collision: does the target patch (or leaving the grid) keep the agent in place?
    tgt, outside = target_cells(a_src, displacement[acts])
    x_block = blocker_input(ring(as_float(pu[src, tgt])), outside)
    blocker = fit_logistic(x_block, (a_dst == a_src).astype(float), rng, epochs, batch, lr, l2)

    # 4) Terminal: does entering this patch end the episode? (observed episode-end signal)
    entered = np.flatnonzero(moved)
    x_term = ring(as_float(pu[src[entered], a_dst[entered]]))
    terminal = fit_logistic(x_term, done[entered].astype(float), rng, epochs, batch, lr, l2)

    # 5) Outcome embedding on a balanced set: pad outcomes (goal images) + off-pad frames.
    pg = patches(goal_imgs)
    g_cells = detector.logit(pg).argmax(axis=1)
    off = np.flatnonzero(~done)
    off = rng.choice(off, size=min(len(off), len(goal_imgs)), replace=False)
    po = as_float(pu[dst[off]])
    o_cells = a_dst[off]
    rings = np.concatenate(
        [ring(pg[np.arange(len(pg)), g_cells]), ring(po[np.arange(len(po)), o_cells])]
    )
    floors = np.stack(
        [floor_stat(pi, [c]) for pi, c in zip(pg, g_cells, strict=True)]
        + [floor_stat(pi, [c]) for pi, c in zip(po, o_cells, strict=True)]
    )
    cm, cp, cs = _pca(rings, cell_dims)
    fm, fp, fs = _pca(floors, floor_dims)
    embed = OutcomeEmbedding(cm, cp, fm, fp, cs, fs)
    return VisionModel(detector, displacement, counts, blocker, terminal, embed)


# ---------------------------------------------------------------- one-step prediction
def predict_next_cell(model: VisionModel, p: np.ndarray, cells: np.ndarray, actions: np.ndarray):
    """Predicted next agent cell from the current patches only (no simulator)."""
    tgt, outside = target_cells(cells, model.displacement[actions])
    blocked = model.blocker.prob(blocker_input(ring(p[np.arange(len(p)), tgt]), outside)) > 0.5
    return np.where(blocked | outside, cells, tgt)
