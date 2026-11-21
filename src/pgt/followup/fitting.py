"""Production fits for the follow-up study: vision, planner, text encoders, raw counts, anchors.

Topology roles: every canonical wall topology is assigned, by a fixed hash, either to the
``fit`` role (vision training boards and anchor banks) or the ``eval`` role (validation/test
banks). Fitting only ever draws ``fit`` topologies, so no fitted component sees an evaluation
topology. Evaluation banks must be restricted to ``topology_role(walls) == "eval"``.

Nothing here generates or reads follow-up validation/test boards.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations

import numpy as np

from pgt import planner as P
from pgt import vision as V
from pgt.data import Config, generate_split
from pgt.followup import encoder as E
from pgt.followup.banks import bank_rng
from pgt.followup.corpus import make_corpus, vocabulary
from pgt.followup.design import ARMS, GLYPHS4, SEEN, TRIMS4, FollowupDesign
from pgt.followup.engine import Budget, Components
from pgt.followup.world4 import Board, certified, render4
from pgt.world import N, canonical_topology, step

ROLE_SALT = "pgt-followup-topology-role-v1"
FIT_ROLE_MODULUS = 4  # 1 in 4 topologies is reserved for fitting; the rest for evaluation

# Fixed fitting hyperparameters (not tuned). Visual learner settings copy the original
# ExpConfig (detector_negatives, epochs, batch, lr, l2, length_penalty); sizes are bounded
# for a CPU fit inside the engine's build cap.
VISION_BOARDS = 48  # 8 boards per unordered glyph pair (6 pairs) -> every combo 24 goal images
WALKS_PER_BOARD_TRIM = 2
DETECTOR_NEGATIVES = 8
VISION_EPOCHS = 3
VISION_BATCH = 256
VISION_LR = 0.5
VISION_L2 = 1e-4
LENGTH_PENALTY = 1e-3
PLANNER_SCENE_CAP = 8192  # distinct frames held by the planner's caches before a reset
TEXT_INIT_STREAM = 7003  # init rng = [corpus_seed, 7003]: identical for every arm of a corpus
MAX_ATTEMPTS_PER_BOARD = 200


# ---------------------------------------------------------------- topology roles
def topology_role(walls) -> str:
    """'fit' or 'eval' from a salted hash of the canonical (D4-invariant) topology."""
    key = repr(canonical_topology(frozenset(tuple(w) for w in walls))).encode()
    h = int.from_bytes(hashlib.sha256(ROLE_SALT.encode() + key).digest()[:8], "little")
    return "fit" if h % FIT_ROLE_MODULUS == 0 else "eval"


def prior_layouts(cfg: Config | None = None) -> list[dict]:
    """Unique ``to_json`` records of the original study's evaluated validation and test layouts,
    regenerated deterministically from the original Config."""
    cfg = cfg or Config()
    out, seen = [], set()
    for split in ("validation", "test"):
        cfg.validate(split)
        for lay in generate_split(cfg, split, unlock_test=split == "test"):
            rec = lay.to_json()
            key = json.dumps(rec, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(rec)
    return out


def fit_boards(
    design: FollowupDesign, name: str, glyph_pairs: list[tuple[str, str]], excluded: set[tuple]
) -> list[Board]:
    """One certified ``fit``-role board per requested glyph pair, from the named stream
    ``fit/<name>``. Topologies are distinct and added to ``excluded``."""
    rng = bank_rng(design, f"fit/{name}")
    out: list[Board] = []
    for _ in range(len(glyph_pairs) * MAX_ATTEMPTS_PER_BOARD):
        if len(out) == len(glyph_pairs):
            break
        k = int(rng.integers(design.min_walls, design.max_walls + 1))
        cells = [divmod(int(i), N) for i in rng.permutation(N * N)[: k + 3]]
        walls = frozenset(cells[:k])
        key = canonical_topology(walls)
        if key in excluded or topology_role(walls) != "fit":
            continue
        board = Board(walls, cells[k], (cells[k + 1], cells[k + 2]), glyph_pairs[len(out)])
        if not certified(board, design.min_route, design.horizon):
            continue
        excluded.add(key)
        out.append(board)
    if len(out) < len(glyph_pairs):
        raise RuntimeError(f"fit/{name}: {len(out)}/{len(glyph_pairs)} boards")
    return out


# ---------------------------------------------------------------- vision
def board_walks(boards: list[Board], rng: np.random.Generator, walks: int, horizon: int):
    """Random-action walks under every trim (four-shape renders); same layout as V.Walks."""
    frames, src, dst, acts, done, pos = [], [], [], [], [], []
    for board in boards:
        for trim in TRIMS4:
            for _ in range(walks):
                cell = board.start
                frames.append(render4(board, cell, trim))
                pos.append(cell)
                for _ in range(horizon):
                    a = int(rng.integers(4))
                    cell = step(board, cell, a)
                    frames.append(render4(board, cell, trim))
                    pos.append(cell)
                    src.append(len(frames) - 2)
                    dst.append(len(frames) - 1)
                    acts.append(a)
                    hit = board.pad_at(cell) is not None
                    done.append(hit)
                    if hit:
                        break
    return V.Walks(
        np.stack(frames),
        np.array(src),
        np.array(dst),
        np.array(acts),
        np.array(done),
        np.array(pos),
    )


def board_goal_images(boards: list[Board]) -> np.ndarray:
    """Both pads x all four trims per training board."""
    return np.stack([render4(b, b.pad(g), t) for b in boards for g in b.glyphs for t in TRIMS4])


def vision_pairs() -> list[tuple[str, str]]:
    pairs = list(combinations(GLYPHS4, 2))
    per = VISION_BOARDS // len(pairs)
    return [p if i % 2 == 0 else p[::-1] for p in pairs for i in range(per)]


def fit_vision(design: FollowupDesign, excluded: set[tuple], budget: Budget) -> V.VisionModel:
    boards = fit_boards(design, "vision", vision_pairs(), excluded)
    budget.check("fit:vision-boards")
    walks = board_walks(
        boards, bank_rng(design, "fit/vision-walks"), WALKS_PER_BOARD_TRIM, design.horizon
    )
    goals = board_goal_images(boards)
    budget.check("fit:vision-data")
    rng = np.random.default_rng([design.visual_seed, design.data_seed])
    model = V.train_vision(
        walks,
        goals,
        rng,
        negatives=DETECTOR_NEGATIVES,
        epochs=VISION_EPOCHS,
        batch=VISION_BATCH,
        lr=VISION_LR,
        l2=VISION_L2,
        cell_dims=design.cell_dims,
        floor_dims=design.floor_dims,
    )
    budget.check("fit:vision")
    return model


class BoundedPlanner:
    """Frozen planner whose private per-frame caches are reset after PLANNER_SCENE_CAP distinct
    frames. Caches are pure functions of (frame, steps_left), so resets never change a plan."""

    def __init__(self, model: V.VisionModel, length_penalty: float, cap: int):
        self.model, self.length_penalty, self.cap = model, length_penalty, cap
        self._reset()

    def _reset(self) -> None:
        self._planner = P.Planner(self.model, self.length_penalty)
        self._keys: set[bytes] = set()

    def __call__(self, image: np.ndarray, goal: np.ndarray, steps_left: int):
        key = hashlib.blake2b(image.tobytes(), digest_size=16).digest()
        if key not in self._keys:
            if len(self._keys) >= self.cap:
                self._reset()
            self._keys.add(key)
        return self._planner(image, goal, steps_left)


# ---------------------------------------------------------------- anchors
def anchor_bank(design: FollowupDesign, bank: int, excluded: set[tuple]):
    """``n_anchors_per_seen_combo`` boards per SEEN combo from stream ``fit/anchor-<bank>``.
    The anchor glyph's partner and pad position come from the same named stream."""
    rng = bank_rng(design, f"fit/anchor-{bank}/glyphs")
    combos, pairs = [], []
    for combo in SEEN:
        for _ in range(design.n_anchors_per_seen_combo):
            others = [g for g in GLYPHS4 if g != combo[0]]
            partner = others[int(rng.integers(len(others)))]
            pair = (combo[0], partner) if rng.integers(2) == 0 else (partner, combo[0])
            combos.append(combo)
            pairs.append(pair)
    boards = fit_boards(design, f"anchor-{bank}", pairs, excluded)
    return list(zip(combos, boards, strict=True))


# ---------------------------------------------------------------- text
class BagOfWords:
    """Counts over the fixed arm-independent vocabulary; out-of-vocabulary words are dropped."""

    def __init__(self, vocab: tuple[str, ...]):
        self.index = {w: i for i, w in enumerate(vocab)}

    def __call__(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), len(self.index)))
        for r, text in enumerate(texts):
            for w in text.split():
                if w in self.index:
                    x[r, self.index[w]] += 1
        return x


def fit_text(design: FollowupDesign, bow: BagOfWords, corpus: int, arm: str) -> E.Params:
    counts = Counter((p.left, p.right) for p in make_corpus(design, corpus, arm))
    keys = sorted(counts)
    w = np.array([counts[k] for k in keys], dtype=float)
    init = E.init_params(
        np.random.default_rng([corpus, TEXT_INIT_STREAM]),
        len(bow.index),
        design.text_hidden,
        design.text_out,
    )
    params, _ = E.train(
        init,
        bow([k[0] for k in keys]),
        bow([k[1] for k in keys]),
        w / w.sum(),
        design.text_epochs,
        design.text_lr,
    )
    return params


# ---------------------------------------------------------------- entry point
def fit_components(design: FollowupDesign, budget: Budget) -> Components:
    """Fit every frozen component from ``fit``-role topologies and the text corpora only."""
    excluded: set[tuple] = set()  # shared: vision and anchor banks get distinct topologies
    model = fit_vision(design, excluded, budget)
    planner = BoundedPlanner(model, LENGTH_PENALTY, PLANNER_SCENE_CAP)
    anchors = {}
    for bank in design.anchor_banks:
        anchors[bank] = anchor_bank(design, bank, excluded)
        budget.check(f"fit:anchors-{bank}")
    bow = BagOfWords(vocabulary())
    text = {}
    for corpus in design.corpus_seeds:
        for arm in ARMS:
            text[(corpus, arm)] = fit_text(design, bow, corpus, arm)
            budget.check(f"fit:text-{corpus}-{arm}")

    def encode(corpus: int, arm: str, texts: list[str]) -> np.ndarray:
        return E.forward(text[(corpus, arm)], bow(texts))[0]

    def anchor_boards(bank: int):
        return list(anchors[bank])

    return Components(
        vision=model, planner=planner, encode=encode, raw=bow, anchor_boards=anchor_boards
    )
