"""Frozen planner in the learned factored visual state.

Inputs are the current rendered image, a goal vector in the outcome-embedding space and the
number of real steps left. Nothing else: no layout, simulator, oracle, future frame or state id
is reachable from here (this module imports only numpy and ``pgt.vision``; tested).

Search: a breadth-first beam over imagined agent cells, de-duplicated by imagined state (at most
49 states, all 4 actions expanded per level, depth <= steps left, so at most 12). Imagination
stops at the first cell the learned terminal detector flags; that imagined first terminal is the
sequence's outcome, so a route can never be credited for reaching the goal after a wrong pad.
Non-terminal imagined end cells are off-pad (timeout) outcomes. Score = squared distance between
the predicted outcome embedding and the goal + ``length_penalty`` x imagined steps; ties go to the
earliest-found state and lowest action index. The first action of the best sequence is executed
in the real environment and the planner is re-run on the next real frame (receding horizon).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pgt import vision as V


@dataclass(frozen=True)
class Scene:
    """Learned factored state of one frame: agent cell + static per-cell predictions."""

    agent: int
    nxt: np.ndarray  # (49, 4) predicted next cell per action
    terminal: np.ndarray  # (49,) predicted: entering this cell ends the episode
    outcome: np.ndarray  # (49, k) predicted outcome embedding if the episode ends here


def perceive(model: V.VisionModel, image: np.ndarray) -> Scene:
    p = V.patches(image)
    agent = int(model.agent_cell(p))
    cells = np.repeat(np.arange(V.CELLS), 4)
    actions = np.tile(np.arange(4), V.CELLS)
    frames = np.broadcast_to(p, (len(cells), *p.shape))  # same frame for every (cell, action)
    nxt = V.predict_next_cell(model, frames, cells, actions).reshape(V.CELLS, 4)
    rings = V.ring(p)
    terminal = model.terminal.prob(rings) > 0.5
    total = p.sum(axis=0)
    floors = np.empty((V.CELLS, p.shape[1]), dtype=np.float64)
    for c in range(V.CELLS):  # floor elsewhere = mean over cells other than agent and outcome
        excl = {agent, c}
        m = (total - sum(p[e] for e in excl)) / (V.CELLS - len(excl))
        floors[c] = m - m.mean()
    return Scene(agent, nxt, terminal, model.embed(rings, floors))


@dataclass(frozen=True)
class Plan:
    action: int
    cell: int  # imagined outcome cell
    terminal: bool  # imagined outcome is a predicted pad (else off-pad / timeout)
    depth: int  # imagined steps to the outcome
    outcome: np.ndarray  # predicted outcome embedding
    score: float


@dataclass(frozen=True)
class Reach:
    depth: np.ndarray  # (49,) first imagined arrival depth, -1 if unreached
    first: np.ndarray  # (49,) first action of the earliest sequence reaching the cell


def imagine(scene: Scene, steps_left: int) -> Reach:
    """Beam over imagined states; terminal cells absorb (first terminal ends imagination)."""
    depth = np.full(V.CELLS, -1)
    first = np.full(V.CELLS, -1)
    frontier = []
    for a in range(4):
        c = int(scene.nxt[scene.agent, a])
        if depth[c] < 0:
            depth[c], first[c] = 1, a
            frontier.append(c)
    for d in range(2, steps_left + 1):
        live = [c for c in frontier if not scene.terminal[c]]
        if not live:
            break
        succ = scene.nxt[live]  # (B, 4), expanded in batch
        frontier = []
        for i, c in enumerate(live):
            for s in succ[i]:
                if depth[s] < 0:
                    depth[s], first[s] = d, first[c]
                    frontier.append(int(s))
    return Reach(depth, first)


def choose(scene: Scene, reach: Reach, goal: np.ndarray, length_penalty: float) -> Plan:
    cand = np.flatnonzero(reach.depth > 0)
    dist = ((scene.outcome[cand] - goal) ** 2).sum(axis=1)
    score = dist + length_penalty * reach.depth[cand]
    best = cand[np.lexsort((cand, reach.first[cand], reach.depth[cand], score))[0]]
    return Plan(
        int(reach.first[best]),
        int(best),
        bool(scene.terminal[best]),
        int(reach.depth[best]),
        scene.outcome[best],
        float(score.min()),
    )


class Planner:
    """Caches perception and imagination per (frame, steps left); goals only change scoring."""

    def __init__(self, model: V.VisionModel, length_penalty: float):
        self.model = model
        self.length_penalty = length_penalty
        self._scenes: dict[bytes, Scene] = {}
        self._reach: dict[tuple[bytes, int], Reach] = {}

    def __call__(self, image: np.ndarray, goal: np.ndarray, steps_left: int) -> Plan:
        key = hashlib.blake2b(image.tobytes(), digest_size=16).digest()
        scene = self._scenes.get(key)
        if scene is None:
            scene = self._scenes[key] = perceive(self.model, image)
        reach = self._reach.get((key, steps_left))
        if reach is None:
            reach = self._reach[(key, steps_left)] = imagine(scene, steps_left)
        return choose(scene, reach, goal, self.length_penalty)
