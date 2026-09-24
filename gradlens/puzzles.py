"""Autodiff puzzles: learn backprop by debugging broken graphs.

Each puzzle builds a small network whose backward pass has been sabotaged by
a single corrupted gradient -- the kind of silent bug that makes real training
mysteriously stall. The forward pass is perfectly fine. Your job: find WHICH
backward step (index, in loss-first order) first carries the corrupted
gradient, then:

    from gradlens.puzzles import load_puzzle, list_puzzles
    p = load_puzzle("p1")
    print(p.story)
    print(p.diff())          # which parameter gradients disagree (your compass)
    ...use debug_backward / step_backward / gradcheck on p.loss...
    p.check(7)               # True / False

The sabotage is injected via tensor hooks (gradients rewritten as they flow),
and every puzzle is deterministic: the clean reference graph is rebuilt from
the same seed, so ``diff()`` always isolates the damage.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple

from .engine import Tensor
from .debug import debug_backward
from .nn import MLP, Tanh, mse_loss


class Puzzle:
    """One sabotaged graph. ``loss`` is live: backward it, film it, step it."""

    def __init__(self, pid: str, title: str, difficulty: str, story: str,
                 build: Callable[[bool], Tuple[Tensor, List[Tensor], Optional[Tensor]]]):
        self.id, self.title, self.difficulty, self.story = pid, title, difficulty, story
        self._build = build
        self.loss, self.params, culprit = build(False)
        self._ref_loss, self._ref_params = build(True)[:2]

        trace = debug_backward(self.loss, raise_on_anomaly=False)
        if culprit is None:
            raise ValueError(f"puzzle {pid}: sabotage missing in the dirty build")
        indices = [i for i, s in enumerate(trace) if s.node is culprit]
        self._culprit_index = indices[0] if indices else 0

    # -- API ----------------------------------------------------------------

    def check(self, answer: int) -> bool:
        """Is ``answer`` the backward-step index of the sabotaged gradient?"""
        return int(answer) == self._culprit_index

    @property
    def n_steps(self) -> int:
        return len(debug_backward(self.loss, raise_on_anomaly=False))

    @property
    def answer_length(self) -> int:
        """Hint: how many steps the backward pass has (answer is < this)."""
        return self.n_steps

    def diff(self) -> str:
        """Compare parameter gradients against the clean reference build.

        Names WHICH parameters end up wrong -- never WHICH op did it.
        That part is your job. (Builds two fresh graphs: gradients accumulate
        across backward calls, so reusing an old graph would double-count.)
        """
        live_loss, live_params, _ = self._build(False)
        ref_loss, ref_params, _ = self._build(True)
        ref_loss.backward()
        live_loss.backward()
        lines = [f"puzzle {self.id}: parameter gradients, sabotaged vs clean"]
        any_bad = False
        for p, q in zip(live_params, ref_params):
            delta = float(np.linalg.norm((p.grad - q.grad).ravel()))
            bad = not np.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-7)
            any_bad |= bad
            lines.append(f"  {p.label or 'param':<22} |diff|={delta:.3e}"
                         + ("   <-- WRONG" if bad else ""))
        if not any_bad:
            lines.append("  (no parameter disagrees?!)")
        return "\n".join(lines)

    def __repr__(self):
        return (f"Puzzle({self.id} '{self.title}' [{self.difficulty}], "
                f"{self.n_steps} backward steps)")


# ---------------------------------------------------------------------------
# puzzle builders: build(clean) -> (loss, params, culprit_or_None)
# clean=True rebuilds the identical network WITHOUT the sabotage hook
# ---------------------------------------------------------------------------

def _net_puzzle(pid, title, difficulty, story, corrupt, layer_idx,
                hidden, seed):
    """Generic deep-net puzzle: sabotage the gradient flowing INTO the
    output tensor of ``layer_idx`` (0-based among Linear layers)."""

    def build(clean: bool):
        np.random.seed(seed)                       # identical weights
        rng = np.random.default_rng(seed)          # identical data
        x = Tensor(rng.standard_normal((8, 2)), label="x")
        target = Tensor(rng.standard_normal((8, 1)), label="target")
        model = MLP(2, list(hidden), 1, act=Tanh())

        out = x
        culprit = None
        linears = [l for l in model.net.layers if hasattr(l, "weight")]
        seen = 0
        for layer in model.net.layers:
            out = layer(out)
            if hasattr(layer, "weight"):
                if seen == layer_idx and not clean:
                    culprit = out
                    culprit.register_hook(corrupt)
                seen += 1
        return mse_loss(out, target), model.parameters(), culprit

    return Puzzle(pid, title, difficulty, story, build)


def _nan_puzzle():
    def build(clean: bool):
        np.random.seed(4)
        rng = np.random.default_rng(4)
        x = Tensor(np.abs(rng.standard_normal((8, 2))) + 0.5, label="x")
        w = Tensor(rng.standard_normal((2, 4)) * 0.5, requires_grad=True, label="w")
        h = (x @ w).tanh()
        if not clean:
            h.register_hook(lambda g: g * float("nan"))
        v = Tensor(rng.standard_normal((4, 1)) * 0.5, requires_grad=True, label="v")
        loss = ((h @ v) ** 2).mean()
        return loss, [w, v], (None if clean else h)

    return Puzzle("p5", "The NaN Injector", "medium",
                  "loss is nan, yet nothing in the forward pass overflows and "
                  "no log(0) exists. The poison enters during BACKWARD, at "
                  "exactly one step. Find step zero of the outbreak.",
                  build)


_PUZZLE_MAKERS: List[Callable[[], Puzzle]] = [
    lambda: _net_puzzle(
        "p1", "The Halved Gradient", "easy",
        "loss falls, then stalls at exactly 2x the expected floor. "
        "Somewhere on the road from loss to weights, half of one gradient "
        "never arrives. Find the backward step where it leaks.",
        lambda g: g * 0.5, layer_idx=1, hidden=[8, 8], seed=0),
    lambda: _net_puzzle(
        "p2", "The Sign Flip", "easy",
        "training diverges no matter how small the learning rate. "
        "Exactly one gradient arrow points the wrong way. Walk the backward "
        "pass and find where the direction flips.",
        lambda g: -g, layer_idx=0, hidden=[10], seed=1),
    lambda: _net_puzzle(
        "p3", "The Vanishing Valve", "medium",
        "the last layer learns; every layer before it is frozen solid. "
        "A valve deep in the net multiplies one passing gradient by 1e-4.",
        lambda g: g * 1e-4, layer_idx=2, hidden=[16, 16, 16], seed=2),
    lambda: _net_puzzle(
        "p4", "The Exploding Amplifier", "medium",
        "a few good steps, then loss spikes to the sky, forever. "
        "Something amplifies one gradient by 100x.",
        lambda g: g * 100.0, layer_idx=0, hidden=[12, 12], seed=3),
    _nan_puzzle,
]


def load_puzzle(pid: str) -> Puzzle:
    """Build puzzle ``pid`` (``'p1'`` .. ``'p5'``)."""
    table = {f"p{i + 1}": make for i, make in enumerate(_PUZZLE_MAKERS)}
    if pid not in table:
        raise KeyError(f"unknown puzzle {pid!r}; available: {sorted(table)}")
    return table[pid]()


def list_puzzles() -> str:
    """One-line overview of every puzzle."""
    rows = []
    for i, make in enumerate(_PUZZLE_MAKERS, 1):
        p = make()
        rows.append(f"p{i}  [{p.difficulty:<6}] {p.title:<24} "
                    f"({p.n_steps} backward steps)")
    return "\n".join(rows)
