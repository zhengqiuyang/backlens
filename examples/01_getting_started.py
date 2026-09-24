"""Five-minute tour: forward pass, backward pass, gradients, graph export.

Run:  python examples/01_getting_started.py
"""

import numpy as np

from gradlens import Tensor, no_grad
from gradlens.debug import debug_backward
from gradlens.viz import save_graph_md, save_graph_html


def main():
    # ------------------------------------------------------------------
    # 1. a small differentiable expression, in plain numpy-backed tensors
    # ------------------------------------------------------------------
    x = Tensor([2.0, 3.0], label="x")
    w = Tensor([0.5, -1.0], requires_grad=True, label="w")
    b = Tensor(0.1, requires_grad=True, label="b")

    # y = tanh(x*w + b); loss = sum(y^2) + w.w
    y = (x * w + b).tanh()
    loss = (y * y).sum() + (w * w).sum()
    print(f"loss = {loss.item():.6f}")

    # ------------------------------------------------------------------
    # 2. backward -- and the whole backward pass, recorded
    # ------------------------------------------------------------------
    trace = debug_backward(loss)          # runs loss.backward() + records
    print("\nbackward pass, loss-first order:")
    print(trace.report())

    print(f"\ndL/dw = {w.grad}   dL/db = {b.grad}")

    # ------------------------------------------------------------------
    # 3. inspect any step of the backward pass, like stack frames
    # ------------------------------------------------------------------
    print(f"\nthe 2nd op to receive its gradient was: {trace[1].op}"
          f" (|grad| = {trace[1].grad_norm:.4g})")

    # ------------------------------------------------------------------
    # 4. export the computation graph; the .md renders natively on GitHub
    # ------------------------------------------------------------------
    save_graph_md(loss, "graph.md", direction="LR")
    save_graph_html(loss, "graph.html", direction="LR")
    print("\nwrote graph.md (paste into any GitHub README/issue) and graph.html")

    # ------------------------------------------------------------------
    # 5. no_grad zones behave like PyTorch's
    # ------------------------------------------------------------------
    with no_grad():
        inference = (x * w + b).tanh()
    assert not inference.requires_grad
    print("no_grad inference works: requires_grad =", inference.requires_grad)


if __name__ == "__main__":
    main()
