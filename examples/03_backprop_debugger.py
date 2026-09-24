"""The headline demo: treat backpropagation like something you can debug.

Part A -- step_backward(): advance the backward pass ONE op at a time and
watch gradients appear in topological order.

Part B -- a training run that goes NaN (squared logits + big lr). PyTorch
would hand you a dead loss and a stack trace that points at the framework;
BackLens raises GradientAnomalyError with the exact op where the gradient
first became NaN, plus the full backward trace.

Run:  python examples/03_backprop_debugger.py
"""

import numpy as np

from backlens import Tensor, MLP, Tanh, cross_entropy, Adam
from backlens.debug import step_backward, debug_backward, GradientAnomalyError


def part_a_step_through_backward():
    print("=" * 68)
    print("A. step_backward(): the backward pass, one op at a time")
    print("=" * 68)

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((4, 3)), label="batch")
    model = MLP(3, [4], 2, act=Tanh())
    logits = model(x)
    loss = cross_entropy(logits, np.array([0, 1, 0, 1]))

    print(f"loss = {loss.item():.6f} -- now stepping through backward:\n")
    for i, node in enumerate(step_backward(loss)):
        g = np.linalg.norm(node.grad.ravel())
        name = node.label or f"op:{node._op}"
        print(f"  step {i:2d}  {name:<24} shape={str(node.shape):<8} |grad|={g:.4f}")
    print("\n(each line is one op's backward closure, executed in topo order)")


def part_b_catch_the_nan():
    print()
    print("=" * 68)
    print("B. a training run that goes NaN -- caught and located")
    print("=" * 68)
    print("the bug: a hacky loss on cubed logits + a big learning rate.")
    print("watch max|grad| blow up, then training die:\n")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 2)) * 3
    y = (x[:, 0] * x[:, 1] > 0).astype(int)

    model = MLP(2, [32], 2, act=Tanh())
    opt = Adam(model.parameters(), lr=0.5)          # too hot

    for step in range(100):
        logits = model(Tensor(x))
        loss = cross_entropy((logits * logits * logits), y)
        opt.zero_grad()
        trace = debug_backward(loss, raise_on_anomaly=False)
        max_g = max(s.grad_norm for s in trace)
        print(f"  step {step:3d}   loss {loss.item():>12.4e}   max|g| {max_g:>10.3e}")
        if trace.first_anomaly() is not None:
            err = GradientAnomalyError(trace)
            print("\n" + "!" * 68)
            print(err)
            print("!" * 68)
            print("\nPyTorch would show you a nan loss and shrug.")
            print("BackLens shows the op where the gradient first broke,")
            print("and everything contaminated downstream.")
            return
        opt.step()
    print("\nno anomaly happened (tune lr up if you want the drama)")


if __name__ == "__main__":
    part_a_step_through_backward()
    part_b_catch_the_nan()
