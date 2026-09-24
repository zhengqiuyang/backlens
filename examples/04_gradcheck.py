"""Trust, but verify: finite-difference gradient checking.

Gradcheck compares every analytic gradient from backward() against central
differences. Use it to verify the engine, your custom loss, or your intuition
about a derivative.

Run:  python examples/04_gradcheck.py
"""

import numpy as np

from backlens import Tensor, MLP, Tanh, cross_entropy
from backlens.check import gradcheck


def main():
    rng = np.random.default_rng(0)

    # 1. a hand-rolled function -- check your calculus
    x = Tensor(rng.standard_normal((3, 3)) + 2, requires_grad=True, label="x")
    result = gradcheck(lambda t: (t * t).tanh().sum() + t.exp().mean(), x)
    print(result.report())

    # 2. the same machinery verifies the whole MLP + cross-entropy pipeline
    model = MLP(2, [4], 3, act=Tanh())
    xs = rng.standard_normal((5, 2))
    ys = rng.integers(0, 3, 5)

    def loss_on(w1, w2):
        # rewire the model to the two tensors gradcheck perturbs
        model.net.layers[0].weight = w1
        model.net.layers[2].weight = w2
        return cross_entropy(model(Tensor(xs)), ys)

    w1 = Tensor(model.net.layers[0].weight.data.copy(), requires_grad=True, label="layer1.W")
    w2 = Tensor(model.net.layers[2].weight.data.copy(), requires_grad=True, label="layer2.W")
    result = gradcheck(loss_on, w1, w2)
    print()
    print(result.report())

    # 3. and it FAILS loudly when gradients are wrong (hook sabotage: x2)
    bad = Tensor(np.array([1.0, 2.0]), requires_grad=True, label="sabotaged")
    bad.register_hook(lambda g: g * 2.0)
    result = gradcheck(lambda t: (t * t).sum(), bad)
    print()
    print(result.report())


if __name__ == "__main__":
    main()
