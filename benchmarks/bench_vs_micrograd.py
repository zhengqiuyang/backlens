"""Head-to-head benchmark: GradLens (numpy-vectorized) vs micrograd (scalar).

Both engines train the SAME 2-16-16-1 relu MLP with the same MSE loss, the
same initialization, and the same full-batch SGD steps, on the same synthetic
dataset -- micrograd one scalar ``Value`` at a time, GradLens on whole arrays.

(The pip release of micrograd is 0.1.0, whose ``Value`` predates ``exp``/``log``,
so the benchmark uses MSE -- expressible with both engines' primitives.)

Requires:  pip install micrograd
Run:       python benchmarks/bench_vs_micrograd.py
"""

import time

import numpy as np

from gradlens import Tensor, MLP, ReLU, mse_loss, SGD

try:
    from micrograd.engine import Value
    from micrograd.nn import MLP as MicroMLP
except ImportError:
    raise SystemExit("pip install micrograd  (or: pip install gradlens[dev])")


def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = (x[:, 0] * x[:, 1] > 0).astype(np.float64)   # XOR, the classic
    return x, y


def init_common(sizes, seed=3):
    """One shared weight dump: list of (W (nin,nout), b (nout,))."""
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal((a, b)) * 0.5, rng.standard_normal(b) * 0.1)
        for a, b in zip(sizes[:-1], sizes[1:])
    ]


def build_gradlens(sizes, init):
    # micrograd's hidden neurons are relu, last layer linear -- mirror that
    model = MLP(sizes[0], list(sizes[1:-1]), sizes[-1], act=ReLU())
    linears = [l for l in model.net.layers if hasattr(l, "weight")]
    for l, (w, b) in zip(linears, init):
        l.weight.data = w.copy()
        l.bias.data = b.copy()
    return model


def build_micrograd(sizes, init):
    model = MicroMLP(sizes[0], sizes[1:])
    # micrograd: layer = nout Neurons, each with nin weights (row-major W[i,j])
    for layer, (w, b) in zip(model.layers, init):
        for j, neuron in enumerate(layer.neurons):
            for i, wi in enumerate(neuron.w):
                wi.data = float(w[i, j])
            neuron.b.data = float(b[j])
    return model


def run_gradlens(x, y, steps, lr, sizes):
    model = build_gradlens(sizes, init_common(sizes))
    opt = SGD(model.parameters(), lr=lr)
    xs, ys = Tensor(x), Tensor(y.reshape(-1, 1))
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = mse_loss(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.perf_counter() - t0, float(loss.item())


def run_micrograd(x, y, steps, lr, sizes):
    model = build_micrograd(sizes, init_common(sizes))
    xs = [list(map(Value, row)) for row in x]
    ys = [float(v) for v in y]
    params = model.parameters()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = sum((model(xi) - t) ** 2 for xi, t in zip(xs, ys)) * (1.0 / len(xs))
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
    return time.perf_counter() - t0, float(loss.data)


def main():
    x, y = make_data()
    sizes = [2, 16, 16, 1]
    steps, lr = 50, 0.05

    gl_time, gl_loss = run_gradlens(x, y, steps, lr, sizes)
    print(f"gradlens : {steps} steps in {gl_time:7.3f}s   final loss {gl_loss:.4f}")

    mg_time, mg_loss = run_micrograd(x, y, steps, lr, sizes)
    print(f"micrograd: {steps} steps in {mg_time:7.3f}s   final loss {mg_loss:.4f}")

    print(f"\nspeedup  : {mg_time / gl_time:6.1f}x   (same net, same init, same optimizer)")
    if gl_loss == mg_loss:
        print("note: identical final losses -- both engines compute identical math")


if __name__ == "__main__":
    main()
