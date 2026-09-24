"""Train an MLP on the 3-class spiral dataset -- the classic hard-2D toy.

Full training of a 2-32-32-3 tanh network in a few seconds on CPU, plus:
  * a live ASCII decision boundary in the terminal (no matplotlib needed),
  * per-parameter gradient norms across training via GradientMonitor.

Run:  python examples/02_spiral_classifier.py
"""

import numpy as np

from gradlens import Tensor, MLP, Tanh, cross_entropy, Adam
from gradlens.debug import GradientMonitor


def spiral(n=120, classes=3, seed=1):
    rng = np.random.default_rng(seed)
    X = np.zeros((n * classes, 2))
    y = np.zeros(n * classes, dtype=int)
    for c in range(classes):
        sl = slice(c * n, (c + 1) * n)
        r = np.linspace(0.0, 1.0, n)
        t = (c * 4 / classes + 4 * r) + rng.standard_normal(n) * 0.2
        X[sl] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[sl] = c
    return X, y


def ascii_boundary(model, n_grid=32):
    """Classify a grid and render it as ASCII characters."""
    xs = np.linspace(-1.2, 1.2, n_grid)
    grid = np.array([[a, b] for b in xs for a in xs])
    z = model(Tensor(grid)).data.reshape(n_grid, n_grid, -1).argmax(axis=2)
    chars = "123"
    return "\n".join("".join(chars[c] for c in row) for row in z)


def main():
    np.random.seed(42)
    X, y = spiral()

    model = MLP(2, [32, 32], 3, act=Tanh())
    opt = Adam(model.parameters(), lr=0.05)
    monitor = GradientMonitor(model)          # hooks: no training-loop changes

    for epoch in range(201):
        logits = model(Tensor(X))
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        monitor.tick()
        opt.step()

        if epoch % 40 == 0:
            acc = float(np.mean(logits.data.argmax(1) == y))
            print(f"epoch {epoch:3d}   loss {loss.item():.4f}   acc {acc:.2f}")

    acc = float(np.mean(model(Tensor(X)).data.argmax(1) == y))
    print(f"\nfinal accuracy: {acc:.2%}")

    print("\ndecision boundary (1/2/3 = predicted class):")
    print(ascii_boundary(model))

    print("\ngradient norms over training:")
    print(monitor.summary())


if __name__ == "__main__":
    main()
