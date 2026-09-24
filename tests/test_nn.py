import numpy as np
import pytest

from gradlens import (
    Tensor, MLP, Linear, Sequential, ReLU, Tanh,
    mse_loss, binary_cross_entropy, cross_entropy,
    SGD, Adam, no_grad,
)


def spiral_data(n_points=100, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)
    for c in range(n_classes):
        start, end = c * n_points, (c + 1) * n_points
        r = np.linspace(0.0, 1.0, n_points)
        t = (c * 4 / n_classes + 4 * r) + rng.standard_normal(n_points) * 0.2
        X[start:end] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[start:end] = c
    return X, y


# ------------------------------------------------------------------ modules

def test_module_parameter_discovery():
    model = MLP(2, [8], 3)
    params = model.parameters()
    shapes = sorted(p.shape for p in params)
    # two Linear layers: W(2,8) b(8,) W(8,3) b(3,)
    assert shapes == [(2, 8), (3,), (8,), (8, 3)]
    assert all(p.requires_grad for p in params)


def test_linear_forward_shape_and_default_bias():
    layer = Linear(4, 3)
    assert layer.bias is not None and layer.bias.shape == (3,)
    out = layer(Tensor(np.random.randn(10, 4)))
    assert out.shape == (10, 3)


def test_linear_without_bias():
    layer = Linear(4, 3, bias=False)
    x = np.random.randn(2, 4)
    out = layer(Tensor(x))
    assert layer.bias is None
    assert len(layer.parameters()) == 1
    assert np.allclose(out.data, x @ layer.weight.data)


def test_sequential_repr_and_forward():
    net = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
    out = net(Tensor(np.random.randn(5, 2)))
    assert out.shape == (5, 1)
    assert "Linear" in repr(net)


# ------------------------------------------------------------------ losses

def test_mse_loss_value_and_grad():
    pred = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    target = Tensor(np.array([0.0, 2.0]))
    loss = mse_loss(pred, target)
    assert np.isclose(loss.item(), 0.5)
    loss.backward()
    assert np.allclose(pred.grad, [1.0, 0.0])


def test_cross_entropy_matches_manual_computation():
    logits = Tensor(np.array([[2.0, 0.0], [0.0, 3.0]]), requires_grad=True)
    targets = np.array([0, 1])
    loss = cross_entropy(logits, targets)
    z = np.array([[2.0, 0.0], [0.0, 3.0]])
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    expected = -np.mean([logp[0, 0], logp[1, 1]])
    assert np.isclose(loss.item(), expected)
    loss.backward()
    # softmax - onehot, divided by N
    p = np.exp(logp)
    expected_grad = (p - np.eye(2)[targets]) / 2
    assert np.allclose(logits.grad, expected_grad)


def test_cross_entropy_numerical_stability_with_huge_logits():
    logits = Tensor(np.array([[1000.0, -1000.0], [500.0, 1000.0]]), requires_grad=True)
    loss = cross_entropy(logits, np.array([0, 1]))
    assert np.isfinite(loss.item())
    loss.backward()
    assert np.all(np.isfinite(logits.grad))


def test_binary_cross_entropy_value():
    z = 0.0
    expected = np.log(2.0)
    logits = Tensor(np.array([z]), requires_grad=True)
    target = Tensor(np.array([1.0]))
    assert np.isclose(binary_cross_entropy(logits, target).item(), expected)


def test_bce_stability_with_large_logits():
    logits = Tensor(np.array([800.0, -800.0]), requires_grad=True)
    target = Tensor(np.array([1.0, 0.0]))
    loss = binary_cross_entropy(logits, target)
    assert np.isfinite(loss.item())
    loss.backward()
    assert np.all(np.isfinite(logits.grad))


# ------------------------------------------------------------------ optimizers

def test_sgd_descends_on_quadratic():
    x = Tensor(np.array([10.0]), requires_grad=True)
    opt = SGD([x], lr=0.1)
    for _ in range(50):
        opt.zero_grad()
        (x * x).sum().backward()
        opt.step()
    assert abs(x.item()) < 1e-3


def test_sgd_momentum_descends():
    x = Tensor(np.array([10.0]), requires_grad=True)
    opt = SGD([x], lr=0.01, momentum=0.9)
    for _ in range(200):
        opt.zero_grad()
        (x * x).sum().backward()
        opt.step()
    assert abs(x.item()) < 1e-2


def test_adam_descends_fast():
    x = Tensor(np.array([10.0]), requires_grad=True)
    opt = Adam([x], lr=0.1)
    first = x.item()
    for _ in range(500):
        opt.zero_grad()
        (x * x).sum().backward()
        opt.step()
    # Adam's long second-moment memory (b2=0.999) slows the tail on a
    # quadratically shrinking gradient; 10 -> O(0.1) is healthy descent.
    assert abs(x.item()) < 0.5 < abs(first)


# ------------------------------------------------------------------ end-to-end

def test_mlp_learns_spiral():
    np.random.seed(42)
    X, y = spiral_data(60)
    model = MLP(2, [32, 32], 3, act=Tanh())
    opt = Adam(model.parameters(), lr=0.05)

    first = None
    for _ in range(150):
        logits = model(Tensor(X))
        loss = cross_entropy(logits, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    logits = model(Tensor(X))
    acc = float(np.mean(np.argmax(logits.data, axis=1) == y))
    assert first > 0.9
    assert acc > 0.9, f"accuracy {acc:.2f} should exceed 0.9 after training"


def test_no_grad_inference_skip_graph():
    model = MLP(2, [4], 2)
    with no_grad():
        out = model(Tensor(np.random.randn(3, 2)))
    assert out._prev == ()
    assert out.requires_grad is False
