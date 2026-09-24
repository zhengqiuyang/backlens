import numpy as np
import pytest

from backlens import Tensor, no_grad, is_grad_enabled, set_op_hook, clear_op_hooks
from backlens.check import gradcheck


def make(seed=0, *shapes):
    rng = np.random.default_rng(seed)
    return [Tensor(rng.standard_normal(s) + 0.5, requires_grad=True, label=f"x{i}")
            for i, s in enumerate(shapes)]


# --------------------------------------------------------------- arithmetic

def test_add_sub_mul_div_grads():
    a, b = make(1, (3,), (3,))
    out = ((a * b) + (a / b) - b).sum()
    out.backward()
    assert np.allclose(a.grad, b.data + 1 / b.data)
    assert np.allclose(b.grad, a.data - a.data / b.data**2 - 1)


def test_pow_grad():
    x, = make(2, (4,))
    (x ** 3).sum().backward()
    assert np.allclose(x.grad, 3 * x.data**2)


def test_scalar_broadcasting_add():
    x, = make(3, (2, 3))
    (x + 5.0).sum().backward()
    assert np.allclose(x.grad, 1.0)


def test_vector_broadcasting_grads():
    # (3,2) * (2,) -> grad of the (2,) operand must sum over the batch
    a, b = make(4, (3, 2), (2,))
    (a * b).sum().backward()
    assert np.allclose(a.grad, np.tile(b.data, (3, 1)))
    assert np.allclose(b.grad, a.data.sum(axis=0))


def test_matmul_grads_2d():
    a, b = make(5, (4, 3), (3, 5))
    (a @ b).sum().backward()
    assert np.allclose(a.grad, np.ones((4, 5)) @ b.data.T)
    assert np.allclose(b.grad, a.data.T @ np.ones((4, 5)))


def test_matmul_vector_cases():
    v, W = make(6, (3,), (3, 2))
    out = (v @ W).sum()
    out.backward()
    assert np.allclose(v.grad, np.ones(2) @ W.data.T)
    assert np.allclose(W.grad, np.outer(v.data, np.ones(2)))

    W2, u = make(7, (2, 3), (3,))
    (W2 @ u).sum().backward()
    assert np.allclose(W2.grad, np.outer(np.ones(2), u.data))
    assert np.allclose(u.grad, W2.data.T @ np.ones(2))

    p, q = make(8, (4,), (4,))
    (p @ q).backward()
    assert np.allclose(p.grad, q.data)
    assert np.allclose(q.grad, p.data)


# --------------------------------------------------------------- functions

@pytest.mark.parametrize("build", [
    lambda x: x.exp().sum(),
    lambda x: x.tanh().sum(),
    lambda x: x.relu().sum(),
    lambda x: x.sigmoid().sum(),
    lambda x: (x * x).sum() + x.mean(),
    lambda x: x.abs().sum(),
])
def test_elementwise_gradcheck(build):
    x, = make(9, (3, 4))
    assert gradcheck(build, x), build


def test_log_gradcheck_with_positive_input():
    x = Tensor(np.abs(np.random.default_rng(9).standard_normal((3, 2))) + 0.5,
               requires_grad=True)
    assert gradcheck(lambda t: t.log().sum(), x)


def test_relu_zero_grad_at_negatives():
    x = Tensor(np.array([-2.0, -0.0, 3.0]), requires_grad=True)
    x.relu().sum().backward()
    assert np.allclose(x.grad, [0.0, 0.0, 1.0])


# --------------------------------------------------------------- reductions

def test_sum_axis_gradcheck():
    x, = make(10, (3, 4))
    assert gradcheck(lambda x: (x * x).sum(axis=0).sum() + x.mean(), x)


def test_sum_keepdims_axis_gradcheck():
    x, = make(11, (2, 3))
    assert gradcheck(lambda x: (x.tanh().sum(axis=1, keepdims=True) * 2.0).sum(), x)


def test_max_grad_routes_to_argmax():
    x = Tensor(np.array([[1.0, 5.0], [3.0, 2.0]]), requires_grad=True)
    (x.max() * 3.0).backward()
    assert np.allclose(x.grad, [[0, 3], [0, 0]])


# --------------------------------------------------------------- shape ops

def test_reshape_transpose_grads():
    x, = make(12, (2, 3))
    out = x.reshape(3, 2).T.sum()
    out.backward()
    assert np.allclose(x.grad, np.ones((2, 3)))


def test_grad_accumulates_across_backward_calls():
    x, = make(13, (2,))
    for _ in range(3):
        (x * 2.0).sum().backward()
    assert np.allclose(x.grad, [6.0, 6.0])


def test_zero_grad_and_detach():
    x, = make(14, (2,))
    y = x * 3.0
    y.sum().backward()
    x.zero_grad()
    assert x.grad is None
    d = y.detach()
    assert d._prev == () and d.requires_grad is False


# --------------------------------------------------------------- no_grad

def test_no_grad_context_and_decorator():
    x, = make(15, (2,))
    with no_grad():
        y = x * 2.0
    assert y._prev == () and y.requires_grad is False

    @no_grad
    def f(t):
        return t + 1.0
    assert f(x).requires_grad is False
    assert is_grad_enabled()


def test_grad_disabled_between_uses():
    x, = make(16, (2,))
    with no_grad():
        _ = x * 2.0
    y = x * 2.0
    assert y.requires_grad


# --------------------------------------------------------------- hooks

def test_tensor_hook_can_rewrite_grad():
    x, = make(17, (3,))
    x.register_hook(lambda g: g * 0.0)
    (x * x).sum().backward()
    assert np.allclose(x.grad, 0.0)


def test_op_hook_fires_for_matching_op():
    x, = make(18, (3,))
    calls = []
    set_op_hook("tanh", lambda g: calls.append(float(np.sum(g))) or None)
    try:
        x.tanh().sum().backward()
    finally:
        clear_op_hooks()
    assert len(calls) == 1


# --------------------------------------------------------------- misc

def test_backward_requires_scalar():
    x, = make(19, (3,))
    with pytest.raises(RuntimeError):
        (x * 2.0).backward()


def test_trace_matches_expected_ops():
    a, b = make(20, (2,), (2,))
    loss = ((a * b).tanh() + 1.0).sum()
    trace = loss.backward(return_trace=True)
    ops = [n._op or "leaf" for n in trace]
    # loss-first order: sum -> add -> tanh -> mul -> leaves
    assert ops[0] == "sum" and ops[1] == "add" and ops[2] == "tanh" and ops[3] == "mul"
    assert ops.count("leaf") >= 2


def test_labels_and_repr():
    x = Tensor([1.0, 2.0], label="demo")
    assert "demo" in repr(x)
    assert x.shape == (2,) and x.item is not None
