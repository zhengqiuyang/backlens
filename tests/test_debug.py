import numpy as np
import pytest

from backlens import Tensor, MLP, cross_entropy, Adam
from backlens.debug import (
    debug_backward, step_backward, BackwardTrace, GradientAnomalyError, GradientMonitor,
)


def build_loss(seed=0):
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((5, 3)), label="x")           # data leaf, no grad
    w = Tensor(rng.standard_normal((3, 2)) * 0.5, requires_grad=True, label="w")
    b = Tensor(np.zeros(2), requires_grad=True, label="b")
    h = (x @ w + b).tanh()
    v = Tensor(rng.standard_normal((2, 1)) * 0.5, requires_grad=True, label="v")
    loss = ((h @ v) ** 2).mean()
    return loss, {"x": x, "w": w, "b": b, "v": v}


# ------------------------------------------------------------------ trace

def test_trace_records_every_node_loss_first():
    loss, _ = build_loss()
    trace = debug_backward(loss, raise_on_anomaly=False)
    assert isinstance(trace, BackwardTrace)
    # every node that receives a gradient, in reverse-topo (loss-first) order
    expected = [n for n in reversed(loss._topo()) if n.requires_grad or n is loss]
    assert len(trace) == len(expected)
    assert trace[0].op == "mean"                    # loss node itself first
    ops = [s.op for s in trace]
    assert ops == [n._op or "leaf" for n in expected]


def test_trace_steps_have_grads_and_norms():
    loss, _ = build_loss()
    trace = debug_backward(loss)
    for s in trace:
        assert s.grad_norm >= 0.0
        assert s.shape == s.node.shape
    assert not trace.first_anomaly()


def test_trace_report_renders_table():
    loss, _ = build_loss()
    text = debug_backward(loss).report()
    assert "step" in text and "op" in text and "|grad|" in text
    assert "mean" in text and "matmul" in text and "tanh" in text


# ------------------------------------------------------------------ anomaly

def make_nan_loss():
    # log of zero -> -inf -> * 0 -> nan on the backward path
    x = Tensor(np.array([1.0, 0.0]), requires_grad=True, label="bad_x")
    y = (x.log() * Tensor(np.array([1.0, 0.0]))).sum()
    return y, x


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_anomaly_error_raised_with_trace():
    loss, _ = make_nan_loss()
    with pytest.raises(GradientAnomalyError) as info:
        debug_backward(loss)
    err = info.value
    assert err.trace.first_anomaly() is not None
    assert "first anomaly" in str(err)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_anomaly_can_be_downgraded_to_report():
    loss, _ = make_nan_loss()
    trace = debug_backward(loss, raise_on_anomaly=False)
    first = trace.first_anomaly()
    assert first is not None
    assert any("NaN" in n or "Inf" in n for n in first.notes)


def test_exploding_grad_flagged():
    w = Tensor(np.array([1e3]), requires_grad=True)
    trace = debug_backward(((w * 1e3) ** 3).sum(), raise_on_anomaly=False)
    assert any("exploding" in n for s in trace for n in s.notes)


def test_vanishing_grad_flagged():
    w = Tensor(np.array([1e-4]), requires_grad=True)
    trace = debug_backward((w * w * w).sum(), raise_on_anomaly=False)
    assert any("vanishing" in n for s in trace for n in s.notes)


# ------------------------------------------------------------------ stepper

def test_step_backward_yields_same_grads_as_plain_backward():
    loss1, _ = build_loss(seed=3)
    loss2, _ = build_loss(seed=3)
    loss1.backward()
    seen = [n for n in step_backward(loss2)]
    assert all(n.grad is not None for n in seen)
    # compare grads on the parameter leaves
    for a, b in zip(loss1._topo(), loss2._topo()):
        if a.grad is not None:
            assert np.allclose(a.grad, b.grad)


def test_step_backward_one_op_at_a_time():
    loss, _ = build_loss()
    gen = step_backward(loss)
    first = next(gen)
    assert first is loss or first._op == "mean"
    # loss.grad must be seeded before the first yield
    assert loss.grad is not None


def test_step_backward_requires_scalar():
    x = Tensor(np.ones(3))
    with pytest.raises(RuntimeError):
        list(step_backward(x * 2.0))


# ------------------------------------------------------------------ monitor

def test_gradient_monitor_records_history():
    np.random.seed(7)
    model = MLP(2, [8], 3)
    opt = Adam(model.parameters(), lr=0.05)
    mon = GradientMonitor(model)

    X = np.random.randn(20, 2)
    y = np.random.randint(0, 3, 20)
    for _ in range(5):
        loss = cross_entropy(model(Tensor(X)), y)
        opt.zero_grad()
        loss.backward()
        mon.tick()
        opt.step()

    assert set(mon.history).issuperset({"Linear(2,8).W", "Linear(8,3).W"})
    for series in mon.history.values():
        assert len(series) == 5
    text = mon.summary()
    assert "last |g|" in text and "Linear(2,8).W" in text
