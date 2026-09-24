"""Tests for the 0.4 unique-feature set: ONNX export, training gradient
audit HTML, autodiff puzzles, and the new Conv2d/MaxPool2d modules."""

import json
import os

import numpy as np
import pytest

from gradlens import (
    Tensor, MLP, Tanh, ReLU, Sequential, Linear, Conv2d, MaxPool2d, Flatten,
    SGD, Adam, mse_loss,
)
from gradlens.debug import GradientMonitor
from gradlens.puzzles import list_puzzles, load_puzzle

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


# --------------------------------------------------------------- nn modules

def test_conv2d_maxpool2d_modules():
    model = Sequential(Conv2d(1, 4, 3, pad=1), ReLU(), MaxPool2d(2, 2),
                       Conv2d(4, 8, 3), ReLU(), Flatten(), Linear(8 * 5 * 5, 3))
    params = model.parameters()
    assert len(params) == 6                       # 2 convs + 1 linear, W+b each
    x = Tensor(np.random.default_rng(0).standard_normal((2, 1, 14, 14)))
    assert model(x).shape == (2, 3)

    opt = SGD(params, lr=0.05)
    target = Tensor(np.random.default_rng(1).standard_normal((2, 3)))
    loss = mse_loss(model(x), target)
    loss.backward()
    for p in params:
        assert p.grad is not None and np.all(np.isfinite(p.grad))
    opt.step()


def test_conv2d_repr_and_no_bias():
    c = Conv2d(3, 6, 5, stride=2, bias=False)
    assert "bias=False" in repr(c) and c.bias is None
    assert len(c.parameters()) == 1


# --------------------------------------------------------------- export

def run_ort(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x.astype(np.float32)})[0]


def test_export_mlp_parity(tmp_path):
    from gradlens.onnx_export import export_onnx
    np.random.seed(0)
    model = MLP(2, [8, 8], 3, act=Tanh())
    x = np.random.default_rng(2).standard_normal((7, 2))
    path = os.path.join(str(tmp_path), "m.onnx")
    export_onnx(model, x, path)
    assert np.allclose(run_ort(path, x), model(Tensor(x)).data, atol=1e-5)


def test_export_cnn_parity(tmp_path):
    from gradlens.onnx_export import export_onnx
    np.random.seed(1)
    model = Sequential(Conv2d(1, 4, 3, pad=1), ReLU(), MaxPool2d(2, 2),
                       Conv2d(4, 8, 3, pad=1), ReLU(),
                       MaxPool2d(3, 3), Flatten(), Linear(8 * 2 * 2, 10))   # 14->conv->14->pool->7->conv->7->pool3s3->2
    x = np.random.default_rng(3).standard_normal((1, 1, 14, 14))
    path = os.path.join(str(tmp_path), "c.onnx")
    export_onnx(model, x, path)
    ours = model(Tensor(x)).data
    assert ours.shape == run_ort(path, x).shape
    assert np.allclose(run_ort(path, x), ours, atol=1e-5)


def test_export_roundtrip_through_loader(tmp_path):
    from gradlens.onnx_export import export_onnx
    from gradlens.onnx_loader import load_onnx
    np.random.seed(2)
    model = MLP(3, [16], 4, act=ReLU())
    x = np.random.default_rng(4).standard_normal((5, 3))
    path = os.path.join(str(tmp_path), "rt.onnx")
    export_onnx(model, x, path)
    reloaded = load_onnx(path)
    assert np.allclose(reloaded(Tensor(x)).data, model(Tensor(x)).data, atol=1e-6)
    # parameters survived with their values
    assert len(reloaded.parameters()) == 4


def test_export_rejects_nonexportable_op(tmp_path):
    from gradlens.onnx_export import export_onnx, ExportError

    class Odd:
        pass

    class MaxModule(MLP):
        def forward(self, x):
            return super().forward(x) + x.abs().max() * 0.0

    np.random.seed(3)
    model = MaxModule(2, [4], 2)
    x = np.zeros((3, 2))
    with pytest.raises(ExportError, match="max"):
        export_onnx(model, x, os.path.join(str(tmp_path), "bad.onnx"))


# --------------------------------------------------------------- audit html

def test_monitor_records_loss_and_writes_html(tmp_path):
    np.random.seed(5)
    model = MLP(2, [8], 2)
    mon = GradientMonitor(model)
    opt = Adam(model.parameters(), lr=0.01)
    x = Tensor(np.random.default_rng(6).standard_normal((16, 2)))
    y = Tensor(np.random.default_rng(7).standard_normal((16, 2)))
    for _ in range(10):
        loss = mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        mon.tick(loss)
        opt.step()

    assert len(mon.loss_history) == 10
    path = os.path.join(str(tmp_path), "audit.html")
    mon.save_html(path, title="unit test run")
    html = open(path, encoding="utf-8").read()
    assert "__AUDIT_JSON__" not in html
    assert "const DATA" in html
    data = json.loads(html[html.index("const DATA = ") + 13:
                          html.index(";\n", html.index("const DATA = "))])
    assert data["title"] == "unit test run"
    assert len(data["params"]) == len(mon.history)
    assert data["steps"] == 10 and len(data["loss"]) == 10
    assert all(p["verdict"] == "healthy" for p in data["params"])


def test_monitor_html_flags_anomalies(tmp_path):
    np.random.seed(6)
    model = MLP(2, [4], 1)
    mon = GradientMonitor(model)
    # fabricate a sick training history: one param explodes, one vanishes
    names = list(mon.history)
    mon.history[names[0]] = [1.0] * 5 + [5e3] * 5        # exploding
    mon.history[names[1]] = [1e-9] * 10                   # vanishing
    mon.loss_history = [1.0, float("nan"), 1.0, 1.0]      # NaN mid-training
    path = os.path.join(str(tmp_path), "sick.html")
    mon.save_html(path)
    data = json.loads(open(path, encoding="utf-8").read()
                      .split("const DATA = ")[1].split(";\n")[0])
    verdicts = {p["name"]: p["verdict"] for p in data["params"]}
    assert verdicts[names[0]] == "exploding?"
    assert verdicts[names[1]] == "vanishing?"
    assert any(v == "NaN/Inf!" for v in verdicts.values()) or \
        any(not np.isfinite(v) for v in data["loss"])


# --------------------------------------------------------------- puzzles

def test_list_puzzles_has_five():
    listing = list_puzzles()
    for i in range(1, 6):
        assert f"p{i}" in listing


@pytest.mark.parametrize("pid", ["p1", "p2", "p3", "p4"])
def test_net_puzzles_selfcheck(pid):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = load_puzzle(pid)
    assert p.check(p._culprit_index)
    assert not p.check((p._culprit_index + 1) % p.n_steps)
    assert 0 < p._culprit_index < p.n_steps


def test_puzzle_diff_isolates_upstream_damage():
    p = load_puzzle("p1")
    text = p.diff()
    lines = text.splitlines()
    wrong = [l for l in lines if "WRONG" in l]
    clean = [l for l in lines[2:] if "WRONG" not in l and "diff" in l]
    # the last layer sits downstream of the sabotage: its gradients are exact
    assert clean and all("|diff|=0.000e+00" in l for l in clean)
    assert wrong                                   # upstream params are wrong


def test_puzzle_nan_poisons_gradients():
    p = load_puzzle("p5")
    assert np.isfinite(p.loss.item())              # forward is finite
    live_loss, live_params, _ = p._build(False)
    live_loss.backward()
    assert any(not np.all(np.isfinite(q.grad)) for q in live_params)


def test_puzzles_are_deterministic():
    a, b = load_puzzle("p3"), load_puzzle("p3")
    assert a._culprit_index == b._culprit_index
    assert a.n_steps == b.n_steps
