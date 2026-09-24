"""Tests for the 0.4 unique-feature set: ONNX export, training gradient
audit HTML, autodiff puzzles, and the new Conv2d/MaxPool2d modules."""

import json
import os

import numpy as np
import pytest

from backlens import (
    Tensor, MLP, Tanh, ReLU, Sequential, Linear, Conv2d, MaxPool2d, Flatten,
    SGD, Adam, mse_loss,
)
from backlens.debug import GradientMonitor
from backlens.puzzles import list_puzzles, load_puzzle

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
    from backlens.onnx_export import export_onnx
    np.random.seed(0)
    model = MLP(2, [8, 8], 3, act=Tanh())
    x = np.random.default_rng(2).standard_normal((7, 2))
    path = os.path.join(str(tmp_path), "m.onnx")
    export_onnx(model, x, path)
    assert np.allclose(run_ort(path, x), model(Tensor(x)).data, atol=1e-5)


def test_export_cnn_parity(tmp_path):
    from backlens.onnx_export import export_onnx
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
    from backlens.onnx_export import export_onnx
    from backlens.onnx_loader import load_onnx
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
    from backlens.onnx_export import export_onnx, ExportError

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

def test_list_puzzles_has_six():
    listing = list_puzzles()
    for i in range(1, 7):
        assert f"p{i}" in listing


@pytest.mark.parametrize("pid", ["p1", "p2", "p3", "p4", "p6"])
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


def test_puzzle_silent_detach_severs_gradients():
    p = load_puzzle("p6")
    live_loss, live_params, _ = p._build(False)
    live_loss.backward()
    grads = {q.label: q.grad for q in live_params}
    assert grads["w1"] is None and grads["w2"] is None    # severed branch
    assert grads["w3"] is not None                        # surviving branch
    text = p.diff()
    assert text.count("NO GRADIENT AT ALL") == 2
    assert "w3" in text and "WRONG" not in [l for l in text.splitlines()
                                            if l.strip().startswith("w3")][0]


# --------------------------------------------------------------- verify_onnx

def test_verify_onnx_passes_on_export(tmp_path):
    from backlens.onnx_export import export_onnx, verify_onnx
    np.random.seed(10)
    model = MLP(2, [8], 3, act=Tanh())
    x = np.random.default_rng(11).standard_normal((5, 2))
    path = os.path.join(str(tmp_path), "v.onnx")
    export_onnx(model, x, path)
    report = verify_onnx(model, x, path, n_inputs=3)
    assert report
    assert report.max_diff < 1e-4
    assert "PASSED" in report.report()


def test_verify_onnx_detects_corrupted_file(tmp_path):
    import onnx
    from backlens.onnx_export import export_onnx, verify_onnx
    np.random.seed(12)
    model = MLP(2, [8], 3, act=Tanh())
    x = np.random.default_rng(13).standard_normal((5, 2))
    good = os.path.join(str(tmp_path), "good.onnx")
    export_onnx(model, x, good)

    proto = onnx.load(good)
    proto.graph.initializer[0].raw_data = (
        np.frombuffer(proto.graph.initializer[0].raw_data, dtype=np.float32) + 5.0
    ).tobytes()
    bad = os.path.join(str(tmp_path), "bad.onnx")
    onnx.save(proto, bad)

    report = verify_onnx(model, x, bad, n_inputs=2)
    assert not report
    assert report.max_diff > 1e-3
    assert "FAILED" in report.report()


def test_verify_onnx_exports_to_temp_when_no_path():
    from backlens.onnx_export import verify_onnx
    np.random.seed(14)
    model = MLP(3, [4], 2, act=Tanh())
    x = np.random.default_rng(15).standard_normal((4, 3))
    report = verify_onnx(model, x)          # no path: export to a temp file
    assert report
    assert report.max_diff < 1e-4


def test_verify_onnx_loaded_onnx_model_roundtrip(tmp_path):
    from backlens.onnx_export import export_onnx, verify_onnx
    from backlens.onnx_loader import load_onnx
    np.random.seed(16)
    model = MLP(2, [8], 2, act=Tanh())
    x = np.random.default_rng(17).standard_normal((5, 2))
    path = os.path.join(str(tmp_path), "rt.onnx")
    export_onnx(model, x, path)
    reloaded = load_onnx(path)
    # the reloaded ONNX model must agree with onnxruntime on the same file
    report = verify_onnx(reloaded, x, path, n_inputs=2)
    assert report
