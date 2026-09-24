"""Tests for the torch bridge (no torch required: numpy dicts duck-type) and
the GradientFlowTimeline."""

import json
import os

import numpy as np
import pytest

from backlens import Tensor, MLP, Tanh, cross_entropy, Adam
from backlens.nn import mse_loss
from backlens.torch_bridge import load_torch_state_dict, _to_numpy
from backlens.flow import GradientFlowTimeline


# ------------------------------------------------------------------ bridge

def fake_torch_state_dict(seed=0, hidden=(8, 8), out=1):
    """A torch-shaped state dict for a 2-hidden-...-out tanh MLP, as numpy.

    Weights use torch's (out, in) convention."""
    rng = np.random.default_rng(seed)
    sizes = [2, *hidden, out]
    sd = {}
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        sd[f"net.{2 * i}.weight"] = rng.standard_normal((b, a))
        sd[f"net.{2 * i}.bias"] = rng.standard_normal(b)
    return sd


def test_bridge_maps_by_order_and_shape():
    model = MLP(2, [8, 8], 1, act=Tanh())
    sd = fake_torch_state_dict(3)
    mapping = load_torch_state_dict(model, sd)
    assert [m[1] for m in mapping] == [k for k in sd
                                       if sd[k].dtype.kind == "f"]
    # values landed in order; 2D weights arrive transposed (torch (out,in))
    assert np.allclose(model.net.layers[0].weight.data, sd["net.0.weight"].T)
    assert np.allclose(model.net.layers[4].bias.data, sd["net.4.bias"])


def test_bridge_clears_stale_gradients():
    model = MLP(2, [4], 1, act=Tanh())
    model.parameters()[0].grad = np.ones_like(model.parameters()[0].data)
    load_torch_state_dict(model, fake_torch_state_dict(1, hidden=(4,), out=1))
    assert all(p.grad is None for p in model.parameters())


def test_bridge_rejects_shape_mismatch():
    model = MLP(2, [8], 1, act=Tanh())
    bad = fake_torch_state_dict(4)
    bad["net.4.weight"] = np.zeros((7, 8))          # wrong shape
    with pytest.raises(ValueError, match="shape mismatch|count mismatch"):
        load_torch_state_dict(model, bad)


def test_bridge_rejects_count_mismatch():
    model = MLP(2, [8, 8], 1, act=Tanh())
    short = {k: v for k, v in fake_torch_state_dict(5).items()
             if "net.4" not in k}
    with pytest.raises(ValueError, match="count mismatch"):
        load_torch_state_dict(model, short)


def test_bridge_non_floating_tensors_skipped():
    """Integer buffers (e.g. BN num_batches_tracked) are ignored."""
    model = MLP(2, [4], 1, act=Tanh())
    sd = fake_torch_state_dict(6, hidden=(4,), out=1)
    sd["num_batches_tracked"] = np.array(42, dtype=np.int64)
    mapping = load_torch_state_dict(model, sd)
    assert all("tracked" not in t for _, t in mapping)


def test_to_numpy_accepts_numpy_and_duck_typed():
    # dtype is preserved (integer buffers must stay non-floating)
    arr = _to_numpy(np.ones((2, 2), dtype=np.float32))
    assert arr.dtype == np.float32

    class FakeTorchTensor:                          # duck-typed torch.Tensor
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((3,), dtype=np.float32)

    assert _to_numpy(FakeTorchTensor()).shape == (3,)


# ------------------------------------------------------------------ timeline

def _train_and_record(snapshot_every=20, steps=60):
    np.random.seed(7)
    rng = np.random.default_rng(8)
    X = rng.standard_normal((60, 2))
    y = rng.integers(0, 3, 60)
    model = MLP(2, [8, 8], 3, act=Tanh())
    opt = Adam(model.parameters(), lr=0.05)
    tl = GradientFlowTimeline()
    for step in range(steps):
        loss = cross_entropy(model(Tensor(X)), y)
        opt.zero_grad()
        loss.backward()
        if step % snapshot_every == 0:
            tl.snapshot(loss, step=step)
        opt.step()
    return tl, model


def test_timeline_records_shares():
    tl, _ = _train_and_record()
    assert len(tl.snapshots) == 3
    for snap in tl.snapshots:
        assert 99.0 < sum(snap["shares"].values()) < 101.0
    assert tl.snapshots[0]["step"] == 0 and tl.snapshots[-1]["step"] == 40


def test_timeline_report_and_html(tmp_path):
    tl, _ = _train_and_record(snapshot_every=10, steps=30)
    text = tl.report()
    assert "timeline" in text and "->" in text
    path = os.path.join(str(tmp_path), "tl.html")
    tl.save_html(path, title="unit")
    html = open(path, encoding="utf-8").read()
    assert "__TL_JSON__" not in html and "const DATA" in html
    data = json.loads(html[html.index("const DATA = ") + 13:
                          html.index(";\n", html.index("const DATA = "))])
    assert data["names"] and len(data["matrix"]) == 3
    # each snapshot's stacked shares sum to ~100
    for row in data["matrix"]:
        assert 99.0 < sum(v or 0 for v in row) < 101.0


def test_timeline_loss_object_accepted():
    tl = GradientFlowTimeline()
    model = MLP(2, [4], 1, act=Tanh())
    x = Tensor(np.random.default_rng(9).standard_normal((5, 2)))
    y = Tensor(np.random.default_rng(10).standard_normal((5, 1)))
    loss = mse_loss(model(x), y)
    loss.backward()
    tl.snapshot(loss)                       # works after a normal backward
    assert len(tl.snapshots) == 1
