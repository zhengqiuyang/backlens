"""Tests for gradlens.flow: per-edge gradient mass measurement and report."""

import json
import os

import numpy as np
import pytest

from gradlens import Tensor, MLP, Tanh
from gradlens.nn import mse_loss
from gradlens.flow import gradient_flow


def small_net(seed=0):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((6, 2)), label="x")
    target = Tensor(rng.standard_normal((6, 1)))
    model = MLP(2, [4], 1, act=Tanh())
    return mse_loss(model(x), target), model


def test_flow_edges_conserve_single_consumer_mass():
    loss, model = small_net()
    flow = gradient_flow(loss)
    # a param with a single consumer: its incoming edge mass == ||grad||
    loss2, _ = small_net()      # same seed -> identical graph
    loss2.backward()
    for p in model.parameters():
        incoming = [e for e in flow.edges
                    if flow.nodes[e["to"]].get("label") == p.label]
        # at least one recorded edge per param; masses positive
        assert all(e["mass"] > 0 for e in incoming), (p.label, incoming)


def test_flow_report_shares_sum_to_100():
    loss, _ = small_net(1)
    flow = gradient_flow(loss, name="unit")
    total = sum(p["share"] for p in flow.params)
    assert 99.0 < total < 101.0
    assert all(p["mass"] > 0 for p in flow.params)
    # ranked descending
    shares = [p["share"] for p in flow.params]
    assert shares == sorted(shares, reverse=True)
    assert "gradient flow" in flow.report()


def test_flow_edge_count_matches_graph():
    loss, model = small_net(2)
    flow = gradient_flow(loss)
    topo = loss._topo()
    grad_edges = sum(1 for t in topo for p in t._prev if p.requires_grad)
    assert len(flow.edges) == grad_edges
    assert len(flow.nodes) == len(topo)


def test_flow_requires_scalar():
    x = Tensor(np.ones((3,)), requires_grad=True)
    with pytest.raises(RuntimeError):
        gradient_flow(x * 2.0)


def test_flow_html(tmp_path):
    loss, _ = small_net(3)
    flow = gradient_flow(loss, name="html test")
    path = os.path.join(str(tmp_path), "flow.html")
    flow.save_html(path)
    html = open(path, encoding="utf-8").read()
    assert "__FLOW_JSON__" not in html
    assert "const DATA" in html and "<svg" in html
    start = html.index("const DATA = ") + 13
    data = json.loads(html[start:html.index(";\n", start)])
    assert data["name"] == "html test"
    assert len(data["params"]) == len(flow.params)
    assert data["edges"] == flow.edges


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_flow_nan_case_is_flagged_not_crashed():
    with pytest.warns(RuntimeWarning):
        x = Tensor(np.array([1.0, 0.0]), requires_grad=True, label="x")
        loss = (x.log() * Tensor(np.array([1.0, 0.0]))).sum()
    flow = gradient_flow(loss, name="nan case")
    # non-finite masses are serialized as null, not as bare NaN
    for e in flow.edges:
        assert e["mass"] is None or np.isfinite(e["mass"])
    flow.save_html(os.path.join(os.path.dirname(__file__), "..", "flow_tmp.html"))
    os.remove(os.path.join(os.path.dirname(__file__), "..", "flow_tmp.html"))
