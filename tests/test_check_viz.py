import os

import numpy as np
import pytest

from gradlens import Tensor
from gradlens.check import gradcheck
from gradlens.viz import to_mermaid, save_graph_md, to_html, save_graph_html


def build(seed=0):
    rng = np.random.default_rng(seed)
    a = Tensor(rng.standard_normal((3, 2)) + 2, requires_grad=True, label="a")
    b = Tensor(rng.standard_normal((2, 4)) + 2, requires_grad=True, label="b")
    loss = ((a @ b).tanh() * 1.5).sum()
    loss.backward()
    return loss, a, b


# ------------------------------------------------------------------ gradcheck

def test_gradcheck_passes_on_correct_engine():
    x = Tensor(np.random.default_rng(1).standard_normal((3, 3)) + 1.5,
               requires_grad=True, label="x")
    result = gradcheck(lambda t: (t * t).tanh().sum() + t.mean(), x)
    assert result
    assert "PASSED" in result.report()


def test_gradcheck_catches_wrong_grad():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    # a hook that silently doubles the gradient flowing into x
    x.register_hook(lambda g: g * 2.0)

    result = gradcheck(lambda t: (t + t).sum(), x)   # true grad: [2, 2]
    assert not result
    assert "FAILED" in result.report()


def test_gradcheck_multiple_inputs():
    rng = np.random.default_rng(2)
    a = Tensor(rng.standard_normal((2, 3)) + 2, requires_grad=True, label="a")
    b = Tensor(rng.standard_normal((3, 2)) + 2, requires_grad=True, label="b")
    result = gradcheck(lambda x, y: (x @ y).sum() + (x * x).mean(), a, b)
    assert result
    assert len(result.reports) == 2


def test_gradcheck_rejects_non_scalar_output():
    x = Tensor(np.ones(3), requires_grad=True)
    with pytest.raises(ValueError):
        gradcheck(lambda t: t * 2.0, x)


def test_gradcheck_rejects_gradless_input():
    x = Tensor(np.ones(3))
    with pytest.raises(ValueError):
        gradcheck(lambda t: t.sum(), x)


# ------------------------------------------------------------------ mermaid

def test_mermaid_has_nodes_and_edges():
    loss, a, b = build()
    src = to_mermaid(loss)
    assert src.startswith("graph TD")
    assert src.count("-->") == len(loss._topo()[0]._prev) or src.count("-->") > 0
    assert "tanh" in src and "matmul" in src and "sum" in src
    assert '"a' in src  # labels appear


def test_mermaid_shows_grad_norms_after_backward():
    loss, _, _ = build()
    assert "|g|=" in to_mermaid(loss)
    # before backward there are no grads
    rng = np.random.default_rng(5)
    x = Tensor(rng.standard_normal(3), requires_grad=True)
    y = (x * x).sum()
    assert "|g|=" not in to_mermaid(y)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mermaid_flags_anomalous_nodes():
    x = Tensor(np.array([1.0, 0.0]), requires_grad=True)
    loss = (x.log() * Tensor(np.array([1.0, 0.0]))).sum()
    loss.backward()
    src = to_mermaid(loss)
    assert "classDef anomaly" in src


def test_mermaid_escapes_special_characters():
    x = Tensor(np.array([1.0]), requires_grad=True, label='we"ird<name>')
    loss = (x * 2.0).sum()
    loss.backward()
    src = to_mermaid(loss)
    assert '&lt;' in src and "&gt;" in src


def test_save_graph_md_writes_file(tmp_path):
    loss, _, _ = build()
    path = os.path.join(str(tmp_path), "graph.md")
    content = save_graph_md(loss, path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == content
    assert content.startswith("```mermaid") and content.endswith("```\n")


def test_save_graph_html_writes_file(tmp_path):
    loss, _, _ = build()
    path = os.path.join(str(tmp_path), "graph.html")
    save_graph_html(loss, path)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    assert "mermaid" in html and "graph TD" in html


def test_html_is_self_contained_string():
    loss, _, _ = build()
    html = to_html(loss)
    assert html.startswith("<!DOCTYPE html>")
    assert "cdn.jsdelivr.net/npm/mermaid" in html


def test_direction_lr():
    loss, _, _ = build()
    assert to_mermaid(loss, direction="LR").startswith("graph LR")
