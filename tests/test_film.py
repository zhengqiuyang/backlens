import json
import os

import numpy as np
import pytest

from backlens import Tensor, MLP, Tanh, cross_entropy
from backlens.debug import debug_backward
from backlens.film import film_backward, BackwardFilm


def build(seed=0):
    rng = np.random.default_rng(seed)
    x = Tensor(rng.standard_normal((5, 3)), label="x")
    w = Tensor(rng.standard_normal((3, 2)) * 0.5, requires_grad=True, label="w")
    b = Tensor(np.zeros(2), requires_grad=True, label="b")
    loss = ((x @ w + b).tanh() ** 2).mean()
    return loss


def nan_build():
    x = Tensor(np.array([1.0, 0.0]), requires_grad=True, label="bad_x")
    return (x.log() * Tensor(np.array([1.0, 0.0]))).sum()


def test_film_matches_debug_trace():
    loss1, loss2 = build(1), build(1)
    trace = debug_backward(loss1, raise_on_anomaly=False)
    film = film_backward(loss2)
    assert len(film.steps) == len(trace)
    assert [s["op"] for s in film.steps] == [s.op for s in trace]
    assert [s["gn"] for s in film.steps] == pytest.approx(
        [s.grad_norm for s in trace], nan_ok=True, rel=1e-12, abs=1e-12
    )


def test_film_graph_structure_valid():
    loss = build()
    film = film_backward(loss)
    n = len(film.nodes)
    assert n == len(loss._topo())
    assert len(film.edges) == sum(len(t._prev) for t in loss._topo())
    assert all(0 <= a < n and 0 <= b < n for a, b in film.edges)
    assert all(0 <= s["node"] < n for s in film.steps)
    kinds = {nd["kind"] for nd in film.nodes}
    assert kinds <= {"data", "param", "op"}


def test_film_json_roundtrip():
    film = film_backward(build(), name="unit test")
    data = json.loads(film.to_json())
    assert data["name"] == "unit test"
    assert len(data["steps"]) == len(film.steps)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_film_json_stays_finite():
    pytest.importorskip("math")
    film = film_backward(nan_build())
    for s in film.steps:
        assert s["gn"] is None or np.isfinite(s["gn"])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_film_anomaly_detection():
    film = film_backward(nan_build())
    assert film.first_anomaly_step is not None
    flagged = [s for s in film.steps if s["nan"] or s["inf"]]
    assert flagged
    assert film.first_anomaly_step == flagged[0]["index"]


def test_film_html_contents(tmp_path):
    film = film_backward(build())
    path = os.path.join(str(tmp_path), "film.html")
    film.save_html(path)
    html = open(path, encoding="utf-8").read()
    assert "__FILM_JSON__" not in html          # template fully filled
    assert "const FILM =" in html
    assert "<svg" in html and "anomaly-banner" in html
    assert "btn-play" in html and "scrub" in html
    # embedded JSON is parseable and safe inside <script>
    start = html.index("const FILM = ") + len("const FILM = ")
    end = html.index(";\n", start)
    data = json.loads(html[start:end].strip())
    assert len(data["nodes"]) == len(film.nodes)


def test_film_save_json(tmp_path):
    film = film_backward(build())
    path = os.path.join(str(tmp_path), "film.json")
    film.save_json(path)
    data = json.load(open(path, encoding="utf-8"))
    assert data == json.loads(film.to_json())


def test_film_mlp_end_to_end(tmp_path):
    np.random.seed(0)
    x = Tensor(np.random.randn(6, 2), label="batch")
    model = MLP(2, [4], 3, act=Tanh())
    loss = cross_entropy(model(x), np.array([0, 1, 2, 0, 1, 2]))
    film = film_backward(loss, name="mlp")
    assert isinstance(film, BackwardFilm)
    assert "nodes" in film.to_json()
    assert film.first_anomaly_step is None
    film.save_html(os.path.join(str(tmp_path), "ok.html"))
