"""Export-side tests for the v0.7 engine ops (slice/gather/expand/clip/sqrt)
and the honest refusal of data-dependent `where`."""

import numpy as np
import pytest

from backlens import Tensor, where
from backlens.check import gradcheck  # noqa: F401

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from backlens.onnx_export import ExportError, export_onnx, verify_onnx  # noqa: E402


class CompositeNet:
    """slice -> gather(repeats) -> expand -> clip -> sqrt -> sum."""

    def __call__(self, x):
        s = x.slice([0], [4], [1])
        g = s.gather(np.array([0, 2, 2, 1]), 1)
        e = x.mean(axis=1, keepdims=True).abs().expand((x.shape[0], 4))
        c = g.clip(-1.0, 1.0)
        return (c.abs() + e).sqrt().sum(axis=1)


def test_composite_export_parity(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 6))
    path = str(tmp_path / "composite.onnx")
    export_onnx(CompositeNet(), x[:2], path)
    # note: expand bakes the traced batch (2) into the graph, so verify
    # must use the same batch size -- a documented static-shape limitation
    report = verify_onnx(CompositeNet(), x[:2], path, n_inputs=4)
    assert report
    assert report.max_diff < 1e-5


def test_where_with_python_cond_refuses_export(tmp_path):
    class WhereNet:
        def __call__(self, x):
            c = x.slice([0], [4], [1])
            return where(c.data > 0, c, c * 0.5).sum(axis=1)

    with pytest.raises(ExportError, match="data-dependent"):
        export_onnx(WhereNet(), np.zeros((2, 6)), str(tmp_path / "w.onnx"))


def test_clip_single_side_export(tmp_path):
    class ClipLo:
        def __call__(self, x):
            return x.clip(0.0).sum(axis=1)          # only lower bound

    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 5))
    path = str(tmp_path / "clip.onnx")
    export_onnx(ClipLo(), x[:1], path)
    assert verify_onnx(ClipLo(), x, path, n_inputs=2)
