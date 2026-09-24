"""Tests for gradlens.onnx_loader: programmatic ONNX models per op,
onnxruntime parity, error handling, and end-to-end gradient flow.

All tests skip gracefully when the optional ``onnx`` package is absent.
"""

import os

import numpy as np
import pytest

from gradlens import Tensor
from gradlens.nn import cross_entropy

onnx = pytest.importorskip("onnx")

from gradlens.onnx_loader import (  # noqa: E402
    OnnxModel, OnnxOpError, analyze_onnx, load_onnx,
)


# ------------------------------------------------------------------ helpers

def make_model(nodes, inputs, outputs, initializers=(), opset=13):
    graph = onnx.helper.make_graph(
        list(nodes), "test", list(inputs), list(outputs),
        initializer=list(initializers),
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", opset)]
    )
    # pin an IR version every onnxruntime accepts
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def tensor_input(name, shape, dtype=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def tensor_output(name, shape, dtype=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def init(name, arr):
    return onnx.numpy_helper.from_array(arr.astype(np.float32), name)


def init_int64(name, arr):
    return onnx.numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name)


def run_model(model, *arrays, out_names=None):
    m = OnnxModel(model)
    env = {}
    if out_names is not None:
        m.output_names = list(out_names)
    return m(*[Tensor(a.astype(np.float64)) for a in arrays])


ORT = pytest.importorskip("onnxruntime")


def run_ort(model, *arrays):
    sess = ORT.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, arrays)))


# ------------------------------------------------------------------ math ops

def test_add_with_broadcast():
    a = np.random.default_rng(0).standard_normal((2, 3))
    b = np.random.default_rng(1).standard_normal(3)
    model = make_model(
        [onnx.helper.make_node("Add", ["A", "B"], ["Y"])],
        [tensor_input("A", [2, 3]), tensor_input("B", [3])],
        [tensor_output("Y", [2, 3])],
    )
    out = run_model(model, a, b)
    assert np.allclose(out.data, a + b)


def test_gemm_transB_alpha_beta():
    rng = np.random.default_rng(2)
    a, b, c = rng.standard_normal((2, 3)), rng.standard_normal((4, 3)), rng.standard_normal(4)
    model = make_model(
        [onnx.helper.make_node("Gemm", ["A", "B", "C"], ["Y"],
                               alpha=0.5, beta=2.0, transB=1)],
        [tensor_input("A", [2, 3]), tensor_input("B", [4, 3]),
         tensor_input("C", [4])],
        [tensor_output("Y", [2, 4])],
    )
    out = run_model(model, a, b, c)
    assert np.allclose(out.data, 0.5 * (a @ b.T) + 2.0 * c)


def test_softmax_opset13_last_axis():
    x = np.random.default_rng(3).standard_normal((2, 5)) * 3
    model = make_model(
        [onnx.helper.make_node("Softmax", ["X"], ["Y"], axis=-1)],
        [tensor_input("X", [2, 5])], [tensor_output("Y", [2, 5])],
    )
    out = run_model(model, x)
    e = np.exp(x - x.max(-1, keepdims=True))
    assert np.allclose(out.data, e / e.sum(-1, keepdims=True), atol=1e-12)


def test_softmax_legacy_opset11_coerces_2d():
    x = np.random.default_rng(4).standard_normal((1, 4, 2, 2))
    model = make_model(
        [onnx.helper.make_node("Softmax", ["X"], ["Y"])],   # default axis=1
        [tensor_input("X", [1, 4, 2, 2])], [tensor_output("Y", [1, 4, 2, 2])],
        opset=11,
    )
    out = run_model(model, x)
    flat = x.reshape(1, -1)
    e = np.exp(flat - flat.max(-1, keepdims=True))
    expected = (e / e.sum(-1, keepdims=True)).reshape(x.shape)
    assert np.allclose(out.data, expected, atol=1e-12)


def test_reduce_sum_axes_keepdims():
    x = np.random.default_rng(5).standard_normal((2, 3, 4))
    model = make_model(
        [onnx.helper.make_node("ReduceSum", ["X"], ["Y"], axes=[1], keepdims=0)],
        [tensor_input("X", [2, 3, 4])], [tensor_output("Y", [2, 4])],
        opset=11,      # axes stays an attribute below opset 13
    )
    assert np.allclose(run_model(model, x).data, x.sum(1))


def test_reduce_mean_opset18_axes_as_input():
    x = np.random.default_rng(6).standard_normal((2, 3, 4))
    model = make_model(
        [onnx.helper.make_node("ReduceMean", ["X", "axes"], ["Y"], keepdims=1)],
        [tensor_input("X", [2, 3, 4]), tensor_input("axes", [1], onnx.TensorProto.INT64)],
        [tensor_output("Y", [2, 1, 4])],
        initializers=[init_int64("axes", [1])],
        opset=18,      # axes-as-input landed at opset 18 for ReduceMean
    )
    # axes arrives via an initializer-backed input; the input list excludes it
    m = OnnxModel(model)
    assert m.input_names == ["X"]
    assert np.allclose(m(Tensor(x)).data, x.mean(1, keepdims=True))


def test_global_average_pool():
    x = np.random.default_rng(7).standard_normal((1, 3, 4, 4))
    model = make_model(
        [onnx.helper.make_node("GlobalAveragePool", ["X"], ["Y"])],
        [tensor_input("X", [1, 3, 4, 4])], [tensor_output("Y", [1, 3, 1, 1])],
    )
    assert np.allclose(run_model(model, x).data, x.mean((2, 3), keepdims=True))


# ------------------------------------------------------------------ shape ops

def test_reshape_zero_and_minus_one():
    x = np.random.default_rng(8).standard_normal((2, 3, 4))
    model = make_model(
        [onnx.helper.make_node("Reshape", ["X", "shape"], ["Y"])],
        [tensor_input("X", [2, 3, 4]), tensor_input("shape", [2], onnx.TensorProto.INT64)],
        [tensor_output("Y", [2, 12])],
        initializers=[init_int64("shape", [0, -1])],
    )
    assert run_model(model, x).shape == (2, 12)


def test_flatten_transpose_concat_squeeze_unsqueeze():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 3, 4))
    model = make_model(
        [
            onnx.helper.make_node("Flatten", ["X"], ["F"], axis=1),
            onnx.helper.make_node("Transpose", ["F"], ["T"], perm=[1, 0]),
            onnx.helper.make_node("Concat", ["T", "T"], ["C"], axis=0),
            onnx.helper.make_node("Unsqueeze", ["C"], ["U"], axes=[2]),
            onnx.helper.make_node("Squeeze", ["U"], ["S"], axes=[2]),
        ],
        [tensor_input("X", [2, 3, 4])],
        [tensor_output("S", [24, 2])],
        opset=11,      # Squeeze/Unsqueeze keep `axes` as an attribute here
    )
    m = OnnxModel(model)
    out = m(Tensor(x))
    assert out.shape == (24, 2)   # (2,3,4)->(2,12)->(12,2)->cat->(24,2)->(24,2,1)->(24,2)

    model2 = make_model(
        [onnx.helper.make_node("Unsqueeze", ["X"], ["Y"], axes=[0])],
        [tensor_input("X", [2, 3])], [tensor_output("Y", [1, 2, 3])],
        opset=11,
    )
    assert run_model(model2, x[:, :, 0]).shape == (1, 2, 3)


def test_constant_and_identity_nodes():
    model = make_model(
        [
            onnx.helper.make_node("Constant", [], ["c"], value_floats=[1.5, 2.5]),
            onnx.helper.make_node("Identity", ["c"], ["Y"]),
        ],
        [], [tensor_output("Y", [2])],
    )
    m = OnnxModel(model)
    assert m.input_names == []
    out = m()
    assert np.allclose(out.data, [1.5, 2.5])


# ------------------------------------------------------------------ conv / pool vs onnxruntime

def test_conv_parity_with_onnxruntime_asymmetric_pads():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((1, 2, 7, 6)).astype(np.float32)
    w = rng.standard_normal((3, 2, 3, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    model = make_model(
        [onnx.helper.make_node("Conv", ["X", "W", "B"], ["Y"],
                               kernel_shape=[3, 3], strides=[2, 2],
                               pads=[1, 0, 2, 1])],
        [tensor_input("X", [1, 2, 7, 6]), tensor_input("W", [3, 2, 3, 3]),
         tensor_input("B", [3])],
        [tensor_output("Y", [1, 3, 3, 4])],
    )
    ref = run_ort(model, x, w, b)
    out = run_model(model, x, w, b)
    assert out.shape == ref[0].shape
    assert np.allclose(out.data, ref[0], atol=1e-5)


def test_conv_auto_pad_same_upper_parity():
    rng = np.random.default_rng(11)
    x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
    w = rng.standard_normal((4, 1, 3, 3)).astype(np.float32)
    model = make_model(
        [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], auto_pad="SAME_UPPER")],
        [tensor_input("X", [1, 1, 8, 8]), tensor_input("W", [4, 1, 3, 3])],
        [tensor_output("Y", [1, 4, 8, 8])],
    )
    ref = run_ort(model, x, w)
    out = run_model(model, x, w)
    assert out.shape == ref[0].shape
    assert np.allclose(out.data, ref[0], atol=1e-5)


def test_maxpool_parity_with_onnxruntime():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)
    model = make_model(
        [onnx.helper.make_node("MaxPool", ["X"], ["Y"], kernel_shape=[3, 3],
                               strides=[2, 2], pads=[1, 1, 1, 1])],
        [tensor_input("X", [2, 3, 9, 9])], [tensor_output("Y", [2, 3, 5, 5])],
    )
    ref = run_ort(model, x)
    out = run_model(model, x)
    assert out.shape == ref[0].shape
    assert np.allclose(out.data, ref[0], atol=1e-6)


# ------------------------------------------------------------------ errors & analysis

def test_unsupported_op_raises_with_names():
    model = make_model(
        [onnx.helper.make_node("LSTM", ["X", "W", "R"], ["Y"])],
        [tensor_input("X", [1]), tensor_input("W", [1]), tensor_input("R", [1])],
        [tensor_output("Y", [1])],
    )
    with pytest.raises(OnnxOpError, match="LSTM"):
        OnnxModel(model)


def test_analyze_onnx(tmp_path):
    model = make_model(
        [onnx.helper.make_node("Relu", ["X"], ["Y"]),
         onnx.helper.make_node("Dropout", ["Y", "ratio"], ["Z"])],
        [tensor_input("X", [2, 2]), tensor_input("ratio", [1])],
        [tensor_output("Z", [2, 2])],
        initializers=[init("ratio", np.array([0.5]))],
    )
    path = os.path.join(str(tmp_path), "m.onnx")
    onnx.save(model, path)
    report = analyze_onnx(path)
    assert report["ops"]["Relu"] == 1
    assert report["unsupported"] == ["Dropout"]
    assert report["loadable"] is False


def test_load_onnx_from_file(tmp_path):
    rng = np.random.default_rng(13)
    w = rng.standard_normal((3, 2))
    model = make_model(
        [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"])],
        [tensor_input("X", [4, 3]), tensor_input("W", [3, 2])],
        [tensor_output("Y", [4, 2])],
        initializers=[init("W", w)],
    )
    path = os.path.join(str(tmp_path), "m.onnx")
    onnx.save(model, path)
    m = load_onnx(path)
    x = rng.standard_normal((4, 3))
    assert np.allclose(m(Tensor(x)).data, x @ w, atol=1e-6)


# ------------------------------------------------------------------ training

def test_parameters_are_trainable_end_to_end():
    """A tiny ONNX MLP: initializers become params; gradients flow and
    finite differences agree with backward through the whole loaded graph."""
    rng = np.random.default_rng(14)
    w1 = rng.standard_normal((4, 3)) * 0.3
    b1 = rng.standard_normal(3) * 0.1
    w2 = rng.standard_normal((3, 2)) * 0.3
    model = make_model(
        [
            onnx.helper.make_node("Gemm", ["X", "W1", "B1"], ["h"]),
            onnx.helper.make_node("Relu", ["h"], ["hr"]),
            onnx.helper.make_node("Gemm", ["hr", "W2"], ["Y"]),
        ],
        [tensor_input("X", [4, 4])],
        [tensor_output("Y", [4, 2])],
        initializers=[init("W1", w1), init("B1", b1), init("W2", w2)],
    )
    m = OnnxModel(model)
    assert len(m.parameters()) == 3
    assert [p.label for p in m.parameters()] == ["W1", "B1", "W2"]

    x = Tensor(rng.standard_normal((4, 4)))
    loss = (m(x) ** 2).sum()
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and np.all(np.isfinite(p.grad))

    # numeric check of dL/dW2 through the loaded graph
    eps = 1e-6
    i, j = 1, 0
    w2p = m._static["W2"]
    w2p.data[i, j] += eps
    plus = float((m(x) ** 2).sum().data)
    w2p.data[i, j] -= 2 * eps
    minus = float((m(x) ** 2).sum().data)
    w2p.data[i, j] += eps
    numeric = (plus - minus) / (2 * eps)
    assert abs(numeric - m._static["W2"].grad[i, j]) < 1e-4


ASSET = os.path.join(os.path.dirname(__file__), "..", "assets", "mnist-8.onnx")
has_mnist = os.path.exists(ASSET)


@pytest.mark.skipif(not has_mnist, reason="assets/mnist-8.onnx not present")
class TestRealMnistModel:
    def test_forward_parity_with_onnxruntime(self):
        m = load_onnx(ASSET)
        x = np.random.default_rng(0).standard_normal((1, 1, 28, 28)).astype(np.float32)
        ref = run_ort(m._proto, x)[0]
        out = m(Tensor(x.astype(np.float64)))
        assert out.shape == ref.shape
        assert np.allclose(out.data, ref, atol=1e-4)

    def test_backward_reaches_every_parameter(self):
        m = load_onnx(ASSET)
        x = np.random.default_rng(1).standard_normal((1, 1, 28, 28))
        loss = cross_entropy(m(Tensor(x)), np.array([3]))
        loss.backward()
        grads = {p.label: float(np.linalg.norm(p.grad)) for p in m.parameters()}
        assert all(np.isfinite(g) for g in grads.values())
        assert all(g > 0 for g in grads.values()), grads

    def test_input_gradient_matches_finite_differences(self):
        m = load_onnx(ASSET)
        x = np.random.default_rng(2).standard_normal((1, 1, 28, 28))
        target = np.array([7])
        xt = Tensor(x, requires_grad=True)
        cross_entropy(m(xt), target).backward()

        rng = np.random.default_rng(3)
        picks = [(int(rng.integers(28)), int(rng.integers(28))) for _ in range(5)]
        eps = 1e-5
        for (i, j) in picks:
            xp = x.copy(); xp[0, 0, i, j] += eps
            xm = x.copy(); xm[0, 0, i, j] -= eps
            num = (cross_entropy(m(Tensor(xp)), target).item()
                   - cross_entropy(m(Tensor(xm)), target).item()) / (2 * eps)
            assert abs(num - xt.grad[0, 0, i, j]) < 1e-4, (i, j, num, xt.grad[0, 0, i, j])
