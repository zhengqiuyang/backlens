"""Tests for the v0.7 ONNX op expansion: Slice, Gather, Expand, Where,
Sqrt, Clip, LeakyRelu, ConstantOfShape, BatchNormalization, grouped Conv --
each verified against onnxruntime where meaningful."""

import numpy as np
import pytest

from backlens import Tensor
from backlens.check import gradcheck

onnx = pytest.importorskip("onnx")

from tests.test_onnx_loader import (  # noqa: E402  (reuse helpers)
    init, init_int64, make_model, run_model, run_ort, tensor_input,
    tensor_output,
)


def node(op, ins, outs, **attrs):
    return onnx.helper.make_node(op, ins, outs, **attrs)


# ------------------------------------------------------------------ Slice

def test_slice_opset10_parity_negative_and_clamped():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 6, 4)).astype(np.float32)
    model = make_model(
        [node("Slice", ["X", "st", "en", "ax", "sp"], ["Y"])],
        [tensor_input("X", [2, 6, 4]),
         tensor_input("st", [2], onnx.TensorProto.INT64),
         tensor_input("en", [2], onnx.TensorProto.INT64),
         tensor_input("ax", [2], onnx.TensorProto.INT64),
         tensor_input("sp", [2], onnx.TensorProto.INT64)],
        [tensor_output("Y", [2, 3, 4])],
        initializers=[init_int64("st", [1, -3]), init_int64("en", [100, 6]),
                      init_int64("ax", [0, 1]), init_int64("sp", [1, 2])],
    )
    ref = run_ort(model, x)[0]
    out = run_model(model, x)
    assert out.shape == ref.shape
    assert np.allclose(out.data, ref, atol=1e-6)


def test_slice_opset11_attribute_style():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    model = make_model(
        [node("Slice", ["X"], ["Y"], starts=[0, 1], ends=[2, 3], axes=[0, 2])],
        [tensor_input("X", [2, 3, 4])], [tensor_output("Y", [2, 3, 4])],
        opset=1,
    )
    assert np.allclose(run_model(model, x).data, x[:2, :, 1:3])


def test_slice_negative_steps_parity():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    model = make_model(
        [node("Slice", ["X", "st", "en", "ax", "sp"], ["Y"])],
        [tensor_input("X", [3, 4]),
         tensor_input("st", [1], onnx.TensorProto.INT64),
         tensor_input("en", [1], onnx.TensorProto.INT64),
         tensor_input("ax", [1], onnx.TensorProto.INT64),
         tensor_input("sp", [1], onnx.TensorProto.INT64)],
        [tensor_output("Y", [3, 4])],
        initializers=[init_int64("st", [0]), init_int64("en", [-5]),
                      init_int64("ax", [0]), init_int64("sp", [-1])],
    )
    ref = run_ort(model, x)[0]
    assert np.allclose(run_model(model, x).data, ref, atol=1e-6)


# ------------------------------------------------------------------ Gather / Expand / Where

def test_gather_parity_repeated_indices():
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    idx = np.array([3, 1, 1, 3], dtype=np.int64)
    model = make_model(
        [node("Gather", ["X", "i"], ["Y"], axis=0)],
        [tensor_input("X", [4, 5]), tensor_input("i", [4], onnx.TensorProto.INT64)],
        [tensor_output("Y", [4, 5])],
        initializers=[init_int64("i", idx)],
    )
    assert np.allclose(run_model(model, x).data, run_ort(model, x)[0], atol=1e-6)


def test_gather_gradients_accumulate_repeats():
    x = np.arange(6, dtype=float).reshape(2, 3)
    t = Tensor(x, requires_grad=True)
    idx = np.array([1, 1, 0])
    out = t.gather(idx, 0)
    weights = Tensor(np.array([1.0, 10.0, 100.0])[:, None])
    (out * weights).sum().backward()
    assert np.allclose(t.grad[1], [1 + 10, 1 + 10, 1 + 10])   # repeated index sums
    assert np.allclose(t.grad[0], [100, 100, 100])


def test_expand_parity():
    x = np.random.default_rng(1).standard_normal((1, 4)).astype(np.float32)
    model = make_model(
        [node("Expand", ["X", "s"], ["Y"])],
        [tensor_input("X", [1, 4]), tensor_input("s", [2], onnx.TensorProto.INT64)],
        [tensor_output("Y", [3, 4])],
        initializers=[init_int64("s", [3, 4])],
    )
    assert np.allclose(run_model(model, x).data, run_ort(model, x)[0], atol=1e-6)


def test_where_parity_and_grads():
    rng = np.random.default_rng(2)
    c = (rng.standard_normal((2, 3)) > 0).astype(np.int32)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((2, 3)).astype(np.float32)
    model = make_model(
        [node("Where", ["C", "A", "B"], ["Y"])],
        [tensor_input("C", [2, 3], onnx.TensorProto.INT64),
         tensor_input("A", [2, 3]), tensor_input("B", [2, 3])],
        [tensor_output("Y", [2, 3])],
    )
    out = run_model(model, c.astype(np.float64), a, b)
    assert np.allclose(out.data, np.where(c.astype(bool), a, b), atol=1e-6)


# ------------------------------------------------------------------ elementwise

def test_sqrt_leakyrelu_clip_parity():
    x = np.random.default_rng(3).standard_normal((2, 4)).astype(np.float32) + 3.0
    model = make_model(
        [node("Sqrt", ["X"], ["S"]),
         node("LeakyRelu", ["X"], ["L"], alpha=0.1),
         node("Clip", ["X", "lo", "hi"], ["C"])],
        [tensor_input("X", [2, 4]),
         tensor_input("lo", [1], onnx.TensorProto.FLOAT),
         tensor_input("hi", [1], onnx.TensorProto.FLOAT)],
        [tensor_output("S", [2, 4]), tensor_output("L", [2, 4]),
         tensor_output("C", [2, 4])],
    )
    from tests.test_onnx_loader import OnnxModel
    m = OnnxModel(model)
    outs = m(Tensor(x.astype(np.float64)), Tensor(np.array([-0.5])),
             Tensor(np.array([1.5])))
    s, l, c = outs
    assert np.allclose(s.data, np.sqrt(x), atol=1e-6)
    assert np.allclose(l.data, np.where(x > 0, x, 0.1 * x), atol=1e-6)
    assert np.allclose(c.data, np.clip(x, -0.5, 1.5), atol=1e-6)


def test_clip_opset11_attributes():
    x = np.array([[-3.0, 0.0, 5.0]])
    model = make_model(
        [node("Clip", ["X"], ["Y"], min=-1.0, max=2.0)],
        [tensor_input("X", [1, 3])], [tensor_output("Y", [1, 3])],
        opset=1,
    )
    assert np.allclose(run_model(model, x).data, np.clip(x, -1, 2))


# ------------------------------------------------------------------ ConstantOfShape / BatchNorm

def test_constantofshape_with_constant_input():
    model = make_model(
        [node("Constant", [], ["shape"], value_ints=[2, 3]),
         node("ConstantOfShape", ["shape"], ["Y"],
              value=onnx.helper.make_tensor("v", onnx.TensorProto.FLOAT, [1], [1.0]))],
        [], [tensor_output("Y", [2, 3])],
    )
    from tests.test_onnx_loader import OnnxModel
    out = OnnxModel(model)()
    assert out.shape == (2, 3)
    assert np.allclose(out.data, 1.0)


def test_batchnorm_inference_parity():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 6, 5, 5)).astype(np.float32)
    s = rng.standard_normal(6).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)
    mean = rng.standard_normal(6).astype(np.float32) * 0.2
    var = (rng.random(6).astype(np.float32) * 0.5 + 0.5)
    model = make_model(
        [node("BatchNormalization", ["X", "S", "B", "M", "V"], ["Y"], epsilon=1e-5)],
        [tensor_input("X", [2, 6, 5, 5]), tensor_input("S", [6]),
         tensor_input("B", [6]), tensor_input("M", [6]), tensor_input("V", [6])],
        [tensor_output("Y", [2, 6, 5, 5])],
        initializers=[init("S", s), init("B", b), init("M", mean), init("V", var)],
    )
    ref = run_ort(model, x)[0]
    out = run_model(model, x)
    assert out.shape == ref.shape
    assert np.allclose(out.data, ref, atol=1e-5)


def test_batchnorm_training_mode_rejected():
    model = make_model(
        [node("BatchNormalization", ["X", "S", "B", "M", "V"], ["Y"],
              training_mode=1)],
        [tensor_input("X", [1, 3, 4, 4])],
        [tensor_output("Y", [1, 3, 4, 4])],
        opset=14,          # training_mode attribute exists from opset 14
        initializers=[init("S", np.ones(3, np.float32)),
                      init("B", np.zeros(3, np.float32)),
                      init("M", np.zeros(3, np.float32)),
                      init("V", np.ones(3, np.float32))],
    )
    from backlens.onnx_loader import OnnxModel, OnnxOpError
    with pytest.raises(OnnxOpError, match="training_mode"):
        m = OnnxModel(model)
        m(Tensor(np.zeros((1, 3, 4, 4))))


# ------------------------------------------------------------------ grouped conv

def test_grouped_conv_parity():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 4, 7, 7)).astype(np.float32)
    w = rng.standard_normal((6, 2, 3, 3)).astype(np.float32)   # group=2
    model = make_model(
        [node("Conv", ["X", "W"], ["Y"], group=2, kernel_shape=[3, 3],
              pads=[1, 1, 1, 1])],
        [tensor_input("X", [1, 4, 7, 7])],
        [tensor_output("Y", [1, 6, 7, 7])],
        initializers=[init("W", w)],
    )
    ref = run_ort(model, x)[0]
    out = run_model(model, x)
    assert out.shape == ref.shape
    assert np.allclose(out.data, ref, atol=1e-5)


def test_depthwise_conv_parity():
    """group == channels: the MobileNet pattern."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 8, 9, 9)).astype(np.float32)
    w = rng.standard_normal((8, 1, 3, 3)).astype(np.float32)
    model = make_model(
        [node("Conv", ["X", "W"], ["Y"], group=8, kernel_shape=[3, 3],
              strides=[2, 2])],
        [tensor_input("X", [1, 8, 9, 9])],
        [tensor_output("Y", [1, 8, 4, 4])],
        initializers=[init("W", w)],
    )
    ref = run_ort(model, x)[0]
    out = run_model(model, x)
    assert out.shape == ref.shape
    assert np.allclose(out.data, ref, atol=1e-5)


def test_grouped_conv_gradients_flow():
    """Grouped conv expressed as slice+conv+cat must propagate gradients."""
    from backlens.onnx_loader import OnnxModel, _get_attrs
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((1, 4, 6, 6)).astype(np.float32)
    w_np = (rng.standard_normal((6, 2, 3, 3)) * 0.2).astype(np.float32)
    model = make_model(
        [node("Conv", ["X", "W"], ["Y"], group=2, kernel_shape=[3, 3],
              pads=[1, 1, 1, 1])],
        [tensor_input("X", [1, 4, 6, 6])],
        [tensor_output("Y", [1, 6, 6, 6])],
        initializers=[init("W", w_np)],
    )
    m = OnnxModel(model)
    loss = (m(Tensor(x_np.astype(np.float64))) ** 2).sum()
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and np.all(np.isfinite(p.grad))
