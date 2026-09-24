"""Tests for engine ops added for ONNX support: transpose(axes), cat,
conv2d (im2col/col2im), maxpool2d."""

import numpy as np
import pytest

from backlens import Tensor, cat
from backlens.check import gradcheck


def t(rng, shape, **kw):
    return Tensor(rng.standard_normal(shape), **kw)


# --------------------------------------------------------------- transpose

def test_transpose_general_perm_roundtrip():
    rng = np.random.default_rng(0)
    x, = [t(rng, (2, 3, 4), requires_grad=True)]
    y = x.transpose((2, 0, 1))
    assert y.shape == (4, 2, 3)
    y.sum().backward()
    assert np.allclose(x.grad, np.ones((2, 3, 4)))


def test_transpose_perm_gradcheck():
    rng = np.random.default_rng(1)
    x, = [t(rng, (2, 3), requires_grad=True)]
    assert gradcheck(lambda v: v.transpose((1, 0)).sum() + (v * v).mean(), x)


def test_transpose_negative_axis():
    rng = np.random.default_rng(2)
    x, = [t(rng, (2, 3, 4), requires_grad=True)]
    assert x.transpose((-1, 0, 1)).shape == (4, 2, 3)


# --------------------------------------------------------------- cat

def test_cat_forward_and_gradcheck():
    rng = np.random.default_rng(3)
    a, b, c = (t(rng, s, requires_grad=True) for s in [(2, 3), (1, 3), (4, 3)])
    assert gradcheck(lambda *xs: (cat(xs, axis=0) ** 2).sum() + cat(xs, axis=0).mean(), a, b, c)


def test_cat_axis1_grads():
    rng = np.random.default_rng(4)
    a, b = t(rng, (2, 3), requires_grad=True), t(rng, (2, 5), requires_grad=True)
    out = cat([a, b], axis=1)
    assert out.shape == (2, 8)
    out.sum().backward()
    assert np.allclose(a.grad, 1)
    assert np.allclose(b.grad, 1)


def test_cat_requires_grad_propagates():
    rng = np.random.default_rng(5)
    a, b = t(rng, (2, 2), requires_grad=True), t(rng, (2, 2))
    assert cat([a, b]).requires_grad
    assert not cat([b, b]).requires_grad


# --------------------------------------------------------------- conv2d

def test_conv2d_forward_matches_reference():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 2, 5, 5))
    w = rng.standard_normal((3, 2, 3, 3))
    b = rng.standard_normal(3)
    out = Tensor(x).conv2d(Tensor(w), Tensor(b), stride=1, pad=1)
    # naive reference
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((1, 3, 5, 5))
    for m in range(3):
        for i in range(5):
            for j in range(5):
                ref[0, m, i, j] = (xp[0, :, i:i + 3, j:j + 3] * w[m]).sum() + b[m]
    assert np.allclose(out.data, ref, atol=1e-10)


def test_conv2d_stride2_no_pad():
    rng = np.random.default_rng(7)
    x = Tensor(rng.standard_normal((2, 1, 6, 6)), requires_grad=True)
    w = Tensor(rng.standard_normal((4, 1, 2, 2)), requires_grad=True)
    out = x.conv2d(w, stride=2)
    assert out.shape == (2, 4, 3, 3)
    assert gradcheck(lambda xv: (xv.conv2d(w) ** 2).sum(), x)


def test_conv2d_gradcheck_weight_and_bias():
    rng = np.random.default_rng(8)
    x = Tensor(rng.standard_normal((1, 2, 4, 4)))
    w = Tensor(rng.standard_normal(3) * 0.1 + rng.standard_normal((3, 2, 3, 3)) * 0.1,
               requires_grad=True, label="w")
    b = Tensor(rng.standard_normal(3) * 0.1, requires_grad=True, label="b")
    assert gradcheck(lambda wv, bv: (x.conv2d(wv, bv, pad=1).tanh() ** 2).mean(), w, b)


def test_conv2d_asymmetric_pads():
    rng = np.random.default_rng(9)
    x = rng.standard_normal((1, 1, 4, 4))
    w = rng.standard_normal((1, 1, 2, 2))
    out = Tensor(x).conv2d(Tensor(w), pad=(1, 0, 0, 1))     # top / bottom / left / right
    assert out.shape == (1, 1, 4, 4)                        # padded 5x5, kernel 2
    xp = np.pad(x, ((0, 0), (0, 0), (1, 0), (0, 1)))
    ref = np.zeros((1, 1, 4, 4))
    for i in range(4):
        for j in range(4):
            ref[0, 0, i, j] = (xp[0, :, i:i + 2, j:j + 2] * w[0]).sum()
    assert np.allclose(out.data, ref, atol=1e-10)


def test_conv2d_input_gradcheck_overlapping_windows():
    # stride < kernel: col2im must scatter-add overlapping positions
    rng = np.random.default_rng(10)
    x = Tensor(rng.standard_normal((1, 1, 4, 4)), requires_grad=True)
    w = Tensor(rng.standard_normal((1, 1, 3, 3)) * 0.3, requires_grad=False)
    assert gradcheck(lambda xv: (xv.conv2d(w, stride=2, pad=1) ** 2).sum(), x)


# --------------------------------------------------------------- maxpool2d

def test_maxpool2d_forward():
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    out = Tensor(x).maxpool2d(2, 2)
    assert out.shape == (1, 1, 2, 2)
    assert np.allclose(out.data, [[[5, 7], [13, 15]]])


def test_maxpool2d_stride3_partial_coverage():
    x = np.arange(14 * 14, dtype=float).reshape(1, 1, 14, 14)
    out = Tensor(x).maxpool2d(3, 3)
    assert out.shape == (1, 1, 4, 4)                     # floor((14-3)/3)+1 = 4


def test_maxpool2d_grad_routes_to_argmax():
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    t = Tensor(x, requires_grad=True)
    (t.maxpool2d(2, 2) * 5.0).sum().backward()
    assert np.allclose(t.grad, [[[[0, 0], [0, 5]]]])


def test_maxpool2d_overlapping_windows_gradcheck():
    rng = np.random.default_rng(11)
    x = Tensor(rng.standard_normal((1, 1, 4, 4)), requires_grad=True)
    assert gradcheck(lambda xv: (xv.maxpool2d(2, 1) ** 2).sum(), x)


def test_maxpool2d_with_pads():
    x = np.arange(4, dtype=float).reshape(1, 1, 2, 2)
    out = Tensor(x).maxpool2d(2, 2, pads=(1, 1, 1, 1))
    assert out.shape == (1, 1, 2, 2)
    # padded 4x4 with -inf border; windows pick up the true values
    assert np.allclose(out.data, [[[0.0, 1.0], [2.0, 3.0]]])
