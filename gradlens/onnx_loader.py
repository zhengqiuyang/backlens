"""Run ONNX models as GradLens graphs.

``load_onnx(path)`` parses an ONNX file and rebuilds its forward pass out of
plain engine ops. The result is a callable whose output is an ordinary
:class:`~gradlens.engine.Tensor` -- which means everything GradLens does comes
free for any ONNX model: ``debug_backward`` on its loss, ``gradcheck``,
Mermaid graph export, and an animated Backprop Film of a real pretrained
network's backward pass.

The ``onnx`` package is imported lazily; install with ``pip install gradlens[onnx]``.

Scope: inference + gradient flow for a practical MLP/CNN subset
(MatMul/Gemm, Add/Mul/..., Conv, MaxPool, Reshape/Flatten/Transpose/Concat,
Softmax, reductions). Control flow (If/Loop), training-oriented ops and
grouped/dilated convs are rejected with an explicit, readable error.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Union

from .engine import Tensor, cat

SUPPORTED_OPS = {
    "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Exp", "Log",
    "Relu", "Tanh", "Sigmoid", "Softmax",
    "MatMul", "Gemm", "Pow", "Sum", "Identity", "Constant",
    "Reshape", "Flatten", "Transpose", "Concat", "Squeeze", "Unsqueeze",
    "ReduceSum", "ReduceMean", "GlobalAveragePool",
    "Conv", "MaxPool",
}


class OnnxOpError(Exception):
    """An ONNX graph uses something outside GradLens's supported subset."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_attrs(node) -> Dict[str, object]:
    import onnx.helper
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _softmax(t: Tensor, axis: int = -1) -> Tensor:
    axis = axis % t.ndim
    # detached row-max shift: softmax is exactly shift-invariant, so gradients
    # remain correct (same trick as cross_entropy)
    shift = Tensor(t.data.max(axis=axis, keepdims=True))
    e = (t - shift).exp()
    return e / e.sum(axis=axis, keepdims=True)


def _resolve_shape(orig: Sequence[int], target) -> tuple:
    """ONNX Reshape semantics: 0 keeps the dim, -1 infers it (at most once)."""
    dims = [int(v) for v in np.asarray(target).ravel()]
    out, known, infer_at = list(dims), 1, -1
    for i, v in enumerate(dims):
        if v == -1:
            infer_at = i
        elif v == 0:
            out[i] = orig[i]
            known *= orig[i]
        else:
            known *= v
    if infer_at >= 0:
        rest = max(1, int(np.prod(orig))) // max(1, known)
        out[infer_at] = rest
    return tuple(out)


def _same_pads(size: int, kernel: int, stride: int, mode: str) -> tuple:
    out = -(-size // stride)                       # ceil
    needed = max(0, (out - 1) * stride + kernel - size)
    begin = needed // 2 if mode == "SAME_UPPER" else needed - needed // 2
    return begin, needed - begin


# ---------------------------------------------------------------------------
# op builders: (inputs: List[Tensor], attrs: dict, opset: int) -> Tensor
# constant inputs (shape/axes tensors) arrive as ordinary int Tensors
# ---------------------------------------------------------------------------

def _b_binary(op):
    def build(ins, attrs, opset):
        out = ins[0]
        for other in ins[1:]:
            out = getattr(out, op)(other)
        return out
    return build


def _b_unary(op):
    return lambda ins, attrs, opset: getattr(ins[0], op)()


def _b_softmax(ins, attrs, opset):
    axis = int(attrs.get("axis", 1 if opset < 13 else -1))
    t = ins[0]
    if opset < 13 and t.ndim > 2:
        # legacy semantics: coerce to (N, -1) around the softmax axis=1
        r = t.reshape(t.shape[0], -1)
        return _softmax(r, 1).reshape(t.shape)
    return _softmax(t, axis)


def _b_matmul(ins, attrs, opset):
    return ins[0] @ ins[1]


def _b_gemm(ins, attrs, opset):
    a, b = ins[0], ins[1]
    if int(attrs.get("transA", 0)):
        a = a.transpose()
    if int(attrs.get("transB", 0)):
        b = b.transpose()
    out = (a @ b) * float(attrs.get("alpha", 1.0))
    if len(ins) > 2:
        out = out + ins[2] * float(attrs.get("beta", 1.0))
    return out


def _b_reshape(ins, attrs, opset):
    t, shape = ins[0], ins[1].data
    return t.reshape(_resolve_shape(t.shape, shape))


def _b_flatten(ins, attrs, opset):
    t = ins[0]
    axis = int(attrs.get("axis", 1)) % (t.ndim + 1)
    lead = int(np.prod(t.shape[:axis])) if axis else 1
    return t.reshape(lead, -1)


def _b_transpose(ins, attrs, opset):
    perm = attrs.get("perm")
    return ins[0].transpose(tuple(perm) if perm else None)


def _b_concat(ins, attrs, opset):
    return cat(ins, axis=int(attrs.get("axis", 0)))


def _b_reduce(op):
    def build(ins, attrs, opset):
        t = ins[0]
        axes = attrs.get("axes")
        if axes is None and len(ins) > 1:          # opset 13+: axes as input
            axes = ins[1].data
        axes = tuple(int(a) for a in axes) if axes is not None else None
        keep = bool(int(attrs.get("keepdims", 1)))
        return getattr(t, op)(axis=axes, keepdims=keep)
    return build


def _b_gap(ins, attrs, opset):
    t = ins[0]
    return t.mean(axis=tuple(range(2, t.ndim)), keepdims=True)


def _b_conv(ins, attrs, opset):
    x, w = ins[0], ins[1]
    b = ins[2] if len(ins) > 2 else None
    if int(attrs.get("group", 1)) != 1:
        raise OnnxOpError("Conv: grouped convolution is not supported")
    dil = list(attrs.get("dilations", [1, 1]))
    if dil != [1, 1]:
        raise OnnxOpError(f"Conv: dilations {dil} != [1, 1] are not supported")
    strides = [int(s) for s in attrs.get("strides", [1, 1])]
    if strides[0] != strides[1]:
        raise OnnxOpError(f"Conv: anisotropic strides {strides} are not supported")
    stride = strides[0]
    kernel = w.shape[2:]
    ks = attrs.get("kernel_shape")
    if ks and tuple(ks) != tuple(kernel):
        raise OnnxOpError(f"Conv: kernel_shape {ks} disagrees with weight {w.shape}")
    auto = attrs.get("auto_pad", b"NOTSET")
    auto = auto.decode() if isinstance(auto, bytes) else auto
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        ph = _same_pads(x.shape[2], kernel[0], stride, auto)
        pw = _same_pads(x.shape[3], kernel[1], stride, auto)
        pads = (ph[0], ph[1], pw[0], pw[1])
    elif auto == "VALID":
        pads = (0, 0, 0, 0)
    else:
        # ONNX pads = [H_begin, W_begin, H_end, W_end]; engine wants
        # (H_begin, H_end, W_begin, W_end) -- reorder (verified vs onnxruntime)
        p = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
        pads = (p[0], p[2], p[1], p[3])
    return x.conv2d(w, b, stride=stride, pad=pads)


def _b_maxpool(ins, attrs, opset):
    x = ins[0]
    auto = attrs.get("auto_pad", b"NOTSET")
    auto = auto.decode() if isinstance(auto, bytes) else auto
    if auto not in ("NOTSET", "VALID"):
        raise OnnxOpError(f"MaxPool: auto_pad {auto} is not supported")
    if int(attrs.get("ceil_mode", 0)) != 0:
        raise OnnxOpError("MaxPool: ceil_mode != 0 is not supported")
    dil = list(attrs.get("dilations", [1, 1]))
    if dil != [1, 1]:
        raise OnnxOpError(f"MaxPool: dilations {dil} != [1, 1] are not supported")
    ks = [int(k) for k in attrs["kernel_shape"]]
    if ks[0] != ks[1]:
        raise OnnxOpError(f"MaxPool: non-square kernel {ks} is not supported")
    strides = [int(s) for s in attrs.get("strides", [1, 1])]
    if strides[0] != strides[1]:
        raise OnnxOpError(f"MaxPool: anisotropic strides {strides} are not supported")
    p = [int(v) for v in attrs.get("pads", [0, 0, 0, 0])]
    pads = (p[0], p[2], p[1], p[3])   # ONNX [Hb,Wb,He,We] -> engine (Hb,He,Wb,We)
    return x.maxpool2d(ks[0], strides[0], pads)


def _b_squeeze(ins, attrs, opset):
    t = ins[0]
    axes = attrs.get("axes")
    if axes is None and len(ins) > 1:
        axes = ins[1].data
    if axes is None:
        axes = [i for i, s in enumerate(t.shape) if s == 1]
    keep = [s for i, s in enumerate(t.shape) if i not in {int(a) for a in axes}]
    return t.reshape(keep)


def _b_unsqueeze(ins, attrs, opset):
    t = ins[0]
    axes = attrs.get("axes")
    if axes is None and len(ins) > 1:
        axes = ins[1].data
    shape = list(t.shape)
    for a in sorted(int(a) for a in axes):
        shape.insert(a, 1)
    return t.reshape(shape)


def _b_pow(ins, attrs, opset):
    t, e = ins[0], ins[1]
    if e.size != 1:
        raise OnnxOpError("Pow: only scalar exponents are supported")
    return t ** float(e.data.ravel()[0])


# ---------------------------------------------------------------------------
# model wrapper
# ---------------------------------------------------------------------------

class OnnxModel:
    """A loaded ONNX model, callable and fully inside the GradLens engine.

    Float initializers become trainable ``Tensor`` parameters (labels keep
    their ONNX names), so the model can be fine-tuned with GradLens optimizers
    and inspected with ``debug_backward`` / ``film_backward``.
    """

    def __init__(self, proto):
        import onnx
        import onnx.numpy_helper

        self._proto = proto
        graph = proto.graph
        self.opset = max(
            (o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")),
            default=13,
        )

        unsupported = sorted({n.op_type for n in graph.node} - SUPPORTED_OPS)
        if unsupported:
            raise OnnxOpError(
                f"model uses ops outside GradLens's subset: {', '.join(unsupported)}"
            )

        # initializers -> named tensors; float ones are trainable parameters
        self._static: Dict[str, Tensor] = {}
        self._params: List[Tensor] = []
        for init in graph.initializer:
            arr = onnx.numpy_helper.to_array(init)
            if init.data_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                self._static[init.name] = Tensor(arr.astype(np.int64), label=init.name)
            else:
                t = Tensor(arr.astype(np.float64), requires_grad=True, label=init.name)
                self._static[init.name] = t
                self._params.append(t)

        # callable inputs = graph inputs that are not initializers
        init_names = {i.name for i in graph.initializer}
        self.input_names = [i.name for i in graph.input if i.name not in init_names]
        self.output_names = [o.name for o in graph.output]
        self.ops = [(n.op_type, _get_attrs(n), list(n.input), list(n.output))
                    for n in graph.node]

    # -- running ---------------------------------------------------------------

    def __call__(self, *inputs: Union[Tensor, np.ndarray]) -> Tensor:
        env: Dict[str, Tensor] = dict(self._static)
        for name, value in zip(self.input_names, inputs):
            env[name] = value if isinstance(value, Tensor) else Tensor(value, label=name)

        for op_type, attrs, ins, outs in self.ops:
            if op_type == "Constant":
                import onnx.numpy_helper
                t = None
                if "value" in attrs:
                    t = Tensor(onnx.numpy_helper.to_array(attrs["value"]).astype(np.float64))
                elif "value_float" in attrs:
                    t = Tensor(float(attrs["value_float"]))
                elif "value_floats" in attrs:
                    t = Tensor(np.asarray(attrs["value_floats"], dtype=np.float64))
                elif "value_int" in attrs:
                    t = Tensor(int(attrs["value_int"]))
                elif "value_ints" in attrs:
                    t = Tensor(np.asarray(attrs["value_ints"], dtype=np.int64))
                if t is None:
                    raise OnnxOpError("Constant without a supported value attribute")
                t.label = outs[0]
                env[outs[0]] = t
                continue
            try:
                missing = [n for n in ins if n and n not in env]
                if missing:
                    raise OnnxOpError(f"unresolved inputs: {missing}")
                tensors = [env[n] for n in ins if n]
                out = _build_op(op_type, tensors, attrs, self.opset)
            except OnnxOpError:
                raise
            except Exception as e:                              # noqa: BLE001
                raise OnnxOpError(f"while executing {op_type} {ins} -> {outs}: {e}") from e
            if len(outs) == 1:
                env[outs[0]] = out
            else:
                for name, t in zip(outs, out):
                    env[name] = t

        results = [env[name] for name in self.output_names]
        return results[0] if len(results) == 1 else results

    # -- module-ish API ----------------------------------------------------------

    def parameters(self) -> List[Tensor]:
        return list(self._params)

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def __repr__(self):
        counts: Dict[str, int] = {}
        for op_type, *_ in self.ops:
            counts[op_type] = counts.get(op_type, 0) + 1
        body = ", ".join(f"{k}x{v}" for k, v in counts.items())
        return f"OnnxModel({len(self._params)} params, ops: {body})"


# ---------------------------------------------------------------------------
# registry & entry points
# ---------------------------------------------------------------------------

def _build_op(op_type, tensors, attrs, opset):
    builder = _REGISTRY.get(op_type)
    if builder is None:
        raise OnnxOpError(f"op '{op_type}' is not supported")
    return builder(tensors, attrs, opset)


_REGISTRY = {
    "Add": _b_binary("__add__"),
    "Sub": _b_binary("__sub__"),
    "Mul": _b_binary("__mul__"),
    "Div": _b_binary("__truediv__"),
    "Neg": _b_unary("__neg__"),
    "Abs": _b_unary("abs"),
    "Exp": _b_unary("exp"),
    "Log": _b_unary("log"),
    "Relu": _b_unary("relu"),
    "Tanh": _b_unary("tanh"),
    "Sigmoid": _b_unary("sigmoid"),
    "Softmax": _b_softmax,
    "MatMul": _b_matmul,
    "Gemm": _b_gemm,
    "Pow": _b_pow,
    "Sum": _b_binary("__add__"),
    "Identity": lambda ins, attrs, opset: ins[0],
    "Reshape": _b_reshape,
    "Flatten": _b_flatten,
    "Transpose": _b_transpose,
    "Concat": _b_concat,
    "Squeeze": _b_squeeze,
    "Unsqueeze": _b_unsqueeze,
    "ReduceSum": _b_reduce("sum"),
    "ReduceMean": _b_reduce("mean"),
    "GlobalAveragePool": _b_gap,
    "Conv": _b_conv,
    "MaxPool": _b_maxpool,
}


def load_onnx(path) -> OnnxModel:
    """Load an ONNX file into a callable :class:`OnnxModel`."""
    import onnx
    return OnnxModel(onnx.load(str(path)))


def analyze_onnx(path) -> Dict[str, object]:
    """Report op coverage of an ONNX model without executing it."""
    import onnx
    proto = onnx.load(str(path))
    counts: Dict[str, int] = {}
    for n in proto.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    unsupported = sorted(set(counts) - SUPPORTED_OPS)
    return {"ops": counts, "unsupported": unsupported,
            "loadable": not unsupported}
