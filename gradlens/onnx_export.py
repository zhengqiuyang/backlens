"""Export GradLens models to ONNX -- train here, deploy anywhere.

``export_onnx(model, dummy_input, "model.onnx")`` runs the model's forward
pass once, walks the recorded engine graph, and re-emits it as a standard
ONNX file: parameters become initializers (float32), engine ops become ONNX
ops. The result runs in onnxruntime, Netron, and any ONNX-compatible runtime
-- and loads back into GradLens via :func:`gradlens.onnx_loader.load_onnx`,
closing the loop.

The mapping is 1:1 (MatMul+Add, no Gemm fusion); ops without an ONNX
counterpart in the graph (e.g. ``max``) are rejected up front with an
explicit error. Exports target opset 11 (attribute-style reductions) for
the widest runtime compatibility; models are verified with ``onnx.checker``.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from .engine import Tensor, no_grad
from .nn import Module


class ExportError(Exception):
    """The graph contains an op that cannot be represented in ONNX."""


# engine op -> ONNX op type, for simple 1:1 mappings
_SIMPLE = {
    "add": "Add", "mul": "Mul", "div": "Div",
    "matmul": "MatMul",
    "relu": "Relu", "tanh": "Tanh", "sigmoid": "Sigmoid",
    "exp": "Exp", "log": "Log", "abs": "Abs", "neg": "Neg",
}


def export_onnx(module: Module, dummy_input, path: str,
                name: str = "gradlens_model", opset: int = 11,
                dynamic_batch: bool = True):
    """Export ``module`` to ``path`` after tracing one forward pass.

    ``dummy_input`` is array-like with the model's input shape. With
    ``dynamic_batch=True`` (default) the batch axis becomes a dynamic
    dimension, so one export serves any batch size. Returns the ONNX
    ModelProto.
    """
    import onnx
    import onnx.helper as h
    import onnx.numpy_helper as nh

    # a normal forward pass: the engine graph must be recorded op by op
    x = Tensor(np.asarray(dummy_input, dtype=np.float64))
    out = module(x)
    topo = out._topo()

    allowed = set(_SIMPLE) | {"reshape", "transpose", "cat",
                              "sum", "mean", "conv2d", "maxpool2d"}
    unsupported = sorted({
        t._op for t in topo
        if t._op and not (t._op in allowed or t._op.startswith("pow"))
    })
    if unsupported:
        raise ExportError(f"graph contains non-exportable ops: {', '.join(unsupported)}")

    names: Dict[int, str] = {id(x): "input"}
    initializers = []
    nodes: List = []
    used_names = {"input"}

    def unique(nm: str) -> str:
        base, k = nm, 0
        while nm in used_names:
            k += 1
            nm = f"{base}_{k}"
        used_names.add(nm)
        return nm

    def add_init(nm: str, arr: np.ndarray) -> str:
        initializers.append(nh.from_array(np.asarray(arr), nm))
        return nm

    def const_float(nm: str, arr: np.ndarray) -> str:
        """Unique-ify ``nm`` and register the initializer; returns the name."""
        return add_init(unique(nm), np.asarray(arr, dtype=np.float32))

    def const_int64(nm: str, arr) -> str:
        return add_init(unique(nm), np.asarray(arr, dtype=np.int64))

    for i, t in enumerate(topo):
        if t is x:
            continue
        op = t._op
        if not t._prev:                       # leaf: parameter or constant
            nm = unique(t.label or f"p{i}")
            names[id(t)] = nm
            add_init(nm, t.data.astype(np.float32))
            continue
        nm = unique(f"t{i}")
        names[id(t)] = nm
        ins = [names[id(p)] for p in t._prev]

        if op in _SIMPLE:
            nodes.append(h.make_node(_SIMPLE[op], ins, [nm]))
        elif op == "reshape":
            shape_name = const_int64(f"{nm}_shape", t._attrs["shape"])
            nodes.append(h.make_node("Reshape", [ins[0], shape_name], [nm]))
        elif op == "transpose":
            perm = t._attrs.get("axes")
            nodes.append(h.make_node("Transpose", ins, [nm],
                                     perm=list(perm) if perm else None))
        elif op == "cat":
            nodes.append(h.make_node("Concat", ins, [nm],
                                     axis=int(t._attrs["axis"])))
        elif op in ("sum", "mean"):
            onnx_op = "ReduceSum" if op == "sum" else "ReduceMean"
            axes = t._attrs.get("axis")
            nodes.append(h.make_node(onnx_op, ins, [nm],
                                     axes=list(axes) if axes is not None else None,
                                     keepdims=int(bool(t._attrs.get("keepdims")))))
        elif op.startswith("pow"):
            exp_name = const_float(f"{nm}_exp", float(t._attrs["exponent"]))
            nodes.append(h.make_node("Pow", [ins[0], exp_name], [nm]))
        elif op == "conv2d":
            attrs = t._attrs
            kh, kw = attrs["kernel"]
            ph_b, ph_e, pw_b, pw_e = attrs["pads"]
            nodes.append(h.make_node(
                "Conv", ins, [nm],
                kernel_shape=[kh, kw],
                strides=[attrs["stride"], attrs["stride"]],
                pads=[ph_b, pw_b, ph_e, pw_e],   # ONNX [Hb, Wb, He, We]
            ))
        elif op == "maxpool2d":
            attrs = t._attrs
            k, s = attrs["kernel"], attrs["stride"]
            ph_b, ph_e, pw_b, pw_e = attrs["pads"]
            nodes.append(h.make_node(
                "MaxPool", ins, [nm],
                kernel_shape=[k, k], strides=[s, s],
                pads=[ph_b, pw_b, ph_e, pw_e],
            ))
        else:                                  # pragma: no cover - guarded above
            raise ExportError(f"op '{op}' is not exportable")

    def value_info(nm, shape):
        dims = ["N" if (axis == 0 and dynamic_batch) else d
                for axis, d in enumerate(shape)]
        return h.make_tensor_value_info(nm, onnx.TensorProto.FLOAT, dims)

    graph = h.make_graph(
        nodes, name,
        [value_info("input", x.shape)],
        [value_info(names[id(out)], out.shape)],
        initializer=initializers,
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", opset)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model
