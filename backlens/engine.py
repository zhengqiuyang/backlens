"""BackLens autograd engine.

A numpy-backed, vectorized reverse-mode automatic differentiation engine in the
spirit of micrograd -- but operating on arrays instead of scalars, with
first-class observability: every op is named, traceable, and hookable, so the
backward pass can be recorded, stepped through, and inspected.

Design notes
------------
* ``Tensor`` wraps a ``numpy.ndarray``. Each op records its inputs (``_prev``),
  an op name (``_op``), and a closure (``_backward``) that propagates gradients.
* ``backward()`` runs the classic topological-sort loop, but can optionally
  return the *trace*: the list of nodes in the exact order their ``_backward``
  closures executed. That trace is what :mod:`backlens.debug` turns into a
  step-by-step view of backpropagation.
* Hooks: ``tensor.register_hook(fn)`` fires when the gradient w.r.t. that
  tensor is finalized; ``set_op_hook(op, fn)`` fires for every node of a given
  op type. Hooks may return a replacement gradient.
"""

from __future__ import annotations

import contextlib
import functools
import numpy as np
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# global grad mode
# ---------------------------------------------------------------------------

_grad_enabled = True


class _NoGrad:
    def __enter__(self) -> None:
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *exc) -> None:
        global _grad_enabled
        _grad_enabled = self._prev


def no_grad(fn=None):
    """Disable gradient tracking inside a ``with`` block or function decorator."""
    if fn is None:
        return _NoGrad()
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _NoGrad():
            return fn(*args, **kwargs)
    return wrapper


def is_grad_enabled() -> bool:
    return _grad_enabled


# per-op-type hooks: op name -> list of callables(grad ndarray) -> Optional[ndarray]
_OP_HOOKS: Dict[str, List[Callable[[np.ndarray], Optional[np.ndarray]]]] = {}


def set_op_hook(op: str, fn: Callable[[np.ndarray], Optional[np.ndarray]]) -> None:
    """Register a hook fired for every node of op type ``op`` during backward."""
    _OP_HOOKS.setdefault(op, []).append(fn)


def clear_op_hooks() -> None:
    _OP_HOOKS.clear()


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing numpy broadcasting.

    This is the piece that makes PyTorch-style broadcasting work in reverse:
    dimensions that were expanded get their gradients summed.
    """
    if grad.shape == shape:
        return grad
    # extra leading axes added by broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # axes that were size-1 in the original and got stretched
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


Arrayable = Union[float, int, "Tensor", np.ndarray]


class Tensor:
    """An ndarray with a tape node attached.

    Args:
        data: array-like (or another Tensor, copied by reference to keep the
            common ``Tensor(other.data)`` pattern cheap).
        requires_grad: if True, ``.grad`` accumulates during ``backward()``.
        label: optional name used by graph visualization and debug reports.
    """

    def __init__(self, data: Arrayable, requires_grad: bool = False, label: Optional[str] = None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self.label = label
        self._backward: Callable[[], None] = lambda: None
        self._prev: Tuple["Tensor", ...] = ()
        self._op: str = ""          # "" for leaves
        self._attrs: Dict = {}      # op attributes (axis, pads, kernel, ...)
        self._hooks: List[Callable[[np.ndarray], Optional[np.ndarray]]] = []

    # -- basic introspection -------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def register_hook(self, fn: Callable[[np.ndarray], Optional[np.ndarray]]) -> None:
        """Call ``fn(grad)`` when this tensor's gradient is finalized.

        If ``fn`` returns an ndarray, the gradient is replaced by it.
        """
        self._hooks.append(fn)

    def parameters(self) -> List["Tensor"]:
        return []

    # -- arithmetic ----------------------------------------------------------

    @staticmethod
    def _ensure(other: Arrayable) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def _make(self, data, parents: Iterable["Tensor"], op: str,
              attrs: Optional[Dict] = None) -> "Tensor":
        """Create a result node wired into the graph (honors no_grad)."""
        out = Tensor(data)
        if _grad_enabled:
            parents = tuple(parents)
            out._prev = parents
            out._op = op
            out._attrs = dict(attrs or {})
            out.requires_grad = any(p.requires_grad for p in parents)
        return out

    def __add__(self, other):
        other = self._ensure(other)
        out = self._make(self.data + other.data, (self, other), "add")

        def _backward():
            if self.requires_grad:
                self._acc(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(out.grad, other.shape))
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = self._make(-self.data, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self._acc(-out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-self._ensure(other))

    def __rsub__(self, other):
        return self._ensure(other) + (-self)

    def __mul__(self, other):
        other = self._ensure(other)
        out = self._make(self.data * other.data, (self, other), "mul")

        def _backward():
            if self.requires_grad:
                self._acc(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(out.grad * self.data, other.shape))
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._ensure(other)
        out = self._make(self.data / other.data, (self, other), "div")

        def _backward():
            if self.requires_grad:
                self._acc(_unbroadcast(out.grad / other.data, self.shape))
            if other.requires_grad:
                other._acc(_unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape))
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return self._ensure(other) / self

    def __pow__(self, exponent: Union[int, float]):
        assert isinstance(exponent, (int, float)), "only scalar exponents are supported"
        out = self._make(self.data ** exponent, (self,), f"pow{exponent}", {"exponent": exponent})

        def _backward():
            if self.requires_grad:
                self._acc(out.grad * exponent * (self.data ** (exponent - 1)))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self._matmul(other)

    def __rmatmul__(self, other):
        return self._ensure(other)._matmul(self)

    def _matmul(self, other: "Tensor") -> "Tensor":
        a, b = self, other
        if a.ndim > 2 or b.ndim > 2:
            raise NotImplementedError("matmul supports 1D/2D operands only")
        out = self._make(a.data @ b.data, (a, b), "matmul")

        def _backward():
            if a.ndim == 1 and b.ndim == 1:            # dot -> scalar
                if a.requires_grad:
                    a._acc(out.grad * b.data)
                if b.requires_grad:
                    b._acc(out.grad * a.data)
            elif a.ndim == 1:                          # (n,) @ (n,m) -> (m,)
                if a.requires_grad:
                    a._acc(out.grad @ b.data.T)
                if b.requires_grad:
                    b._acc(np.outer(a.data, out.grad))
            elif b.ndim == 1:                          # (k,n) @ (n,) -> (k,)
                if a.requires_grad:
                    a._acc(np.outer(out.grad, b.data))
                if b.requires_grad:
                    b._acc(a.data.T @ out.grad)
            else:                                      # (k,n) @ (n,m) -> (k,m)
                if a.requires_grad:
                    a._acc(out.grad @ b.data.T)
                if b.requires_grad:
                    b._acc(a.data.T @ out.grad)
        out._backward = _backward
        return out

    # -- elementwise functions ----------------------------------------------

    def exp(self):
        out = self._make(np.exp(self.data), (self,), "exp")

        def _backward():
            if self.requires_grad:
                self._acc(out.grad * out.data)
        out._backward = _backward
        return out

    def log(self):
        out = self._make(np.log(self.data), (self,), "log")

        def _backward():
            if self.requires_grad:
                self._acc(out.grad / self.data)
        out._backward = _backward
        return out

    def tanh(self):
        out = self._make(np.tanh(self.data), (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self._acc(out.grad * (1.0 - out.data ** 2))
        out._backward = _backward
        return out

    def relu(self):
        out = self._make(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._acc(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = self._make(s, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self._acc(out.grad * out.data * (1.0 - out.data))
        out._backward = _backward
        return out

    def abs(self):
        out = self._make(np.abs(self.data), (self,), "abs")

        def _backward():
            if self.requires_grad:
                self._acc(out.grad * np.sign(self.data))
        out._backward = _backward
        return out

    # -- reductions / shape ---------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum",
                          {"axis": axis, "keepdims": keepdims})

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                self._acc(np.broadcast_to(grad, self.shape).copy())
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = self._make(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean",
                          {"axis": axis, "keepdims": keepdims})

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                n = self.size / out.size
                self._acc(np.broadcast_to(grad, self.shape).copy() / n)
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        """Maximum. Gradient routes entirely to the (first) argmax position."""
        if axis is not None:
            raise NotImplementedError("max supports global reduction only")
        out = self._make(np.max(self.data), (self,), "max")
        idx = np.unravel_index(np.argmax(self.data), self.shape)

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad.item() if out.grad.ndim == 0 else float(out.grad.ravel()[0])
                self._acc(grad)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = self._make(self.data.reshape(shape), (self,), "reshape", {"shape": tuple(shape)})

        def _backward():
            if self.requires_grad:
                self._acc(out.grad.reshape(self.shape))
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        """Transpose; ``axes=None`` reverses dimensions (numpy semantics)."""
        if axes is not None:
            axes = tuple(a % self.ndim for a in axes)
        out = self._make(np.transpose(self.data, axes), (self,), "transpose",
                          {"axes": axes})

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self._acc(np.transpose(out.grad))
                else:
                    inverse = tuple(np.argsort(axes))
                    self._acc(np.transpose(out.grad, inverse))
        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # -- autodiff core --------------------------------------------------------

    def conv2d(self, weight: "Tensor", bias: Optional["Tensor"] = None,
               stride: int = 1, pad=0) -> "Tensor":
        """2D convolution via im2col + matmul (group=1, NCHW layout).

        self: (N, C, H, W); weight: (M, C, kH, kW); bias: (M,).
        ``pad`` is symmetric pixels or an ONNX-style 4-tuple
        ``(H_begin, H_end, W_begin, W_end)``. Backward uses col2im
        scatter-add, so gradients flow to input, weight, and bias exactly.
        """
        a, w = self, weight
        if a.ndim != 4 or w.ndim != 4:
            raise NotImplementedError("conv2d supports NCHW x (M,C,kH,kW) only")
        if isinstance(pad, int):
            pads = (pad, pad, pad, pad)
        else:
            pads = tuple(int(p) for p in pad)
        n, c, h, wd = a.shape
        m, cw, kh, kw = w.shape
        if c != cw:
            raise ValueError(f"channel mismatch: input {c} vs weight {cw}")
        ho = (h + pads[0] + pads[1] - kh) // stride + 1
        wo = (wd + pads[2] + pads[3] - kw) // stride + 1

        xp = np.pad(a.data, ((0, 0), (0, 0), (pads[0], pads[1]), (pads[2], pads[3])))
        # im2col: (N, C*kH*kW, ho*wo), one row per (channel, kernel offset)
        cols = np.empty((n, c * kh * kw, ho * wo))
        order = []  # (channel, i, j) for each row, reused by col2im
        idx = 0
        for ci in range(c):
            for i in range(kh):
                for j in range(kw):
                    patch = xp[:, ci, i:i + stride * ho:stride, j:j + stride * wo:stride]
                    cols[:, idx, :] = patch.reshape(n, -1)
                    order.append((ci, i, j))
                    idx += 1

        wmat = w.data.reshape(m, -1)                     # (M, C*kH*kW)
        out_data = np.matmul(wmat, cols)                 # (N, M, ho*wo)
        if bias is not None:
            out_data = out_data + bias.data.reshape(1, m, 1)
        out = self._make(out_data.reshape(n, m, ho, wo),
                         (a, w) + ((bias,) if bias is not None else ()), "conv2d",
                         {"stride": stride, "pads": pads, "kernel": (kh, kw)})

        def _backward():
            g = out.grad.reshape(n, m, ho * wo)          # (N, M, L)
            if w.requires_grad:
                gw = np.einsum("nml,nkl->mk", g, cols).reshape(w.shape)
                w._acc(gw)
            if bias is not None and bias.requires_grad:
                bias._acc(g.sum(axis=(0, 2)))
            if a.requires_grad:
                gcols = np.matmul(wmat.T, g)             # (N, C*kH*kW, L)
                gxp = np.zeros_like(xp)
                for idx, (ci, i, j) in enumerate(order):
                    gxp[:, ci, i:i + stride * ho:stride, j:j + stride * wo:stride] += \
                        gcols[:, idx, :].reshape(n, ho, wo)
                a._acc(gxp[:, :, pads[0]:pads[0] + h, pads[2]:pads[2] + wd])
        out._backward = _backward
        return out

    def maxpool2d(self, kernel: int, stride: int, pads=(0, 0, 0, 0)) -> "Tensor":
        """2D max pooling, NCHW. Gradient routes to each window's argmax."""
        a = self
        if a.ndim != 4:
            raise NotImplementedError("maxpool2d supports NCHW only")
        pads = tuple(int(p) for p in pads)
        n, c, h, wd = a.shape
        ho = (h + pads[0] + pads[1] - kernel) // stride + 1
        wo = (wd + pads[2] + pads[3] - kernel) // stride + 1

        xp = np.pad(a.data, ((0, 0), (0, 0), (pads[0], pads[1]), (pads[2], pads[3])),
                    constant_values=-np.inf)
        out_data = np.empty((n, c, ho, wo))
        arg = np.empty((n, c, ho, wo, 2), dtype=int)     # argmax position per window
        for i in range(ho):
            for j in range(wo):
                win = xp[:, :, i * stride:i * stride + kernel,
                         j * stride:j * stride + kernel].reshape(n, c, -1)
                am = win.argmax(axis=-1)
                out_data[:, :, i, j] = np.take_along_axis(win, am[..., None], axis=-1)[..., 0]
                arg[:, :, i, j] = np.stack(np.unravel_index(am, (kernel, kernel)), axis=-1)

        out = self._make(out_data, (a,), "maxpool2d",
                         {"kernel": kernel, "stride": stride, "pads": pads})

        def _backward():
            if a.requires_grad:
                gxp = np.full_like(xp, 0.0)
                for i in range(ho):
                    for j in range(wo):
                        gi, gj = arg[:, :, i, j, 0], arg[:, :, i, j, 1]
                        rows = i * stride + gi
                        cols_ = j * stride + gj
                        np.add.at(gxp, (np.arange(n)[:, None], np.arange(c)[None, :],
                                        rows, cols_), out.grad[:, :, i, j])
                a._acc(gxp[:, :, pads[0]:pads[0] + h, pads[2]:pads[2] + wd])
        out._backward = _backward
        return out

    def _acc(self, grad: np.ndarray) -> None:
        """Accumulate ``grad`` into ``self.grad`` (running hooks first)."""
        for fn in self._hooks:
            replacement = fn(grad)
            if replacement is not None:
                grad = replacement
        for fn in _OP_HOOKS.get(self._op, ()):
            replacement = fn(grad)
            if replacement is not None:
                grad = replacement
        self.grad = grad if self.grad is None else self.grad + grad

    def _topo(self) -> List["Tensor"]:
        """Topological order (leaves -> root) of the graph below this node."""
        topo, visited = [], set()
        def visit(t: "Tensor"):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:
                    visit(p)
                topo.append(t)
        visit(self)
        return topo

    def backward(self, return_trace: bool = False) -> Optional[List["Tensor"]]:
        """Run reverse-mode autodiff from this (scalar-valued) node.

        Like PyTorch, gradients ACCUMULATE: calling ``backward()`` twice adds
        the gradients of both passes. Call ``zero_grad()`` on parameters (or
        the optimizer) between training steps.

        Args:
            return_trace: if True, also return the nodes in the exact order
                their ``_backward`` closures executed (i.e. reverse topo order).
                This is the raw material for :mod:`backlens.debug`.
        """
        if self.size != 1:
            raise RuntimeError(
                f"backward() requires a scalar loss, got shape {self.shape}; "
                "call .sum() or .mean() first"
            )
        topo = self._topo()
        seed = np.ones_like(self.data)
        self.grad = seed if self.grad is None else self.grad + seed

        trace: List[Tensor] = []
        for node in reversed(topo):
            if node.grad is None:
                continue  # node receives no gradient from this loss
            node._backward()
            if return_trace:
                trace.append(node)
        return trace if return_trace else None

    # -- misc -----------------------------------------------------------------

    def __repr__(self) -> str:
        label = f"{self.label!r} " if self.label else ""
        req = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({label}{self.data!r}{req})"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self.data == Tensor._ensure(other).data


def cat(tensors, axis: int = 0) -> "Tensor":
    """Concatenate tensors along ``axis`` (all other dims must match)."""
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("cat needs at least one tensor")
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis))
    if _grad_enabled:
        out._prev = tensors
        out._op = "cat"
        out._attrs = {"axis": axis}
        out.requires_grad = any(t.requires_grad for t in tensors)

    def _backward():
        sizes = [t.shape[axis] for t in tensors]
        splits = np.cumsum(sizes)[:-1]
        grads = np.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t._acc(g)
    out._backward = _backward
    return out
