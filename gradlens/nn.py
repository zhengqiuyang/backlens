"""Neural-network building blocks on top of the GradLens engine.

A deliberately small PyTorch-flavored API: ``Module``/``Linear``/activations,
``Sequential``/``MLP``, common losses, and SGD/Adam optimizers. Everything is
built from engine ops only, so every layer stays fully traceable by
:mod:`gradlens.debug` and :mod:`gradlens.viz`.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Optional

from .engine import Tensor, no_grad


class Module:
    """Base class. Subclasses assign ``Tensor`` attributes; parameters are
    discovered by walking ``__dict__`` recursively (like micrograd/PyTorch)."""

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params += v.parameters()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params += item.parameters()
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}()"


class Linear(Module):
    """Affine layer ``y = x @ W + b`` with Kaiming-uniform-ish init."""

    def __init__(self, n_in: int, n_out: int, bias: bool = True):
        self.weight = Tensor(
            np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in),
            requires_grad=True, label=f"Linear({n_in},{n_out}).W",
        )
        self.bias = (
            Tensor(np.zeros(n_out), requires_grad=True, label=f"Linear({n_in},{n_out}).b")
            if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        n_in, n_out = self.weight.shape
        return f"Linear(n_in={n_in}, n_out={n_out}, bias={self.bias is not None})"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Conv2d(Module):
    """NCHW convolution over 2D inputs (group=1)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 stride: int = 1, pad: int = 0, bias: bool = True):
        self.stride, self.pad = stride, pad
        scale = np.sqrt(2.0 / (in_ch * kernel * kernel))
        self.weight = Tensor(
            np.random.randn(out_ch, in_ch, kernel, kernel) * scale,
            requires_grad=True, label=f"Conv2d({in_ch},{out_ch},k{kernel}).W")
        self.bias = (
            Tensor(np.zeros(out_ch), requires_grad=True,
                   label=f"Conv2d({in_ch},{out_ch},k{kernel}).b")
            if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weight, self.bias,
                        stride=self.stride, pad=self.pad)

    def __repr__(self):
        c, _, kh, kw = self.weight.shape
        return (f"Conv2d({c}, {kh}x{kw}, stride={self.stride}, "
                f"pad={self.pad}, bias={self.bias is not None})")


class MaxPool2d(Module):
    def __init__(self, kernel: int, stride: int):
        self.kernel, self.stride = kernel, stride

    def forward(self, x: Tensor) -> Tensor:
        return x.maxpool2d(self.kernel, self.stride)

    def __repr__(self):
        return f"MaxPool2d(k={self.kernel}, stride={self.stride})"


class Flatten(Module):
    """Flatten trailing dims after ``start_dim`` (default: NCHW -> (N, -1))."""

    def __init__(self, start_dim: int = 1):
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        lead = int(np.prod(x.shape[:self.start_dim])) if self.start_dim else 1
        return x.reshape(lead, -1)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        inner = " -> ".join(repr(l) for l in self.layers)
        return f"Sequential({inner})"


class MLP(Module):
    """Multilayer perceptron: ``MLP(n_in, [n_hidden...], n_out, act=Tanh)``."""

    def __init__(self, n_in: int, hidden: Iterable[int], n_out: int, act: Optional[Module] = None):
        act = act or Tanh()
        sizes = [n_in, *list(hidden), n_out]
        layers: List[Module] = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act)
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def __repr__(self):
        return f"MLP{self.net!r}"


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()


def binary_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """Numerically stable BCE from raw logits (targets in {0, 1}).

    Uses the identity ``bce(z, t) = relu(z) - z*t + log(1 + exp(-|z|))``,
    where ``exp(-|z|) <= 1`` so the composition never overflows.
    """
    per_example = logits.relu() - logits * target + ((-logits.abs()).exp() + 1.0).log()
    return per_example.mean()


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Softmax cross-entropy from raw logits.

    ``logits``: (N, C); ``targets``: int array (N,) of class indices.
    Implemented with a max-shift for numerical stability, out of plain engine
    ops so every step of the loss shows up in the computation graph.
    """
    assert logits.ndim == 2, "cross_entropy expects (N, C) logits"
    targets = np.asarray(targets).astype(int).ravel()
    n, c = logits.shape
    assert targets.shape == (n,), f"targets shape {targets.shape} != ({n},)"

    shift = logits - logits.max()          # scalar max, detached (constant shift)
    logsumexp = shift.exp().sum(axis=1, keepdims=True).log()
    log_probs = shift - logsumexp          # (N, C) log-softmax
    picked = (log_probs * Tensor(np.eye(c)[targets])).sum() / n
    return -picked


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self._velocity):
            if p.grad is None:
                continue
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v


class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] = self.b1 * self._m[i] + (1 - self.b1) * g
            self._v[i] = self.b2 * self._v[i] + (1 - self.b2) * g * g
            m_hat = self._m[i] / (1 - self.b1 ** self.t)
            v_hat = self._v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
