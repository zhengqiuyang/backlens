"""Bridge PyTorch checkpoints into BackLens models -- torch NOT required.

``load_torch_state_dict(model, state_dict)`` copies weights from a PyTorch
``state_dict`` (any object with ``.items()`` yielding tensors that answer to
``.detach().cpu().numpy()`` -- plain numpy dicts work too, which is what makes
this testable without torch) into a BackLens model with a mirroring
architecture, matching parameters by ORDER and SHAPE.

The mapping is deliberately conservative, with one well-defined convenience:
torch's ``nn.Linear`` stores weights as ``(out, in)`` while BackLens uses
``(in, out)``, so 2-D tensors that match after transposition are transposed
automatically (and noted in the returned mapping). Anything else is refused
with a full report.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .nn import Module


def _to_numpy(t) -> np.ndarray:
    """Accept torch tensors (duck-typed) or plain numpy arrays.

    Preserves dtype: callers need the original kind to tell float weights
    from integer buffers.
    """
    if hasattr(t, "detach"):                       # torch.Tensor
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def _is_floating(arr: np.ndarray) -> bool:
    return arr.dtype.kind == "f"


def load_torch_state_dict(model: Module, state_dict,
                          strict: bool = True) -> List[Tuple[str, str]]:
    """Load a PyTorch state dict into ``model`` (order + shape matching).

    Parameters are matched positionally: the i-th floating tensor of the
    state dict fills the i-th parameter of the model, requiring equal
    shapes. Returns the mapping ``[(model_param_label, torch_name), ...]``.

    Args:
        model: a BackLens model mirroring the torch architecture.
        state_dict: a torch ``state_dict()``, or any dict of arrays with
            equivalent ordering (shape-compatible numpy dicts work too).
        strict: if True (default), mismatched shapes or counts raise
            ``ValueError`` with a full report instead of loading partially.

    Typical use::

        model = backlens.MLP(784, [128, 64], 10, act=backlens.ReLU())
        backlens.load_torch_state_dict(model, torch_mlp.state_dict())
        # now: debug / film / export the SAME weights without torch installed
    """
    torch_items = []
    for name, t in state_dict.items():
        arr = _to_numpy(t)
        if _is_floating(arr):
            torch_items.append((name, arr.astype(np.float64)))
    params = model.parameters()

    report_lines: List[str] = []
    errors: List[str] = []
    mapping: List[Tuple[str, str]] = []

    if strict and len(torch_items) != len(params):
        errors.append(
            f"count mismatch: {len(torch_items)} floating torch tensors vs "
            f"{len(params)} model parameters"
        )

    for i, (p, (tname, tarr)) in enumerate(zip(params, torch_items)):
        transpose = False
        if p.shape != tarr.shape:
            if p.ndim == 2 and tarr.ndim == 2 and tarr.T.shape == p.shape:
                transpose = True                 # torch (out, in) -> (in, out)
            else:
                msg = (f"#{i}: model {p.label or '?'} {p.shape} != "
                       f"torch '{tname}' {tarr.shape}")
                if strict:
                    errors.append(msg)
                else:
                    report_lines.append(f"  SKIPPED {msg}")
                    continue
        p.data = tarr.T.copy() if transpose else tarr.copy()
        p.grad = None
        label = p.label or f"param{i}"
        suffix = " [transposed]" if transpose else ""
        mapping.append((label, tname))
        report_lines.append(f"  {label:<24} <- '{tname}' {p.shape}{suffix}")

    if errors:
        detail = "\n".join(errors + report_lines)
        raise ValueError(f"load_torch_state_dict failed:\n{detail}")
    return mapping
