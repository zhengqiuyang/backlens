"""Finite-difference gradient checking.

The trusted referee for any autodiff engine: perturb every element of the
inputs by ``eps``, compare central-difference slopes against the analytic
gradients from ``backward()``. PyTorch keeps its ``gradcheck`` tucked away in
``torch.autograd``; here it is a first-class teaching tool -- one call, a
readable report, per-input verdicts.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Sequence

from .engine import Tensor


class InputReport:
    def __init__(self, name: str, max_abs_err: float, max_rel_err: float, passed: bool):
        self.name = name
        self.max_abs_err = max_abs_err
        self.max_rel_err = max_rel_err
        self.passed = passed

    def __repr__(self):
        flag = "PASS" if self.passed else "FAIL"
        return (f"[{flag}] {self.name}: max_abs_err={self.max_abs_err:.3e} "
                f"max_rel_err={self.max_rel_err:.3e}")


class GradcheckResult:
    def __init__(self, reports: List[InputReport]):
        self.reports = reports

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.reports)

    def report(self) -> str:
        lines = ["gradcheck " + ("PASSED" if self.passed else "FAILED")]
        for r in self.reports:
            lines.append(f"  {r!r}")
        return "\n".join(lines)

    def __bool__(self):
        return self.passed


def gradcheck(
    fn: Callable[..., Tensor],
    *inputs: Tensor,
    eps: float = 1e-6,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> GradcheckResult:
    """Verify analytic gradients of ``fn(*inputs)`` against central differences.

    ``fn`` must return a scalar ``Tensor``. Each input must have
    ``requires_grad=True``. Returns a :class:`GradcheckResult` that is truthy
    when every input passes; call ``.report()`` for a human-readable summary.

    Example:
        x = Tensor(np.random.randn(3, 2), requires_grad=True)
        result = gradcheck(lambda x: (x * x).sum() + x.tanh().mean(), x)
        assert result
    """
    reports: List[InputReport] = []
    for idx, x in enumerate(inputs):
        if not x.requires_grad:
            raise ValueError(f"input #{idx} must have requires_grad=True")

        # analytic
        out = fn(*inputs)
        if out.size != 1:
            raise ValueError("fn must return a scalar Tensor; add .sum() or .mean()")
        out.backward()
        analytic = np.array(x.grad, dtype=np.float64, copy=True)
        for other in inputs:
            other.grad = None  # grads accumulate; reset for the next input

        # numeric (central differences), element by element
        numeric = np.zeros_like(x.data)
        flat_x = x.data.reshape(-1)
        flat_n = numeric.reshape(-1)
        for i in range(flat_x.size):
            orig = flat_x[i]
            flat_x[i] = orig + eps
            f_plus = float(fn(*inputs).data.item())
            flat_x[i] = orig - eps
            f_minus = float(fn(*inputs).data.item())
            flat_x[i] = orig
            flat_n[i] = (f_plus - f_minus) / (2.0 * eps)

        err = np.abs(analytic - numeric)
        denom = np.maximum(np.abs(analytic), np.abs(numeric))
        rel = np.where(denom > 0, err / np.maximum(denom, 1e-12), err)
        passed = bool(np.all((err < atol) | (rel < rtol)))
        name = x.label or f"input#{idx}"
        reports.append(InputReport(name, float(err.max()), float(rel.max()), passed))

    return GradcheckResult(reports)
