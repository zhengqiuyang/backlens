"""X-ray vision for the backward pass.

This is BackLens's headline feature. PyTorch lets you *run* ``backward()``;
micrograd doesn't scale past scalars. BackLens records the backward pass as it
happens and lets you:

* ``debug_backward(loss)`` -> a :class:`BackwardTrace` of every op, in the
  exact order its gradient was finalized, with per-node grad norms and
  automatic anomaly flags (NaN / Inf / exploding / vanishing).
* ``step_backward(loss)`` -> a generator that advances backward **one op at a
  time**, so you can literally walk a classroom through backpropagation.
* ``GradientMonitor(model)`` -> per-parameter gradient norms recorded across a
  training run (hooks, no changes to the training loop).
* :class:`GradientAnomalyError` raised at the first NaN/Inf, carrying the
  trace -- like ``torch.autograd.set_detect_anomaly(True)`` but readable.
"""

from __future__ import annotations

import json
import math
import numpy as np
from typing import Dict, Iterator, List, Optional

from .engine import Tensor
from .nn import Module

_EXPLODE_NORM = 1e3   # grads above this get an "exploding" note
_VANISH_NORM = 1e-7   # non-zero grads below this get a "vanishing" note


def _fmt(v: float) -> str:
    if v == 0.0:
        return "0"
    if math.isnan(v) or math.isinf(v):
        return str(v)
    exp = math.floor(math.log10(abs(v)))
    if -3 <= exp < 4:
        return f"{v:.4g}"
    return f"{v:.2e}"


class BackwardStep:
    """Snapshot of one node's gradient, taken right after its ``_backward`` ran."""

    __slots__ = ("index", "op", "label", "shape", "grad_norm", "grad_absmax",
                 "grad_absmin", "notes", "node")

    def __init__(self, index: int, node: Tensor):
        g = node.grad if node.grad is not None else np.zeros(())
        finite = g[np.isfinite(g)]
        self.index = index
        self.op = node._op or "leaf"
        self.label = node.label or ""
        self.shape = node.shape
        with np.errstate(over="ignore", invalid="ignore"):
            self.grad_norm = float(np.linalg.norm(g.ravel())) if g.size else 0.0
            self.grad_absmax = float(np.max(np.abs(g))) if g.size else 0.0
        self.grad_absmin = float(np.min(np.abs(finite))) if finite.size else 0.0
        self.notes: List[str] = []
        self.node = node
        if not np.all(np.isfinite(g)):
            has_nan = bool(np.any(np.isnan(g)))
            self.notes.append("NaN in grad" if has_nan else "Inf in grad")
        elif self.grad_absmax > _EXPLODE_NORM:
            self.notes.append("exploding?")
        elif 0.0 < self.grad_norm < _VANISH_NORM:
            self.notes.append("vanishing?")
        if node.grad is None:
            self.notes.append("no grad path from loss")

    def __repr__(self):
        name = self.label or self.op
        notes = f"  [{'; '.join(self.notes)}]" if self.notes else ""
        return f"#{self.index:<3} {name} shape={self.shape} |g|={_fmt(self.grad_norm)}{notes}"


class BackwardTrace:
    """The backward pass as a list of :class:`BackwardStep`, loss-first order."""

    def __init__(self, steps: List[BackwardStep]):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, i):
        return self.steps[i]

    def first_anomaly(self) -> Optional[BackwardStep]:
        """First step whose gradient contains NaN/Inf -- the contamination source."""
        for s in self.steps:
            if "NaN in grad" in s.notes or "Inf in grad" in s.notes:
                return s
        return None

    def report(self, max_rows: Optional[int] = None) -> str:
        """Human-readable table of the whole backward pass."""
        header = (f"{'step':<5} {'op':<12} {'label':<22} {'shape':<12} "
                  f"{'|grad|':<10} {'max|g|':<10} notes")
        lines = [header, "-" * len(header)]
        steps = self.steps if max_rows is None else self.steps[:max_rows]
        for s in steps:
            label = (s.label[:20] + "..") if len(s.label) > 22 else s.label
            lines.append(
                f"{s.index:<5} {s.op:<12} {label:<22} {str(s.shape):<12} "
                f"{_fmt(s.grad_norm):<10} {_fmt(s.grad_absmax):<10} {'; '.join(s.notes)}"
            )
        skipped = len(self.steps) - len(steps)
        if skipped > 0:
            lines.append(f"... ({skipped} more steps)")
        anomaly = self.first_anomaly()
        if anomaly:
            lines.append("")
            lines.append(
                f"first anomaly: step #{anomaly.index} op={anomaly.op} "
                f"label={anomaly.label or '-'} -> everything downstream is contaminated"
            )
        return "\n".join(lines)


class GradientAnomalyError(RuntimeError):
    """Raised when the backward pass produces NaN/Inf gradients.

    Carries the full :class:`BackwardTrace`; ``str(err)`` prints the report up
    to and including the first anomaly.
    """

    def __init__(self, trace: BackwardTrace):
        self.trace = trace
        first = trace.first_anomaly()
        super().__init__(
            "NaN/Inf gradient detected during backward.\n"
            f"first anomaly at step #{first.index} (op={first.op}, "
            f"label={first.label or '-'})\n"
            f"gradient magnitude there: |g|={_fmt(first.grad_norm)}\n"
            "full trace up to anomaly:\n" + trace.report(max_rows=first.index + 1)
        )


def debug_backward(loss: Tensor, raise_on_anomaly: bool = True) -> BackwardTrace:
    """Run ``loss.backward()`` and record every step of it.

    Args:
        loss: scalar ``Tensor`` (usually the training loss).
        raise_on_anomaly: if True (default), raise :class:`GradientAnomalyError`
            when any gradient is NaN/Inf, with the trace attached.
    """
    raw = loss.backward(return_trace=True)
    trace = BackwardTrace([BackwardStep(i, node) for i, node in enumerate(raw)])
    if raise_on_anomaly and trace.first_anomaly() is not None:
        raise GradientAnomalyError(trace)
    return trace


def step_backward(loss: Tensor) -> Iterator[Tensor]:
    """Yield graph nodes one at a time as their gradients are finalized.

    A true stepper for backpropagation: each ``next()`` executes exactly one
    op's backward closure, then hands you the node so you can inspect
    ``.grad``, ``.data``, ``._op`` before continuing.

    Example:
        for node in step_backward(loss):
            print(node._op, node.grad)
    """
    if loss.size != 1:
        raise RuntimeError("step_backward() requires a scalar loss")
    topo = loss._topo()
    seed = np.ones_like(loss.data)
    loss.grad = seed if loss.grad is None else loss.grad + seed
    for node in reversed(topo):
        if node.grad is None:
            continue
        node._backward()
        yield node


class GradientMonitor:
    """Record per-parameter gradient norms across training steps (via hooks).

    Example:
        mon = GradientMonitor(model)
        for step in range(n):
            loss.backward(); opt.step()
            mon.tick(loss)                    # close out one training step
        print(mon.summary())
        mon.save_html("audit.html")           # training gradient audit report
    """

    def __init__(self, module: Module):
        self.history: Dict[str, List[float]] = {}
        self.loss_history: List[float] = []
        self._pending: Dict[str, float] = {}
        for p in module.parameters():
            name = p.label or f"param{id(p) % 10000}"
            self.history[name] = []
            p.register_hook(self._make_recorder(name))

    def _make_recorder(self, name: str):
        def record(grad: np.ndarray) -> None:
            with np.errstate(over="ignore", invalid="ignore"):
                self._pending[name] = float(np.linalg.norm(grad.ravel()))
        return record

    def tick(self, loss=None) -> None:
        """Seal the current training step's norms (and optional loss) into history."""
        for name, series in self.history.items():
            series.append(self._pending.get(name, 0.0))
        self._pending.clear()
        if loss is not None:
            self.loss_history.append(
                loss.item() if isinstance(loss, Tensor) else float(loss)
            )

    def summary(self) -> str:
        lines = [f"{'parameter':<28} {'last |g|':<10} {'max |g|':<10} trend (every 10%)"]
        lines.append("-" * 78)
        for name, series in self.history.items():
            last = series[-1] if series else 0.0
            mx = max(series) if series else 0.0
            sampled = [series[int(i * (len(series) - 1))] for i in np.linspace(0, 1, 8)] if len(series) > 1 else series
            trend = " ".join(_fmt(v) for v in sampled)
            lines.append(f"{name:<28} {_fmt(last):<10} {_fmt(mx):<10} {trend}")
        return "\n".join(lines)

    # -- HTML audit report -------------------------------------------------

    _NONFINITE = float("nan")

    def _safe_series(self, series: List[float]) -> List[float]:
        return [v if np.isfinite(v) else self._NONFINITE for v in series]

    def save_html(self, path: str, title: str = "BackLens training gradient audit") -> None:
        """Write a self-contained audit report: per-parameter gradient norm
        heatmap + sparklines, loss curve (if recorded), anomaly flags."""
        params = []
        for name, series in self.history.items():
            safe = self._safe_series(series)
            finite = [v for v in safe if np.isfinite(v)]
            lo = min(finite) if finite else 0.0
            hi = max(finite) if finite else 1.0
            verdict = "healthy"
            if any(not np.isfinite(v) for v in safe):
                verdict = "NaN/Inf!"
            elif hi > _EXPLODE_NORM:
                verdict = "exploding?"
            elif len(safe) >= 5 and 0 < safe[-1] < _VANISH_NORM and hi < _VANISH_NORM * 1e2:
                verdict = "vanishing?"
            params.append({
                "name": name, "series": safe,
                "last": safe[-1] if safe else 0.0,
                "max": hi, "verdict": verdict,
            })
        data = {"title": title, "params": params,
                "loss": self._safe_series(self.loss_history),
                "steps": max((len(s) for s in self.history.values()), default=0)}
        payload = json.dumps(data).replace("</", "<\\/")
        html = _AUDIT_TEMPLATE.replace("__AUDIT_JSON__", payload)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


_AUDIT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BackLens gradient audit</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#c9d1d9;
          --dim:#8b949e; --blue:#58a6ff; --red:#f85149; --orange:#d29922; --green:#3fb950; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-monospace, Consolas, monospace; }
  header { padding:16px 20px 6px; }
  h1 { margin:0; font-size:18px; } h1 .lens { color: var(--blue); }
  #meta { color: var(--dim); margin-top:2px; }
  main { padding: 0 20px 30px; }
  .loss-card, .param-card { background:var(--panel); border:1px solid var(--border);
    border-radius:8px; padding:10px 14px; margin-top:14px; }
  .card-title { display:flex; justify-content:space-between; align-items:baseline; gap:10px; }
  .card-title b { font-size:13.5px; }
  .verdict { font-size:12px; padding:1px 8px; border-radius:10px; border:1px solid var(--border); }
  .verdict.bad { color:var(--red); border-color:var(--red); }
  .verdict.warn { color:var(--orange); border-color:var(--orange); }
  .verdict.ok { color:var(--green); border-color:#2e6b45; }
  .stats { color:var(--dim); font-size:12px; margin:4px 0 8px; }
  svg { display:block; width:100%; height:46px; }
  table { border-collapse:collapse; width:100%; margin-top:8px; font-size:12px; }
  th, td { text-align:left; padding:4px 8px; border-bottom:1px solid #21262d; }
  th { color:var(--dim); font-weight:500; }
  td.bad { color: var(--red); font-weight:600; }
  footer { padding:8px 20px; color:var(--dim); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1><span class="lens">BackLens</span> training gradient audit</h1>
  <div id="meta"></div>
</header>
<main id="main"></main>
<footer>generated by <code>GradientMonitor.save_html</code> &mdash; single file, zero dependencies</footer>
<script>
const DATA = __AUDIT_JSON__;
document.getElementById('meta').textContent =
  `${DATA.title}  ·  ${DATA.params.length} parameters  ·  ${DATA.steps} training steps` +
  (DATA.loss.length ? `  ·  loss ${DATA.loss[0].toPrecision(3)} -> ${DATA.loss[DATA.loss.length-1].toPrecision(3)}` : '');

function fmt(v) {
  if (v == null || !isFinite(v)) return 'NaN';
  if (v === 0) return '0';
  if (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3) return v.toExponential(1);
  return String(Math.round(v * 1000) / 1000);
}
const NS = 'http://www.w3.org/2000/svg';
function spark(series, color, W, H, logScale) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const vals = series.map(v => isFinite(v) ? v : null);
  const finite = vals.filter(v => v !== null);
  if (!finite.length) return svg;
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (logScale) { lo = Math.log10(Math.max(lo, 1e-12)); hi = Math.log10(Math.max(hi, 1e-12)); }
  if (hi <= lo) hi = lo + 1;
  const pts = [];
  series.forEach((v, i) => {
    const px = series.length === 1 ? W/2 : (i / (series.length - 1)) * (W - 8) + 4;
    if (v === null) { pts.push(null); return; }
    let t = logScale ? Math.log10(Math.max(v, 1e-12)) : v;
    const py = H - 5 - ((t - lo) / (hi - lo)) * (H - 10);
    pts.push([px, py]);
  });
  // segments split at NaN
  let d = '', pen = false;
  pts.forEach(p => {
    if (!p) { pen = false; return; }
    d += (pen ? ' L ' : ' M ') + p[0].toFixed(1) + ' ' + p[1].toFixed(1);
    pen = true;
  });
  svg.appendChild(document.createElementNS(NS, 'path'))
     .setAttribute('d', d);
  svg.lastChild.setAttribute('stroke', color);
  svg.lastChild.setAttribute('fill', 'none');
  svg.lastChild.setAttribute('stroke-width', '1.8');
  return svg;
}
const main = document.getElementById('main');

if (DATA.loss.length) {
  const card = document.createElement('div');
  card.className = 'loss-card';
  card.innerHTML = `<div class="card-title"><b>loss</b></div>
    <div class="stats">${DATA.loss.length} recorded values</div>`;
  card.appendChild(spark(DATA.loss, '#d29922', 900, 60, false));
  main.appendChild(card);
}

for (const p of DATA.params) {
  const bad = p.verdict !== 'healthy';
  const card = document.createElement('div');
  card.className = 'param-card';
  const cls = p.verdict.includes('NaN') ? 'bad' : (p.verdict === 'healthy' ? 'ok' : 'warn');
  card.innerHTML = `<div class="card-title"><b>${p.name}</b>
      <span class="verdict ${cls}">${p.verdict}</span></div>
    <div class="stats">last |g| = ${fmt(p.last)} &nbsp; max |g| = ${fmt(p.max)} &nbsp; (log-scale sparkline)</div>`;
  card.appendChild(spark(p.series, cls === 'bad' ? '#f85149' : '#58a6ff', 900, 46, true));
  main.appendChild(card);
}
</script>
</body>
</html>
"""
