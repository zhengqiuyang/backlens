"""Gradient flow: which parameters eat the gradient?

Backpropagation distributes the loss's gradient through the graph like a
current through a circuit. ``gradient_flow(loss)`` measures the exact mass
on every edge (by diffing each parent's gradient around every op's backward
closure) and renders a Sankey-style map: edge thickness = gradient mass,
parameter nodes ranked by the share they receive.

This answers, at a glance, the question every practitioner eventually asks:
"why is that layer not learning?" -- because almost no gradient reaches it.

Zero dependencies: one self-contained HTML file, hand-rolled SVG.
"""

from __future__ import annotations

import json
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from .engine import Tensor


def _fin(x: float) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


class GradientFlow:
    """Per-edge gradient masses for one backward pass."""

    def __init__(self, name: str, nodes: List[dict], edges: List[dict],
                 params: List[dict]):
        self.name = name
        self.nodes = nodes
        self.edges = edges          # {"from", "to", "mass"}
        self.params = params        # ranked: {"name","mass","share"}

    # -- data ----------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "nodes": self.nodes,
                           "edges": self.edges, "params": self.params}, indent=1)

    def report(self, top: int = 10) -> str:
        lines = [f"gradient flow '{self.name}': parameter shares"]
        for p in self.params[:top]:
            bar = "#" * int(round(p["share"] / 2.5)) if p["share"] else ""
            lines.append(f"  {p['name']:<26} {p['share']:5.1f}%  |g|={p['mass']:.3e}  {bar}")
        return "\n".join(lines)

    def save_html(self, path: str) -> None:
        data = self.to_json().replace("</", "<\\/")
        html = _FLOW_TEMPLATE.replace("__FLOW_JSON__", data)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)


def gradient_flow(loss: Tensor, name: str = "gradient flow") -> GradientFlow:
    """Run backward on ``loss`` and measure the gradient mass of every edge.

    Each op's backward closure is wrapped: the gradients of its parents are
    snapshotted before and diffed after, so the increment attributed to that
    edge is exact (multiple consumers accumulate incrementally).
    """
    if loss.size != 1:
        raise RuntimeError("gradient_flow requires a scalar loss")
    topo = loss._topo()
    index = {id(t): i for i, t in enumerate(topo)}

    # reset the graph's gradients, then run backward while measuring
    for t in topo:
        t.grad = None
    seed = np.ones_like(loss.data)
    loss.grad = seed if loss.grad is None else loss.grad + seed

    edge_mass: Dict[Tuple[int, int], float] = {}
    for node in reversed(topo):
        if node.grad is None:
            continue
        before = {id(p): (p.grad.copy() if p.grad is not None else None)
                  for p in node._prev if p.requires_grad}
        node._backward()
        for p in node._prev:
            if not p.requires_grad:
                continue
            now = p.grad
            inc = now if before[id(p)] is None else now - before[id(p)]
            with np.errstate(over="ignore", invalid="ignore"):
                mass = float(np.linalg.norm(inc.ravel()))
            key = (id(p), id(node))
            edge_mass[key] = edge_mass.get(key, 0.0) + mass

    def kind(t: Tensor) -> str:
        if t._prev:
            return "op"
        return "param" if t.requires_grad else "data"

    nodes = [{"id": i, "op": t._op or "leaf", "label": t.label or "",
              "shape": list(t.shape), "kind": kind(t)}
             for i, t in enumerate(topo)]
    edges = [{"from": index[a], "to": index[b], "mass": _fin(m)}
             for (a, b), m in sorted(
                 edge_mass.items(),
                 key=lambda kv: -(kv[1] if math.isfinite(kv[1]) else math.inf),
             )]

    # parameter mass = norm of the parameter's final gradient
    param_entries = []
    for t in topo:
        if not t._prev and t.requires_grad and t.grad is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                param_entries.append((t.label or f"param{index[id(t)]}",
                                      float(np.linalg.norm(t.grad.ravel()))))
    denom = sum(m for _, m in param_entries) or 1.0
    params = [{"name": n, "mass": m, "share": 100.0 * m / denom}
              for n, m in sorted(param_entries, key=lambda x: -x[1])]

    return GradientFlow(name, nodes, edges, params)


# ---------------------------------------------------------------------------
# the viewer: Sankey-style layered map, edge width ~ log(mass)
# ---------------------------------------------------------------------------

_FLOW_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BackLens gradient flow</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --fg:#c9d1d9;
          --dim:#8b949e; --blue:#58a6ff; --green:#3fb950; --red:#f85149; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-monospace, Consolas, monospace; }
  header { padding:16px 20px 4px; }
  h1 { margin:0; font-size:18px; } h1 .lens { color:var(--blue); }
  #meta { color:var(--dim); margin-top:2px; }
  main { display:flex; border-top:1px solid var(--border);
         border-bottom:1px solid var(--border); }
  #graph-wrap { flex:1; overflow:auto; max-height:74vh; }
  aside { width:400px; max-height:74vh; overflow:auto; background:var(--panel);
          border-left:1px solid var(--border); padding:12px 14px; }
  h3 { margin:4px 0 8px; font-size:14px; }
  .bar-row { margin:6px 0; }
  .bar-label { display:flex; justify-content:space-between; font-size:12px;
               color:var(--dim); margin-bottom:2px; }
  .bar-label b { color:var(--fg); font-weight:600; }
  .bar-track { background:#21262d; border-radius:4px; height:12px; overflow:hidden; }
  .bar-fill { height:100%; background:linear-gradient(90deg,#1f6feb,#58a6ff);
              border-radius:4px; }
  .bar-fill.top { background:linear-gradient(90deg,#2ea043,#3fb950); }
  footer { padding:8px 20px; color:var(--dim); font-size:12px; }
  svg text { font-family:inherit; }
  .node rect.box { fill:#161b22; stroke:#3d444d; stroke-width:1.2; rx:7; }
  .node.kind-data rect.box { fill:#14202e; stroke:#2f4a68; }
  .node.kind-param rect.box { fill:#142519; stroke:#2e6b45; }
  .node .title { fill:#e6edf3; font-weight:600; }
  .node.kind-param .title { fill:#7ee2a8; }
  .node.kind-data .title { fill:#79c0ff; }
  .node .sub { fill:var(--dim); font-size:10.5px; }
  .node .mass { fill:var(--blue); font-size:11px; }
  .node.kind-param .mass { fill:var(--green); }
</style>
</head>
<body>
<header>
  <h1><span class="lens">BackLens</span> gradient flow &mdash; who eats the gradient?</h1>
  <div id="meta"></div>
</header>
<main>
  <div id="graph-wrap"><svg id="graph" xmlns="http://www.w3.org/2000/svg"></svg></div>
  <aside id="panel"><h3>parameter gradient shares</h3></aside>
</main>
<footer>generated by <code>backlens.flow.gradient_flow</code> &mdash; single file, zero dependencies; edge thickness &prop; log |gradient mass|</footer>
<script>
const DATA = __FLOW_JSON__;

const nodes = DATA.nodes, edges = DATA.edges;
const depth = nodes.map(() => 0);
edges.forEach(e => { depth[e.to] = Math.max(depth[e.to], depth[e.from] + 1); });
const cols = [];
nodes.forEach((n, i) => { (cols[depth[i]] || (cols[depth[i]] = [])).push(i); });

const NW = 170, NH = 60, COLW = 230, VGAP = 28, PAD = 40;
const colH = c => c.length * NH + (c.length - 1) * VGAP;
const H = Math.max(...cols.map(colH)) + 2 * PAD;
const W = cols.length * COLW + 2 * PAD;
const pos = [];
cols.forEach((ids, ci) => {
  const y0 = (H - colH(ids)) / 2 + PAD;
  ids.forEach((idx, ri) => { pos[idx] = { x: PAD + ci * COLW, y: y0 + ri * (NH + VGAP) }; });
});

const massOf = i => {
  let m = 0;
  edges.forEach(e => { if (e.to === i && e.mass) m = Math.max(m, e.mass); });
  return m;
};
const maxMass = Math.max(...edges.map(e => e.mass || 0), 1e-12);
function widthOf(m) {
  if (!m || !isFinite(m)) return 0.8;
  if (m <= 0) return 0.8;
  const t = Math.min(1, Math.max(0, (Math.log10(m) - Math.log10(maxMass) + 6) / 6));
  return 0.8 + t * 15;
}
function fmtG(v) {
  if (v == null || !isFinite(v)) return '\u221e';
  if (v === 0) return '0';
  if (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3) return v.toExponential(1);
  return String(Math.round(v * 1000) / 1000);
}

const svg = document.getElementById('graph');
svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
svg.setAttribute('width', Math.min(W, 1500));
svg.style.minWidth = Math.min(W, 1100) + 'px';
const NS = 'http://www.w3.org/2000/svg';
const mk = (tag, attrs) => {
  const el = document.createElementNS(NS, tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
};

// edges: gradient flows right -> left (consumer -> producer)
edges.forEach(e => {
  const p1 = pos[e.from], p2 = pos[e.to];
  const x1 = p1.x + NW, y1 = p1.y + NH / 2, x2 = p2.x, y2 = p2.y + NH / 2;
  const mx = (x1 + x2) / 2;
  const intoParam = nodes[e.to].kind === 'param';
  svg.appendChild(mk('path', {
    class: 'edge',
    d: `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`,
    fill: 'none',
    stroke: intoParam ? 'var(--green)' : 'var(--blue)',
    'stroke-width': widthOf(e.mass).toFixed(1),
    'stroke-opacity': intoParam ? '0.75' : '0.45',
  }));
});

// nodes
nodes.forEach((n, i) => {
  const p = pos[i];
  const g = mk('g', { class: `node kind-${n.kind}`, transform: `translate(${p.x},${p.y})` });
  g.appendChild(mk('rect', { class: 'box', width: NW, height: NH, rx: 7 }));
  const title = n.label || n.op;
  g.appendChild(mk('text', { class: 'title', x: 10, y: 19 })).textContent =
    title.length > 23 ? title.slice(0, 22) + '\u2026' : title;
  g.appendChild(mk('text', { class: 'sub', x: 10, y: 34 })).textContent =
    `${n.op}  ${JSON.stringify(n.shape)}`;
  g.appendChild(mk('text', { class: 'mass', x: 10, y: 49 })).textContent =
    '|g\u2192| ' + fmtG(massOf(i));
  svg.appendChild(g);
});

// side panel: ranked parameter shares
const panel = document.getElementById('panel');
DATA.params.forEach((p, i) => {
  const row = document.createElement('div');
  row.className = 'bar-row';
  row.innerHTML = `<div class="bar-label"><b>${p.name}</b>` +
    `<span>${p.share.toFixed(1)}% &nbsp; |g|=${fmtG(p.mass)}</span></div>` +
    `<div class="bar-track"><div class="bar-fill${i === 0 ? ' top' : ''}" ` +
    `style="width:${Math.max(1.5, p.share).toFixed(1)}%"></div></div>`;
  panel.appendChild(row);
});

document.getElementById('meta').textContent =
  `${DATA.name}  \u00b7  ${nodes.length} nodes  \u00b7  ${edges.length} gradient edges  ` +
  `\u00b7  top parameter: ${DATA.params[0] ? DATA.params[0].name + ' ' + DATA.params[0].share.toFixed(1) + '%' : '-'}  ` +
  `\u00b7  gradient flows right \u2192 left`;
</script>
</body>
</html>
"""
