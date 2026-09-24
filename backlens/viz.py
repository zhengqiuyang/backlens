"""Computation-graph export: Mermaid + standalone HTML.

micrograd's ``draw_graph`` needs a system-wide Graphviz install -- the single
most common "it doesn't work on my machine" report. BackLens exports plain
`Mermaid <https://mermaid.js.org>`_ text instead, which GitHub renders natively
inside ```` ```mermaid ```` fences in READMEs, issues and PRs, and which any
browser can render from the generated HTML file with zero local dependencies.

Nodes show: label / op, shape, data norm, and gradient norm (after backward).
Nodes with NaN/Inf gradients are flagged red so a sick graph is visible at a
glance.
"""

from __future__ import annotations

import html as _html
import math
import numpy as np
from typing import Dict, Optional

from .engine import Tensor


def _esc(text: str) -> str:
    """Mermaid node text lives inside double quotes; neutralize specials."""
    return text.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")


def _fmt(v: float) -> str:
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    if -2 <= exp < 4:
        return f"{v:.3g}"
    return f"{v:.1e}"


def _node_text(t: Tensor, show: str) -> str:
    parts = []
    if show in ("both", "label"):
        parts.append(t.label or "")
    if show in ("both", "op"):
        parts.append(t._op or "leaf")
    parts.append(str(t.shape))
    if show in ("both", "value"):
        with np.errstate(over="ignore", invalid="ignore"):
            parts.append(f"|x|={_fmt(float(np.linalg.norm(t.data.ravel())))}")
    if show in ("both", "grad") and t.grad is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            parts.append(f"|g|={_fmt(float(np.linalg.norm(t.grad.ravel())))}")
    # escape each part, then join with mermaid line breaks
    return "<br/>".join(_esc(p) for p in parts if p)


def to_mermaid(root: Tensor, direction: str = "TD", show: str = "both") -> str:
    """Render the graph below ``root`` as Mermaid ``graph`` source.

    Args:
        root: any ``Tensor``; the full subgraph feeding it is drawn.
        direction: ``TD`` (top-down, default) or ``LR``.
        show: ``"both"`` | ``"label"`` | ``"op"`` | ``"value"`` | ``"grad"`` --
            which info lines to display. ``"both"`` shows label+op+shape+|x|+|g|.
    """
    topo = root._topo()
    ids: Dict[int, str] = {}
    kinds: Dict[str, str] = {}
    lines = [f"graph {direction}"]

    for i, t in enumerate(topo):
        ids[id(t)] = f"n{i}"
        if t.grad is not None and not np.all(np.isfinite(t.grad)):
            kinds[ids[id(t)]] = "anomaly"
        elif t._prev:
            kinds[ids[id(t)]] = "op"
        elif t.requires_grad:
            kinds[ids[id(t)]] = "param"
        else:
            kinds[ids[id(t)]] = "data"
        lines.append(f'    {ids[id(t)]}["{_node_text(t, show)}"]')

    seen_edges = set()
    for t in topo:
        for p in t._prev:
            edge = (ids[id(p)], ids[id(t)])
            if edge not in seen_edges:      # a tensor reused by the same op
                seen_edges.add(edge)
                lines.append(f"    {edge[0]} --> {edge[1]}")

    lines.append("    classDef op fill:#fff,stroke:#666")
    lines.append("    classDef param fill:#efe,stroke:#4a9")
    lines.append("    classDef data fill:#eef,stroke:#68a")
    lines.append("    classDef anomaly fill:#fdd,stroke:#c00,stroke-width:2.5px")
    for node_id, kind in kinds.items():
        lines.append(f"    class {node_id} {kind}")
    return "\n".join(lines)


def save_graph_md(root: Tensor, path: str, direction: str = "TD", show: str = "both") -> str:
    """Write ``path`` as a Markdown file whose mermaid fence GitHub renders.

    Returns the file content (handy for pasting straight into a README).
    """
    content = (
        f"```mermaid\n{to_mermaid(root, direction=direction, show=show)}\n```\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BackLens graph</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  body {{ font-family: ui-monospace, Consolas, monospace; margin: 2rem; background: #fafafa; }}
  .mermaid {{ background: white; }}
</style>
</head>
<body>
<h3>BackLens computation graph</h3>
<div class="mermaid">
{graph}
</div>
<script>mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});</script>
</body>
</html>
"""


def to_html(root: Tensor, direction: str = "TD", show: str = "both") -> str:
    """Self-contained HTML string embedding the Mermaid graph (CDN script)."""
    return _HTML_TEMPLATE.format(graph=to_mermaid(root, direction=direction, show=show))


def save_graph_html(root: Tensor, path: str, direction: str = "TD", show: str = "both") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_html(root, direction=direction, show=show))
