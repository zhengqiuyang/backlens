"""BackLens -- a numpy autograd engine with X-ray vision into backpropagation.

Quick tour::

    from backlens import Tensor, MLP, cross_entropy, Adam
    from backlens.debug import debug_backward, step_backward, GradientMonitor
    from backlens.check import gradcheck
    from backlens.viz import save_graph_md
    from backlens.film import film_backward

    model = MLP(2, [16, 16], 3)
    opt = Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        logits = model(Tensor(X))
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    trace = debug_backward(loss)      # every op, in backward order
    save_graph_md(loss, "graph.md")   # renders on GitHub, no graphviz needed
    film_backward(loss).save_html("film.html")   # animated replay in browser
"""

from .engine import Tensor, cat, where, no_grad, is_grad_enabled, set_op_hook, clear_op_hooks
from . import nn
from .nn import (
    Module, Linear, Conv2d, MaxPool2d, Flatten, Sequential, MLP,
    Tanh, ReLU, Sigmoid,
    mse_loss, binary_cross_entropy, cross_entropy,
    Optimizer, SGD, Adam,
)
from .debug import (
    BackwardTrace, BackwardStep, GradientAnomalyError,
    debug_backward, step_backward, GradientMonitor,
)
from .check import gradcheck, GradcheckResult
from .viz import to_mermaid, save_graph_md, to_html, save_graph_html
from .film import film_backward, BackwardFilm
from .onnx_loader import load_onnx, analyze_onnx, OnnxModel, OnnxOpError
from .onnx_export import export_onnx, verify_onnx, VerifyReport, ExportError
from .flow import gradient_flow, GradientFlow, GradientFlowTimeline
from .torch_bridge import load_torch_state_dict

__version__ = "0.7.0"
__all__ = [
    "Tensor", "cat", "where", "no_grad", "is_grad_enabled", "set_op_hook", "clear_op_hooks",
    "nn", "Module", "Linear", "Conv2d", "MaxPool2d", "Flatten",
    "Sequential", "MLP", "Tanh", "ReLU", "Sigmoid",
    "mse_loss", "binary_cross_entropy", "cross_entropy",
    "Optimizer", "SGD", "Adam",
    "BackwardTrace", "BackwardStep", "GradientAnomalyError",
    "debug_backward", "step_backward", "GradientMonitor",
    "gradcheck", "GradcheckResult",
    "to_mermaid", "save_graph_md", "to_html", "save_graph_html",
    "film_backward", "BackwardFilm",
    "load_onnx", "analyze_onnx", "OnnxModel", "OnnxOpError",
    "export_onnx", "verify_onnx", "VerifyReport", "ExportError",
    "gradient_flow", "GradientFlow", "GradientFlowTimeline",
    "load_torch_state_dict",
]
