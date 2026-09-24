"""Train here, deploy anywhere: GradLens -> ONNX -> onnxruntime.

Trains a classifier in GradLens, exports it to a standard ONNX file, runs it
in onnxruntime, and checks that both engines agree. Also produces a training
gradient audit report (docs/demo_audit.html).

Run:  python examples/07_export_onnx.py
"""

import os

import numpy as np

from gradlens import Tensor, MLP, Tanh, Adam
from gradlens.debug import GradientMonitor
from gradlens.nn import cross_entropy


def spiral(n=100, classes=3, seed=1):
    rng = np.random.default_rng(seed)
    X = np.zeros((n * classes, 2))
    y = np.zeros(n * classes, dtype=int)
    for c in range(classes):
        sl = slice(c * n, (c + 1) * n)
        r = np.linspace(0.0, 1.0, n)
        t = (c * 4 / classes + 4 * r) + rng.standard_normal(n) * 0.2
        X[sl] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[sl] = c
    return X, y


def main():
    from gradlens.onnx_export import export_onnx

    X, y = spiral()
    np.random.seed(42)
    model = MLP(2, [32, 32], 3, act=Tanh())
    opt = Adam(model.parameters(), lr=0.05)
    mon = GradientMonitor(model)

    print("training in GradLens (numpy, CPU)...")
    for epoch in range(201):
        loss = cross_entropy(model(Tensor(X)), y)
        opt.zero_grad()
        loss.backward()
        mon.tick(loss)
        opt.step()
    acc = float(np.mean(model(Tensor(X)).data.argmax(1) == y))
    print(f"trained accuracy: {acc:.2%}   final loss {loss.item():.4f}")

    # ---------------------------------------------------------------
    # export to ONNX and run in onnxruntime
    # ---------------------------------------------------------------
    path = "spiral.onnx"
    export_onnx(model, X[:1], path)
    print(f"\nexported {path} -- drag it into https://netron.app to inspect")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(None, {"input": X.astype(np.float32)})[0]
    diff = float(np.abs(ort_logits - model(Tensor(X)).data).max())
    ort_acc = float(np.mean(ort_logits.argmax(1) == y))
    print(f"onnxruntime accuracy: {ort_acc:.2%}   max logit diff vs GradLens: {diff:.2e}")
    assert ort_acc == acc and diff < 1e-4

    # ---------------------------------------------------------------
    # full circle: load the exported file back into GradLens
    # ---------------------------------------------------------------
    from gradlens.onnx_loader import load_onnx
    reloaded = load_onnx(path)
    back = reloaded(Tensor(X)).data
    print(f"roundtrip (GradLens -> ONNX -> GradLens) max diff: "
          f"{float(np.abs(back - model(Tensor(X)).data).max()):.2e}")

    # ---------------------------------------------------------------
    # training gradient audit report (another zero-dependency HTML)
    # ---------------------------------------------------------------
    audit = os.path.join(os.path.dirname(__file__), "..", "docs", "demo_audit.html")
    mon.save_html(str(audit), title="spiral MLP training -- gradient audit")
    print(f"wrote {os.path.normpath(audit)}")

    os.remove(path)
    print("\ntrain here, deploy anywhere: done.")


if __name__ == "__main__":
    main()
