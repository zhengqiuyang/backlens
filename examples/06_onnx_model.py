"""Load a real ONNX model into GradLens -- run it, check it, film its backward.

Uses the pretrained MNIST CNN from the ONNX Model Zoo (assets/mnist-8.onnx,
a LeNet-style Conv/MaxPool network). If the asset is missing (or the onnx
package is not installed), a small conv net is built programmatically instead,
so the example always runs.

Run:  python examples/06_onnx_model.py
"""

import os
import warnings

import numpy as np

from gradlens import Tensor, SGD
from gradlens.nn import cross_entropy
from gradlens.debug import debug_backward
from gradlens.film import film_backward

ASSET = os.path.join(os.path.dirname(__file__), "..", "assets", "mnist-8.onnx")


def get_model():
    if os.path.exists(ASSET):
        from gradlens.onnx_loader import load_onnx, analyze_onnx
        report = analyze_onnx(ASSET)
        print(f"assets/mnist-8.onnx: ops={report['ops']}")
        print(f"loadable={report['loadable']} unsupported={report['unsupported']}")
        return load_onnx(ASSET), (1, 1, 28, 28)
    # fallback: build a tiny ONNX conv net from scratch (needs the onnx pkg)
    import onnx, onnx.helper
    from gradlens.onnx_loader import OnnxModel
    rng = np.random.default_rng(0)

    def init(name, arr):
        return onnx.numpy_helper.from_array(arr.astype(np.float32), name)

    w1 = rng.standard_normal((4, 1, 3, 3)).astype(np.float32) * 0.1
    b1 = rng.standard_normal(4).astype(np.float32) * 0.1
    w2 = rng.standard_normal((8, 4, 3, 3)).astype(np.float32) * 0.1
    b2 = rng.standard_normal(8).astype(np.float32) * 0.1
    w3 = rng.standard_normal((8 * 5 * 5, 10)).astype(np.float32) * 0.1
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Conv", ["x", "w1", "b1"], ["c1"],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            onnx.helper.make_node("Relu", ["c1"], ["r1"]),
            onnx.helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2],
                                  strides=[2, 2]),
            onnx.helper.make_node("Conv", ["p1", "w2", "b2"], ["c2"],
                                  kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            onnx.helper.make_node("Relu", ["c2"], ["r2"]),
            onnx.helper.make_node("MaxPool", ["r2"], ["p2"], kernel_shape=[2, 2],
                                  strides=[2, 2]),
            onnx.helper.make_node("Flatten", ["p2"], ["f"]),
            onnx.helper.make_node("Gemm", ["f", "w3"], ["y"]),
        ],
        "tiny-conv",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 12, 12])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 10])],
        initializer=[init("w1", w1), init("b1", b1), init("w2", w2),
                     init("b2", b2), init("w3", w3)],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model.ir_version = 9
    print("(asset missing -- using a programmatic tiny conv net instead)")
    return OnnxModel(model), (1, 1, 12, 12)


def main():
    model, in_shape = get_model()
    print(model)

    # ---------------------------------------------------------------
    # 1. forward + parity against onnxruntime, if available
    # ---------------------------------------------------------------
    x = np.random.default_rng(0).standard_normal(in_shape)
    logits = model(Tensor(x))
    print(f"\nforward output shape: {logits.shape}  requires_grad={logits.requires_grad}")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model._proto.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        ref = sess.run(None, {model.input_names[0]: x.astype(np.float32)})[0]
        diff = float(np.abs(logits.data - ref).max())
        print(f"parity with onnxruntime: max abs diff = {diff:.2e}  (< 1e-4 = pass)")
        assert diff < 1e-4
    except ImportError:
        print("(onnxruntime not installed; skipping parity check)")

    # ---------------------------------------------------------------
    # 2. fine-tune the loaded, pretrained model (overfit one sample)
    # ---------------------------------------------------------------
    label = np.array([3])
    opt = SGD(model.parameters(), lr=0.01)
    print("\nfine-tuning on one sample (proving gradients update real weights):")
    first = None
    for step in range(60):
        loss = cross_entropy(model(Tensor(x)), label)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 15 == 0 or step == 59:
            print(f"  step {step:2d}   loss {loss.item():.4f}")
    print(f"  loss {first:.4f} -> {loss.item():.4f}")

    # ---------------------------------------------------------------
    # 3. film one backward pass through the pretrained network
    # ---------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_loss = cross_entropy(model(Tensor(x)), label)
        film = film_backward(final_loss, name="ONNX mnist-8: backward through a real CNN")
    film.save_html("onnx_film.html")
    film.save_html(os.path.join(os.path.dirname(__file__), "..", "docs", "demo_onnx.html"))
    print(f"\n{film.summary()}")
    print("wrote onnx_film.html and docs/demo_onnx.html -- open in any browser")

    # the full backward trace also works as a table:
    trace = debug_backward(final_loss, raise_on_anomaly=False)
    ops_in_backward = [s.op for s in trace]
    print("\nbackward ops (first 12):", ops_in_backward[:12], "...")


if __name__ == "__main__":
    main()
