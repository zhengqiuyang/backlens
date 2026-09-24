"""Backprop Film: record a backward pass, replay it in the browser.

This trains a tiny MLP, then records one full backward pass and writes a
self-contained interactive player: play/pause, single-step, scrub, and watch
gradients light up the computation graph op by op. A second film shows the
NaN scenario -- the culprit node glows red.

Run:   python examples/05_backprop_film.py [--open]
"""

import argparse
import warnings

import numpy as np

from backlens import Tensor, MLP, Tanh, cross_entropy, Adam
from backlens.film import film_backward


def normal_film(path):
    np.random.seed(0)
    x = Tensor(np.random.randn(16, 2), label="batch")
    y = np.random.randint(0, 3, 16)
    model = MLP(2, [8], 3, act=Tanh())

    # a few training steps so gradients have interesting magnitudes
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(30):
        loss = cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    loss = cross_entropy(model(x), y)
    film = film_backward(loss, name="MLP + cross-entropy (after 30 steps)")
    film.save_html(path)
    print(film.summary())
    print(f"wrote {path}")


def nan_film(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # log(0) in the forward pass -> NaN gradients, flagged red in the film
        x = Tensor(np.array([1.0, 0.0, 2.0, 0.5]), requires_grad=True, label="x")
        loss = (x.log() * Tensor(np.array([1.0, 1.0, 1.0, 1.0]))).sum()
        film = film_backward(loss, name="log(0) bug -- NaN gradients")
    film.save_html(path)
    print(film.summary())
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the films in a browser")
    args = ap.parse_args()

    normal_film("film.html")
    print()
    nan_film("film_nan.html")

    if args.open:
        import webbrowser
        webbrowser.open("file://film.html")
    else:
        print("\nopen film.html / film_nan.html in any browser (no server needed)")
