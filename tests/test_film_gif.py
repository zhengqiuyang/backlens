"""Tests for BackwardFilm.save_gif (requires the optional pillow extra)."""

import os

import numpy as np
import pytest

from backlens import Tensor, MLP, Tanh, cross_entropy
from backlens.film import film_backward

pil = pytest.importorskip("PIL")


def build_film():
    np.random.seed(0)
    x = Tensor(np.random.randn(6, 2), label="batch")
    y = np.random.randint(0, 3, 6)
    model = MLP(2, [4], 3, act=Tanh())
    return film_backward(cross_entropy(model(x), y), name="gif test")


def test_save_gif_writes_animation(tmp_path):
    film = build_film()
    path = os.path.join(str(tmp_path), "film.gif")
    film.save_gif(path, duration_ms=80, final_hold_ms=300)
    from PIL import Image
    with Image.open(path) as im:
        assert im.format == "GIF"
        frames = getattr(im, "n_frames", 1)
        # one frame per backward step + the initial (pre-backward) state
        assert frames == len(film.steps) + 1
        im.seek(0)
        w, h = im.size
        assert w > 300 and h > 200


def test_save_gif_frames_advance(tmp_path):
    """Per-step durations and the final hold must be honored."""
    film = build_film()
    path = os.path.join(str(tmp_path), "film.gif")
    film.save_gif(path, duration_ms=60, final_hold_ms=900)
    from PIL import Image
    im = Image.open(path)
    im.seek(0)
    assert im.info["duration"] == 60
    im.seek(im.n_frames - 1)
    assert im.info["duration"] == 900      # final frame holds longer
