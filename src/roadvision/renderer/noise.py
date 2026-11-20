"""Procedural noise helpers.

All functions take an explicit NumPy Generator. No global random state is used,
so rendering order cannot change the output.
"""

from __future__ import annotations

import cv2
import numpy as np


def _resize(layer: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(layer, (width, height), interpolation=cv2.INTER_CUBIC)


def fractal_noise(
    rng: np.random.Generator,
    height: int,
    width: int,
    octaves: int = 5,
    base_cells: int = 4,
    persistence: float = 0.55,
) -> np.ndarray:
    """Multi-octave value noise in [0, 1]."""
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    norm = 0.0
    for octave in range(octaves):
        cells = base_cells * (2**octave)
        gh = max(2, int(np.ceil(height / cells)))
        gw = max(2, int(np.ceil(width / cells)))
        layer = rng.random((gh, gw), dtype=np.float32)
        total += amplitude * _resize(layer, height, width)
        norm += amplitude
        amplitude *= persistence
    total /= max(norm, 1e-6)
    lo, hi = float(total.min()), float(total.max())
    if hi - lo < 1e-8:
        return np.zeros_like(total)
    return (total - lo) / (hi - lo)


def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return array
    return cv2.GaussianBlur(array, (0, 0), sigmaX=sigma, sigmaY=sigma)


def soft_blobs(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
    min_radius: float,
    max_radius: float,
    blur_sigma: float = 18.0,
) -> np.ndarray:
    """Soft blob field in [0, 1], useful for shadows, dampness and puddles."""
    small_h = max(24, height // 6)
    small_w = max(24, width // 6)
    canvas = np.zeros((small_h, small_w), dtype=np.float32)
    for _ in range(count):
        cx = rng.uniform(0, small_w)
        cy = rng.uniform(0, small_h)
        radius = rng.uniform(min_radius, max_radius) * small_w
        axes = (
            max(1, int(radius)),
            max(1, int(radius * rng.uniform(0.45, 1.0))),
        )
        angle = rng.uniform(0, 180)
        cv2.ellipse(
            canvas,
            (int(cx), int(cy)),
            axes,
            angle,
            0,
            360,
            1.0,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    canvas = _resize(canvas, height, width)
    canvas = gaussian_blur(canvas, blur_sigma)
    lo, hi = float(canvas.min()), float(canvas.max())
    if hi - lo < 1e-8:
        return np.zeros_like(canvas)
    return np.clip((canvas - lo) / (hi - lo), 0.0, 1.0)


def speckle(
    rng: np.random.Generator,
    height: int,
    width: int,
    density: float,
    bright: float = 1.0,
) -> np.ndarray:
    """Sparse aggregate-like dots, slightly blurred."""
    canvas = np.zeros((height, width), dtype=np.float32)
    count = int(height * width * density)
    if count <= 0:
        return canvas
    xs = rng.integers(0, width, count)
    ys = rng.integers(0, height, count)
    canvas[ys, xs] = rng.uniform(0.4, 1.0, count).astype(np.float32) * bright
    return gaussian_blur(canvas, 0.7)
