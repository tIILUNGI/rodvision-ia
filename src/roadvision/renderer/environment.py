"""Lighting, shadow and wet-surface effects.

These effects change appearance but are not opaque occluders. Ground-truth
visibility is therefore unaffected by them; robustness questions can use them
without corrupting instance geometry.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config.models import Lighting, RenderConfig, Weather
from .geometry import RoadGeometry
from .noise import fractal_noise, gaussian_blur, soft_blobs


def apply_lighting(
    image: np.ndarray, lighting: Lighting, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    if lighting == Lighting.OVERCAST:
        gain = rng.uniform(0.94, 1.03)
        contrast = rng.uniform(0.90, 0.98)
        cast = np.array([-0.008, 0.0, 0.012], dtype=np.float32)
    elif lighting == Lighting.GOLDEN:
        gain = rng.uniform(0.98, 1.06)
        contrast = rng.uniform(0.98, 1.06)
        cast = np.array([0.045, 0.018, -0.028], dtype=np.float32)
    else:  # harsh
        gain = rng.uniform(1.02, 1.12)
        contrast = rng.uniform(1.10, 1.22)
        cast = np.array([0.020, 0.010, -0.015], dtype=np.float32)

    out = image * gain
    out = (out - 0.5) * contrast + 0.5
    out = out + cast
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "lighting_gain": round(float(gain), 6),
        "lighting_contrast": round(float(contrast), 6),
    }


def apply_shadows(
    image: np.ndarray,
    geometry: RoadGeometry,
    strength: float,
    coverage: float,
    strong: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = image.shape[:2]
    field = soft_blobs(
        rng,
        h,
        w,
        count=int(rng.integers(4, 9)),
        min_radius=0.18,
        max_radius=0.55,
        blur_sigma=30.0,
    )
    dapple = fractal_noise(rng, h, w, octaves=4, base_cells=14)
    field = np.clip(0.70 * field + 0.30 * dapple, 0.0, 1.0)
    # Smooth, partially transparent canopy shadow. A hard threshold here
    # produces camouflage-like islands, which is a realism failure.
    mask = gaussian_blur(field, 3.0 if strong else 10.0)
    mask = np.clip((mask - 0.18) / 0.82, 0.0, 1.0)
    factor = 1.0 - strength * mask
    out = image * factor[..., None]
    actual = float((mask > 0.15).mean())
    return out.astype(np.float32), mask, round(actual, 6)


def apply_weather(
    image: np.ndarray,
    geometry: RoadGeometry,
    weather: Weather,
    puddle_strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    out = image.copy()
    if weather == Weather.DRY:
        return out, 0.0

    if weather == Weather.DAMP:
        out *= rng.uniform(0.90, 0.96)
        sheen = soft_blobs(rng, h, w, 3, 0.18, 0.42, blur_sigma=28.0)
        out += (sheen * 0.035)[..., None]
        return np.clip(out, 0.0, 1.0), round(float(sheen.mean()), 6)

    # Wet: darker substrate, standing water, reflections.
    out *= rng.uniform(0.76, 0.87)
    puddles = soft_blobs(
        rng,
        h,
        w,
        count=int(rng.integers(3, 7)),
        min_radius=0.14,
        max_radius=0.40,
        blur_sigma=12.0,
    )
    level = float(np.quantile(puddles, 0.55 + 0.25 * (1.0 - puddle_strength)))
    water = np.clip((puddles - level) / max(1e-6, 1.0 - level), 0.0, 1.0)
    water = gaussian_blur(water, 2.0)
    water_color = np.array([0.11, 0.15, 0.18], dtype=np.float32)
    out = out * (1.0 - 0.52 * water[..., None]) + water_color * (0.52 * water[..., None])
    # A cool sky reflection and a few sharp glints.
    reflection = fractal_noise(rng, h, w, octaves=4, base_cells=9)
    glint = np.clip((reflection - 0.72) * 2.8, 0.0, 1.0) * water
    out += glint[..., None] * np.array([0.20, 0.22, 0.24], dtype=np.float32)
    # Fine ripples.
    ripple = np.sin(np.linspace(0, 18 * np.pi, w, dtype=np.float32))[None, :]
    out += (ripple * water * 0.012)[..., None]
    actual = float((water > 0.15).mean())
    return np.clip(out, 0.0, 1.0).astype(np.float32), round(actual, 6)


def finalize_image(
    image: np.ndarray,
    config: RenderConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = image.astype(np.float32)
    if config.blur_sigma > 0:
        out = gaussian_blur(out, config.blur_sigma)
    if config.grain_sigma > 0:
        grain = rng.normal(0.0, config.grain_sigma / 255.0, out.shape[:2])
        out += grain[..., None]
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2.0) / (w / 2.0)
    ny = (yy - h / 2.0) / (h / 2.0)
    radius = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
    vignette = 1.0 - config.vignette * (radius**2)
    out *= vignette[..., None]
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out
