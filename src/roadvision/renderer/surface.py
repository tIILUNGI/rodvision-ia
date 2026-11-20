"""Procedural asphalt and verge rendering.

The asphalt is generated in the road plane and warped through the same
homography used for defect geometry. Texture therefore compresses with distance,
which is a strong depth cue that image-space noise cannot provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config.models import RenderConfig
from .geometry import RoadGeometry
from .noise import fractal_noise, gaussian_blur, speckle


@dataclass
class SurfaceResult:
    image: np.ndarray
    road_mask: np.ndarray
    attributes: dict = field(default_factory=dict)


def _tinted_gray(tone: float, warmth: float) -> np.ndarray:
    return np.array(
        [tone * (1.0 + 0.06 * warmth), tone, tone * (1.0 - 0.05 * warmth)],
        dtype=np.float32,
    )


def _plane_asphalt(rng: np.random.Generator, size: int = 1024) -> np.ndarray:
    tone = rng.uniform(0.34, 0.49)
    warmth = rng.uniform(-0.7, 0.7)
    coarse = fractal_noise(rng, size, size, octaves=6, base_cells=3, persistence=0.60)
    medium = fractal_noise(rng, size, size, octaves=6, base_cells=12, persistence=0.52)
    fine = fractal_noise(rng, size, size, octaves=5, base_cells=48, persistence=0.45)
    grain = rng.normal(0.0, 0.018, (size, size)).astype(np.float32)
    base = _tinted_gray(tone, warmth)

    texture = np.empty((size, size, 3), dtype=np.float32)
    for channel in range(3):
        texture[..., channel] = (
            base[channel]
            + 0.125 * (coarse - 0.5)
            + 0.065 * (medium - 0.5)
            + 0.032 * (fine - 0.5)
            + grain
        )

    bright = speckle(rng, size, size, density=0.0042, bright=1.0)
    dark = speckle(rng, size, size, density=0.0030, bright=1.0)
    texture += bright[..., None] * np.array([0.22, 0.21, 0.18], dtype=np.float32)
    texture -= dark[..., None] * np.array([0.16, 0.16, 0.17], dtype=np.float32)

    # Longitudinal tyre-polish bands, expressed in road-plane coordinates.
    u = np.linspace(0.0, 1.0, size, dtype=np.float32)
    wear = np.zeros((size, size), dtype=np.float32)
    for centre in (0.30, 0.70):
        wear += np.exp(-((u - centre) ** 2) / (2 * 0.035**2))[None, :]
    texture *= (1.0 - 0.075 * np.clip(wear, 0, 1))[..., None]

    # A few darker tar-rich streaks.
    streak = fractal_noise(rng, size, size, octaves=4, base_cells=6)
    texture -= (np.clip(streak - 0.68, 0, 1) ** 1.5)[..., None] * 0.075

    return np.clip(texture, 0.0, 1.0).astype(np.float32), {
        "base_tone": round(float(tone), 6),
        "warmth": round(float(warmth), 6),
    }


def render_surface(
    geometry: RoadGeometry,
    config: RenderConfig,
    rng: np.random.Generator,
) -> SurfaceResult:
    h, w = config.height, config.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    road_mask = geometry.road_mask(w, h) > 0

    # --- perspective-correct asphalt ---------------------------------------
    plane_size = 1024
    plane_texture, attributes = _plane_asphalt(rng, plane_size)
    scale = np.array(
        [[plane_size, 0.0, 0.0], [0.0, plane_size, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = geometry.matrix @ scale
    warped = cv2.warpPerspective(
        np.clip(plane_texture * 255.0, 0, 255).astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32) / 255.0

    # Image-space microtexture keeps the near field from looking too smooth.
    micro = fractal_noise(rng, h, w, octaves=4, base_cells=80)
    warped += (micro[..., None] - 0.5) * 0.060
    image_space_dots = speckle(rng, h, w, density=0.0016, bright=1.0)
    warped += image_space_dots[..., None] * np.array([0.17, 0.16, 0.14])

    # --- verge / background -------------------------------------------------
    grass_tone = rng.uniform(0.16, 0.30)
    grass_warm = rng.uniform(-0.5, 0.7)
    bg_noise = fractal_noise(rng, h, w, octaves=5, base_cells=6)
    bg_fine = fractal_noise(rng, h, w, octaves=4, base_cells=30)
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[..., 0] = grass_tone * (1.05 + 0.14 * grass_warm) + 0.10 * bg_noise + 0.04 * bg_fine
    bg[..., 1] = grass_tone * 1.18 + 0.11 * bg_noise + 0.03 * bg_fine
    bg[..., 2] = grass_tone * (0.72 - 0.10 * grass_warm) + 0.07 * bg_noise
    shoulder = cv2.dilate(road_mask.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    shoulder &= ~road_mask
    gravel = fractal_noise(rng, h, w, octaves=5, base_cells=24)
    shoulder_color = np.array([0.34, 0.31, 0.26], dtype=np.float32)
    bg[shoulder] = bg[shoulder] * 0.30 + shoulder_color * (
        0.70 + 0.35 * gravel[shoulder, None]
    )

    image = np.where(road_mask[..., None], warped, bg)

    # Atmospheric darkening toward the far edge.
    distance = (yy / max(h - 1, 1)).astype(np.float32)
    image = image * (0.90 + 0.10 * distance)[..., None]

    # Road lip: a narrow lighter edge, then a dark curb shadow.
    road_u8 = road_mask.astype(np.uint8)
    edge = (cv2.dilate(road_u8, np.ones((3, 3), np.uint8)) - road_u8) > 0
    image[edge] = np.array([0.47, 0.45, 0.41], dtype=np.float32)
    curb_shadow = (cv2.dilate(road_u8, np.ones((9, 9), np.uint8)) > 0) & ~road_mask
    image[curb_shadow] *= 0.72

    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    attributes["verge_tone"] = round(float(grass_tone), 6)
    attributes["texture_mode"] = "perspective_warped"
    return SurfaceResult(image=image, road_mask=road_mask, attributes=attributes)
