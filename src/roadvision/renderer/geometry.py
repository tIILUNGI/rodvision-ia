"""Projective road-plane geometry.

Scene objects are planned in a unit road plane. A single homography maps the
plane to the image, so the pixel-space masks and boxes are exactly the
projection that was rendered.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RoadGeometry:
    width: int
    height: int
    perspective: float = 0.55

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def __post_init__(self) -> None:
        p = float(np.clip(self.perspective, 0.0, 1.0))
        far_inset = 0.10 + 0.20 * p
        top_y = -0.10 * self.height
        bottom_y = 1.06 * self.height
        near_left = -0.10 * self.width
        near_right = 1.10 * self.width
        far_left = far_inset * self.width
        far_right = (1.0 - far_inset) * self.width
        self._quad = np.array(
            [
                [far_left, top_y],
                [far_right, top_y],
                [near_right, bottom_y],
                [near_left, bottom_y],
            ],
            dtype=np.float32,
        )
        src = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
        )
        self._matrix = cv2.getPerspectiveTransform(src, self._quad)

    @property
    def road_quad(self) -> np.ndarray:
        return self._quad.copy()

    def project(self, points_uv: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_uv, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self._matrix).reshape(-1, 2)
        if not np.isfinite(out).all():
            raise ValueError("non-finite projected geometry")
        return out

    def project_one(self, u: float, v: float) -> tuple[float, float]:
        x, y = self.project(np.array([[u, v]], dtype=np.float32))[0]
        return float(x), float(y)

    def pixels_per_uv_at(self, v: float, u: float = 0.5) -> float:
        """Approximate horizontal pixels per one unit of road-plane u."""
        eps = 1e-4
        a = self.project_one(max(0.0, u - eps), v)
        b = self.project_one(min(1.0, u + eps), v)
        return float(np.hypot(b[0] - a[0], b[1] - a[1]) / (2 * eps))

    def road_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = np.round(self._quad).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        return mask

    def to_pixel_polygon(self, points_uv: np.ndarray) -> np.ndarray:
        pts = self.project(points_uv)
        return np.round(pts).astype(np.int32)
