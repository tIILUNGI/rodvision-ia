"""Opaque occluders.

An occluder is a physical object placed over the road. It has its own mask and
changes the visible fraction of every defect it covers. The target coverage is
achieved by a deterministic offset search, not by trial-and-error during
painting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config.models import OccluderType
from ..generator.scene import PlannedOccluder
from .geometry import RoadGeometry
from .noise import fractal_noise, gaussian_blur


@dataclass
class OccluderRender:
    object_id: str
    occluder_type: OccluderType
    mask: np.ndarray
    attributes: dict = field(default_factory=dict)


def _sheet_polygon(
    width: int,
    height: int,
    rng: np.random.Generator,
    points: int = 24,
) -> np.ndarray:
    """Irregular rounded rectangle in local coordinates around the origin."""
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    # Superellipse gives a sheet-like shape with softened corners.
    exponent = 2.0 / 4.0
    cos = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** exponent
    sin = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** exponent
    radial = 1.0 + 0.11 * rng.normal(size=points)
    x = (width / 2.0) * cos * radial
    y = (height / 2.0) * sin * radial
    return np.stack([x, y], axis=1)


def _leaf_cluster_mask(
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    count = int(rng.integers(8, 15))
    for _ in range(count):
        cx = int(rng.uniform(0.15, 0.85) * width)
        cy = int(rng.uniform(0.15, 0.85) * height)
        rx = int(rng.uniform(0.10, 0.22) * width)
        ry = int(rng.uniform(0.08, 0.18) * height)
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(canvas, (cx, cy), (max(4, rx), max(3, ry)), angle, 0, 360, 255, -1)
    return canvas


def _build_occluder_mask(
    local_polygon: np.ndarray | None,
    leaf_patch: np.ndarray | None,
    center: tuple[float, float],
    occluder_type: OccluderType,
    h: int,
    w: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if occluder_type == OccluderType.LEAF_LITTER:
        assert leaf_patch is not None
        lh, lw = leaf_patch.shape
        x0 = int(round(center[0] - lw / 2))
        y0 = int(round(center[1] - lh / 2))
        x1, y1 = x0 + lw, y0 + lh
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x1), min(h, y1)
        if sx1 > sx0 and sy1 > sy0:
            mask[sy0:sy1, sx0:sx1] = leaf_patch[
                sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0
            ]
    else:
        assert local_polygon is not None
        pts = np.round(local_polygon + np.array(center, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA)
    return mask


def render_occluder(
    image: np.ndarray,
    planned: PlannedOccluder,
    target_mask: np.ndarray,
    geometry: RoadGeometry,
    rng: np.random.Generator,
) -> OccluderRender:
    h, w = geometry.height, geometry.width
    target = target_mask.astype(bool)
    ys, xs = np.nonzero(target)
    if len(xs) == 0:
        raise ValueError("occluder target has an empty mask")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(8, x2 - x1), max(8, y2 - y1)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    sheet_w = int(max(28, bw * 1.18 + 12))
    sheet_h = int(max(28, bh * 1.45 + 12))

    if planned.type == OccluderType.LEAF_LITTER:
        leaf_full = _leaf_cluster_mask(sheet_w, sheet_h, rng)
        polygon_full = None
    else:
        leaf_full = None
        polygon_full = _sheet_polygon(sheet_w, sheet_h, rng)

    direction = np.deg2rad(planned.direction_deg)
    unit = np.array([np.cos(direction), np.sin(direction)], dtype=np.float64)
    target_fraction = float(planned.target_visible_fraction)
    target_pixels = max(1, int(target.sum()))

    def coverage(scale: float, offset: float) -> tuple[float, np.ndarray]:
        if polygon_full is not None:
            polygon = polygon_full * scale
        else:
            polygon = None
        if leaf_full is not None:
            sw = max(8, int(round(leaf_full.shape[1] * scale)))
            sh = max(8, int(round(leaf_full.shape[0] * scale)))
            leaf = cv2.resize(leaf_full, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            leaf = None
        c = (center[0] + unit[0] * offset, center[1] + unit[1] * offset)
        mask = _build_occluder_mask(polygon, leaf, c, planned.type, h, w)
        visible = int((target & (mask == 0)).sum())
        return visible / target_pixels, mask

    # The sheet is scanned across the target from both sides. A binary search
    # would be cheaper but is only valid when coverage is monotone; leaves and
    # irregular sheets are not. Shrinking the sheet is the fallback when the
    # requested visible fraction cannot be hit with the initial size.
    best_vf = 1.0
    best_scale = 1.0
    best_offset = 0.0
    best_mask = np.zeros((h, w), np.uint8)
    for scale in (1.0, 0.90, 0.82, 0.74, 0.66, 0.58):
        max_offset = float(np.hypot(sheet_w, sheet_h)) * scale
        candidates = np.linspace(-max_offset, max_offset, 65)
        for candidate in candidates:
            vf, mask = coverage(scale, float(candidate))
            if abs(vf - target_fraction) < abs(best_vf - target_fraction):
                best_vf, best_scale, best_offset, best_mask = (
                    vf,
                    scale,
                    float(candidate),
                    mask,
                )
        # Local refinement around the best candidate, three decimal decades.
        step = 2.0 * max_offset / 64.0
        for _ in range(3):
            lo = best_offset - step
            hi = best_offset + step
            for candidate in np.linspace(lo, hi, 21):
                vf, mask = coverage(best_scale, float(candidate))
                if abs(vf - target_fraction) < abs(best_vf - target_fraction):
                    best_vf = vf
                    best_offset = float(candidate)
                    best_mask = mask
            step /= 10.0
        if abs(best_vf - target_fraction) <= 0.012:
            break

    actual_vf = best_vf
    mask = best_mask
    offset = best_offset

    m = mask > 0
    if not m.any():
        raise RuntimeError("occluder mask is empty")

    # Drop shadow around the opaque object.
    shadow = (cv2.dilate(mask, np.ones((15, 15), np.uint8)) > 0) & ~m
    shadow_f = gaussian_blur(shadow.astype(np.float32), 5.0)
    image[:] = image * (1.0 - 0.35 * shadow_f[..., None])

    # Opaque object texture.
    texture = fractal_noise(rng, h, w, octaves=5, base_cells=18)
    ys_m, xs_m = np.nonzero(m)
    bx1, bx2 = int(xs_m.min()), int(xs_m.max()) + 1
    by1, by2 = int(ys_m.min()), int(ys_m.max()) + 1
    if planned.type == OccluderType.DEBRIS_SHEET:
        base = np.array([0.075, 0.105, 0.130], dtype=np.float32)
        color = base + (texture[..., None] - 0.5) * 0.050
        folds = fractal_noise(rng, h, w, octaves=4, base_cells=30)
        color += np.clip(folds - 0.62, 0, 1)[..., None] * 0.16
        # Creases: a bright ridge next to a dark valley, like a folded tarp.
        crease = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(3, 6))):
            angle = float(rng.uniform(0, np.pi))
            length = float(rng.uniform(0.35, 0.85)) * max(bx2 - bx1, by2 - by1)
            cx0 = float(rng.uniform(bx1, bx2))
            cy0 = float(rng.uniform(by1, by2))
            dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
            pts = np.array(
                [[cx0 - dx, cy0 - dy], [cx0, cy0 + rng.normal(0, 12)], [cx0 + dx, cy0 + dy]],
                dtype=np.float32,
            )
            cv2.polylines(crease, [np.round(pts).astype(np.int32)], False, 255, 1, cv2.LINE_AA)
        crease_f = gaussian_blur((crease > 0).astype(np.float32), 2.4)
        ridge = np.clip(crease_f, 0, 1)
        color += ridge[..., None] * np.array([0.045, 0.048, 0.052], dtype=np.float32)
        color -= np.clip(ridge - 0.45, 0, 1)[..., None] * 0.030
        # Slightly lighter dusty film on the upper side of the sheet.
        color += np.clip((0.5 - texture) * 0.5, 0, 1)[..., None] * 0.03
        # Edge wear: the sheet border catches a little light.
        inside = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        border = m & ~inside
        color[border] += np.array([0.07, 0.07, 0.07], dtype=np.float32)
    elif planned.type == OccluderType.CARDBOARD:
        base = np.array([0.38, 0.27, 0.16], dtype=np.float32)
        color = base + (texture[..., None] - 0.5) * 0.10
        corrugation = (np.sin(np.linspace(0, 30 * np.pi, w, dtype=np.float32))[None, :] * 0.5 + 0.5)
        color *= (0.88 + 0.12 * corrugation)[..., None]
        tape = np.zeros((h, w), np.uint8)
        ty = int((by1 + by2) / 2)
        cv2.line(tape, (bx1, ty), (bx2, ty), 255, max(3, (by2 - by1) // 14), cv2.LINE_AA)
        tape_f = gaussian_blur((tape > 0).astype(np.float32), 1.0)
        color = color * (1 - 0.35 * tape_f[..., None]) + np.array(
            [0.55, 0.47, 0.34], dtype=np.float32
        ) * (0.35 * tape_f[..., None])
    else:
        color = np.empty((h, w, 3), dtype=np.float32)
        color[..., 0] = 0.20 + 0.18 * texture
        color[..., 1] = 0.27 + 0.18 * texture
        color[..., 2] = 0.10 + 0.10 * texture
        # Veins within the leaf cluster only.
        veins = np.zeros((h, w), np.uint8)
        for _ in range(14):
            x0 = int(rng.uniform(bx1, bx2))
            y0 = int(rng.uniform(by1, by2))
            x1 = int(np.clip(x0 + rng.normal(0, sheet_w * 0.25), bx1, bx2 - 1))
            y1 = int(np.clip(y0 + rng.normal(0, sheet_h * 0.25), by1, by2 - 1))
            cv2.line(veins, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        color += ((veins > 0) & m)[..., None] * np.array([0.06, 0.08, 0.02], dtype=np.float32)
        # Leaves have a slight sheen where they face the light.
        color += (texture**2)[..., None] * np.array([0.10, 0.12, 0.04], dtype=np.float32)

    image[m] = color[m]
    return OccluderRender(
        object_id=planned.id,
        occluder_type=planned.type,
        mask=mask > 0,
        attributes={
            "target_object_id": planned.target_object_id,
            "target_visible_fraction": round(float(actual_vf), 6),
            "requested_visible_fraction": planned.target_visible_fraction,
            "offset_px": round(float(offset), 4),
            "area_px": int(m.sum()),
        },
    )
