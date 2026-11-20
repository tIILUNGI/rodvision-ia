"""Defect painting.

The public function returns the binary mask that was actually painted. Ground
truth is built from these masks, never from a second estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config.models import DefectClass
from ..generator.scene import PlannedDefect
from .geometry import RoadGeometry
from .noise import fractal_noise, gaussian_blur


@dataclass
class DefectRender:
    object_id: str
    defect_type: DefectClass
    full_mask: np.ndarray
    attributes: dict = field(default_factory=dict)


def _poly_mask(height: int, width: int, points: np.ndarray) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255, lineType=cv2.LINE_AA)
    return mask


def _jagged_line(
    canvas: np.ndarray,
    p1: tuple[float, float] | np.ndarray,
    p2: tuple[float, float] | np.ndarray,
    thickness: int,
    rng: np.random.Generator,
    jitter: float = 1.4,
    value: int = 255,
) -> None:
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    delta = b - a
    length = float(np.hypot(delta[0], delta[1]))
    if length < 1e-6:
        return
    normal = np.array([-delta[1], delta[0]], dtype=np.float64) / length
    mid = (a + b) / 2.0 + normal * float(rng.normal(0.0, jitter))
    cv2.line(
        canvas,
        tuple(np.round(a).astype(int)),
        tuple(np.round(mid).astype(int)),
        value,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )
    cv2.line(
        canvas,
        tuple(np.round(mid).astype(int)),
        tuple(np.round(b).astype(int)),
        value,
        thickness=thickness,
        lineType=cv2.LINE_8,
    )


def _radial_polygon(
    planned: PlannedDefect,
    rng: np.random.Generator,
    points: int = 128,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    radial = np.ones_like(theta)
    for harmonic in range(2, 16):
        amplitude = rng.normal(0.0, 0.075) / (harmonic**0.62)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        radial += amplitude * np.cos(harmonic * theta + phase)
    radial += 0.032 * rng.normal(size=points)
    # A few bite-outs and protrusions make the silhouette read as broken
    # asphalt rather than a smooth ellipse.
    for _ in range(int(rng.integers(3, 8))):
        idx = int(rng.integers(0, points))
        radial[idx] += rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.20)
    radial = np.clip(radial, 0.55, 1.45)
    angle = np.deg2rad(planned.rotation_deg)
    local_x = planned.width_uv * radial * np.cos(theta)
    local_y = planned.height_uv * radial * np.sin(theta)
    x = planned.center_uv[0] + local_x * np.cos(angle) - local_y * np.sin(angle)
    y = planned.center_uv[1] + local_x * np.sin(angle) + local_y * np.cos(angle)
    return np.stack([x, y], axis=1)


def _crack_paths(
    planned: PlannedDefect,
    rng: np.random.Generator,
) -> list[list[tuple[float, float]]]:
    angle = np.deg2rad(planned.rotation_deg)
    total_length = planned.width_uv * 2.0
    steps = int(rng.integers(26, 42))
    step = total_length / steps
    start = np.array(planned.center_uv, dtype=np.float64) - np.array(
        [np.cos(angle), np.sin(angle)]
    ) * (total_length / 2.0)
    points: list[tuple[float, float]] = [tuple(start)]
    current = start.copy()
    local_angle = angle
    for _ in range(steps):
        local_angle += rng.normal(0.0, 0.28)
        current = current + step * np.array([np.cos(local_angle), np.sin(local_angle)])
        points.append((float(current[0]), float(current[1])))
    paths = [points]

    # Branches.
    for _ in range(int(rng.integers(2, 4))):
        lo = max(1, steps // 5)
        hi = max(lo + 1, 4 * steps // 5)
        anchor = points[int(rng.integers(lo, hi))]
        branch_angle = local_angle + rng.choice([-1.0, 1.0]) * rng.uniform(0.40, 0.95)
        branch_steps = max(4, steps // 3)
        branch_step = step * rng.uniform(0.45, 0.85)
        branch = [anchor]
        cx, cy = anchor
        for _ in range(branch_steps):
            branch_angle += rng.normal(0.0, 0.30)
            cx += branch_step * np.cos(branch_angle)
            cy += branch_step * np.sin(branch_angle)
            branch.append((float(cx), float(cy)))
        paths.append(branch)

    # A short secondary fissure running near the main one.
    if len(points) > 10:
        offset = np.array([-np.sin(angle), np.cos(angle)]) * rng.uniform(0.003, 0.007)
        lo = int(len(points) * rng.uniform(0.15, 0.35))
        hi = int(len(points) * rng.uniform(0.60, 0.85))
        secondary = [
            (float(points[i][0] + offset[0]), float(points[i][1] + offset[1]))
            for i in range(lo, hi)
        ]
        if len(secondary) > 2:
            paths.append(secondary)
    return paths


def _draw_crack_mask(
    paths: list[list[tuple[float, float]]],
    planned: PlannedDefect,
    geometry: RoadGeometry,
    width_uv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = geometry.height, geometry.width
    mask = np.zeros((h, w), dtype=np.uint8)
    all_pts = np.concatenate([np.asarray(p) for p in paths], axis=0)
    projected = geometry.project(all_pts)
    cursor = 0
    for path in paths:
        n = len(path)
        pts = projected[cursor : cursor + n]
        cursor += n
        for i in range(n - 1):
            taper = 1.0 - 0.60 * abs((i / max(n - 1, 1)) - 0.5) * 2.0
            v = (path[i][1] + path[i + 1][1]) / 2.0
            base_px = width_uv * geometry.pixels_per_uv_at(v, path[i][0]) * 1.25
            thickness = max(1, min(4, int(round(base_px * taper * rng.uniform(0.75, 1.25)))))
            _jagged_line(mask, pts[i], pts[i + 1], thickness, rng, jitter=0.55)
    return mask


def _alligator_edges(
    planned: PlannedDefect,
    geometry: RoadGeometry,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    h, w = geometry.height, geometry.width
    n = int(rng.integers(36, 52))
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.18, 1.0, n)
    angle0 = np.deg2rad(planned.rotation_deg)
    uv_points = []
    for theta, r in zip(angles, radii, strict=True):
        x = planned.width_uv * r * np.cos(theta)
        y = planned.height_uv * r * np.sin(theta)
        px = planned.center_uv[0] + x * np.cos(angle0) - y * np.sin(angle0)
        py = planned.center_uv[1] + x * np.sin(angle0) + y * np.cos(angle0)
        uv_points.append((px, py))
    # Add points near the centre so the network is not a ring.
    for _ in range(6):
        uv_points.append(
            (
                planned.center_uv[0] + rng.normal(0, planned.width_uv * 0.30),
                planned.center_uv[1] + rng.normal(0, planned.height_uv * 0.30),
            )
        )
    points = geometry.project(np.asarray(uv_points, dtype=np.float32))

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for x, y in points:
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            subdiv.insert((float(x), float(y)))
    triangles = subdiv.getTriangleList()
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    lengths: list[float] = []
    for tri in triangles:
        verts = [
            np.array([tri[0], tri[1]], dtype=np.float64),
            np.array([tri[2], tri[3]], dtype=np.float64),
            np.array([tri[4], tri[5]], dtype=np.float64),
        ]
        for a, b in ((verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])):
            pa = (int(round(a[0])), int(round(a[1])))
            pb = (int(round(b[0])), int(round(b[1])))
            key = (min(pa, pb), max(pa, pb))
            if key in seen:
                continue
            seen.add(key)
            length = float(np.hypot(*(b - a)))
            if length < 2.0:
                continue
            edges.append((a, b))
            lengths.append(length)
    if lengths:
        median = float(np.median(lengths))
        edges = [
            (a, b)
            for a, b in edges
            if float(np.hypot(*(b - a))) <= max(median * 1.35, 9.0)
        ]

    # A broken boundary, but only short segments so it does not become a thick
    # continuous outline.
    perimeter_uv = []
    for theta in np.linspace(0, 2 * np.pi, 44, endpoint=False):
        r = rng.uniform(0.88, 1.08)
        x = planned.width_uv * r * np.cos(theta)
        y = planned.height_uv * r * np.sin(theta)
        px = planned.center_uv[0] + x * np.cos(angle0) - y * np.sin(angle0)
        py = planned.center_uv[1] + x * np.sin(angle0) + y * np.cos(angle0)
        perimeter_uv.append((px, py))
    perimeter = geometry.project(np.asarray(perimeter_uv, dtype=np.float32))
    for i in range(len(perimeter)):
        if rng.random() < 0.55:
            edges.append((perimeter[i], perimeter[(i + 1) % len(perimeter)]))
    return edges, len(points)


def _paint_pothole(
    image: np.ndarray,
    mask: np.ndarray,
    planned: PlannedDefect,
    rng: np.random.Generator,
) -> dict:
    h, w = mask.shape
    m = mask > 0
    ys, xs = np.nonzero(m)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bh = max(1, y2 - y1)
    bw = max(1, x2 - x1)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    depth = dist / max(float(dist.max()), 1.0)
    tex = fractal_noise(rng, h, w, octaves=6, base_cells=20)
    mottle = fractal_noise(rng, h, w, octaves=4, base_cells=5)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    y_norm = np.clip((yy - y1) / bh, 0.0, 1.0)
    x_norm = np.clip((xx - x1) / bw, 0.0, 1.0)

    deep = rng.uniform(0.028, 0.055)
    # Deep floor: dark, mottled, slightly warmer at the bottom.
    value = (
        deep
        + 0.055 * (1.0 - depth)
        + 0.060 * (tex - 0.5)
        + 0.040 * (mottle - 0.5)
        + 0.022 * y_norm
    )
    interior = np.empty((h, w, 3), dtype=np.float32)
    interior[..., 0] = value * 1.24
    interior[..., 1] = value * 0.98
    interior[..., 2] = value * 0.78
    image[m] = interior[m]

    # The exposed inner wall is much lighter on the top/left side; this is the
    # strongest depth cue in the image.
    erosion = int(rng.integers(5, 10))
    inner = cv2.erode(mask, np.ones((erosion, erosion), np.uint8)) > 0
    wall = m & ~inner
    wall_value = (
        0.09
        + 0.17 * (1.0 - y_norm)
        + 0.055 * (1.0 - x_norm)
        + 0.085 * (tex - 0.5)
        + 0.045 * (mottle - 0.5)
    )
    wall_value = np.clip(wall_value, 0.045, 0.42)
    wall_color = np.stack(
        [wall_value * 1.25, wall_value * 1.02, wall_value * 0.84], axis=-1
    )
    image[wall] = wall_color[wall]

    # Shadow under the overhanging lip.
    top_shadow = m & (y_norm < 0.11)
    image[top_shadow] *= 0.58
    # Deep shadow on the lower-right wall.
    deep_shadow = (cv2.erode(mask, np.ones((erosion + 5, erosion + 5), np.uint8)) == 0) & m
    deep_shadow &= (y_norm > 0.55) | (x_norm > 0.62)
    image[deep_shadow] *= 0.55

    # Loose stones and aggregate in the bottom half.
    stones = np.zeros((h, w), dtype=np.uint8)
    stone_vals = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(rng.integers(16, 34))):
        cx = int(rng.integers(x1, x2))
        cy = int(rng.integers(y1 + int(0.28 * bh), y2))
        if not m[cy, cx]:
            continue
        ax = int(rng.integers(1, max(2, bw // 16)))
        ay = int(rng.integers(1, max(2, bh // 18)))
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(stones, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
        cv2.ellipse(
            stone_vals,
            (cx, cy),
            (ax, ay),
            angle,
            0,
            360,
            float(rng.uniform(0.12, 0.42)),
            -1,
        )
    stones = (stones > 0) & m
    if stones.any():
        sv = stone_vals[stones]
        image[stones] = np.stack(
            [sv * 1.28 + 0.04, sv * 1.08 + 0.03, sv * 0.86 + 0.02], axis=-1
        )

    # A few interior crevices.
    for _ in range(int(rng.integers(1, 4))):
        a = (rng.uniform(x1, x2), rng.uniform(y1, y2))
        b = (rng.uniform(x1, x2), rng.uniform(y1, y2))
        overlay = np.zeros((h, w), dtype=np.uint8)
        _jagged_line(overlay, a, b, 1, rng, jitter=1.2)
        overlay = overlay > 0
        image[overlay & m] *= 0.50

    # Broken chunks of asphalt around the lip, kept outside the counted mask.
    outer = (cv2.dilate(mask, np.ones((13, 13), np.uint8)) > 0) & ~m
    chunk_noise = gaussian_blur(rng.random((h, w)).astype(np.float32), 0.7)
    chunks = outer & (chunk_noise > 0.64)
    if chunks.any():
        cv2_chunk_values = 0.16 + 0.18 * chunk_noise
        image[chunks] = np.stack(
            [
                (cv2_chunk_values * 1.25)[chunks],
                (cv2_chunk_values * 1.02)[chunks],
                (cv2_chunk_values * 0.84)[chunks],
            ],
            axis=-1,
        )
    # Dark seam where the broken lip meets intact asphalt.
    seam = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & ~m
    image[seam] *= 0.68

    return {
        "depth_contrast": round(float(deep), 6),
        "mask_pixels": int(m.sum()),
        "rim_px": erosion,
        "stone_count": int((stones > 0).sum()),
    }


def _paint_crack(
    image: np.ndarray,
    mask: np.ndarray,
    planned: PlannedDefect,
    rng: np.random.Generator,
) -> dict:
    m = mask > 0
    halo = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & ~m
    image[halo] = image[halo] * 0.78
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    depth = dist / max(float(dist.max()), 1.0)
    dark = rng.uniform(0.028, 0.052)
    color = np.empty((*mask.shape, 3), dtype=np.float32)
    color[..., 0] = dark * 1.12 + 0.010 * (1.0 - depth)
    color[..., 1] = dark * 0.98 + 0.010 * (1.0 - depth)
    color[..., 2] = dark * 0.86 + 0.009 * (1.0 - depth)
    image[m] = image[m] * 0.12 + color[m] * 0.88
    # Occasional lighter exposed aggregate on one side of the fissure.
    edge = m & (dist < 1.15)
    image[edge] += rng.uniform(0.0, 0.020)
    return {"mask_pixels": int(m.sum()), "depth_contrast": round(float(dark), 6)}


def _paint_patch(
    image: np.ndarray,
    mask: np.ndarray,
    planned: PlannedDefect,
    rng: np.random.Generator,
) -> dict:
    h, w = mask.shape
    m = mask > 0
    lighter = bool(rng.random() < 0.55)
    tone = rng.uniform(0.36, 0.48) if lighter else rng.uniform(0.20, 0.28)
    texture = fractal_noise(rng, h, w, octaves=5, base_cells=16)
    grain = rng.normal(0.0, 0.022, (h, w)).astype(np.float32)
    aggregate = (rng.random((h, w)) > 0.9955).astype(np.float32)
    aggregate = gaussian_blur(aggregate, 0.5) * 0.10
    color = np.empty((h, w, 3), dtype=np.float32)
    color[..., 0] = tone + 0.08 * (texture - 0.5) + grain + aggregate
    color[..., 1] = tone * 0.985 + 0.08 * (texture - 0.5) + grain + aggregate
    color[..., 2] = tone * 0.95 + 0.07 * (texture - 0.5) + grain + aggregate
    image[m] = color[m]

    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    seam = m & ~inner
    image[seam] *= 0.55
    # Raised edge highlight along the top and shadow along the bottom.
    shift = max(2, h // 150)
    upper = np.zeros_like(mask)
    upper[shift:] = mask[:-shift]
    lower_edge = m & ~(upper > 0)
    image[lower_edge] *= 0.80
    lower = np.zeros_like(mask)
    lower[:-shift] = mask[shift:]
    upper_edge = m & ~(lower > 0)
    image[upper_edge] += 0.025
    return {"patch_tone": round(float(tone), 6), "lighter": lighter}


def render_defect(
    image: np.ndarray,
    planned: PlannedDefect,
    geometry: RoadGeometry,
    rng: np.random.Generator,
) -> DefectRender:
    h, w = geometry.height, geometry.width
    if planned.type == DefectClass.POTHOLE:
        polygon_uv = _radial_polygon(planned, rng)
        points = geometry.to_pixel_polygon(polygon_uv)
        mask = _poly_mask(h, w, points)
        attributes = _paint_pothole(image, mask, planned, rng)
    elif planned.type == DefectClass.CRACK:
        paths = _crack_paths(planned, rng)
        width_uv = max(planned.height_uv, 0.0018)
        mask = _draw_crack_mask(paths, planned, geometry, width_uv, rng)
        attributes = _paint_crack(image, mask, planned, rng)
    elif planned.type == DefectClass.ALLIGATOR_CRACK:
        edges, node_count = _alligator_edges(planned, geometry, rng)
        mask = np.zeros((h, w), dtype=np.uint8)
        thickness = max(1, min(2, int(round(geometry.pixels_per_uv_at(planned.center_uv[1]) * 0.0019))))
        for a, b in edges:
            _jagged_line(mask, a, b, thickness, rng, jitter=1.4)
        attributes = _paint_crack(image, mask, planned, rng)
        attributes["node_count"] = int(node_count)
        attributes["edge_count"] = int(len(edges))
    elif planned.type == DefectClass.PATCH:
        angle = np.deg2rad(planned.rotation_deg)
        local = np.array(
            [
                [-planned.width_uv, -planned.height_uv],
                [planned.width_uv, -planned.height_uv],
                [planned.width_uv, planned.height_uv],
                [-planned.width_uv, planned.height_uv],
            ],
            dtype=np.float32,
        )
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        local = local @ rotation.T
        local += np.array(planned.center_uv, dtype=np.float32)
        local += rng.normal(0.0, 0.004, local.shape).astype(np.float32)
        points = geometry.to_pixel_polygon(local)
        mask = _poly_mask(h, w, points)
        attributes = _paint_patch(image, mask, planned, rng)
    else:  # pragma: no cover
        raise ValueError(f"unsupported defect {planned.type}")

    if int(mask.sum()) == 0:
        raise RuntimeError(f"defect {planned.id} produced an empty mask")
    return DefectRender(
        object_id=planned.id,
        defect_type=planned.type,
        full_mask=mask > 0,
        attributes=attributes,
    )
