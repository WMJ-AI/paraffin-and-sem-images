"""Stem mask, pith center, and geometry clamping."""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.geometry import LineGeometry


def detect_stem_mask(img: np.ndarray) -> tuple[np.ndarray, float] | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    contour = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(contour) < img.shape[0] * img.shape[1] * 0.02:
        return None

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radius = float(dist.max())
    return mask, radius


def detect_pith(img: np.ndarray, stem_mask: np.ndarray) -> tuple[float, float]:
    """Locate pith: darkest high-distance interior point."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5).astype(np.float64)
    score = dist * (255.0 - gray) * (stem_mask > 0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(x), float(y)


def _dt_peak(stem_mask: np.ndarray) -> tuple[tuple[float, float], float, np.ndarray]:
    dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return (float(x), float(y)), float(dist.max()), dist


def _pith_from_inner_circle(
    img: np.ndarray,
    stem_mask: np.ndarray,
    dist: np.ndarray,
) -> tuple[float, float] | None:
    """Center of the pith–xylem ring. Survives 4X crops that clip the outer bark."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem_r = float(dist.max())
    if stem_r < 40:
        return None
    h, w = gray.shape
    scale = min(1.0, 960.0 / max(h, w))
    sw, sh = int(w * scale), int(h * scale)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (7, 7), 0)
    small_stem = cv2.resize(stem_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    min_r = max(8, int(0.055 * stem_r * scale))
    max_r = max(min_r + 4, int(0.34 * stem_r * scale))
    circles = cv2.HoughCircles(
        small,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=float(max(min_r, 12)),
        param1=70,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    yy, xx = np.ogrid[:sh, :sw]
    best = None
    best_score = -1e9
    for c in circles[0]:
        cx_s, cy_s, r_s = float(c[0]), float(c[1]), float(c[2])
        ix, iy = int(round(cx_s)), int(round(cy_s))
        if not (0 <= ix < sw and 0 <= iy < sh) or small_stem[iy, ix] == 0:
            continue
        d2 = (xx - cx_s) ** 2 + (yy - cy_s) ** 2
        inner = (d2 <= (0.65 * r_s) ** 2) & (small_stem > 0)
        ring = (d2 >= r_s ** 2) & (d2 <= (1.35 * r_s) ** 2) & (small_stem > 0)
        if int(inner.sum()) < 40 or int(ring.sum()) < 40:
            continue
        pale = float(small[inner].mean()) - float(small[ring].mean())
        r_frac = (r_s / scale) / stem_r
        score = pale - 80.0 * abs(r_frac - 0.16)
        if score > best_score:
            best_score = score
            best = (cx_s / scale, cy_s / scale)
    if best is None or best_score < 3.0:
        return None
    return best


def _pith_low_vessel(
    img: np.ndarray,
    stem_mask: np.ndarray,
    dist: np.ndarray,
) -> tuple[float, float] | None:
    """4X pith: pale tissue with few vessels, not the outer bark."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stem_r = float(dist.max())
    if stem_r < 40:
        return None
    k = int(max(21, (0.07 * stem_r) // 2 * 2 + 1))
    white = ((gray >= 205) & (stem_mask > 0)).astype(np.uint8) * 255
    dens = cv2.blur(white.astype(np.float32), (k, k)) / 255.0
    pale = gray.astype(np.float32) / 255.0
    # Do not weight by whole-stem DT: 4X crops clip the disk and shift the peak.
    score = (1.0 - dens) * (0.35 + 0.65 * pale)
    score[stem_mask == 0] = 0
    score[dist < 0.10 * stem_r] = 0
    if float(score.max()) <= 0:
        return None
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(x), float(y)


def detect_pith_center(
    img: np.ndarray,
    stem_mask: np.ndarray,
    mag: int = 2,
) -> tuple[float, float]:
    """Pith for the green-line start.

    2X: stem is a full disk → DT peak, optionally refined by the inner ring.
    4X: the crop often clips the disk; DT peak of the visible mass is still the
    most stable CV start. (Hough on vessel pores is worse.)
    """
    dt_pith, stem_r, dist = _dt_peak(stem_mask)
    if mag <= 2:
        ring = _pith_from_inner_circle(img, stem_mask, dist)
        if ring is not None and math.hypot(ring[0] - dt_pith[0], ring[1] - dt_pith[1]) < 0.10 * stem_r:
            return (0.45 * dt_pith[0] + 0.55 * ring[0], 0.45 * dt_pith[1] + 0.55 * ring[1])
        return dt_pith
    return dt_pith


def _ray_boundary(
    stem_mask: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), -math.sin(rad)
    h, w = stem_mask.shape
    last = (float(cx), float(cy))
    for t in range(1, int(max(h, w))):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if x < 0 or y < 0 or x >= w or y >= h or stem_mask[y, x] == 0:
            return last
        last = (float(x), float(y))
    return last


def estimate_green_angle(img: np.ndarray, stem_mask: np.ndarray, pith: tuple[float, float]) -> float:
    """Pick radial angle with the clearest layer boundaries."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy = pith
    best_angle = -25.0
    best_score = -1.0

    for angle in np.arange(-80, 81, 5):
        rad = math.radians(float(angle))
        dx, dy = math.cos(rad), -math.sin(rad)
        h, w = gray.shape
        values = []
        for t in range(10, int(min(h, w) * 0.45)):
            x = int(round(cx + t * dx))
            y = int(round(cy + t * dy))
            if 0 <= x < w and 0 <= y < h and stem_mask[y, x] > 0:
                values.append(float(gray[y, x]))
            else:
                break
        if len(values) < 30:
            continue
        arr = np.array(values, dtype=np.float32)
        smooth = cv2.GaussianBlur(arr.reshape(1, -1), (1, 15), 0).flatten()
        grad = np.abs(np.diff(smooth))
        score = float(np.percentile(grad, 90))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def point_on_ray(origin: tuple[float, float], angle_deg: float, dist: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    ox, oy = origin
    return (ox + dist * math.cos(rad), oy - dist * math.sin(rad))


def clamp_point_to_stem(
    stem_mask: np.ndarray,
    origin: tuple[float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    ox, oy = origin
    px, py = point
    angle = math.degrees(math.atan2(-(py - oy), px - ox))
    max_dist = math.hypot(px - ox, py - oy)
    boundary = _ray_boundary(stem_mask, ox, oy, angle)
    bd = math.hypot(boundary[0] - ox, boundary[1] - oy)
    scale = min(1.0, (bd - 2.0) / max(max_dist, 1e-6))
    return (ox + (px - ox) * scale, oy + (py - oy) * scale)


def detect_inner_ring_radius(
    img: np.ndarray,
    stem_mask: np.ndarray,
    pith: tuple[float, float],
) -> float:
    """Radius of the ring nearest to pith (first tissue ring around 髓心)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cx, cy = pith
    dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    iy, ix = int(round(cy)), int(round(cx))
    iy = int(np.clip(iy, 0, dist.shape[0] - 1))
    ix = int(np.clip(ix, 0, dist.shape[1] - 1))
    stem_r = float(dist[iy, ix])
    if stem_r < 30:
        stem_r = float(dist.max())

    radii: list[float] = []
    for angle in range(0, 360, 10):
        rad = math.radians(float(angle))
        dx, dy = math.cos(rad), -math.sin(rad)
        h, w = gray.shape
        vals: list[float] = []
        max_t = int(max(stem_r * 0.55, 30))
        for t in range(1, max_t):
            x = int(round(cx + t * dx))
            y = int(round(cy + t * dy))
            if x < 0 or y < 0 or x >= w or y >= h or stem_mask[y, x] == 0:
                break
            vals.append(float(gray[y, x]))
        if len(vals) < 20:
            continue
        arr = np.array(vals, dtype=np.float32)
        smooth = cv2.GaussianBlur(arr.reshape(1, -1), (1, 15), 0).flatten()
        pith_level = float(np.median(smooth[: max(5, len(smooth) // 12)]))
        thr = pith_level + max(8.0, 0.12 * (float(smooth.max()) - pith_level))
        hit = None
        run = 0
        for i in range(5, len(smooth)):
            if smooth[i] >= thr:
                run += 1
                if run >= 3:
                    hit = float(i - 2)
                    break
            else:
                run = 0
        if hit is None:
            grad = np.abs(np.diff(smooth))
            search = grad[: max(10, int(len(grad) * 0.4))]
            if len(search):
                hit = float(int(np.argmax(search)) + 1)
        if hit is not None and 0.05 * stem_r <= hit <= 0.35 * stem_r:
            radii.append(hit)

    if radii:
        r = float(np.median(radii))
    else:
        r = stem_r * 0.14
    return float(np.clip(r, stem_r * 0.07, stem_r * 0.30))


def clamp_geometry(geom: LineGeometry, stem_mask: np.ndarray) -> LineGeometry:
    """Keep green at pith→bark; keep layer line from ring to exact bark edge."""
    cx, cy = geom.center
    green_angle = geom.green_angle_deg
    seg_angle = geom.seg_angle_deg

    g_end = _ray_boundary(stem_mask, cx, cy, green_angle)
    bark_end_pt = _ray_boundary(stem_mask, cx, cy, seg_angle)
    bark_r = math.hypot(bark_end_pt[0] - cx, bark_end_pt[1] - cy)

    def clamp_along_seg(pt: tuple[float, float], min_r: float = 0.0) -> tuple[float, float]:
        px, py = pt
        r = math.hypot(px - cx, py - cy)
        r = float(np.clip(r, min_r, max(bark_r - 1.0, min_r)))
        return point_on_ray((cx, cy), seg_angle, r)

    p0_raw = (geom.seg_x0, geom.seg_y0 or geom.seg_y)
    r0 = math.hypot(p0_raw[0] - cx, p0_raw[1] - cy)
    r0 = float(np.clip(r0, max(20.0, bark_r * 0.07), bark_r * 0.30))
    p0 = point_on_ray((cx, cy), seg_angle, r0)

    p1 = clamp_along_seg((geom.xylem_end, geom.xylem_end_y or geom.seg_y), min_r=r0 + 5.0)
    p2 = clamp_along_seg((geom.phloem_end, geom.phloem_end_y or geom.seg_y), min_r=r0 + 5.0)
    # Force outermost endpoint exactly on bark edge (not beyond)
    p3 = bark_end_pt

    # Ensure order along ray: r0 < r1 < r2 < bark
    r1 = math.hypot(p1[0] - cx, p1[1] - cy)
    r2 = math.hypot(p2[0] - cx, p2[1] - cy)
    if r1 <= r0 + 2:
        r1 = r0 + max(20.0, (bark_r - r0) * 0.55)
        p1 = point_on_ray((cx, cy), seg_angle, r1)
    if r2 <= r1 + 2:
        r2 = r1 + max(10.0, (bark_r - r1) * 0.45)
        p2 = point_on_ray((cx, cy), seg_angle, min(r2, bark_r - 4.0))
    if math.hypot(p2[0] - cx, p2[1] - cy) >= bark_r - 2:
        p2 = point_on_ray((cx, cy), seg_angle, bark_r - 6.0)

    return LineGeometry(
        center=(cx, cy),
        green_start=(cx, cy),
        green_end=g_end,
        seg_y=float(p0[1]),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(p3[0]),
        green_angle_deg=math.degrees(math.atan2(-(g_end[1] - cy), g_end[0] - cx)),
        seg_angle_deg=seg_angle,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(p3[1]),
    )


def choose_green_endpoint(
    stem_mask: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pick pith (inner) vs bark (outer) endpoints using distance transform."""
    dist = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    d0 = dist[int(round(p0[1])), int(round(p0[0]))]
    d1 = dist[int(round(p1[1])), int(round(p1[0]))]
    if d0 >= d1:
        return p0, p1
    return p1, p0
