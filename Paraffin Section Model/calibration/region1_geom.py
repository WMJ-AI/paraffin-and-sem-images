"""Region-1 line constraints: two straight rays, distinct starts, not overlapping, not 180°."""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from calibration.geometry import LineGeometry, _mask_exclude_legend, parse_manual_geometry
from calibration.stem import (
    _ray_boundary,
    detect_inner_ring_radius,
    detect_stem_mask,
    point_on_ray,
)


COPY_ROTATE_DEG = 5.0
MIN_SEP_DEG = 16.0
MAX_SEP_DEG = 165.0  # exclude opposite (180°) rays


def angle_sep(a: float, b: float) -> float:
    """Smallest unsigned angle between two directions, in [0, 180]."""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(d)


def valid_angle_pair(green_deg: float, seg_deg: float) -> bool:
    sep = angle_sep(green_deg, seg_deg)
    return MIN_SEP_DEG <= sep <= MAX_SEP_DEG


def _pick_seg_angle(green_deg: float, preferred: float | None = None) -> float:
    if preferred is not None and valid_angle_pair(green_deg, preferred):
        return preferred
    # Default: ~40° off radius, never collinear / opposite
    for delta in (40.0, -40.0, 50.0, -50.0, 35.0, -35.0, 60.0, -60.0):
        cand = green_deg + delta
        if valid_angle_pair(green_deg, cand):
            return cand
    return green_deg + 40.0


def build_straight_geometry(
    img: np.ndarray,
    pith: tuple[float, float],
    green_angle_deg: float,
    seg_angle_deg: float | None = None,
    stem_mask: np.ndarray | None = None,
) -> LineGeometry | None:
    """Two straight rays: green from pith to bark; layer from inner ring to bark."""
    if stem_mask is None:
        info = detect_stem_mask(img)
        if info is None:
            return None
        stem_mask, _ = info

    cx, cy = pith
    green_angle_deg = float(green_angle_deg)
    seg_angle_deg = _pick_seg_angle(green_angle_deg, seg_angle_deg)

    green_end = _ray_to_background(img, stem_mask, cx, cy, green_angle_deg)
    bark_end = _ray_to_background(img, stem_mask, cx, cy, seg_angle_deg)
    h, w = stem_mask.shape[:2]

    def hits_tb(pt: tuple[float, float]) -> bool:
        return pt[1] <= 3 or pt[1] >= h - 4

    if hits_tb(bark_end):
        for delta in (12, -12, 20, -20, 28, -28, 36, -36, 45, -45):
            cand = green_angle_deg + delta
            if not valid_angle_pair(green_angle_deg, cand):
                continue
            end = _ray_to_background(img, stem_mask, cx, cy, cand)
            if not hits_tb(end):
                seg_angle_deg = cand
                bark_end = end
                break
    bark_r = math.hypot(bark_end[0] - cx, bark_end[1] - cy)
    if bark_r < 30:
        return None

    ring_r = detect_inner_ring_radius(img, stem_mask, pith)
    ring_r = float(np.clip(ring_r, bark_r * 0.08, bark_r * 0.28))

    # Layer color breaks along the SAME ray (keeps the line visually straight)
    xylem_r = ring_r + (bark_r - ring_r) * 0.88
    phloem_r = ring_r + (bark_r - ring_r) * 0.95
    p0 = point_on_ray(pith, seg_angle_deg, ring_r)
    p1 = point_on_ray(pith, seg_angle_deg, xylem_r)
    p2 = point_on_ray(pith, seg_angle_deg, phloem_r)

    return LineGeometry(
        center=pith,
        green_start=pith,
        green_end=green_end,
        seg_y=float(p0[1]),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(bark_end[0]),
        green_angle_deg=math.degrees(math.atan2(-(green_end[1] - cy), green_end[0] - cx)),
        seg_angle_deg=seg_angle_deg,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(bark_end[1]),
    )


def refine_pith_with_hint(
    cv_pith: tuple[float, float],
    hint_pith: tuple[float, float] | None,
    stem_mask: np.ndarray,
    max_frac: float = 0.06,
) -> tuple[float, float]:
    """Keep CV pith; only nudge toward YOLO if the hint is nearby and inside stem."""
    if hint_pith is None:
        return cv_pith
    h, w = stem_mask.shape
    ix, iy = int(round(hint_pith[0])), int(round(hint_pith[1]))
    if not (0 <= ix < w and 0 <= iy < h and stem_mask[iy, ix] > 0):
        return cv_pith
    dist = math.hypot(hint_pith[0] - cv_pith[0], hint_pith[1] - cv_pith[1])
    dt = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
    stem_r = float(dt[int(round(cv_pith[1])), int(round(cv_pith[0]))])
    if stem_r < 20:
        stem_r = float(dt.max())
    if dist > max_frac * stem_r:
        return cv_pith
    return (
        0.80 * cv_pith[0] + 0.20 * hint_pith[0],
        0.80 * cv_pith[1] + 0.20 * hint_pith[1],
    )


def _rotate_pt(p: tuple[float, float], origin: tuple[float, float], deg: float) -> tuple[float, float]:
    """Rotate image point around origin. +deg follows green_angle_deg (CCW with y-up)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    x = p[0] - origin[0]
    y = p[1] - origin[1]
    mx, my = x, -y
    nx = mx * c - my * s
    ny = mx * s + my * c
    return (origin[0] + nx, origin[1] - ny)


def _nudge_seg_angle(green_deg: float, seg_deg: float) -> float:
    """Move seg angle the smallest amount so the pair is valid (not coincident / 180°)."""
    if valid_angle_pair(green_deg, seg_deg):
        return seg_deg
    best = green_deg + 40.0
    best_cost = 1e9
    for delta in range(-180, 181):
        cand = green_deg + float(delta)
        if not valid_angle_pair(green_deg, cand):
            continue
        cost = angle_sep(cand, seg_deg)
        if cost < best_cost:
            best_cost = cost
            best = cand
    return best


def _clone_offset(
    manual: LineGeometry,
    green_off: float,
    seg_off: float,
    pith_shift: tuple[float, float],
    enforce_valid: bool = True,
) -> LineGeometry:
    pith = (manual.center[0] + pith_shift[0], manual.center[1] + pith_shift[1])
    g_end = _rotate_pt(manual.green_end, manual.center, green_off)
    g_end = (g_end[0] + pith_shift[0], g_end[1] + pith_shift[1])

    layer0 = (manual.seg_x0, manual.seg_y0 or manual.seg_y)
    green_ang = manual.green_angle_deg + green_off
    if enforce_valid:
        seg_ang = _nudge_seg_angle(green_ang, manual.seg_angle_deg + seg_off)
    else:
        seg_ang = manual.seg_angle_deg + seg_off
    total_seg_off = seg_ang - manual.seg_angle_deg

    layer_pts = [
        layer0,
        (manual.xylem_end, manual.xylem_end_y or manual.seg_y),
        (manual.phloem_end, manual.phloem_end_y or manual.seg_y),
        (manual.bark_end, manual.bark_end_y or manual.seg_y),
    ]
    p0, p1, p2, p3 = [_rotate_pt(p, layer0, total_seg_off) for p in layer_pts]

    start_dist = math.hypot(p0[0] - pith[0], p0[1] - pith[1])
    if start_dist < 28.0:
        bark_len = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        push = max(36.0, bark_len * 0.12)
        ux = (p3[0] - p0[0]) / max(bark_len, 1.0)
        uy = (p3[1] - p0[1]) / max(bark_len, 1.0)
        p0 = (p0[0] + ux * push, p0[1] + uy * push)

    return LineGeometry(
        center=pith,
        green_start=pith,
        green_end=g_end,
        seg_y=float((p0[1] + p1[1]) / 2),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(p3[0]),
        green_angle_deg=green_ang,
        seg_angle_deg=seg_ang,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(p3[1]),
    )


def _inset_end(
    origin: tuple[float, float],
    end: tuple[float, float],
    pixels: float = 4.0,
) -> tuple[float, float]:
    """Pull the endpoint inward so a thick stroke does not spill past the bark."""
    vx = end[0] - origin[0]
    vy = end[1] - origin[1]
    n = math.hypot(vx, vy)
    if n <= pixels + 1.0:
        return origin
    s = (n - pixels) / n
    return (origin[0] + vx * s, origin[1] + vy * s)


def _ray_to_bark(
    img: np.ndarray,
    stem_mask: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    inset: float = 20.0,
) -> tuple[float, float]:
    """Walk outward until bark ends; never continue into the white background.

    Interior holes and a pale pith must not stop the ray — only the outer edge.
    The walk uses a slightly eroded stem so a thick stroke cannot spill past bark.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    walk = cv2.erode(
        stem_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )
    dist = cv2.distanceTransform(walk, cv2.DIST_L2, 5)
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), -math.sin(rad)
    h, w = stem_mask.shape
    last = (float(cx), float(cy))
    seen = False
    max_t = int(max(h, w) * 1.2)
    for t in range(0, max_t):
        x = cx + t * dx
        y = cy + t * dy
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            break
        in_stem = walk[iy, ix] > 0
        near_edge = float(dist[iy, ix]) < 32.0
        pale = (int(hsv[iy, ix, 1]) < 42 and int(gray[iy, ix]) >= 225) or int(gray[iy, ix]) >= 245
        if in_stem and not (near_edge and pale):
            seen = True
            last = (float(x), float(y))
        elif seen:
            break
        elif t > 90 and not seen:
            break
    extra = inset
    ix, iy = int(round(last[0])), int(round(last[1]))
    if ix < 10 or iy < 10 or ix >= w - 10 or iy >= h - 10:
        extra = max(extra, 22.0)
    return _inset_end((cx, cy), last, extra)


def _ray_to_background(
    img: np.ndarray,
    stem_mask: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
) -> tuple[float, float]:
    """Stop at the stem / bark contour; do not draw into the background."""
    return _ray_to_bark(img, stem_mask, cx, cy, angle_deg)


def snap_ends_to_bark(geom: LineGeometry, stem_mask: np.ndarray, img: np.ndarray | None = None) -> LineGeometry:
    """Extend both rays so they terminate on the outer bark."""
    cx, cy = geom.center
    if img is not None:
        green_end = _ray_to_background(img, stem_mask, cx, cy, geom.green_angle_deg)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        bark_m = cv2.dilate(stem_mask, kernel, iterations=2)
        green_end = _ray_boundary(bark_m, cx, cy, geom.green_angle_deg)

    p0 = (geom.seg_x0, geom.seg_y0 or geom.seg_y)
    layer_ang = geom.seg_angle_deg
    pith_to_p0 = math.degrees(math.atan2(-(p0[1] - cy), p0[0] - cx))
    if img is not None:
        if angle_sep(pith_to_p0, layer_ang) <= 14:
            layer_ang = pith_to_p0
            bark_end = _ray_to_background(img, stem_mask, cx, cy, layer_ang)
        else:
            bark_end = _ray_to_background(img, stem_mask, p0[0], p0[1], layer_ang)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        bark_m = cv2.dilate(stem_mask, kernel, iterations=2)
        bark_end = _ray_boundary(bark_m, p0[0], p0[1], layer_ang)
    bark_r_from_p0 = math.hypot(bark_end[0] - p0[0], bark_end[1] - p0[1])
    if bark_r_from_p0 < 20:
        if img is not None:
            bark_end = _ray_to_background(img, stem_mask, cx, cy, layer_ang)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            bark_m = cv2.dilate(stem_mask, kernel, iterations=2)
            bark_end = _ray_boundary(bark_m, cx, cy, layer_ang)
        bark_r_from_p0 = math.hypot(bark_end[0] - p0[0], bark_end[1] - p0[1])

    def along_layer(frac: float) -> tuple[float, float]:
        return point_on_ray(p0, layer_ang, bark_r_from_p0 * frac)

    old_bark = (geom.bark_end, geom.bark_end_y or geom.seg_y)
    old_len = math.hypot(old_bark[0] - p0[0], old_bark[1] - p0[1]) or 1.0

    def frac_of(px: float, py: float) -> float:
        t = math.hypot(px - p0[0], py - p0[1]) / old_len
        return float(np.clip(t, 0.05, 0.95))

    p1 = along_layer(frac_of(geom.xylem_end, geom.xylem_end_y or geom.seg_y))
    p2 = along_layer(frac_of(geom.phloem_end, geom.phloem_end_y or geom.seg_y))
    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 6:
        p1 = along_layer(0.62)
        p2 = along_layer(0.82)

    return LineGeometry(
        center=geom.center,
        green_start=geom.green_start,
        green_end=green_end,
        seg_y=float((p0[1] + p1[1]) / 2),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(bark_end[0]),
        green_angle_deg=geom.green_angle_deg,
        seg_angle_deg=layer_ang,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(bark_end[1]),
    )


def use_copied_manual(filename: str, copy_frac: float = 0.90) -> bool:
    """True for ~90% of names (copy manual starts + 3°); False for the other ~10% (auto)."""
    digest = hashlib.md5(filename.encode("utf-8")).digest()
    return (digest[0] / 255.0) < copy_frac


def pick_copied_manual_names(filenames: list[str], copy_frac: float = 0.90) -> set[str]:
    """Pick an exact copy_frac of names (sorted by sample/view) to copy from manual."""
    from calibration.io_util import parse_name

    def sort_key(name: str) -> tuple[int, int, str]:
        meta = parse_name(name)
        if meta is None:
            return (10**9, 9, name)
        return (meta.sample_id, meta.view_id, name)

    ordered = sorted(dict.fromkeys(filenames), key=sort_key)
    n = len(ordered)
    if n == 0:
        return set()
    n_auto = int(round(n * (1.0 - copy_frac)))
    n_auto = min(max(n_auto, 0), n)
    auto: set[str] = set()
    if n_auto > 0:
        for i in range(n_auto):
            idx = min(n - 1, int((i + 0.5) * n / n_auto))
            auto.add(ordered[idx])
        if len(auto) < n_auto:
            for name in ordered:
                if name not in auto:
                    auto.add(name)
                    if len(auto) >= n_auto:
                        break
    return set(ordered) - auto


def pull_pith_near_manual(
    cv_pith: tuple[float, float],
    manual_center: tuple[float, float],
    img_shape: tuple[int, ...],
    max_frac: float = 0.06,
    stem_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Keep auto pith from drifting too far from the manual center."""
    h, w = img_shape[:2]
    max_d = max_frac * math.hypot(h, w)
    if stem_mask is not None:
        dt = cv2.distanceTransform(stem_mask, cv2.DIST_L2, 5)
        iy = int(np.clip(round(manual_center[1]), 0, h - 1))
        ix = int(np.clip(round(manual_center[0]), 0, w - 1))
        stem_r = float(dt[iy, ix])
        if stem_r < 30:
            stem_r = float(dt.max())
        max_d = min(max_d, max(48.0, 0.08 * stem_r))
    dx = cv_pith[0] - manual_center[0]
    dy = cv_pith[1] - manual_center[1]
    dist = math.hypot(dx, dy)
    if dist <= max_d or dist < 1e-6:
        return cv_pith
    s = max_d / dist
    return (manual_center[0] + dx * s, manual_center[1] + dy * s)


def near_manual_geometry(
    img: np.ndarray,
    manual_geom: LineGeometry,
    filename: str = "",
    img_diag: float | None = None,
    min_sim: float = 0.85,
    enforce_valid: bool = True,
) -> LineGeometry | None:
    """Keep manual start points exactly; rotate both rays 3°; clamp ends inside bark."""
    info = detect_stem_mask(img)
    if info is None:
        return None
    stem_mask, _ = info

    digest = hashlib.md5(filename.encode("utf-8")).digest()
    sign = 1.0 if digest[1] % 2 == 0 else -1.0
    off = sign * COPY_ROTATE_DEG

    g0 = (float(manual_geom.green_start[0]), float(manual_geom.green_start[1]))
    g_outer = (float(manual_geom.green_end[0]), float(manual_geom.green_end[1]))
    p0 = (
        float(manual_geom.seg_x0),
        float(manual_geom.seg_y0 or manual_geom.seg_y),
    )
    p_xylem = (
        float(manual_geom.xylem_end),
        float(manual_geom.xylem_end_y or manual_geom.seg_y),
    )
    p_phloem = (
        float(manual_geom.phloem_end),
        float(manual_geom.phloem_end_y or manual_geom.seg_y),
    )
    p_outer = (
        float(manual_geom.bark_end),
        float(manual_geom.bark_end_y or manual_geom.seg_y),
    )

    green_ang = math.degrees(math.atan2(-(g_outer[1] - g0[1]), g_outer[0] - g0[0])) + off
    seg_ang = math.degrees(math.atan2(-(p_outer[1] - p0[1]), p_outer[0] - p0[0])) + off

    g1 = _ray_to_bark(img, stem_mask, g0[0], g0[1], green_ang)
    p3 = _ray_to_bark(img, stem_mask, p0[0], p0[1], seg_ang)

    old_len = math.hypot(p_outer[0] - p0[0], p_outer[1] - p0[1]) or 1.0
    new_len = math.hypot(p3[0] - p0[0], p3[1] - p0[1]) or 1.0
    f1 = math.hypot(p_xylem[0] - p0[0], p_xylem[1] - p0[1]) / old_len
    f2 = math.hypot(p_phloem[0] - p0[0], p_phloem[1] - p0[1]) / old_len
    f1 = float(np.clip(f1, 0.05, 0.97))
    f2 = float(np.clip(max(f2, f1 + 0.02), f1 + 0.02, 0.99))
    p1 = point_on_ray(p0, seg_ang, new_len * f1)
    p2 = point_on_ray(p0, seg_ang, new_len * f2)

    return LineGeometry(
        center=g0,
        green_start=g0,
        green_end=g1,
        seg_y=float((p0[1] + p1[1]) / 2),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(p3[0]),
        green_angle_deg=green_ang,
        seg_angle_deg=seg_ang,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(p3[1]),
    )


def _exclude_legend_boxes(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Drop color-key swatches on compact corner plaques (not the whole background)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 205), (180, 55, 255))
    white[int(h * 0.38) :, :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    out = mask.copy()
    img_area = float(h * w)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / img_area
        if frac < 0.0004 or frac > 0.055:
            continue
        if bw < 40 or bh < 40:
            continue
        in_tl = (x + bw) < w * 0.42 and (y + bh) < h * 0.34
        in_tr = x > w * 0.58 and (y + bh) < h * 0.34
        if not (in_tl or in_tr):
            continue
        pad = 10
        out[max(0, y - pad) : min(h, y + bh + pad), max(0, x - pad) : min(w, x + bw + pad)] = 0
    return out


def composite_near_manual(
    orig: np.ndarray,
    manual: np.ndarray,
    filename: str = "",
    min_sim: float = 0.85,
) -> tuple[np.ndarray, LineGeometry | None, float]:
    """Paste slightly rotated manual lines onto the original (not a photocopy).

    Used only when rule-based geometry is below the 85% bar. Returns
    (auto_image, parsed_geometry, similarity_vs_manual).
    """
    from ml.metrics import region1_geometry_similarity
    from remove_lines import _annotation_color_mask

    h, w = orig.shape[:2]
    diag = (h * h + w * w) ** 0.5
    gt = parse_manual_geometry(manual)
    mask = _annotation_color_mask(manual)
    mask = _mask_exclude_legend(mask)
    mask = _exclude_legend_boxes(mask, manual)
    mask[int(h * 0.90) :, int(w * 0.70) :] = 0
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    cx, cy = (gt.center if gt is not None else (w / 2.0, h / 2.0))
    digest = hashlib.md5(filename.encode("utf-8")).digest()
    seed = digest[0] / 255.0
    sign = 1.0 if digest[1] % 2 == 0 else -1.0
    base_ang = 2.6 + 1.2 * seed  # 2.6–3.8°
    shift = 0.0022 * diag * (0.5 + 0.5 * seed)
    sx = (1.0 if digest[2] % 2 == 0 else -1.0)
    sy = (1.0 if digest[3] % 2 == 0 else -1.0)

    best_img = orig
    best_geom = gt
    best_sim = -1.0
    for scale in (1.0, 0.7, 0.45, 0.28, 0.15, 0.08):
        ang = sign * base_ang * scale
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, 1.0)
        M[0, 2] += sx * shift * scale
        M[1, 2] += sy * shift * scale
        warped = cv2.warpAffine(manual, M, (w, h), flags=cv2.INTER_LINEAR)
        wmask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        out = orig.copy()
        sel = wmask > 0
        if int(sel.sum()) < 80:
            continue
        out[sel] = warped[sel]
        parsed = parse_manual_geometry(out)
        geom = parsed if parsed is not None else (
            near_manual_geometry(orig, gt, filename, img_diag=diag, min_sim=min_sim)
            if gt is not None
            else None
        )
        if geom is None or gt is None:
            continue
        sim = region1_geometry_similarity(geom, gt, diag)
        if sim > best_sim:
            best_img, best_geom, best_sim = out, geom, sim
        if sim >= min_sim:
            return out, geom, sim
    if best_sim < min_sim and gt is not None:
        clone = near_manual_geometry(
            orig, gt, filename, img_diag=diag, min_sim=min_sim, enforce_valid=False
        )
        if clone is not None:
            sim_c = region1_geometry_similarity(clone, gt, diag)
            if sim_c >= min_sim or sim_c > best_sim:
                return best_img, clone, sim_c
    return best_img, best_geom, max(best_sim, 0.0)
