"""Parse and draw calibration line geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from remove_lines import _annotation_color_mask


@dataclass
class LineGeometry:
    center: tuple[float, float]
    green_start: tuple[float, float]
    green_end: tuple[float, float]
    seg_y: float
    seg_x0: float
    xylem_end: float
    phloem_end: float
    bark_end: float
    green_angle_deg: float
    seg_angle_deg: float = 0.0
    seg_y0: float = 0.0
    xylem_end_y: float = 0.0
    phloem_end_y: float = 0.0
    bark_end_y: float = 0.0


def _mask_exclude_legend(mask: np.ndarray) -> np.ndarray:
    """Punch only legend corners; do not wipe the right side (rays often go there)."""
    h, w = mask.shape[:2]
    out = mask.copy()
    ch, cw = int(h * 0.12), int(w * 0.18)
    out[:ch, :cw] = 0
    out[:ch, w - cw :] = 0
    return out


def _punch_legend_plaques(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Zero color-key swatches only on corner legend plaques, not stem tissue."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    ch, cw = int(h * 0.22), int(w * 0.30)
    corner = np.zeros_like(white)
    corner[:ch, :cw] = white[:ch, :cw]
    corner[:ch, w - cw :] = white[:ch, w - cw :]
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(corner, connectivity=8)
    out = mask.copy()
    img_area = float(h * w)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / img_area
        if frac < 0.0003 or frac > 0.08:
            continue
        if bw < 40 or bh < 40:
            continue
        pad = 14
        out[max(0, y - pad) : min(h, y + bh + pad), max(0, x - pad) : min(w, x + bw + pad)] = 0
    return out


def _line_like_mask(mask: np.ndarray, min_area: int = 400, min_aspect: float = 3.5) -> np.ndarray:
    """Keep the longest thin annotation stroke; drop legend squares and stained tissue."""
    if mask is None or int(cv2.countNonZero(mask)) < min_area:
        return np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    work = cv2.dilate(mask, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    best_i = 0
    best_score = 0.0
    for i in range(1, n):
        _x, _y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1.0)
        if aspect < min_aspect:
            continue
        score = float(area) * aspect
        if score > best_score:
            best_score = score
            best_i = i
    if best_i == 0:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == best_i] = 255
    return cv2.bitwise_and(out, mask)


def _color_layers(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)
    # Tight ranges matching 人工标定 ink, not stained tissue
    green = cv2.inRange(hsv, (40, 140, 70), (90, 255, 255))
    orange = cv2.inRange(hsv, (5, 100, 130), (28, 255, 255))
    blue = cv2.inRange(hsv, (95, 120, 90), (135, 255, 255))
    red = cv2.inRange(hsv, (0, 110, 90), (12, 255, 255)) | cv2.inRange(
        hsv, (170, 110, 90), (180, 255, 255)
    )
    red |= ((r > 150) & (g < 110) & (b < 110) & (r > g + 50) & (r > b + 50)).astype(np.uint8) * 255
    green = _punch_legend_plaques(_mask_exclude_legend(green), img)
    orange = _punch_legend_plaques(_mask_exclude_legend(orange), img)
    blue = _punch_legend_plaques(_mask_exclude_legend(blue), img)
    red = _punch_legend_plaques(_mask_exclude_legend(red), img)
    return green, orange, blue, red


def _diff_color_layers(
    manual: np.ndarray,
    orig: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ink = pixels that changed vs the original and are saturated (not tissue)."""
    d = cv2.absdiff(manual, orig)
    mag = d[:, :, 0].astype(np.int32) + d[:, :, 1].astype(np.int32) + d[:, :, 2].astype(np.int32)
    hsv = cv2.cvtColor(manual, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ink = ((mag > 45) & (s > 65)).astype(np.uint8) * 255
    green = ((ink > 0) & (h >= 40) & (h <= 90) & (s > 80)).astype(np.uint8) * 255
    orange = ((ink > 0) & (h >= 5) & (h <= 28) & (s > 90) & (v > 90)).astype(np.uint8) * 255
    blue = ((ink > 0) & (h >= 95) & (h <= 135) & (s > 80)).astype(np.uint8) * 255
    b, g, r = cv2.split(manual)
    red = (
        ((ink > 0) & (((h <= 12) | (h >= 170)) & (s > 90) & (v > 70)))
        | ((ink > 0) & (r > 150) & (g < 110) & (b < 110) & (r > g + 40) & (r > b + 40))
    ).astype(np.uint8) * 255
    green = _punch_legend_plaques(_mask_exclude_legend(green), manual)
    orange = _punch_legend_plaques(_mask_exclude_legend(orange), manual)
    blue = _punch_legend_plaques(_mask_exclude_legend(blue), manual)
    red = _punch_legend_plaques(_mask_exclude_legend(red), manual)
    return green, orange, blue, red


def _fit_line_endpoints(mask: np.ndarray, max_points: int = 5000) -> tuple[tuple[float, float], tuple[float, float]] | None:
    h, w = mask.shape[:2]
    work = _mask_exclude_legend(mask.copy())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    work = cv2.dilate(work, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    best = None
    best_score = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        score = area * max(aspect, 1.0)
        if area > 500 and score > best_score:
            best_score = score
            best = labels == i
    if best is None:
        ys, xs = np.where(work > 0)
    else:
        ys, xs = np.where(best)
    if len(xs) < 50:
        return None
    if len(xs) > max_points:
        step = len(xs) / float(max_points)
        idx = (np.arange(max_points) * step).astype(np.int32)
        xs, ys = xs[idx], ys[idx]
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vx.item()), float(vy.item())])
    direction /= np.linalg.norm(direction) + 1e-6
    origin = np.array([float(x0.item()), float(y0.item())])
    proj = (pts - origin) @ direction
    p_start = origin + direction * proj.min()
    p_end = origin + direction * proj.max()
    return (float(p_start[0]), float(p_start[1])), (float(p_end[0]), float(p_end[1]))


def _stroke_inner_outer(
    mask: np.ndarray,
    dist: np.ndarray | None,
    pith: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Inner = stroke pixel nearest the pith; outer = farthest pixel from inner."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    if pith is not None:
        d2 = (xs.astype(np.float64) - pith[0]) ** 2 + (ys.astype(np.float64) - pith[1]) ** 2
        i0 = int(np.argmin(d2))
    elif dist is not None:
        i0 = int(np.argmax(dist[ys, xs]))
    else:
        return None
    start = (float(xs[i0]), float(ys[i0]))
    d2e = (xs.astype(np.float64) - start[0]) ** 2 + (ys.astype(np.float64) - start[1]) ** 2
    i1 = int(np.argmax(d2e))
    end = (float(xs[i1]), float(ys[i1]))
    if math.hypot(end[0] - start[0], end[1] - start[1]) < 20:
        return None
    if pith is not None:
        if math.hypot(end[0] - pith[0], end[1] - pith[1]) < math.hypot(
            start[0] - pith[0], start[1] - pith[1]
        ):
            start, end = end, start
    return start, end


def _ray_unit(angle_deg: float) -> tuple[float, float]:
    rad = math.radians(float(angle_deg))
    return math.cos(rad), -math.sin(rad)


def _mask_ray_span(
    mask: np.ndarray,
    origin: tuple[float, float],
    angle_deg: float,
    max_perp: float = 36.0,
) -> tuple[float, float] | None:
    """Along-ray [t_inner, t_outer] of a color stroke near the given ray."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return None
    ux, uy = _ray_unit(angle_deg)
    ox, oy = origin
    dx = xs.astype(np.float64) - ox
    dy = ys.astype(np.float64) - oy
    along = dx * ux + dy * uy
    perp = dx * (-uy) + dy * ux
    keep = (along > 2.0) & (np.abs(perp) < max_perp)
    t = along[keep]
    if t.size < 8:
        t = along[along > 2.0]
    if t.size < 8:
        return None
    return float(np.percentile(t, 1)), float(np.percentile(t, 99))


def _pt_on_ray(origin: tuple[float, float], angle_deg: float, dist: float) -> tuple[float, float]:
    ux, uy = _ray_unit(angle_deg)
    return (origin[0] + dist * ux, origin[1] + dist * uy)


def _nearest_ink_color(
    bgr: np.ndarray,
    colors: dict[str, tuple[int, int, int]],
    max_d: float = 58.0,
) -> str | None:
    best_name = None
    best_d = max_d
    b = bgr.astype(np.float32)
    for name, col in colors.items():
        d = float(np.linalg.norm(b - np.array(col, dtype=np.float32)))
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def _layer_runs_along_segment(
    img: np.ndarray,
    inner: tuple[float, float],
    outer: tuple[float, float],
    extra: float = 1.35,
    orange: np.ndarray | None = None,
    blue: np.ndarray | None = None,
    red: np.ndarray | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Walk inner→outer using ink masks. Blue wins over orange (masks overlap)."""
    vx, vy = outer[0] - inner[0], outer[1] - inner[1]
    base = math.hypot(vx, vy)
    if base < 20:
        return None
    ux, uy = vx / base, vy / base
    length = base * extra
    n = max(80, int(round(length)))
    h, w = img.shape[:2]
    labels: list[str | None] = []
    ts: list[float] = []
    rad = 3
    none_streak = 0
    for i in range(n + 1):
        t = length * i / n
        x = int(round(inner[0] + t * ux))
        y = int(round(inner[1] + t * uy))
        ts.append(t)
        if not (rad <= x < w - rad and rad <= y < h - rad):
            labels.append(None)
            none_streak += 1
            if t > base and none_streak > 12:
                break
            continue
        y0, y1 = y - rad, y + rad + 1
        x0, x1 = x - rad, x + rad + 1
        lab = None
        on_orange = orange is not None and int(orange[y0:y1, x0:x1].max()) > 0
        on_blue = blue is not None and int(blue[y0:y1, x0:x1].max()) > 0
        on_red = red is not None and int(red[y0:y1, x0:x1].max()) > 0
        if t <= base + 10:
            # Inside the orange stroke: do not let a noisy blue mask steal the xylem.
            if on_blue and not on_orange:
                lab = "blue"
            elif on_orange:
                lab = "orange"
            elif on_blue:
                lab = "blue"
            elif on_red:
                lab = "red"
        else:
            if on_red:
                lab = "red"
            elif on_blue:
                lab = "blue"
            elif on_orange:
                lab = "orange"
        if lab is None and t <= base + 10:
            lab = _nearest_ink_color(
                img[y, x], {"orange": COLOR_ORANGE, "blue": COLOR_BLUE, "red": COLOR_RED}, max_d=80.0
            )
        labels.append(lab)
        if lab is None:
            none_streak += 1
            if t > base and none_streak > 12:
                break
        else:
            none_streak = 0

    def _first_last(name: str, t_min: float = -1.0) -> tuple[float, float] | None:
        hits = [ts[i] for i, lab in enumerate(labels) if lab == name and ts[i] >= t_min]
        if len(hits) < 3:
            return None
        return float(hits[0]), float(hits[-1])

    orange_run = _first_last("orange")
    if orange_run is None:
        return None
    t0, t_or_end = orange_run
    blue_run = _first_last("blue", t_or_end - 15.0)
    red_run = _first_last("red", t_or_end - 15.0)
    if blue_run is not None:
        t1 = min(t_or_end, blue_run[0])
        t2 = blue_run[1]
    else:
        t1 = t_or_end
        t2 = red_run[0] if red_run is not None else t_or_end
    if red_run is not None and red_run[1] > t2:
        t3 = red_run[1]
    else:
        t3 = max(t2, t_or_end)
    t1 = max(t0 + 4.0, t1)
    t2 = max(t1, t2)
    t3 = max(t2, t3)
    return {
        "p0": (inner[0] + t0 * ux, inner[1] + t0 * uy),
        "p1": (inner[0] + t1 * ux, inner[1] + t1 * uy),
        "p2": (inner[0] + t2 * ux, inner[1] + t2 * uy),
        "p3": (inner[0] + t3 * ux, inner[1] + t3 * uy),
        "t0": t0,
        "t1": t1,
        "t2": t2,
        "t3": t3,
    }


def ink_stroke_measure(geom: "LineGeometry") -> dict[str, tuple[float, float] | float]:
    """Starts and color lengths from a 6-point geometry."""
    g0 = geom.green_start
    g1 = geom.green_end
    p0 = (float(geom.seg_x0), float(geom.seg_y0 or geom.seg_y))
    p1 = (float(geom.xylem_end), float(geom.xylem_end_y or geom.seg_y))
    p2 = (float(geom.phloem_end), float(geom.phloem_end_y or geom.seg_y))
    p3 = (float(geom.bark_end), float(geom.bark_end_y or geom.seg_y))

    def _len(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    return {
        "green_start": g0,
        "orange_start": p0,
        "blue_start": p1,
        "red_start": p2,
        "green_end": g1,
        "orange_end": p1,
        "blue_end": p2,
        "red_end": p3,
        "green_len": _len(g0, g1),
        "orange_len": _len(p0, p1),
        "blue_len": _len(p1, p2),
        "red_len": _len(p2, p3),
    }


def _best_stroke_mask(
    mask: np.ndarray,
    dist: np.ndarray | None,
    pith: tuple[float, float] | None,
    min_area: int = 350,
    min_aspect: float = 3.2,
) -> np.ndarray:
    """Pick the long ink stroke whose inner end sits near the pith."""
    if mask is None or int(cv2.countNonZero(mask)) < min_area:
        return np.zeros_like(mask) if mask is not None else np.zeros((1, 1), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = cv2.dilate(mask, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    h, w = mask.shape[:2]
    stem_r = float(dist.max()) if dist is not None else float(min(h, w) * 0.4)
    best_i = 0
    best_score = -1.0
    fallback_i = 0
    fallback_pin = 1e18
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        ys, xs = np.where((labels == i) & (mask > 0))
        if len(xs) < 20:
            continue
        if dist is not None:
            dts = dist[ys, xs]
            i0 = int(np.argmax(dts))
        elif pith is not None:
            d2p = (xs.astype(np.float64) - pith[0]) ** 2 + (ys.astype(np.float64) - pith[1]) ** 2
            i0 = int(np.argmin(d2p))
        else:
            i0 = 0
        inner = (float(xs[i0]), float(ys[i0]))
        d2 = (xs.astype(np.float64) - inner[0]) ** 2 + (ys.astype(np.float64) - inner[1]) ** 2
        length = float(np.sqrt(d2.max()))
        if length < max(120.0, 0.22 * stem_r):
            continue
        thickness = float(len(xs)) / max(length, 1.0)
        stroke_aspect = length / max(thickness, 1.0)
        if stroke_aspect < 8.0:
            continue
        ink_y1 = int(ys.max())
        if ink_y1 < 0.10 * h:
            continue
        if pith is not None:
            pin = math.hypot(inner[0] - pith[0], inner[1] - pith[1])
        else:
            pin = 0.0
        if pin < fallback_pin:
            fallback_pin = pin
            fallback_i = i
        if pith is not None and pin > 0.72 * max(stem_r, 1.0):
            continue
        touches_border = x <= 2 or y <= 2 or (x + bw) >= w - 3 or (y + bh) >= h - 3
        border_pen = 0.15 if touches_border and min(bw, bh) < 0.25 * min(h, w) else 1.0
        pin_pen = 1.0 / (1.0 + (pin / max(stem_r, 1.0)) ** 2)
        score = stroke_aspect * math.sqrt(float(len(xs))) * (length / max(stem_r, 1.0)) * pin_pen * border_pen
        if score > best_score:
            best_score = score
            best_i = i
    if best_i == 0:
        best_i = fallback_i
    if best_i == 0:
        return _line_like_mask(mask, min_area=min_area, min_aspect=min_aspect)
    out = np.zeros_like(mask)
    out[labels == best_i] = 255
    return cv2.bitwise_and(out, mask)


def _ray_from_pith(
    mask: np.ndarray,
    pith: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Inner/outer of a stroke, keeping only the arm on one side of the pith."""
    ends = _fit_line_endpoints(mask)
    if ends is None:
        return None
    a, b = ends
    da = math.hypot(a[0] - pith[0], a[1] - pith[1])
    db = math.hypot(b[0] - pith[0], b[1] - pith[1])
    same_side = (a[0] - pith[0]) * (b[0] - pith[0]) + (a[1] - pith[1]) * (b[1] - pith[1]) > 0
    if same_side:
        inner, outer = (a, b) if da <= db else (b, a)
    else:
        outer = a if da >= db else b
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            inner = pith
        else:
            vx, vy = outer[0] - pith[0], outer[1] - pith[1]
            dots = (xs.astype(np.float64) - pith[0]) * vx + (ys.astype(np.float64) - pith[1]) * vy
            same = dots > 0
            if not np.any(same):
                inner = pith
            else:
                d2 = (xs[same].astype(np.float64) - pith[0]) ** 2 + (
                    ys[same].astype(np.float64) - pith[1]
                ) ** 2
                i = int(np.argmin(d2))
                inner = (float(xs[same][i]), float(ys[same][i]))
    return inner, outer


def _outermost_stroke_run(
    mask: np.ndarray,
    pith: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Inner/outer sitting on the orange ink, using the bark-reaching run."""
    ends = _fit_line_endpoints(mask)
    if ends is None:
        return None
    a, b = ends
    da = math.hypot(a[0] - pith[0], a[1] - pith[1])
    db = math.hypot(b[0] - pith[0], b[1] - pith[1])
    closer, farther = (a, b) if da <= db else (b, a)
    vx, vy = farther[0] - closer[0], farther[1] - closer[1]
    length = math.hypot(vx, vy)
    if length < 40:
        return None
    ux, uy = vx / length, vy / length
    h, w = mask.shape[:2]
    hits: list[tuple[float, float, float]] = []
    rad = 4
    n = max(40, int(round(length)))
    for i in range(n + 1):
        t = length * i / n
        x = closer[0] + t * ux
        y = closer[1] + t * uy
        xi, yi = int(round(x)), int(round(y))
        if not (rad <= xi < w - rad and rad <= yi < h - rad):
            continue
        if int(mask[yi - rad : yi + rad + 1, xi - rad : xi + rad + 1].max()) > 0:
            hits.append((t, x, y))
    if len(hits) < 10:
        return None
    runs: list[list[tuple[float, float, float]]] = []
    cur = [hits[0]]
    for hit in hits[1:]:
        if hit[0] - cur[-1][0] > 36:
            if len(cur) >= 8:
                runs.append(cur)
            cur = [hit]
        else:
            cur.append(hit)
    if len(cur) >= 8:
        runs.append(cur)
    if not runs:
        return None
    best = max(runs, key=lambda r: r[-1][0] - r[0][0])
    # Snap to real ink pixels — never invent a geometric ring start.
    inner = (float(best[0][1]), float(best[0][2]))
    outer = (float(best[-1][1]), float(best[-1][2]))
    ys, xs = np.where(mask > 0)
    if len(xs) >= 10:
        d2i = (xs.astype(np.float64) - inner[0]) ** 2 + (ys.astype(np.float64) - inner[1]) ** 2
        d2o = (xs.astype(np.float64) - outer[0]) ** 2 + (ys.astype(np.float64) - outer[1]) ** 2
        ii = int(np.argmin(d2i))
        oi = int(np.argmin(d2o))
        inner = (float(xs[ii]), float(ys[ii]))
        outer = (float(xs[oi]), float(ys[oi]))
        # Prefer the end nearer the pith as orange start.
        di = math.hypot(inner[0] - pith[0], inner[1] - pith[1])
        do = math.hypot(outer[0] - pith[0], outer[1] - pith[1])
        if do < di:
            inner, outer = outer, inner
    return inner, outer


def _last_on_ray(
    mask: np.ndarray,
    start: tuple[float, float],
    toward: tuple[float, float],
    gap_max: int = 30,
) -> tuple[float, float]:
    """Last mask hit along start→toward, allowing small gaps in thick ink."""
    vx, vy = toward[0] - start[0], toward[1] - start[1]
    length = math.hypot(vx, vy) or 1.0
    ux, uy = vx / length, vy / length
    h, w = mask.shape[:2]
    work = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    last = (float(toward[0]), float(toward[1]))
    gap = 0
    seen = False
    max_t = int(max(h, w) * 1.15)
    for t in range(0, max_t):
        x = start[0] + t * ux
        y = start[1] + t * uy
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            break
        if int(work[yi, xi]) > 0:
            last = (float(x), float(y))
            gap = 0
            seen = True
        elif seen:
            gap += 1
            if gap > gap_max:
                break
    return last


def parse_manual_geometry(
    manual_img: np.ndarray,
    orig_img: np.ndarray | None = None,
) -> LineGeometry | None:
    """Read manual rays from the annotation image.

    Green inner endpoint is the pith (most interior green pixel). Layer start is
    the inner-ring end of the orange stroke — not a diameter across the stem.
    """
    from calibration.stem import detect_stem_mask

    h, w = manual_img.shape[:2]
    if orig_img is not None:
        green, orange, blue, red = _diff_color_layers(manual_img, orig_img)
        g2, o2, b2, r2 = _color_layers(manual_img)
        green = cv2.bitwise_or(green, g2)
        orange = cv2.bitwise_or(orange, o2)
        blue = cv2.bitwise_or(blue, b2)
        red = cv2.bitwise_or(red, r2)
    else:
        green, orange, blue, red = _color_layers(manual_img)

    ref = orig_img if orig_img is not None else manual_img
    info = detect_stem_mask(ref)
    stem = info[0] if info is not None else None
    dist = None
    pith = None
    if stem is not None:
        stem_bark = cv2.dilate(stem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)), iterations=1)
        green = cv2.bitwise_and(green, stem_bark)
        orange = cv2.bitwise_and(orange, stem_bark)
        blue = cv2.bitwise_and(blue, stem_bark)
        red = cv2.bitwise_and(red, stem_bark)
        dist = cv2.distanceTransform(stem, cv2.DIST_L2, 5)
        yx = np.unravel_index(int(np.argmax(dist)), dist.shape)
        pith = (float(yx[1]), float(yx[0]))

    green_m = _best_stroke_mask(green, dist, pith)
    if int(cv2.countNonZero(green_m)) < 50:
        green_m = green
    orange_m = _best_stroke_mask(orange, dist, pith)
    if int(cv2.countNonZero(orange_m)) < 50:
        orange_m = _best_stroke_mask(orange | red, dist, pith)

    # Green start = ink pixel nearest the pith (full radius, not a truncated fragment).
    if dist is not None:
        green_line = (
            _stroke_inner_outer(green_m, dist, pith)
            or _stroke_inner_outer(green_m, dist, None)
            or _fit_line_endpoints(green_m)
        )
        hint = pith if pith is not None else None
        seg_line = _outermost_stroke_run(orange_m, hint) if hint is not None else None
        if seg_line is None:
            seg_line = (
                _stroke_inner_outer(orange_m, dist, hint)
                or (_ray_from_pith(orange_m, hint) if hint is not None else None)
                or _fit_line_endpoints(orange_m)
            )
    else:
        green_line = _fit_line_endpoints(green_m)
        seg_line = _fit_line_endpoints(orange_m)
    if green_line is None or seg_line is None:
        return None

    g_start, g_end = green_line
    p0, p_outer = seg_line
    ys, xs = np.where(orange_m > 0)
    if len(xs) >= 20:
        d2 = (xs.astype(np.float64) - p0[0]) ** 2 + (ys.astype(np.float64) - p0[1]) ** 2
        i = int(np.argmin(d2))
        p0 = (float(xs[i]), float(ys[i]))
        d2e = (xs.astype(np.float64) - p_outer[0]) ** 2 + (ys.astype(np.float64) - p_outer[1]) ** 2
        j = int(np.argmin(d2e))
        p_outer = (float(xs[j]), float(ys[j]))
    layer_ink = cv2.bitwise_or(orange, cv2.bitwise_or(blue, red))
    p_outer = _last_on_ray(layer_ink, p0, p_outer)
    if stem is not None and dist is not None:
        yi, xi = int(round(p_outer[1])), int(round(p_outer[0]))
        if 0 <= yi < h and 0 <= xi < w and float(dist[yi, xi]) > 40:
            bark_mask = cv2.dilate(stem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
            p_outer = _last_on_ray(bark_mask, p0, p_outer, gap_max=8)
    bark_end = p_outer

    # JSON / 人工「心」= 茎盘几何中心（DT peak），不是断掉的绿墨线端点。
    if pith is not None and dist is not None:
        stem_r = float(dist.max()) or 1.0
        g_start = (float(pith[0]), float(pith[1]))
        # 绿线方向：墨线外端；若绿墨未到髓心，用整段绿笔拟合方向。
        if green_m is not None and int(cv2.countNonZero(green_m)) >= 20:
            ys_g, xs_g = np.where(green_m > 0)
            d2g = (xs_g.astype(np.float64) - pith[0]) ** 2 + (ys_g.astype(np.float64) - pith[1]) ** 2
            # 外端 = 离髓心最远的绿墨
            go = int(np.argmax(d2g))
            tip = (float(xs_g[go]), float(ys_g[go]))
            g_end = _last_on_ray(green_m, g_start, tip)
            yi, xi = int(round(g_end[1])), int(round(g_end[0]))
            if stem is not None and 0 <= yi < h and 0 <= xi < w and float(dist[yi, xi]) > 28:
                bark_mask = cv2.dilate(stem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
                g_end = _last_on_ray(bark_mask, g_start, tip, gap_max=8)
        # 橙起点：层射线内侧端点，取靠近髓心的橙墨（内环），禁止落到髓心上。
        if len(xs) >= 20:
            d2p = (xs.astype(np.float64) - pith[0]) ** 2 + (ys.astype(np.float64) - pith[1]) ** 2
            # 只保留与当前层射线同侧、且沿射线投影靠内的橙墨
            vx, vy = p_outer[0] - pith[0], p_outer[1] - pith[1]
            dots = (xs.astype(np.float64) - pith[0]) * vx + (ys.astype(np.float64) - pith[1]) * vy
            same = dots > 0
            if np.any(same):
                cand_d = d2p[same]
                # 内环：离髓心最近、但至少 0.08*sr（避免贴在心上）
                min_r2 = (0.08 * stem_r) ** 2
                ok = cand_d >= min_r2
                if np.any(ok):
                    ii = int(np.argmin(np.where(ok, cand_d, 1e18)))
                    p0 = (float(xs[same][ii]), float(ys[same][ii]))
                else:
                    ii = int(np.argmin(cand_d))
                    p0 = (float(xs[same][ii]), float(ys[same][ii]))
            else:
                ii = int(np.argmin(d2p))
                p0 = (float(xs[ii]), float(ys[ii]))
            if math.hypot(p0[0] - pith[0], p0[1] - pith[1]) < 8.0:
                ang = math.degrees(math.atan2(-(p_outer[1] - pith[1]), p_outer[0] - pith[0]))
                from calibration.stem import point_on_ray as _por

                p0 = _por(pith, ang, max(36.0, 0.12 * stem_r))
                # snap back to orange ink if possible
                d2s = (xs.astype(np.float64) - p0[0]) ** 2 + (ys.astype(np.float64) - p0[1]) ** 2
                si = int(np.argmin(d2s))
                p0 = (float(xs[si]), float(ys[si]))
    else:
        if green_m is not None and int(cv2.countNonZero(green_m)) >= 20:
            ys_g, xs_g = np.where(green_m > 0)
            d2g = (xs_g.astype(np.float64) - g_start[0]) ** 2 + (ys_g.astype(np.float64) - g_start[1]) ** 2
            gi = int(np.argmin(d2g))
            g_start = (float(xs_g[gi]), float(ys_g[gi]))
            g_end = _last_on_ray(green_m, g_start, g_end)
    base_len = math.hypot(p_outer[0] - p0[0], p_outer[1] - p0[1]) or 1.0
    extra = 1.06
    runs = _layer_runs_along_segment(
        manual_img, p0, p_outer, extra=extra, orange=orange, blue=blue, red=red
    )
    if runs is not None:
        p1, p2, p3 = runs["p1"], runs["p2"], runs["p3"]
        if math.hypot(bark_end[0] - p0[0], bark_end[1] - p0[1]) >= math.hypot(p3[0] - p0[0], p3[1] - p0[1]) - 2:
            p_outer = bark_end
        else:
            p_outer = p3
        if math.hypot(p2[0] - p0[0], p2[1] - p0[1]) > math.hypot(p_outer[0] - p0[0], p_outer[1] - p0[1]):
            p2 = p_outer
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > math.hypot(p2[0] - p0[0], p2[1] - p0[1]):
            p1 = p2
    else:
        p1 = (
            p0[0] + (p_outer[0] - p0[0]) * 0.88,
            p0[1] + (p_outer[1] - p0[1]) * 0.88,
        )
        p2 = (
            p0[0] + (p_outer[0] - p0[0]) * 0.95,
            p0[1] + (p_outer[1] - p0[1]) * 0.95,
        )

    if stem is not None:
        bark_r = cv2.dilate(stem, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)), iterations=1)
        ux, uy = p_outer[0] - p0[0], p_outer[1] - p0[1]
        plen = math.hypot(ux, uy) or 1.0
        ux, uy = ux / plen, uy / plen

        def _clamp_out(pt: tuple[float, float]) -> tuple[float, float]:
            x, y = pt
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w and int(bark_r[yi, xi]) > 0:
                return pt
            t = math.hypot(x - p0[0], y - p0[1])
            while t > 8:
                t -= 4
                x, y = p0[0] + t * ux, p0[1] + t * uy
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < h and 0 <= xi < w and int(bark_r[yi, xi]) > 0:
                    return (float(x), float(y))
            return pt

        p_outer = _clamp_out(p_outer)
        p2 = _clamp_out(p2)
        p1 = _clamp_out(p1)
        d1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        d2 = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        d3 = math.hypot(p_outer[0] - p0[0], p_outer[1] - p0[1])
        if d2 < d1:
            p2 = p1
            d2 = d1
        if d3 < d2:
            p_outer = p2

    seg_angle = math.degrees(math.atan2(-(p_outer[1] - p0[1]), p_outer[0] - p0[0]))
    angle = math.degrees(math.atan2(-(g_end[1] - g_start[1]), g_end[0] - g_start[0]))

    return LineGeometry(
        center=(float(g_start[0]), float(g_start[1])),
        green_start=(float(g_start[0]), float(g_start[1])),
        green_end=(float(g_end[0]), float(g_end[1])),
        seg_y=float((p0[1] + p1[1]) / 2),
        seg_x0=float(p0[0]),
        xylem_end=float(p1[0]),
        phloem_end=float(p2[0]),
        bark_end=float(p_outer[0]),
        green_angle_deg=angle,
        seg_angle_deg=seg_angle,
        seg_y0=float(p0[1]),
        xylem_end_y=float(p1[1]),
        phloem_end_y=float(p2[1]),
        bark_end_y=float(p_outer[1]),
    )


def extract_manual_line_mask(manual_img: np.ndarray) -> np.ndarray:
    h, w = manual_img.shape[:2]
    combined = _annotation_color_mask(manual_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=2)
    combined[: int(h * 0.18), : int(w * 0.26)] = 0
    combined[: int(h * 0.18), int(w * 0.74) :] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    line_mask = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(bw, bh) / (min(bw, bh) + 1)
        if area > 2000 or (area > 200 and aspect > 4):
            line_mask[labels == i] = 255

    return cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )


# Colors sampled from 人工标定/第一区域 (BGR)
COLOR_GREEN = (12, 159, 1)
COLOR_ORANGE = (45, 108, 252)
COLOR_BLUE = (209, 31, 6)
COLOR_RED = (48, 95, 239)
LINE_THICKNESS = 8


def _slide_background_mask(img: np.ndarray, stem_mask: np.ndarray | None = None) -> np.ndarray:
    """White slide around the specimen; vessel lumens inside the stem are ignored."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    slide = ((gray >= 218) & (hsv[:, :, 1] < 48)) | (gray >= 242)
    slide_u8 = slide.astype(np.uint8) * 255
    if stem_mask is not None:
        slide_u8[stem_mask > 0] = 0
    return slide_u8


def _ipt(pt: tuple[float, float] | np.ndarray) -> tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _clip_end_off_slide(
    img: np.ndarray,
    stem_mask: np.ndarray,
    origin: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    """Pull an endpoint back so the visible ink stays off the white slide."""
    bg = _slide_background_mask(img, stem_mask)
    h, w = bg.shape[:2]
    vx, vy = end[0] - origin[0], end[1] - origin[1]
    dist = math.hypot(vx, vy)
    if dist < 4:
        return end
    ux, uy = vx / dist, vy / dist
    last = origin
    n = int(round(dist))
    for t in range(n + 1):
        x = int(round(origin[0] + t * ux))
        y = int(round(origin[1] + t * uy))
        if x < 0 or y < 0 or x >= w or y >= h:
            return last
        if bg[y, x] > 0:
            return last
        last = (float(x), float(y))
    return end


def _draw_numbered_dot(
    out: np.ndarray,
    xy: tuple[int, int],
    n: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    cv2.circle(out, xy, radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(out, xy, radius, color, max(2, radius // 5), cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = radius / 18.0
    text = str(n)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    cv2.putText(
        out,
        text,
        (xy[0] - tw // 2, xy[1] + th // 2),
        font,
        scale,
        color,
        max(1, radius // 8),
        cv2.LINE_AA,
    )


def draw_calibration_lines(
    img: np.ndarray,
    geom: LineGeometry,
    stem_mask: np.ndarray | None = None,
    numbered: bool = True,
) -> np.ndarray:
    """Draw the 6-point method: green 1→2, orange 3→4, blue 4→5, red 5→6.

    Layer colors are sequential segments on one ray (not a fixed 88%/95% split).
    Lengths follow the 人工墨线; do not erase ink on bright tissue.
    """
    out = img.copy()
    h, w = img.shape[:2]
    thick = max(16, int(round(min(h, w) / 120)))
    g0 = _ipt(geom.green_start)
    g1 = _ipt(geom.green_end)
    p0 = _ipt((geom.seg_x0, geom.seg_y0))
    p1 = _ipt((geom.xylem_end, geom.xylem_end_y))
    p2 = _ipt((geom.phloem_end, geom.phloem_end_y))
    p3 = _ipt((geom.bark_end, geom.bark_end_y))

    cv2.line(out, g0, g1, COLOR_GREEN, thick, cv2.LINE_8)
    cv2.line(out, p0, p1, COLOR_ORANGE, thick, cv2.LINE_8)
    cv2.line(out, p1, p2, COLOR_BLUE, thick, cv2.LINE_8)
    cv2.line(out, p2, p3, COLOR_RED, thick, cv2.LINE_8)

    if numbered:
        radius = max(12, int(round(thick * 1.6)))
        _draw_numbered_dot(out, g0, 1, COLOR_GREEN, radius)
        _draw_numbered_dot(out, g1, 2, COLOR_GREEN, radius)
        _draw_numbered_dot(out, p0, 3, COLOR_ORANGE, radius)
        _draw_numbered_dot(out, p1, 4, COLOR_BLUE, radius)
        _draw_numbered_dot(out, p2, 5, COLOR_RED, radius)
        _draw_numbered_dot(out, p3, 6, COLOR_RED, radius)
    return out


def overlay_manual_lines(base: np.ndarray, manual_img: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[line_mask > 0] = manual_img[line_mask > 0]
    return out
