"""Draw auto annotations in a style close to the human teaching marks."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Colors aligned with teaching / Wand-style overview marks
ORANGE = (248, 104, 40)  # 多导管区域框
PURPLE = (144, 112, 248)  # Ai 十字箭头
BLUE = (0, 40, 200)  # TVW 双箭头
GREEN = (0, 220, 80)  # 计入统计的导管腔轮廓
# TVW strokes: 3–5 nearest wall gaps between the boxed pair
TVW_RAINBOW = [
    (40, 140, 255),
    (20, 100, 230),
    (60, 180, 255),
    (30, 80, 200),
    (80, 160, 240),
]


def _arrowhead(draw: ImageDraw.ImageDraw, tip, base_dir, size: float, color, width: int = 2):
    """Filled triangle arrowhead at tip, pointing along base_dir (unit)."""
    d = np.asarray(base_dir, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return
    d = d / n
    ortho = np.array([-d[1], d[0]])
    tip = np.asarray(tip, dtype=float)
    p1 = tip
    p2 = tip - d * size + ortho * size * 0.55
    p3 = tip - d * size - ortho * size * 0.55
    pts = [tuple(map(int, p1)), tuple(map(int, p2)), tuple(map(int, p3))]
    draw.polygon(pts, fill=color)
    draw.line([pts[1], pts[0], pts[2]], fill=color, width=max(1, width))


def draw_double_arrow(
    draw: ImageDraw.ImageDraw,
    p0,
    p1,
    color,
    width: int = 4,
    head: float = 12,
):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 4:
        return
    d = v / L
    # shorten line a bit so heads sit on ends
    a = p0 + d * (head * 0.35)
    b = p1 - d * (head * 0.35)
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    _arrowhead(draw, p0, -d, head, color, width)
    _arrowhead(draw, p1, d, head, color, width)


def draw_ai_cross(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float, color=PURPLE):
    """Purple 4-way double arrow spanning lumen axes (human Ai mark)."""
    rx = max(8.0, rx * 0.85)
    ry = max(8.0, ry * 0.85)
    w = 4 if min(rx, ry) > 25 else 3
    head = 11 if min(rx, ry) > 25 else 8
    draw_double_arrow(draw, (cx - rx, cy), (cx + rx, cy), color, width=w, head=head)
    draw_double_arrow(draw, (cx, cy - ry), (cx, cy + ry), color, width=w, head=head)


def draw_orange_box(draw: ImageDraw.ImageDraw, xyxy, color=ORANGE, width: int = 6):
    x0, y0, x1, y1 = map(int, xyxy)
    for i in range(width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)


def _axis_radii_from_contour(contour: np.ndarray) -> tuple[float, float, float, float]:
    """Return cx, cy, rx, ry from contour moments / PCA-ish bbox."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    # use oriented extents via SVD
    c = pts - np.array([cx, cy])
    if len(c) < 3:
        x, y, w, h = cv2.boundingRect(contour)
        return cx, cy, w / 2.0, h / 2.0
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt[:2].T
    rx = float(np.percentile(np.abs(proj[:, 0]), 92))
    ry = float(np.percentile(np.abs(proj[:, 1]), 92))
    # For cross drawing we use axis-aligned radii (human marks look axis-aligned).
    x, y, w, h = cv2.boundingRect(contour)
    return float(cx), float(cy), w / 2.2, h / 2.2


def _ellipse_pts_from_contour(contour: np.ndarray, scale: float = 1.0) -> np.ndarray | None:
    """拟合椭圆并按 scale 缩小轴长，返回 Nx2 采样点。"""
    pts = contour.reshape(-1, 2)
    if len(pts) < 5:
        return None
    try:
        (ex, ey), (maj, mino), ang = cv2.fitEllipse(pts.astype(np.float32))
    except Exception:
        return None
    if maj < 8 or mino < 8:
        return None
    s = float(np.clip(scale, 0.75, 1.05))
    a, b = maj * 0.5 * s, mino * 0.5 * s
    th = np.deg2rad(ang)
    ct, st = np.cos(th), np.sin(th)
    ts = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    xs = a * np.cos(ts)
    ys = b * np.sin(ts)
    xr = ex + xs * ct - ys * st
    yr = ey + xs * st + ys * ct
    return np.stack([xr, yr], axis=1)


def _polyline_min_gap(a: np.ndarray, b: np.ndarray) -> float:
    p1 = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if len(p1) == 0 or len(p2) == 0:
        return 1e9
    if len(p1) > 120:
        p1 = p1[:: max(1, len(p1) // 120)]
    if len(p2) > 120:
        p2 = p2[:: max(1, len(p2) // 120)]
    d2 = ((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def pair_green_draw_scales(
    c1: np.ndarray,
    c2: np.ndarray,
    min_gap_px: float = 8.0,
    base_scale: float = 0.94,
) -> tuple[float, float]:
    """
    双导管绿圈绘制缩放：先略内缩贴壁，若两圈仍过近/贴合则继续对缩，
    保证绘制椭圆之间至少 min_gap_px。
    """
    e1 = _ellipse_pts_from_contour(c1, base_scale)
    e2 = _ellipse_pts_from_contour(c2, base_scale)
    if e1 is None or e2 is None:
        return base_scale, base_scale
    if _polyline_min_gap(e1, e2) >= min_gap_px:
        return base_scale, base_scale
    lo, hi = 0.82, base_scale
    best = 0.86
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        p1 = _ellipse_pts_from_contour(c1, mid)
        p2 = _ellipse_pts_from_contour(c2, mid)
        if p1 is None or p2 is None:
            break
        if _polyline_min_gap(p1, p2) >= min_gap_px:
            best = mid
            lo = mid
        else:
            hi = mid
    return best, best


def draw_green_contour(
    draw: ImageDraw.ImageDraw,
    contour: np.ndarray,
    color=GREEN,
    width: int = 2,
    scale: float = 1.0,
):
    """Green lumen outline = vessel counted in Ai/Di/Dh（优先绘闭合类椭圆，避免锯齿折线）。"""
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return
    ell = _ellipse_pts_from_contour(contour, scale=scale)
    if ell is not None:
        pts = ell
    seq = [tuple(map(int, p)) for p in pts] + [tuple(map(int, pts[0]))]
    draw.line(seq, fill=color, width=width)


def _oriented_box_xyxy(box, pad: float = 0.0) -> tuple[int, int, int, int]:
    """Axis-aligned bounds of an OrientedBox (cx,cy,w,h,angle)."""
    rad = np.deg2rad(getattr(box, "angle", 0.0))
    hw, hh = box.w / 2.0 + pad, box.h / 2.0 + pad
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    pts = corners @ rot.T + np.array([box.cx, box.cy])
    return (
        int(np.floor(pts[:, 0].min())),
        int(np.floor(pts[:, 1].min())),
        int(np.ceil(pts[:, 0].max())),
        int(np.ceil(pts[:, 1].max())),
    )


def _vessel_in_boxes(v, boxes, pad: float = 40.0) -> bool:
    if not boxes:
        return True
    return any(b.contains_point(v.cx, v.cy, pad=pad) for b in boxes)


def annotate_pairs_human_style(
    rgb: np.ndarray,
    pairs,
    vessels=None,
    guide_boxes=None,
    singles=None,
    pair_draw_scale: float | None = None,
) -> np.ndarray:
    """
    Display:
    - green: 双导管成员 + 其外的单导管
    - orange / purple Ai / TVW: 仅有配对时绘制
    - 无配对：只画绿线（无橙框、无箭头）
    - pair_draw_scale: 若给定（如 1.0），双导管绿圈按该比例绘制，不再自动拉远/拉近
    """
    if rgb.ndim != 3:
        raise ValueError("rgb required")
    h, w = rgb.shape[:2]
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    boxes = list(guide_boxes or []) if pairs else []

    # Green = 已配对腔 + 配对之外的单导管
    draw_scale: dict[str, float] = {}
    draw_vs = []
    seen = set()
    for p in pairs or []:
        if pair_draw_scale is not None:
            s1 = s2 = float(pair_draw_scale)
        else:
            # 仅当两圈几乎贴合时略内缩；已有间距则保持原几何
            s1, s2 = pair_green_draw_scales(
                p.v1.contour, p.v2.contour, min_gap_px=4.0, base_scale=1.0
            )
        draw_scale[p.v1.vessel_id] = s1
        draw_scale[p.v2.vessel_id] = s2
        for v in (p.v1, p.v2):
            if v.vessel_id in seen:
                continue
            draw_vs.append(v)
            seen.add(v.vessel_id)
    if singles is None and vessels is not None:
        from human_rules import select_single_vessels

        singles = select_single_vessels(vessels, pairs or [], w, h)
    for v in singles or []:
        if v.vessel_id in seen:
            continue
        draw_vs.append(v)
        seen.add(v.vessel_id)

    for v in draw_vs:
        draw_green_contour(
            draw, v.contour, GREEN, width=2, scale=draw_scale.get(v.vessel_id, 1.0)
        )

    # 仅双导管：橙框 + Ai 箭头 + TVW 线（无配对则不画）
    if pairs:
        if boxes:
            for b in boxes:
                draw_orange_box(draw, _oriented_box_xyxy(b, pad=0), width=5)
        else:
            for p in pairs:
                x1, y1, bw1, bh1 = cv2.boundingRect(p.v1.contour)
                x2, y2, bw2, bh2 = cv2.boundingRect(p.v2.contour)
                pad = 18
                draw_orange_box(
                    draw,
                    (
                        min(x1, x2) - pad,
                        min(y1, y2) - pad,
                        max(x1 + bw1, x2 + bw2) + pad,
                        max(y1 + bh1, y2 + bh2) + pad,
                    ),
                    width=5,
                )
        for p in pairs:
            for v in (p.v1, p.v2):
                if boxes and not _vessel_in_boxes(v, boxes, pad=55.0):
                    continue
                cx, cy, rx, ry = _axis_radii_from_contour(v.contour)
                draw_ai_cross(draw, cx, cy, rx, ry)
            if p.lines:
                for li, ln in enumerate(p.lines):
                    color = TVW_RAINBOW[li % len(TVW_RAINBOW)]
                    draw.line(
                        [(int(ln.x0), int(ln.y0)), (int(ln.x1), int(ln.y1))],
                        fill=color,
                        width=4,
                    )

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(out)
