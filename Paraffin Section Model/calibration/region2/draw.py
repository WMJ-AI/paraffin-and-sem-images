"""Draw region-2: dashed area boxes + filled vessel lumens (no detection boxes)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.region2.rules import (
    VESSEL_FILL_ALPHA,
    VESSEL_FILL_BGR,
    VESSEL_LINE_THICKNESS,
    VESSEL_OUTLINE_BGR,
)
from calibration.region2.vessels import Region2Result, Vessel

PHLOEM_BOX_BGR = (255, 220, 0)  # cyan dashed, same family as manuals
PITH_BOX_BGR = (200, 40, 180)  # purple
XYLEM_BOX_BGR = (0, 140, 255)  # orange
BOX_THICKNESS = 5


def draw_vessels(img: np.ndarray, vessels: list[Vessel], alpha: float = VESSEL_FILL_ALPHA) -> np.ndarray:
    """Fill each lumen (count unit). Do not draw detection rectangles."""
    overlay = img.copy()
    for v in vessels:
        cv2.drawContours(overlay, [v.contour], -1, VESSEL_FILL_BGR, thickness=-1)
    out = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    thick = max(2, VESSEL_LINE_THICKNESS)
    for v in vessels:
        cv2.drawContours(out, [v.contour], -1, VESSEL_OUTLINE_BGR, thick, cv2.LINE_AA)
    return out


def _dash_line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color, thickness: int) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dash, gap = 22.0, 12.0
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    n = 0.0
    while n < dist:
        a, b = n, min(dist, n + dash)
        q1 = (int(round(x1 + dx * a)), int(round(y1 + dy * a)))
        q2 = (int(round(x1 + dx * b)), int(round(y1 + dy * b)))
        cv2.line(img, q1, q2, color, thickness, cv2.LINE_AA)
        n += dash + gap


def draw_dashed_rect(img: np.ndarray, box: tuple[int, int, int, int], color, thickness: int = BOX_THICKNESS) -> None:
    x, y, bw, bh = box
    x1, y1 = x + bw - 1, y + bh - 1
    _dash_line(img, (x, y), (x1, y), color, thickness)
    _dash_line(img, (x1, y), (x1, y1), color, thickness)
    _dash_line(img, (x1, y1), (x, y1), color, thickness)
    _dash_line(img, (x, y1), (x, y), color, thickness)


def draw_partition_line(
    img: np.ndarray,
    mask: np.ndarray,
    color,
    thickness: int = BOX_THICKNESS,
) -> None:
    """Solid inner tissue line + end caps to the image edge (the partition)."""
    from calibration.region2.areas import polyline_from_mask

    parsed = polyline_from_mask(mask)
    if parsed is None:
        return
    inner, valid, from_left = parsed
    h, w = mask.shape
    ys = np.where(valid > 0)[0]
    if ys.size < 2:
        return
    y0, y1 = int(ys.min()), int(ys.max())
    pts: list[tuple[int, int]] = []
    step = 6
    for y in range(y0, y1 + 1, step):
        if valid[y] > 0:
            pts.append((int(inner[y]), int(y)))
    if valid[y1] > 0:
        last = (int(inner[y1]), int(y1))
        if not pts or pts[-1] != last:
            pts.append(last)
    if len(pts) < 2:
        return
    edge_x = 0 if from_left else w - 1
    cv2.line(img, (edge_x, pts[0][1]), pts[0], color, thickness, cv2.LINE_AA)
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    cv2.line(img, pts[-1], (edge_x, pts[-1][1]), color, thickness, cv2.LINE_AA)


def draw_region2(img: np.ndarray, result: Region2Result, with_stats: bool = True) -> np.ndarray:
    out = img.copy()
    for box in result.xylem_boxes:
        draw_dashed_rect(out, box, XYLEM_BOX_BGR, BOX_THICKNESS + 1)
    if cv2.countNonZero(result.phloem_mask):
        draw_partition_line(out, result.phloem_mask, PHLOEM_BOX_BGR, BOX_THICKNESS + 1)
    if cv2.countNonZero(result.pith_mask):
        draw_partition_line(out, result.pith_mask, PITH_BOX_BGR, BOX_THICKNESS + 1)
    for box in result.phloem_boxes:
        draw_dashed_rect(out, box, PHLOEM_BOX_BGR)
    for box in result.pith_boxes:
        draw_dashed_rect(out, box, PITH_BOX_BGR)
    out = draw_vessels(out, result.vessels)
    if not with_stats:
        return out
    return _draw_stat_lines(out, legend_lines(result))


def legend_lines(result: Region2Result) -> list[str]:
    """Same 5 fields / rounding as 结果呈现表.xlsx region-2 sheets."""
    return legend_lines_from_values(
        int(result.count),
        float(result.lumen_area_um2),
        float(result.xylem_area_um2),
        float(result.phloem_area_um2),
        float(result.pith_area_um2),
    )


def legend_lines_from_values(
    count: int,
    lumen_um2: float,
    xylem_um2: float,
    phloem_um2: float,
    pith_um2: float,
) -> list[str]:
    return [
        f"导管总数量  {int(count)}",
        f"导管腔面积  {lumen_um2:.1f} µm²",
        f"局部木质部面积  {xylem_um2:.1f} µm²",
        f"局部韧皮部面积  {phloem_um2:.1f} µm²",
        f"局部髓面积  {pith_um2:.1f} µm²",
    ]


def replace_legend(orig: np.ndarray, auto: np.ndarray, lines: list[str]) -> np.ndarray:
    """Erase baked-in legend boxes using the original crop, then redraw stats."""
    h, w = auto.shape[:2]
    roi_h, roi_w = min(h, int(h * 0.32)), min(w, int(w * 0.58))
    hsv = cv2.cvtColor(auto[:roi_h, :roi_w], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 240), (180, 50, 255))
    white = cv2.dilate(white, np.ones((9, 9), np.uint8))
    out = auto.copy()
    roi = out[:roi_h, :roi_w]
    src = orig[:roi_h, :roi_w]
    roi[white > 0] = src[white > 0]
    return _draw_stat_lines(out, lines)


def _draw_stat_lines(img: np.ndarray, lines: list[str]) -> np.ndarray:
    h, w = img.shape[:2]
    font_size = max(28, int(w / 90))
    font_path = _chinese_font()
    if font_path is not None:
        from PIL import Image, ImageDraw, ImageFont

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.truetype(font_path, font_size)
        y = 18
        x = 18
        pad = 8
        for text in lines:
            box = draw.textbbox((x, y), text, font=font)
            draw.rectangle(
                (box[0] - pad, box[1] - 4, box[2] + pad, box[3] + 4),
                fill=(255, 255, 255),
            )
            draw.text((x, y), text, font=font, fill=(180, 90, 0))
            y = box[3] + 10
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    scale = max(0.9, w / 2800)
    thick = max(2, int(round(scale * 2)))
    y = int(36 * scale + 18)
    x = 18
    for text in lines:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, VESSEL_OUTLINE_BGR, thick, cv2.LINE_AA)
        y += int(34 * scale + 8)
    return img


def _chinese_font() -> str | None:
    from pathlib import Path

    for p in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if p.exists():
            return str(p)
    return None
