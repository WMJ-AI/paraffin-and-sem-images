"""Vessel lumen segmentation via multi-threshold dark CC + light refine."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from exclusion_rules import (  # noqa: F401 — re-export for callers
    evaluate_exclusion,
    get_exclusion_config,
    is_border_truncated,
    is_crack_like_void,
    is_serrated_debris_void,
    lumen_shape_metrics,
)

# 本地 --target 安装的 ultralytics（segment_hybrid → infer_lumen_seg）
_PYLIBS = Path(__file__).resolve().parent / ".pylibs"
if _PYLIBS.is_dir():
    import sys as _sys

    if str(_PYLIBS) not in _sys.path:
        _sys.path.insert(0, str(_PYLIBS))

# 可选审计日志（默认关闭；绝不作为排除依据）
EXCLUDED_VOID_LOG = Path(__file__).resolve().parent / "output" / "排除_裂隙状腔.csv"


@dataclass
class Vessel:
    vessel_id: str
    image_name: str
    contour: np.ndarray
    area_px: float
    peri_px: float
    cx: float
    cy: float
    bbox: tuple[int, int, int, int]
    status: str = "primary"
    reason: str = ""
    pair_id: str | None = None
    ai_um2: float | None = None
    di_um: float | None = None
    peri_um: float | None = None
    mask: np.ndarray | None = field(default=None, repr=False)


def tissue_height(h: int) -> int:
    return int(round(h * 691 / 768))


def tissue_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    th = tissue_height(h)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[:th, :] = 255
    mask[int(th * 0.92) :, : int(w * 0.22)] = 0
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    hh, ww = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((hh + 2, ww + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def _touches_forbidden(mask: np.ndarray, tissue_h: int) -> bool:
    if mask[:1, :].any() or mask[:, :1].any() or mask[:, -1:].any():
        return True
    return bool(mask[max(0, tissue_h - 1) : tissue_h + 1, :].any())


def _candidate_from_contour(
    cnt: np.ndarray, filled: np.ndarray, t: int, circ: float, source: str
) -> dict | None:
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri <= 0:
        return None
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    return {
        "contour": cnt,
        "area_px": area,
        "peri_px": peri,
        "cx": m["m10"] / m["m00"],
        "cy": m["m01"] / m["m00"],
        "bbox": (x, y, bw, bh),
        "circ": circ,
        "mask": filled,
        "t": t,
        "source": source,
    }


def _from_mask(mask: np.ndarray, tissue_h: int, min_area: float, max_area: float, t: int):
    if _touches_forbidden(mask, tissue_h):
        return None
    filled = _fill_holes(mask)
    # 1px dilate to better approach lumen-wall interface (reduces TVW overestimate)
    filled = cv2.dilate(filled, np.ones((3, 3), np.uint8), iterations=1)
    # but undo if this causes border touch
    if _touches_forbidden(filled, tissue_h):
        filled = _fill_holes(mask)

    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < min_area or area > max_area:
        return None
    peri = float(cv2.arcLength(cnt, True))
    if peri <= 0:
        return None
    circ = 4 * np.pi * area / (peri * peri)
    if circ < 0.14:
        return None
    return _candidate_from_contour(cnt, filled, t, circ, "dark_cc")


def _from_ellipse_recovery(
    gray: np.ndarray,
    frag_mask: np.ndarray,
    tissue: np.ndarray,
    tissue_h: int,
    min_area: float,
    max_area: float,
    t: int,
) -> dict | None:
    """
    Recover debris-filled lumen (e.g. 8(8) right vessel): fragment fails circularity,
    but fitted ellipse is round and darker than surrounding wall ring.
    """
    cnts, _ = cv2.findContours(frag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    frag_a = float(cv2.contourArea(cnt))
    if frag_a < min_area * 0.5 or frag_a > max_area * 1.15:
        return None
    (ex, ey), (maj, mino), ang = cv2.fitEllipse(cnt)
    if maj <= 1 or mino <= 1:
        return None
    ell_ratio = min(maj, mino) / max(maj, mino)
    if ell_ratio < 0.65:
        return None
    ell_area = float(np.pi * (maj * 0.5) * (mino * 0.5))
    if ell_area < min_area or ell_area > max_area:
        return None

    emask = np.zeros_like(gray, dtype=np.uint8)
    cv2.ellipse(emask, ((ex, ey), (maj, mino), ang), 255, -1)
    emask = cv2.bitwise_and(emask, tissue)
    if _touches_forbidden(emask, tissue_h):
        return None
    inside = gray[emask > 0]
    if inside.size < 50:
        return None
    mean_in = float(inside.mean())
    # 仅够黑的空腔算导管（教学 18(8)：浅灰凹陷不算）
    if mean_in > 48:
        return None

    ring = np.zeros_like(gray, dtype=np.uint8)
    cv2.ellipse(ring, ((ex, ey), (maj * 1.28, mino * 1.28), ang), 255, -1)
    cv2.ellipse(ring, ((ex, ey), (maj, mino), ang), 0, -1)
    ring = cv2.bitwise_and(ring, tissue)
    if not ring.any():
        return None
    mean_ring = float(gray[ring > 0].mean())
    if mean_ring - mean_in < 25:
        return None

    ecnts, _ = cv2.findContours(emask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ecnts:
        return None
    ecnt = max(ecnts, key=cv2.contourArea)
    peri = float(cv2.arcLength(ecnt, True))
    circ = 4 * np.pi * ell_area / (peri * peri) if peri > 0 else ell_ratio
    # score slightly below solid dark_cc of same size
    got = _candidate_from_contour(ecnt, emask, t, max(circ, ell_ratio * 0.9), "ellipse_recover")
    if got is None:
        return None
    got["area_px"] = ell_area
    return got


def append_excluded_void_log(rows: list[dict], path: Path = EXCLUDED_VOID_LOG) -> None:
    """可选审计日志：只写不读，不参与任何排除判定。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "cx",
        "cy",
        "ai_um2",
        "aspect",
        "ell_ar",
        "solid",
        "circ",
        "bbox_w",
        "bbox_h",
        "rule",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def segment_dark_lumens(
    gray: np.ndarray,
    um_per_px: float,
    image_name: str,
    min_ai_um2: float | None = None,
    max_ai_um2: float | None = None,
    log_excluded: bool | None = None,
) -> list[Vessel]:
    """
    分割暗腔。排除判定只走 exclusion_rules / learned_rules.json.exclusion，
    与排除_裂隙状腔.csv 无关。
    """
    xcfg = get_exclusion_config()
    min_ai_um2 = xcfg.min_ai_um2 if min_ai_um2 is None else min_ai_um2
    max_ai_um2 = xcfg.max_ai_um2 if max_ai_um2 is None else max_ai_um2
    log_excluded = xcfg.write_exclusion_log if log_excluded is None else log_excluded
    tissue = tissue_mask(gray)
    th = tissue_height(gray.shape[0])
    min_area = min_ai_um2 / (um_per_px**2)
    max_area = max_ai_um2 / (um_per_px**2)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    candidates: list[dict] = []
    for t in (40, 48, 56, 64, 72):
        dark = ((blur <= t) & (tissue > 0)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area * 0.35 or area > max_area * 1.25:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            # 仅丢弃真正贴图像边的连通域；近边完整腔交给 exclusion 的截断判定
            if x <= 1 or y <= 1 or x + bw >= gray.shape[1] - 2 or y + bh >= th - 2:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            got_mask = None
            if area >= min_area * 0.6:
                got_mask = _from_mask(comp, th, min_area, max_area, t)
            # 碎屑/锯齿暗连通域：mask 圆形度差时改用椭圆恢复贴合黑色腔
            got_ell = _from_ellipse_recovery(
                gray, comp, tissue, th, min_area * 0.75, max_area, t
            )
            if got_mask is not None and got_ell is not None:
                if (
                    got_mask["circ"] < 0.55
                    and got_ell["circ"] >= got_mask["circ"] + 0.12
                    and got_ell["area_px"] >= got_mask["area_px"] * 0.85
                ):
                    got = got_ell
                else:
                    got = got_mask
            elif got_mask is not None:
                got = got_mask
            else:
                got = got_ell
            if got is not None:
                candidates.append(got)

    def _score(c: dict) -> float:
        # Prefer solid dark_cc over ellipse recovery when similar
        bonus = 1.0 if c.get("source") == "dark_cc" else 0.85
        return c["circ"] * np.sqrt(c["area_px"]) * bonus

    def _should_replace(old: dict, c: dict) -> bool:
        """同位置多阈值候选：保留可通过 exclusion、且更圆的轮廓，避免高阈值渗漏腔吞掉完整腔。"""
        bad_new, _, _ = evaluate_exclusion(
            c["contour"], gray.shape[1], gray.shape[0], tissue_h=th, cfg=xcfg
        )
        bad_old, _, _ = evaluate_exclusion(
            old["contour"], gray.shape[1], gray.shape[0], tissue_h=th, cfg=xcfg
        )
        if bad_old and not bad_new:
            return True
        if bad_new and not bad_old:
            return False
        if old.get("source") == "ellipse_recover" and c.get("source") == "dark_cc":
            # 仅当暗连通域本身够圆且未排除时才覆盖椭圆恢复
            if (not bad_new) and c["circ"] >= old["circ"] * 0.9:
                return True
            return False
        # 面积增大时圆形度不能明显变差（原 0.85 过松，会让锯齿渗漏腔替换干净腔）
        if c["area_px"] > old["area_px"] * 1.02 and c["circ"] >= old["circ"] * 0.95:
            return True
        if c["circ"] > old["circ"] + 0.05 and c["area_px"] > old["area_px"] * 0.9:
            return True
        return False

    candidates.sort(key=_score, reverse=True)
    kept: list[dict] = []
    for c in candidates:
        clash_i = None
        for i, k in enumerate(kept):
            if (c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2 < 40**2:
                clash_i = i
                break
        if clash_i is None:
            kept.append(c)
        elif _should_replace(kept[clash_i], c):
            kept[clash_i] = c

    vessels: list[Vessel] = []
    excluded_rows: list[dict] = []
    m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
    prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
    idx = 0
    img_h, img_w = gray.shape[:2]
    for c in sorted(kept, key=lambda z: (z["cy"], z["cx"])):
        bad, rule, sm = evaluate_exclusion(
            c["contour"], img_w, img_h, tissue_h=th, cfg=xcfg
        )
        if bad:
            ai = float(c["area_px"]) * (um_per_px**2)
            excluded_rows.append(
                {
                    "image": image_name,
                    "cx": round(float(c["cx"]), 1),
                    "cy": round(float(c["cy"]), 1),
                    "ai_um2": round(ai, 2),
                    "aspect": round(sm["aspect"], 3),
                    "ell_ar": round(sm["ell_ar"], 3),
                    "solid": round(sm["solid"], 3),
                    "circ": round(sm["circ"], 3),
                    "bbox_w": sm["bbox_w"],
                    "bbox_h": sm["bbox_h"],
                    "rule": rule,
                }
            )
            continue
        idx += 1
        src = c.get("source", "dark_cc")
        vessels.append(
            Vessel(
                vessel_id=f"{prefix}-V{idx:02d}",
                image_name=image_name,
                contour=c["contour"],
                area_px=c["area_px"],
                peri_px=c["peri_px"],
                cx=c["cx"],
                cy=c["cy"],
                bbox=c["bbox"],
                status="primary",
                reason=f"{src};T={c['t']};circ={c['circ']:.2f}",
                mask=c["mask"],
            )
        )
    if log_excluded and excluded_rows:
        append_excluded_void_log(excluded_rows)
    # 全局阈值下邻腔被粘连超标/碎屑打断时，在已检出腔旁局部补检（例：4(4) 下方双导管）
    vessels = recover_adjacent_pair_partners(
        gray, vessels, um_per_px=um_per_px, image_name=image_name
    )
    # 锯齿暗连通域轮廓 → 完整闭合类圆（椭圆），避免绿线齿状（例：5(1) 左侧腔）
    vessels = refine_vessels_to_ellipse(vessels, gray)
    return vessels


def ellipse_contour_from_fit(
    contour: np.ndarray,
    n_pts: int = 128,
) -> tuple[np.ndarray, float, float, tuple] | None:
    """
    将腔轮廓拟合为椭圆，返回平滑闭合类圆轮廓、椭圆面积、圆形度代理、ellipse 参数。
    面积/长短轴比偏离过大时返回 None（保持原轮廓）。
    """
    if contour is None or len(contour) < 5:
        return None
    try:
        ellipse = cv2.fitEllipse(contour.astype(np.float32))
    except cv2.error:
        return None
    (ex, ey), (maj, mino), ang = ellipse
    if maj < 10 or mino < 10:
        return None
    ell_ratio = float(min(maj, mino) / max(maj, mino))
    if ell_ratio < 0.52:
        return None
    ell_area = float(np.pi * (maj * 0.5) * (mino * 0.5))
    orig_area = float(cv2.contourArea(contour))
    if orig_area <= 0:
        return None
    if ell_area < orig_area * 0.72 or ell_area > orig_area * 1.45:
        return None

    # 参数方程采样 → 闭合平滑折线（绘制无锯齿）
    a, b = maj * 0.5, mino * 0.5
    theta = np.deg2rad(ang)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ts = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    xs = a * np.cos(ts)
    ys = b * np.sin(ts)
    xr = ex + xs * cos_t - ys * sin_t
    yr = ey + xs * sin_t + ys * cos_t
    ect = np.stack([xr, yr], axis=1).astype(np.float32).reshape(-1, 1, 2)
    circ = float(ell_ratio)  # 轴比越接近 1 越圆
    return ect, ell_area, circ, ellipse


def refine_vessels_to_ellipse(vessels: list[Vessel], gray: np.ndarray) -> list[Vessel]:
    """
    对锯齿/毛刺轮廓用椭圆替换为完整闭合类圆绿边；原已光滑类圆则几乎不变。
    """
    if not vessels:
        return vessels
    th = tissue_height(gray.shape[0])
    out: list[Vessel] = []
    for v in vessels:
        sm = lumen_shape_metrics(v.contour)
        jagged = sm["rough"] >= 1.10 or sm["circ"] < 0.62
        # 非锯齿但实心度尚可时也轻微椭圆化，统一成类圆绘制
        force = sm["solid"] >= 0.85
        if not (jagged or force):
            out.append(v)
            continue
        got = ellipse_contour_from_fit(v.contour)
        if got is None:
            out.append(v)
            continue
        ect, ell_area, _circ, ellipse = got
        # 椭圆内应仍偏暗（避免拟合漂到壁上）
        emask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.ellipse(emask, ellipse, 255, -1)
        if _touches_forbidden(emask, th):
            out.append(v)
            continue
        inside = gray[emask > 0]
        if inside.size < 40 or float(inside.mean()) > 48:
            out.append(v)
            continue
        peri = float(cv2.arcLength(ect, True))
        m = cv2.moments(ect)
        if m["m00"] == 0:
            out.append(v)
            continue
        x, y, bw, bh = cv2.boundingRect(ect.astype(np.int32))
        v.contour = ect.astype(np.int32)
        v.area_px = ell_area
        v.peri_px = peri
        v.cx = float(m["m10"] / m["m00"])
        v.cy = float(m["m01"] / m["m00"])
        v.bbox = (x, y, bw, bh)
        v.mask = emask
        if "ellipse_smooth" not in v.reason:
            v.reason = (v.reason + ";ellipse_smooth") if v.reason else "ellipse_smooth"
        out.append(v)
    return out


def _min_contour_dist_px(c1: np.ndarray, c2: np.ndarray) -> float:
    p1 = c1.reshape(-1, 2).astype(np.float32)
    p2 = c2.reshape(-1, 2).astype(np.float32)
    if len(p1) > 180:
        p1 = p1[:: max(1, len(p1) // 180)]
    if len(p2) > 180:
        p2 = p2[:: max(1, len(p2) // 180)]
    # (n,1,2) vs (1,m,2)
    d = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2)).min()
    return float(d)


def recover_adjacent_pair_partners(
    gray: np.ndarray,
    vessels: list[Vessel],
    um_per_px: float,
    image_name: str,
    min_ai_um2: float = 250.0,
    max_ai_um2: float = 1500.0,
    min_gap_um: float = 1.5,
    max_gap_um: float = 28.0,
) -> list[Vessel]:
    """
    已检出导管旁局部补检漏掉的相邻腔。
    典型失败：邻腔与纤维暗区粘连后 Ai 超上限，或腔内偏亮碎屑把全局暗连通域打散
   （例：4(4) 仅检出上方腔、下方完整邻腔漏检——应成双导管）。
    """
    if not vessels:
        return vessels

    tissue = tissue_mask(gray)
    th = tissue_height(gray.shape[0])
    min_area = min_ai_um2 / (um_per_px**2)
    max_area = max_ai_um2 / (um_per_px**2)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    img_h, img_w = h, w

    occupied = np.zeros((h, w), dtype=np.uint8)
    for v in vessels:
        if v.mask is not None:
            occupied = cv2.bitwise_or(occupied, (v.mask > 0).astype(np.uint8) * 255)
        else:
            cv2.drawContours(occupied, [v.contour], -1, 255, -1)
    # 轻膨胀即可；过重会吃掉紧贴共同壁的邻腔暗区（如 2(3) 右下双导管）
    occupied = cv2.dilate(occupied, np.ones((3, 3), np.uint8), iterations=1)

    existing = list(vessels)
    added: list[dict] = []

    for v in vessels:
        ai_anchor = float(v.area_px) * (um_per_px**2)
        # 小碎腔旁补检易造假双导管；完整双导管侧腔 Ai 常在 250–400
        if ai_anchor < 220.0:
            continue
        # 仅当已有「合格近邻」时跳过：gap 在配对窗口内且 Ai 比不太离谱
        def _has_good_neighbor(anchor: Vessel) -> bool:
            for o in existing:
                if o.vessel_id == anchor.vessel_id:
                    continue
                gap = _min_contour_dist_px(anchor.contour, o.contour) * um_per_px
                if gap < min_gap_um or gap > max_gap_um:
                    continue
                ai_o = float(o.area_px) * (um_per_px**2)
                if ai_o < 180.0:
                    continue
                ratio = max(ai_anchor, ai_o) / max(min(ai_anchor, ai_o), 1e-6)
                if ratio <= 6.0:
                    return True
            return False

        if _has_good_neighbor(v):
            continue

        x, y, bw, bh = v.bbox
        # 四侧 ROI：放大 span，降低切掉邻腔边缘的概率
        side_rois: list[tuple[int, int, int, int]] = []
        hx, hy = int(1.35 * bw), int(1.35 * bh)
        span = int(2.6 * max(bw, bh))
        side_rois.append((int(v.cx - hx), int(y + bh - 8), int(v.cx + hx), int(y + bh + span)))  # down
        side_rois.append((int(v.cx - hx), int(y - span), int(v.cx + hx), int(y + 8)))  # up
        side_rois.append((int(x + bw - 8), int(v.cy - hy), int(x + bw + span), int(v.cy + hy)))  # right
        side_rois.append((int(x - span), int(v.cy - hy), int(x + 8), int(v.cy + hy)))  # left

        local_best: dict | None = None
        local_best_key: tuple | None = None
        for rx0, ry0, rx1, ry1 in side_rois:
            x0, y0 = max(0, rx0), max(0, ry0)
            x1, y1 = min(w, rx1), min(th, ry1)
            if x1 - x0 < 40 or y1 - y0 < 40:
                continue
            for t in (48, 56, 64, 72, 80, 88):
                dark = ((blur <= t) & (tissue > 0)).astype(np.uint8) * 255
                # 不在此用 occupied 挖空：紧贴共同壁时膨胀占位会切碎邻腔
                roi = np.zeros_like(dark)
                roi[y0:y1, x0:x1] = dark[y0:y1, x0:x1]
                roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                # 局部闭运算：粘回碎屑打断的暗腔，同时 ROI 限制避免全局纤维粘连
                roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
                n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=4)
                for i in range(1, n):
                    area = float(stats[i, cv2.CC_STAT_AREA])
                    if area < min_area * 0.35 or area > max_area * 1.15:
                        continue
                    cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] * 0.5
                    cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] * 0.5
                    if any((cx - e.cx) ** 2 + (cy - e.cy) ** 2 < 45**2 for e in existing):
                        continue
                    if any((cx - a["cx"]) ** 2 + (cy - a["cy"]) ** 2 < 45**2 for a in added):
                        continue
                    # 与已占位掩膜重叠过大 → 同腔碎片
                    comp = (labels == i).astype(np.uint8) * 255
                    overlap = cv2.bitwise_and(comp, occupied)
                    if float(np.count_nonzero(overlap)) > 0.28 * max(area, 1):
                        continue
                    # 须在锚点旁侧，避免同腔碎片
                    if (cx - v.cx) ** 2 + (cy - v.cy) ** 2 < (0.30 * max(bw, bh)) ** 2:
                        continue
                    got = _from_mask(comp, th, min_area * 0.55, max_area, t)
                    if got is None:
                        got = _from_ellipse_recovery(
                            gray, comp, tissue, th, min_area * 0.55, max_area, t
                        )
                    if got is None:
                        continue
                    bad, _, _ = evaluate_exclusion(
                        got["contour"], img_w, img_h, tissue_h=th
                    )
                    if bad:
                        continue
                    gap = _min_contour_dist_px(v.contour, got["contour"]) * um_per_px
                    if gap < min_gap_um or gap > max_gap_um:
                        continue
                    ai_a = ai_anchor
                    ai_b = float(got["area_px"]) * (um_per_px**2)
                    if ai_b < 180.0:
                        continue
                    ratio = max(ai_a, ai_b) / max(min(ai_a, ai_b), 1e-6)
                    if ratio > 6.0:
                        continue
                    # 圆形度门槛略降：腔内碎屑时 circ 常偏低但仍是真腔
                    if got["circ"] < 0.38:
                        continue
                    # 优先圆形度高的完整腔，避免高阈值把邻腔扩进纤维（4(4) T=88 虚大）
                    key = (-float(got["circ"]), gap, abs(np.log(ratio)))
                    if local_best_key is None or key < local_best_key:
                        local_best_key = key
                        local_best = got

        if local_best is not None:
            added.append(local_best)
            existing.append(
                Vessel(
                    vessel_id="tmp",
                    image_name=image_name,
                    contour=local_best["contour"],
                    area_px=local_best["area_px"],
                    peri_px=local_best["peri_px"],
                    cx=local_best["cx"],
                    cy=local_best["cy"],
                    bbox=local_best["bbox"],
                    status="primary",
                    reason="adjacent_recover",
                    mask=local_best.get("mask"),
                )
            )
            # 占位，避免下一锚点重复检到同一块
            if local_best.get("mask") is not None:
                occupied = cv2.bitwise_or(
                    occupied, (local_best["mask"] > 0).astype(np.uint8) * 255
                )

    if not added:
        return vessels

    m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
    prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
    merged = list(vessels)
    for c in added:
        merged.append(
            Vessel(
                vessel_id="tmp",
                image_name=image_name,
                contour=c["contour"],
                area_px=c["area_px"],
                peri_px=c["peri_px"],
                cx=c["cx"],
                cy=c["cy"],
                bbox=c["bbox"],
                status="primary",
                reason=f"adjacent_recover;T={c['t']};circ={c['circ']:.2f}",
                mask=c.get("mask"),
            )
        )
    merged.sort(key=lambda z: (z.cy, z.cx))
    out: list[Vessel] = []
    for i, v in enumerate(merged, start=1):
        v.vessel_id = f"{prefix}-V{i:02d}"
        out.append(v)
    return out


def attach_metrics(vessels: list[Vessel], um_per_px: float) -> list[Vessel]:
    for v in vessels:
        v.ai_um2 = v.area_px * (um_per_px**2)
        v.di_um = 2.0 * np.sqrt(v.ai_um2 / np.pi)
        v.peri_um = v.peri_px * um_per_px
    return vessels


def shrink_contour_toward_center(contour: np.ndarray, scale: float) -> np.ndarray:
    """将轮廓点向形心缩放（scale<1 内缩），保持形状。"""
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return np.asarray(contour)
    c = pts.mean(axis=0)
    out = (pts - c) * float(scale) + c
    return out.astype(np.int32).reshape(-1, 1, 2)


def _apply_contour_to_vessel(v: Vessel, contour: np.ndarray) -> None:
    contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
    v.contour = contour
    v.area_px = float(cv2.contourArea(contour))
    v.peri_px = float(cv2.arcLength(contour, True))
    m = cv2.moments(contour)
    if m["m00"] > 1e-6:
        v.cx = float(m["m10"] / m["m00"])
        v.cy = float(m["m01"] / m["m00"])
    x, y, bw, bh = cv2.boundingRect(contour)
    v.bbox = (x, y, bw, bh)
    v.mask = None


def ensure_pair_lumen_gap(
    v1: Vessel,
    v2: Vessel,
    um_per_px: float,
    min_gap_px: float = 8.0,
    min_scale: float = 0.82,
) -> float:
    """
    若双导管两腔轮廓过近/贴合/相交，同步向形心内缩，直到间距 ≥ min_gap_px。
    返回最终间距（px）。用于绘制与 TVW，避免绿圈贴在一起。
    """
    gap = _min_contour_dist_px(v1.contour, v2.contour)
    if gap >= min_gap_px:
        return gap
    c1_raw = v1.contour.copy()
    c2_raw = v2.contour.copy()
    best_scale = min_scale
    for scale in np.linspace(0.98, min_scale, 18):
        t1 = shrink_contour_toward_center(c1_raw, scale)
        t2 = shrink_contour_toward_center(c2_raw, scale)
        g = _min_contour_dist_px(t1, t2)
        if g >= min_gap_px:
            best_scale = float(scale)
            gap = g
            break
        best_scale = float(scale)
        gap = g
    _apply_contour_to_vessel(v1, shrink_contour_toward_center(c1_raw, best_scale))
    _apply_contour_to_vessel(v2, shrink_contour_toward_center(c2_raw, best_scale))
    attach_metrics([v1, v2], um_per_px)
    if "gap_inset" not in (v1.reason or ""):
        v1.reason = (v1.reason + ";gap_inset") if v1.reason else "gap_inset"
    if "gap_inset" not in (v2.reason or ""):
        v2.reason = (v2.reason + ";gap_inset") if v2.reason else "gap_inset"
    return float(gap)


def _vessel_mask_iou(a: Vessel, b: Vessel, shape: tuple[int, int]) -> float:
    ha, wa = shape
    ma = np.zeros((ha, wa), dtype=np.uint8)
    mb = np.zeros((ha, wa), dtype=np.uint8)
    if a.mask is not None and a.mask.shape[:2] == (ha, wa):
        ma = (a.mask > 0).astype(np.uint8)
    else:
        cv2.drawContours(ma, [a.contour], -1, 1, -1)
    if b.mask is not None and b.mask.shape[:2] == (ha, wa):
        mb = (b.mask > 0).astype(np.uint8)
    else:
        cv2.drawContours(mb, [b.contour], -1, 1, -1)
    inter = int(np.logical_and(ma, mb).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(ma, mb).sum())
    return inter / max(union, 1)


def suppress_nested_vessels(
    vessels: list[Vessel],
    shape: tuple[int, int] | None = None,
    *,
    contain_frac: float = 0.72,
) -> list[Vessel]:
    """
    去掉套在更大腔内的小轮廓（绿圈套绿圈）。
    若较小腔 ≥ contain_frac 的像素落在较大腔内，或中心落在较大轮廓内，则丢弃较小者。
    """
    if len(vessels) < 2:
        return vessels
    if shape is None:
        # 从轮廓估计画布
        max_x = max(int(v.contour[:, :, 0].max()) for v in vessels) + 2
        max_y = max(int(v.contour[:, :, 1].max()) for v in vessels) + 2
        shape = (max_y, max_x)
    h, w = shape[:2]
    ranked = sorted(vessels, key=lambda v: float(v.ai_um2 or 0), reverse=True)
    masks: list[np.ndarray] = []
    areas: list[int] = []
    for v in ranked:
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [v.contour], -1, 255, -1)
        masks.append(m)
        areas.append(int(cv2.countNonZero(m)))
    drop: set[int] = set()
    for i in range(len(ranked)):
        if i in drop:
            continue
        for j in range(i + 1, len(ranked)):
            if j in drop:
                continue
            # ranked[i] larger (or equal); test if j nested in i
            inter = int(cv2.countNonZero(cv2.bitwise_and(masks[i], masks[j])))
            if areas[j] > 0 and inter / areas[j] >= contain_frac:
                drop.add(j)
                continue
            # center of smaller inside larger contour
            if (
                cv2.pointPolygonTest(
                    ranked[i].contour, (float(ranked[j].cx), float(ranked[j].cy)), False
                )
                >= 0
            ):
                drop.add(j)
    return [v for idx, v in enumerate(ranked) if idx not in drop]


def merge_vessel_candidates(
    primary: list[Vessel],
    secondary: list[Vessel],
    shape: tuple[int, int],
    iou_thr: float = 0.35,
    image_name: str = "",
) -> list[Vessel]:
    """
    并集合并：primary 优先保留；secondary 中与已有 IoU 低的补入（召回补漏）。
    """
    kept = list(primary)
    for s in secondary:
        if any(_vessel_mask_iou(s, k, shape) >= iou_thr for k in kept):
            continue
        # 中心过近也视为同一腔
        if any((s.cx - k.cx) ** 2 + (s.cy - k.cy) ** 2 < 36**2 for k in kept):
            continue
        # 小腔中心落在已有大腔内 → 套圈，跳过
        if any(
            cv2.pointPolygonTest(k.contour, (float(s.cx), float(s.cy)), False) >= 0
            for k in kept
        ):
            continue
        kept.append(s)
    kept = suppress_nested_vessels(kept, shape)
    m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
    prefix = f"{m.group(1)}({m.group(2)})" if m else (image_name or "img")
    kept.sort(key=lambda z: (z.cy, z.cx))
    out: list[Vessel] = []
    for i, v in enumerate(kept, start=1):
        v.vessel_id = f"{prefix}-V{i:02d}"
        v.pair_id = None
        out.append(v)
    return out


def segment_hybrid(
    gray: np.ndarray,
    um_per_px: float,
    image_name: str,
    rgb: np.ndarray | None = None,
    weights: str | Path | None = None,
    dl_conf: float = 0.25,
    log_excluded: bool | None = None,
) -> list[Vessel]:
    """
    DL 腔检出为主 + 经典多阈值补漏 → exclusion/邻腔补检已在 classic 内完成；
    合并后再做一次邻腔补检与椭圆平滑。
    """
    classic = segment_dark_lumens(
        gray, um_per_px=um_per_px, image_name=image_name, log_excluded=log_excluded
    )
    dl_vessels: list[Vessel] = []
    try:
        from infer_lumen_seg import default_weights, predict_lumen_vessels

        wpath = weights or default_weights()
        if wpath is not None:
            src = rgb if rgb is not None else gray
            dl_raw = predict_lumen_vessels(
                src,
                image_name=image_name,
                um_per_px=um_per_px,
                weights=wpath,
                conf=dl_conf,
            )
            # DL 结果也走 exclusion，避免标尺/裂隙假阳性
            xcfg = get_exclusion_config()
            th = tissue_height(gray.shape[0])
            img_h, img_w = gray.shape[:2]
            for v in dl_raw:
                bad, _, _ = evaluate_exclusion(
                    v.contour, img_w, img_h, tissue_h=th, cfg=xcfg
                )
                if not bad:
                    dl_vessels.append(v)
    except Exception as exc:  # noqa: BLE001 — 无权重/无 ultralytics 时回退经典
        print(f"[hybrid] DL 跳过: {exc}")
        return classic

    if not dl_vessels:
        return classic

    # DL 优先，经典补漏
    merged = merge_vessel_candidates(
        dl_vessels, classic, gray.shape[:2], iou_thr=0.35, image_name=image_name
    )
    merged = recover_adjacent_pair_partners(
        gray, merged, um_per_px=um_per_px, image_name=image_name
    )
    merged = refine_vessels_to_ellipse(merged, gray)
    merged = suppress_nested_vessels(merged, gray.shape[:2])
    return merged


def _box_aabb(box, pad: float = 30.0) -> tuple[int, int, int, int]:
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


def recover_vessels_in_human_boxes(
    gray: np.ndarray,
    vessels: list[Vessel],
    boxes,
    um_per_px: float,
    image_name: str,
    guide_lines=None,
    min_ai_um2: float = 90.0,
    max_ai_um2: float = 1500.0,
) -> list[Vessel]:
    """
    人工蓝框内若不足 2 个腔：在框 ROI 内用更低阈值补检漏腔
    （常见情况：共同壁一侧小腔/碎屑腔被全局 min_ai 滤掉）。
    """
    if not boxes:
        return list(vessels)

    tissue = tissue_mask(gray)
    th = tissue_height(gray.shape[0])
    min_area = min_ai_um2 / (um_per_px**2)
    max_area = max_ai_um2 / (um_per_px**2)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]

    existing = list(vessels)
    added: list[dict] = []

    for box in boxes:
        inside = [v for v in existing if box.contains_point(v.cx, v.cy, pad=22.0)]
        if len(inside) >= 2:
            continue

        # 引导点：框内人工短线中点（偏向共同壁）
        seeds = []
        for p0, p1 in guide_lines or []:
            mid = 0.5 * (np.asarray(p0, float) + np.asarray(p1, float))
            if box.contains_point(float(mid[0]), float(mid[1]), pad=40.0):
                seeds.append(mid)
        if not seeds:
            seeds = [np.array([box.cx, box.cy], dtype=float)]

        x0, y0, x1, y1 = _box_aabb(box, pad=36.0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(th, y1)
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue

        local_cands: list[dict] = []
        for t in (40, 48, 56, 64, 72, 80, 88):
            dark = ((blur <= t) & (tissue > 0)).astype(np.uint8) * 255
            roi = np.zeros_like(dark)
            roi[y0:y1, x0:x1] = dark[y0:y1, x0:x1]
            roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=4)
            for i in range(1, n):
                area = float(stats[i, cv2.CC_STAT_AREA])
                if area < min_area * 0.45 or area > max_area * 1.2:
                    continue
                cx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] * 0.5
                cy = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] * 0.5
                if not box.contains_point(float(cx), float(cy), pad=28.0):
                    continue
                # skip already-known centers
                if any((cx - v.cx) ** 2 + (cy - v.cy) ** 2 < 45**2 for v in existing):
                    continue
                if any((cx - c["cx"]) ** 2 + (cy - c["cy"]) ** 2 < 40**2 for c in local_cands):
                    continue
                comp = (labels == i).astype(np.uint8) * 255
                got = None
                if area >= min_area * 0.55:
                    got = _from_mask(comp, th, min_area, max_area, t)
                if got is None:
                    got = _from_ellipse_recovery(
                        gray, comp, tissue, th, min_area, max_area, t
                    )
                if got is None:
                    continue
                # prefer near human TVW strokes
                dseed = min(float(np.hypot(got["cx"] - s[0], got["cy"] - s[1])) for s in seeds)
                got["seed_dist"] = dseed
                local_cands.append(got)

        if not local_cands:
            continue
        # take best missing partner(s) near wall strokes
        local_cands.sort(
            key=lambda c: (
                c.get("seed_dist", 1e9),
                -c.get("circ", 0) * np.sqrt(c["area_px"]),
            )
        )
        need = 2 - len(inside)
        for c in local_cands:
            if need <= 0:
                break
            if any((c["cx"] - v.cx) ** 2 + (c["cy"] - v.cy) ** 2 < 45**2 for v in existing):
                continue
            # must sit on the other side of wall relative to an existing in-box vessel if any
            if inside:
                ok_side = False
                for v0 in inside:
                    for s in seeds:
                        # candidate and known vessel on opposite sides of stroke mid
                        a = np.array([v0.cx, v0.cy]) - s
                        b = np.array([c["cx"], c["cy"]]) - s
                        if float(np.dot(a, b)) < 0:
                            ok_side = True
                            break
                    if ok_side:
                        break
                if not ok_side and min(c.get("seed_dist", 1e9), 1e9) > 70:
                    continue
            added.append(c)
            # temporary Vessel for clash checks
            existing.append(
                Vessel(
                    vessel_id="tmp",
                    image_name=image_name,
                    contour=c["contour"],
                    area_px=c["area_px"],
                    peri_px=c["peri_px"],
                    cx=c["cx"],
                    cy=c["cy"],
                    bbox=c["bbox"],
                    status="primary",
                    reason="human_box_recover",
                    mask=c.get("mask"),
                )
            )
            need -= 1

    if not added:
        return list(vessels)

    # rebuild ids after merge
    merged = list(vessels)
    for c in added:
        merged.append(
            Vessel(
                vessel_id="pending",
                image_name=image_name,
                contour=c["contour"],
                area_px=c["area_px"],
                peri_px=c["peri_px"],
                cx=c["cx"],
                cy=c["cy"],
                bbox=c["bbox"],
                status="primary",
                reason=f"human_box_recover;T={c['t']};circ={c['circ']:.2f}",
                mask=c.get("mask"),
            )
        )
    merged = attach_metrics(merged, um_per_px)
    merged.sort(key=lambda z: (z.cy, z.cx))
    out: list[Vessel] = []
    for idx, v in enumerate(merged, start=1):
        m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
        prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
        v.vessel_id = f"{prefix}-V{idx:02d}"
        v.pair_id = None
        out.append(v)
    return out
