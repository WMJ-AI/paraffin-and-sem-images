"""
通用排除规则（几何阈值，与文件名/历史 CSV 无关）。

运行时只读 output/learned_rules.json 的 exclusion 段；
绝不读取 output/排除_裂隙状腔.csv（该文件若存在仅为可选审计日志）。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

LEARNED_JSON = Path(__file__).resolve().parent / "output" / "learned_rules.json"


@dataclass
class ExclusionConfig:
    """换图仍有效的通用排除阈值（不绑定具体样品坐标）。"""

    # —— 面积门（教学小腔 Ai×0.8 ≈ 102）—— #
    min_ai_um2: float = 102.0
    max_ai_um2: float = 1500.0
    min_single_ai_um2: float = 102.0

    # —— 图框截断 —— #
    border_margin_px: int = 30

    # —— 细长裂隙（非导管） —— #
    crack_aspect_hi: float = 2.2
    crack_ell_ar_hi: float = 2.6
    crack_solid_lo: float = 0.78
    crack_aspect_vhi: float = 2.5
    crack_circ_lo: float = 0.28
    crack_aspect_mid: float = 2.0
    crack_solid_vlo: float = 0.62
    crack_circ_vlo: float = 0.22

    # —— 锯齿/碎屑/渗漏假腔 —— #
    serr_rough_hi: float = 1.40
    serr_circ_lo: float = 0.40
    serr_rough_mid: float = 1.45
    serr_solid_lo: float = 0.72
    serr_circ_mid: float = 0.35
    serr_mdef_r_hi: float = 1.2
    serr_circ_for_mdef: float = 0.30
    serr_rough_soft: float = 1.35
    serr_circ_soft: float = 0.45
    serr_solid_soft: float = 0.90
    serr_rough_vhi: float = 1.48
    serr_circ_for_vhi: float = 0.50
    leak_solid_lo: float = 0.82
    leak_circ_lo: float = 0.45
    leak_rough_lo: float = 1.20
    long_aspect_hi: float = 1.90
    long_rough_lo: float = 1.30
    long_circ_lo: float = 0.50
    long_solid_lo: float = 0.88

    # —— 审计 —— #
    # CSV 仅可选写出，从不参与判定
    write_exclusion_log: bool = False


# 规则说明（写入 JSON / 文档，供人读；代码用阈值判定）
EXCLUSION_RULE_DOCS: list[dict] = [
    {
        "id": "E0_area",
        "name": "面积门",
        "reject_if": "Ai < min_ai_um2 或 Ai > max_ai_um2",
        "keep_note": "完整导管腔通常落在此区间",
    },
    {
        "id": "E1_border",
        "name": "图框截断",
        "reject_if": "轮廓 bbox 距图像/组织边界 ≤ border_margin_px",
        "keep_note": "必须完整才计入",
    },
    {
        "id": "E2_crack",
        "name": "细长裂隙",
        "reject_if": (
            "(aspect≥crack_aspect_hi 且 ell_ar≥crack_ell_ar_hi 且 solid<crack_solid_lo) 或 "
            "(aspect≥crack_aspect_vhi 且 circ<crack_circ_lo) 或 "
            "(aspect≥crack_aspect_mid 且 solid<crack_solid_vlo 且 circ<crack_circ_vlo)"
        ),
        "keep_note": "层间暗缝、非导管腔",
    },
    {
        "id": "E3_serrated",
        "name": "锯齿碎屑/渗漏假腔",
        "reject_if": (
            "rough/circ/solid/mdef_r 组合超阈（见 exclusion 字段 serr_* / leak_* / long_*）；"
            "典型：贴碎屑走的高频锯齿绿边、渗入纤维缝的齿状块"
        ),
        "keep_note": "略有缺口但仍类圆、高实心度的完整腔保留；随后可椭圆平滑绘制",
    },
    {
        "id": "E4_single_size",
        "name": "单导管绘制门",
        "reject_if": "未配对且 Ai < min_single_ai_um2（不画绿线、不进单导管统计）",
        "keep_note": "避免过小碎腔当单导管",
    },
]


def default_exclusion_dict() -> dict:
    cfg = ExclusionConfig()
    return {
        "description": (
            "通用排除规则：仅按轮廓几何（面积、圆形度、实心度、周长粗糙度、长宽比、是否贴边）判定；"
            "与文件名无关，不读取排除_裂隙状腔.csv"
        ),
        "source": "几何通用规则（非逐图黑名单）",
        "params": asdict(cfg),
        "rules": EXCLUSION_RULE_DOCS,
    }


def load_exclusion_config(path: Path = LEARNED_JSON) -> ExclusionConfig:
    base = ExclusionConfig()
    if not path.exists():
        return base
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return base
    block = data.get("exclusion") or {}
    params = block.get("params") if isinstance(block, dict) else None
    if not isinstance(params, dict):
        # 兼容：若将来把字段摊平到根级
        params = {k: data[k] for k in (f.name for f in fields(ExclusionConfig)) if k in data}
    kwargs = {}
    names = {f.name for f in fields(ExclusionConfig)}
    for k, v in params.items():
        if k in names:
            kwargs[k] = v
    return ExclusionConfig(**{**asdict(base), **kwargs})


_CACHED: ExclusionConfig | None = None


def get_exclusion_config(reload: bool = False) -> ExclusionConfig:
    global _CACHED
    if _CACHED is None or reload:
        _CACHED = load_exclusion_config()
    return _CACHED


def lumen_shape_metrics(contour: np.ndarray) -> dict:
    """bbox 长宽比、椭圆比、实心度、圆形度、周长粗糙度、相对深凹。"""
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    hull = cv2.convexHull(contour)
    solid = area / max(float(cv2.contourArea(hull)), 1.0)
    circ = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
    hull_peri = float(cv2.arcLength(hull, True))
    rough = peri / max(hull_peri, 1e-6)
    hi = cv2.convexHull(contour, returnPoints=False)
    defects = (
        cv2.convexityDefects(contour, hi) if hi is not None and len(hi) >= 3 else None
    )
    max_def = 0.0
    if defects is not None:
        for d in defects[:, 0]:
            depth = float(d[3]) / 256.0
            if depth > 3.0:
                max_def = max(max_def, depth)
    eq_r = float(np.sqrt(area / np.pi)) if area > 0 else 1.0
    mdef_r = max_def / max(eq_r, 1e-6)
    if len(contour) >= 5:
        _center, (ma, MA), _ang = cv2.fitEllipse(contour)
        ell_ar = max(float(ma), float(MA)) / max(min(float(ma), float(MA)), 1e-6)
    else:
        ell_ar = aspect
    return {
        "aspect": float(aspect),
        "ell_ar": float(ell_ar),
        "solid": float(solid),
        "circ": float(circ),
        "rough": float(rough),
        "mdef_r": float(mdef_r),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
    }


def is_crack_like_void(
    contour: np.ndarray, cfg: ExclusionConfig | None = None
) -> tuple[bool, str, dict]:
    """细长裂隙/碎裂暗缝（非导管）。"""
    cfg = cfg or get_exclusion_config()
    m = lumen_shape_metrics(contour)
    a, e, s, c = m["aspect"], m["ell_ar"], m["solid"], m["circ"]
    if a >= cfg.crack_aspect_hi and e >= cfg.crack_ell_ar_hi and s < cfg.crack_solid_lo:
        return True, "细长裂隙(长宽比+椭圆比+低实心度)", m
    if a >= cfg.crack_aspect_vhi and c < cfg.crack_circ_lo:
        return True, "细长裂隙(长宽比+低圆形度)", m
    if a >= cfg.crack_aspect_mid and s < cfg.crack_solid_vlo and c < cfg.crack_circ_vlo:
        return True, "细长裂隙(低实心度+低圆形度)", m
    return False, "", m


def is_serrated_debris_void(
    contour: np.ndarray, cfg: ExclusionConfig | None = None
) -> tuple[bool, str, dict]:
    """锯齿碎屑轮廓 / 渗漏纤维缝假腔。"""
    cfg = cfg or get_exclusion_config()
    m = lumen_shape_metrics(contour)
    r, s, c, d = m["rough"], m["solid"], m["circ"], m["mdef_r"]
    a, e = m["aspect"], m["ell_ar"]
    # 高实心度完整腔：阈值毛刺会使 rough 偏高、circ 偏低，随后可椭圆平滑，不宜当碎屑剔除
    solid_ok = s >= 0.85
    if r >= cfg.serr_rough_hi and c < cfg.serr_circ_lo and not solid_ok:
        return True, "锯齿碎屑腔(高周长比+低圆形度)", m
    if r >= cfg.serr_rough_mid and s < cfg.serr_solid_lo and c < cfg.serr_circ_mid:
        return True, "锯齿碎屑腔(粗糙+低实心度)", m
    if d >= cfg.serr_mdef_r_hi and c < cfg.serr_circ_for_mdef and not solid_ok:
        return True, "锯齿碎屑腔(深凹缺陷)", m
    if r >= cfg.serr_rough_soft and c < cfg.serr_circ_soft and s < 0.85:
        return True, "锯齿碎屑腔(毛刺非类圆)", m
    if r >= cfg.serr_rough_vhi and c < cfg.serr_circ_for_vhi and not solid_ok:
        return True, "锯齿碎屑腔(强毛刺)", m
    if s < cfg.leak_solid_lo and c < cfg.leak_circ_lo and r >= cfg.leak_rough_lo:
        return True, "渗漏低实心度腔", m
    if (
        max(a, e) >= cfg.long_aspect_hi
        and r >= cfg.long_rough_lo
        and c < cfg.long_circ_lo
        and s < cfg.long_solid_lo
    ):
        return True, "细长锯齿非导管", m
    return False, "", m


def is_border_truncated(
    contour: np.ndarray,
    img_w: int,
    img_h: int,
    tissue_h: int | None = None,
    margin: int | None = None,
    cfg: ExclusionConfig | None = None,
) -> tuple[bool, str, dict]:
    """被图像/组织边界截断的不完整腔。"""
    cfg = cfg or get_exclusion_config()
    margin = cfg.border_margin_px if margin is None else margin
    th = tissue_h if tissue_h is not None else img_h
    m = lumen_shape_metrics(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    pts = contour.reshape(-1, 2)
    sides = []
    if y <= margin:
        sides.append("top")
    if x <= margin:
        sides.append("left")
    if x + bw >= img_w - 1 - margin:
        sides.append("right")
    if y + bh >= th - 1 - margin:
        sides.append("bottom")
    if not sides:
        return False, "", m
    flat = 0.0
    if "top" in sides:
        band = pts[pts[:, 1] <= y + 4]
        if len(band):
            flat = max(flat, float(band[:, 0].max() - band[:, 0].min()) / max(bw, 1))
    if "bottom" in sides:
        band = pts[pts[:, 1] >= y + bh - 5]
        if len(band):
            flat = max(flat, float(band[:, 0].max() - band[:, 0].min()) / max(bw, 1))
    if "left" in sides:
        band = pts[pts[:, 0] <= x + 4]
        if len(band):
            flat = max(flat, float(band[:, 1].max() - band[:, 1].min()) / max(bh, 1))
    if "right" in sides:
        band = pts[pts[:, 0] >= x + bw - 5]
        if len(band):
            flat = max(flat, float(band[:, 1].max() - band[:, 1].min()) / max(bh, 1))
    side = "+".join(sides)
    # 近边但轮廓完整（非平直截断）的腔保留；仅当边缘带呈截断状或像素真正贴边时剔除
    touches = bool(
        (pts[:, 1] <= 1).any()
        or (pts[:, 0] <= 1).any()
        or (pts[:, 0] >= img_w - 2).any()
        or (pts[:, 1] >= th - 2).any()
    )
    if flat < 0.45 and not touches:
        return False, "", {**m, "flat_frac": round(flat, 3), "edge": side}
    if flat < 0.35 and touches:
        # 轻微贴边但无明显截断平面 → 仍视为完整腔
        return False, "", {**m, "flat_frac": round(flat, 3), "edge": side}
    return True, f"图框截断不完整({side})", {**m, "flat_frac": round(flat, 3), "edge": side}


def evaluate_exclusion(
    contour: np.ndarray,
    img_w: int,
    img_h: int,
    tissue_h: int | None = None,
    cfg: ExclusionConfig | None = None,
) -> tuple[bool, str, dict]:
    """统一入口：任一排除规则命中即剔除。"""
    cfg = cfg or get_exclusion_config()
    bad, rule, sm = is_crack_like_void(contour, cfg)
    if bad:
        return bad, rule, sm
    bad, rule, sm = is_serrated_debris_void(contour, cfg)
    if bad:
        return bad, rule, sm
    return is_border_truncated(contour, img_w, img_h, tissue_h=tissue_h, cfg=cfg)
