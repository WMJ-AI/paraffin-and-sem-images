"""
通用规则：自动标定（运行时不依赖按文件名的指南）。

- Ai: 不规则闭合轮廓面积
- 自动框选: 几何找相邻对，可保留多对
- TVW: 3–5 条最近壁间距离，主值=均值
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from pair_tvw import TVWLine, VesselPair, find_pairs
from segment import Vessel

LEARNED_JSON = Path(__file__).resolve().parent / "output" / "learned_rules.json"


@dataclass
class HumanRuleConfig:
    min_gap_um: float = 1.2
    max_gap_um: float = 22.0
    max_ai_ratio: float = 3.5
    min_pair_ai_um2: float = 200.0
    min_tvw_um: float = 3.5
    max_tvw_um: float = 22.0
    # 单导管绿线门槛（教学小腔×0.8）；配对成员另受 min_pair_ai_um2 约束
    min_single_ai_um2: float = 102.0
    max_singles: int = 6
    edge_margin_px: float = 35.0
    # 自动多导管区：默认允许多对（学自人工 1–3 框）
    one_multi_per_image: bool = False
    max_multis_per_image: int = 3  # 一图可多框；与 learned_rules 同步
    second_pair_score_ratio: float = 0.60
    # 勿强制每图至少一对：否则远距碎腔会被 nearest 凑成假双导管（如 13(8)）
    require_box_per_image: bool = False
    # 双导管共壁带内若有小暗腔 → 不算双导管（仅单导管）
    reject_pair_with_interstitial_lumen: bool = True
    interstitial_min_cavity_ai_um2: float = 3.0
    interstitial_max_cavity_ai_um2: float = 120.0
    # 导管腔面积下限 = 教学小腔 Ai 的 80%（见 ref_small_lumen_ai_um2）
    min_lumen_ai_um2: float = 102.0
    # 够黑门控（不同 SEM 对比度下需可调；过严会导致整批 v=0）
    require_deep_black_lumen: bool = True
    max_lumen_core_p50: float = 52.0
    min_lumen_very_dark_frac: float = 0.15
    min_lumen_dark_frac: float = 0.35
    min_lumen_ring_contrast: float = 12.0
    # 高反差兜底：core 略亮但相对壁够暗时仍保留
    soft_max_core_p50: float = 58.0
    soft_min_dark_frac: float = 0.40
    soft_min_contrast: float = 25.0


def load_learned_rules(path: Path = LEARNED_JSON) -> HumanRuleConfig:
    """加载 learn_general_rules.py 产出的通用参数；无文件则用默认。"""
    base = HumanRuleConfig()
    if not path.exists():
        return base
    data = json.loads(path.read_text(encoding="utf-8"))
    kwargs = {}
    names = {f.name for f in fields(HumanRuleConfig)}
    for k, v in data.items():
        if k in names:
            kwargs[k] = v
    return HumanRuleConfig(**{**base.__dict__, **kwargs})


DEFAULT_RULES = load_learned_rules()


def _pair_priority(p: VesselPair, img_w: int, img_h: int) -> float:
    """Higher = more like a human-chosen multi-vessel region.
    三管邻近时优先共同壁更短的一对（gap/TVW 越小越好）。
    """
    cx = 0.5 * (p.v1.cx + p.v2.cx)
    cy = 0.5 * (p.v1.cy + p.v2.cy)
    dist_c = np.hypot(cx - img_w / 2, cy - img_h / 2) / max(img_w, img_h)
    gap = float(np.median([ln.length_px for ln in p.lines])) if p.lines else 40.0
    tvw = p.tvw_primary_um
    size = min(p.v1.ai_um2 or 0, p.v2.ai_um2 or 0)
    n_lines = len(p.lines)
    # 壁越近、腔越大 → 越优先成双；不再偏好 TVW≈7.5
    return float(size) - 40.0 * tvw - 0.5 * gap - 200.0 * dist_c + 4.0 * n_lines


def select_primary_pairs(
    pairs: list[VesselPair],
    img_w: int,
    img_h: int,
    rules: HumanRuleConfig = DEFAULT_RULES,
) -> list[VesselPair]:
    """自动框选：按优先级保留互不占用腔的多对（上限 max_multis）。"""
    if not pairs:
        return []
    # 已过 gap/Ai/TVW 几何过滤的对：按优先级保留互不占用腔的对，直到上限。
    # second_pair_score_ratio 仅作极弱相对门槛（默认 0.35），避免刷掉完整第二对。
    ranked = sorted(pairs, key=lambda p: _pair_priority(p, img_w, img_h), reverse=True)
    kept: list[VesselPair] = []
    used: set[str] = set()
    top_score: float | None = None
    # 学参常为 0.6，过严会挤掉真第二对；运行时下限 0.35
    ratio_thr = min(float(rules.second_pair_score_ratio), 0.35)
    for p in ranked:
        ids = {p.v1.vessel_id, p.v2.vessel_id}
        if ids & used:
            continue
        score = _pair_priority(p, img_w, img_h)
        if top_score is None:
            top_score = score
        elif score < top_score * ratio_thr:
            # 相对榜首过弱才丢弃；完整但略小的第二对通常仍高于 0.35×
            continue
        kept.append(p)
        used |= ids
        if len(kept) >= rules.max_multis_per_image:
            break
    return kept


def select_single_vessels(
    vessels: list[Vessel],
    pairs: list[VesselPair],
    img_w: int,
    img_h: int,
    tissue_h: int | None = None,
    rules: HumanRuleConfig = DEFAULT_RULES,
) -> list[Vessel]:
    paired = {p.v1.vessel_id for p in pairs} | {p.v2.vessel_id for p in pairs}
    th = tissue_h or int(round(img_h * 691 / 768))
    cands: list[Vessel] = []
    for v in vessels:
        if v.vessel_id in paired:
            continue
        if (v.ai_um2 or 0) < rules.min_single_ai_um2:
            continue
        if (
            v.cx < rules.edge_margin_px
            or v.cy < rules.edge_margin_px
            or v.cx > img_w - rules.edge_margin_px
            or v.cy > th - rules.edge_margin_px
        ):
            continue
        cands.append(v)
    cands.sort(key=lambda v: v.ai_um2 or 0, reverse=True)
    return cands[: rules.max_singles]


def filter_vessels_by_min_ai(
    vessels: list[Vessel],
    rules: HumanRuleConfig | None = None,
    gray: np.ndarray | None = None,
    um_per_px: float | None = None,
) -> list[Vessel]:
    """剔除 Ai 过低碎腔、绿圈套绿圈、小腔团假腔、非够黑假腔。"""
    rules = rules or DEFAULT_RULES
    floor = float(
        getattr(rules, "min_lumen_ai_um2", None)
        or rules.min_pair_ai_um2
        or 102.0
    )
    kept = [v for v in vessels if (v.ai_um2 or 0) >= floor]
    from segment import suppress_nested_vessels

    kept = suppress_nested_vessels(kept)
    if gray is not None and um_per_px is not None and um_per_px > 0:
        from cavity_cluster import filter_multi_cavity_clusters
        from lumen_darkness import filter_deep_black_lumens

        kept = filter_multi_cavity_clusters(kept, gray, float(um_per_px))
        if getattr(rules, "require_deep_black_lumen", True):
            kept = filter_deep_black_lumens(
                kept,
                gray,
                max_core_p50=float(getattr(rules, "max_lumen_core_p50", 52.0)),
                min_very_dark_frac=float(getattr(rules, "min_lumen_very_dark_frac", 0.15)),
                min_dark_frac=float(getattr(rules, "min_lumen_dark_frac", 0.35)),
                min_contrast=float(getattr(rules, "min_lumen_ring_contrast", 12.0)),
                soft_max_core_p50=float(getattr(rules, "soft_max_core_p50", 58.0)),
                soft_min_dark_frac=float(getattr(rules, "soft_min_dark_frac", 0.40)),
                soft_min_contrast=float(getattr(rules, "soft_min_contrast", 25.0)),
            )
    return kept



def find_pairs_auto(
    vessels: list[Vessel],
    um_per_px: float,
    image_name: str,
    img_w: int,
    img_h: int,
    rules: HumanRuleConfig | None = None,
    gray: np.ndarray | None = None,
) -> list[VesselPair]:
    """客户/生产入口：纯通用规则自动框选+TVW，不读人工指南。"""
    return find_pairs_by_human_rules(
        vessels,
        um_per_px,
        image_name,
        img_w=img_w,
        img_h=img_h,
        rules=rules or DEFAULT_RULES,
        yellow_guide=None,
        gray=gray,
    )


def find_pairs_by_human_rules(
    vessels: list[Vessel],
    um_per_px: float,
    image_name: str,
    img_w: int,
    img_h: int,
    rules: HumanRuleConfig | None = None,
    yellow_guide=None,
    gray=None,
) -> list[VesselPair]:
    """
    通用几何配对 + TVW。
    yellow_guide 仅内部复核/学习时可选；生产请用 find_pairs_auto / yellow_guide=None。
    """
    rules = rules or DEFAULT_RULES
    if yellow_guide is not None and getattr(yellow_guide, "force_zero_pair", False):
        for v in vessels:
            v.pair_id = None
        return []

    def _collect(max_gap, max_tvw, min_iface=5, max_ai_ratio=None, min_ai=None, min_tvw=None):
        ai_ratio = rules.max_ai_ratio if max_ai_ratio is None else max_ai_ratio
        min_ai_u = rules.min_pair_ai_um2 if min_ai is None else min_ai
        min_tvw_u = rules.min_tvw_um if min_tvw is None else min_tvw
        raw = find_pairs(
            vessels,
            um_per_px=um_per_px,
            image_name=image_name,
            min_gap_um=rules.min_gap_um,
            max_gap_um=max_gap,
            min_interface_points=min_iface,
            min_tvw_um=min_tvw_u,
            max_tvw_um=max_tvw,
            max_ai_ratio=ai_ratio,
            min_pair_ai_um2=min_ai_u,
        )
        out: list[VesselPair] = []
        for p in raw:
            a1, a2 = p.v1.ai_um2 or 0, p.v2.ai_um2 or 0
            if min(a1, a2) < min_ai_u:
                continue
            if max(a1, a2) / max(min(a1, a2), 1e-6) > ai_ratio:
                continue
            out.append(p)
        return out

    def _nearest_fallback() -> list[VesselPair]:
        if len(vessels) < 2:
            return []
        best = None
        for i, v1 in enumerate(vessels):
            pts1 = v1.contour.reshape(-1, 2).astype(float)
            for j, v2 in enumerate(vessels):
                if j <= i:
                    continue
                pts2 = v2.contour.reshape(-1, 2).astype(float)
                dmin, i1, i2 = 1e18, 0, 0
                for a, p in enumerate(pts1[::4]):
                    dd = np.sum((pts2[::4] - p) ** 2, axis=1)
                    k = int(np.argmin(dd))
                    d = float(dd[k])
                    if d < dmin:
                        dmin, i1, i2 = d, a * 4, k * 4
                dist = float(np.sqrt(dmin))
                if best is None or dist < best[0]:
                    best = (dist, v1, v2, pts1[i1], pts2[i2])
        assert best is not None
        dist, v1, v2, p1, p2 = best
        m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
        prefix = f"{m.group(1)}({m.group(2)})" if m else image_name
        pid = f"{prefix}-P01"
        tvw_um = float(np.clip(dist * um_per_px, 1.5, 45.0))
        line = TVWLine(
            pair_id=pid,
            image_name=image_name,
            line_id="L1",
            x0=float(p1[0]),
            y0=float(p1[1]),
            x1=float(p2[0]),
            y1=float(p2[1]),
            length_px=dist,
            um_per_px=um_per_px,
            tvw_um=tvw_um,
            quantile=0.5,
        )
        d1, d2 = v1.di_um or 1.0, v2.di_um or 1.0
        cwr1 = (tvw_um / d1) ** 2
        cwr2 = (tvw_um / d2) ** 2
        v1.pair_id = pid
        v2.pair_id = pid
        return [
            VesselPair(
                pair_id=pid,
                image_name=image_name,
                v1=v1,
                v2=v2,
                lines=[line],
                tvw_median_um=tvw_um,
                tvw_mean_um=tvw_um,
                tvw_sd_um=0.0,
                cwr1=cwr1,
                cwr2=cwr2,
                cwr_pair=0.5 * (cwr1 + cwr2),
            )
        ]

    # 先标准门槛；若无对则略降共壁点数门槛再试（漏检_有腔未配上）
    kept = _collect(rules.max_gap_um, rules.max_tvw_um, min_iface=5)
    if not kept and len(vessels) >= 2:
        kept = _collect(rules.max_gap_um, rules.max_tvw_um, min_iface=3)
    has_yellow_boxes = bool(
        yellow_guide is not None and getattr(yellow_guide, "boxes", None)
    )

    if (
        not kept
        and rules.require_box_per_image
        and len(vessels) >= 2
        and not has_yellow_boxes
    ):
        kept = _collect(
            max_gap=min(45.0, rules.max_gap_um * 2.0),
            max_tvw=min(45.0, rules.max_tvw_um * 2.0),
            min_iface=3,
            max_ai_ratio=max(8.0, rules.max_ai_ratio),
            min_ai=max(80.0, rules.min_pair_ai_um2 * 0.5),
            min_tvw=max(1.2, rules.min_tvw_um * 0.5),
        )
    if (
        not kept
        and rules.require_box_per_image
        and len(vessels) >= 2
        and not has_yellow_boxes
    ):
        kept = _nearest_fallback()

    if has_yellow_boxes:
        # 仅内部学习/对照：可选人工框
        from learn_from_review import (
            filter_valid_multi_boxes,
            pair_from_two_largest_in_boxes,
        )
        from segment import recover_vessels_in_human_boxes

        boxes = filter_valid_multi_boxes(list(yellow_guide.boxes))
        yellow_guide.boxes = boxes
        if gray is not None and boxes:
            recovered = recover_vessels_in_human_boxes(
                gray,
                vessels,
                boxes,
                um_per_px,
                image_name,
                guide_lines=getattr(yellow_guide, "lines", None),
            )
            vessels[:] = list(recovered)
        selected = pair_from_two_largest_in_boxes(
            vessels,
            boxes,
            um_per_px,
            image_name,
            pad=55.0,
            guide_lines=getattr(yellow_guide, "lines", None),
        )
    else:
        selected = select_primary_pairs(kept, img_w, img_h, rules)

    if selected:
        from pair_tvw import apply_nearest_tvw_lines
        from segment import ensure_pair_lumen_gap

        # 无人工框时：仅对贴合轮廓做轻度内缩；有人工框则保持检出几何，由人工线定 TVW
        if not has_yellow_boxes:
            for p in selected:
                ensure_pair_lumen_gap(p.v1, p.v2, um_per_px, min_gap_px=4.0)

        selected = apply_nearest_tvw_lines(selected, um_per_px)
        if has_yellow_boxes and yellow_guide is not None:
            from learn_from_review import apply_human_tvw_guides

            selected = apply_human_tvw_guides(
                selected, getattr(yellow_guide, "lines", None), um_per_px
            )
        tvw_lo = 0.2 if has_yellow_boxes else 0.5
        selected = [
            p for p in selected if p.lines and tvw_lo <= p.tvw_mean_um <= 28.0
        ]

    # 共壁带内有小暗腔 → 不成双导管，两侧仅按单导管计
    if (
        selected
        and gray is not None
        and getattr(rules, "reject_pair_with_interstitial_lumen", True)
    ):
        from pair_guards import filter_pairs_without_interstitial

        selected = filter_pairs_without_interstitial(
            selected,
            gray,
            um_per_px,
            min_cavity_ai_um2=float(
                getattr(rules, "interstitial_min_cavity_ai_um2", 3.0)
            ),
            max_cavity_ai_um2=float(
                getattr(rules, "interstitial_max_cavity_ai_um2", 120.0)
            ),
            all_vessels=vessels,
        )

    sel_ids = {p.v1.vessel_id for p in selected} | {p.v2.vessel_id for p in selected}
    for v in vessels:
        if v.vessel_id not in sel_ids:
            v.pair_id = None
    return selected
