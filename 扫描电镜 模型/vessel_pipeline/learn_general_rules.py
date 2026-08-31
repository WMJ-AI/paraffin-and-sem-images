"""
从人工标定笔画归纳**通用规则**（只写参数，不写按图文件名的坐标）。

输出唯一规则文件: output/learned_rules.json
运行时：按图识别 boxes → 配对一致 → 自动画 TVW lines。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from human_rules import find_pairs_by_human_rules
from learn_from_review import filter_valid_multi_boxes, load_human_mark_guides
from pair_tvw import min_contour_distance, _contour_points
from run_from_human import _find_tif, natural_key_name
from run_sample1 import process_image

OUT_JSON = ROOT / "output" / "learned_rules.json"
MARK = ROOT / "output" / "人工标定"
# 优先用人复核修订的双导管区域；若无则退回整棵 人工标定/
MARK_REVISION = next(
    (p for p in MARK.iterdir() if p.is_dir() and "修订" in p.name and "双导管" in p.name),
    None,
)


def main():
    mark_root = MARK_REVISION if MARK_REVISION is not None else MARK
    print(f"学习来源: {mark_root}")
    guides = {
        k: g
        for k, g in load_human_mark_guides(mark_root).items()
        if filter_valid_multi_boxes(g.boxes)
    }

    gaps, ai_mins, ai_ratios, tvws, n_boxes = [], [], [], [], []
    for key, g in sorted(guides.items(), key=lambda kv: natural_key_name(kv[0])):
        g.boxes = filter_valid_multi_boxes(g.boxes)
        path = _find_tif(key)
        if path is None or not g.boxes:
            continue
        rgb, gray, vessels, _, scale = process_image(path)
        um = scale["um_per_px"]
        h, w = gray.shape[:2]
        pairs = find_pairs_by_human_rules(
            vessels, um, path.name, img_w=w, img_h=h, yellow_guide=g, gray=gray
        )
        n_boxes.append(len(g.boxes))
        for p in pairs:
            d, _, _ = min_contour_distance(
                _contour_points(p.v1.contour), _contour_points(p.v2.contour)
            )
            gaps.append(d * um)
            a1, a2 = p.v1.ai_um2 or 0, p.v2.ai_um2 or 0
            ai_mins.append(min(a1, a2))
            if min(a1, a2) > 1:
                ai_ratios.append(max(a1, a2) / min(a1, a2))
            if p.lines:
                tvws.append(p.tvw_mean_um)
        print(f"标定图 boxes={len(g.boxes)} pairs={len(pairs)}")

    if not gaps:
        print("无可用人工对，未写入")
        return

    def q(arr, p):
        return float(np.percentile(arr, p))

    from exclusion_rules import default_exclusion_dict

    # 保留 exclusion / 单导管门槛（不随双导管修订覆盖）
    prev = {}
    if OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    prev_exclusion = prev.get("exclusion")

    src_name = mark_root.name if mark_root != MARK else "人工标定"
    rules = {
        "description": (
            f"通用规则：双导管参数学自人复核「{src_name}」；"
            "按图像几何识别多导管区（boxes），配对后自动画 TVW；与文件名无关"
        ),
        "learned_from": str(mark_root.relative_to(ROOT)) if mark_root.is_relative_to(ROOT) else str(mark_root),
        "workflow": [
            "分割暗腔轮廓，计算 Ai/Di",
            "按 exclusion 通用几何规则剔除裂隙/锯齿/截断/过小碎腔（不读 CSV、不绑文件名）",
            "按 gap/Ai/共同壁几何找候选对并框选多导管区",
            "对每对自动量 3–5 条彼此分离的最短壁间距离作为 TVW",
        ],
        "n_images": len(n_boxes),
        "n_pairs": len(gaps),
        # 分位数学自人复核；下限与 exclusion 面积门对齐，避免把碎腔配成双导管
        "min_gap_um": max(1.2, q(gaps, 5) * 0.85),
        "max_gap_um": min(28.0, max(18.0, q(gaps, 99) * 1.25)),
        "min_pair_ai_um2": max(
            float((prev_exclusion or {}).get("params", {}).get("min_ai_um2", 250.0)) * 0.9,
            q(ai_mins, 5) * 0.85,
        ),
        "max_ai_ratio": min(6.0, max(3.0, q(ai_ratios, 95) * 1.15)) if ai_ratios else 3.5,
        "min_tvw_um": max(2.0, q(tvws, 5) * 0.85) if tvws else 2.0,
        "max_tvw_um": min(28.0, max(16.0, q(tvws, 99) * 1.25)) if tvws else 22.0,
        "max_multis_per_image": int(max(2, min(3, int(max(n_boxes))))),
        "one_multi_per_image": False,
        "second_pair_score_ratio": 0.60,
        "require_box_per_image": False,
        "max_singles": int(prev.get("max_singles", 6)),
        "min_single_ai_um2": float(prev.get("min_single_ai_um2", 320.0)),
        "tvw_n_lines": 5,
        "exclusion": prev_exclusion if isinstance(prev_exclusion, dict) else default_exclusion_dict(),
        "stats": {
            "gap_um_p50": q(gaps, 50),
            "ai_min_p50": q(ai_mins, 50),
            "tvw_p50": q(tvws, 50) if tvws else None,
            "boxes_per_image_p50": float(np.median(n_boxes)),
            "boxes_per_image_max": int(max(n_boxes)),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入唯一规则文件 {OUT_JSON}")
    print(json.dumps(rules, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
