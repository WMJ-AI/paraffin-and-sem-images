"""
基线指标：相对人工修订双导管框的 P/R，以及规则腔检出摘要。

用法:
  python baseline_metrics.py
  python baseline_metrics.py --out output/baseline_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from batch_io import BATCH  # noqa: E402
from human_rules import find_pairs_auto, load_learned_rules  # noqa: E402
from learn_from_review import (  # noqa: E402
    filter_valid_multi_boxes,
    image_key_from_stem,
    load_human_mark_guides,
)
from run_sample1 import process_image  # noqa: E402

MARK_DIR = ROOT / "output" / "人工标定" / "人工 修订 双导管区域"
ATTR_JSON = ROOT / "output" / "错误归因_双导管.json"


def _find_tif(key: str) -> Path | None:
    # key like "6 (8).tif"
    stem = key.replace(".tif", "")
    for f in BATCH.glob("*.tif"):
        if image_key_from_stem(f.stem) == key:
            return f
    # fuzzy: strip leading zeros in paren
    m = __import__("re").match(r"(\d+)\s*\((\d+)\)", stem)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    for f in BATCH.glob("*.tif"):
        kk = image_key_from_stem(f.stem)
        mm = __import__("re").match(r"(\d+)\s*\((\d+)\)", kk)
        if mm and int(mm.group(1)) == a and int(mm.group(2)) == b:
            return f
    return None


def _box_aabb(box, pad: float = 8.0) -> tuple[float, float, float, float]:
    rad = np.deg2rad(getattr(box, "angle", 0.0))
    hw, hh = box.w / 2.0 + pad, box.h / 2.0 + pad
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    pts = corners @ rot.T + np.array([box.cx, box.cy])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(
        pts[:, 1].max()
    )


def _pair_aabb(p, pad: int = 18) -> tuple[float, float, float, float]:
    x1, y1, bw1, bh1 = cv2.boundingRect(p.v1.contour)
    x2, y2, bw2, bh2 = cv2.boundingRect(p.v2.contour)
    return (
        float(min(x1, x2) - pad),
        float(min(y1, y2) - pad),
        float(max(x1 + bw1, x2 + bw2) + pad),
        float(max(y1 + bh1, y2 + bh2) + pad),
    )


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-6)


def _vessels_in_box(vessels, box, pad: float = 30.0) -> int:
    x0, y0, x1, y1 = _box_aabb(box, pad=pad)
    n = 0
    for v in vessels:
        if x0 <= v.cx <= x1 and y0 <= v.cy <= y1:
            n += 1
    return n


def evaluate(mark_root: Path | None = None, iou_thr: float = 0.25) -> dict:
    mark_root = mark_root or MARK_DIR
    guides = load_human_mark_guides(mark_root) if mark_root.exists() else {}
    if not guides:
        # fallback: parent 人工标定
        alt = ROOT / "output" / "人工标定"
        guides = load_human_mark_guides(alt) if alt.exists() else {}

    rules = load_learned_rules()
    hit = miss = fp = 0
    miss_types = {
        "漏检_框内腔不足2(分割/排除过严)": 0,
        "漏检_有腔未配上(共壁/TVW/排序挤掉)": 0,
    }
    fp_types = {
        "误检_框内多余对(应只保留人工对)": 0,
        "误检_框外假双导管": 0,
    }
    details: list[dict] = []
    n_human_boxes = 0
    n_images = 0
    tvw_vals: list[float] = []

    for key, guide in sorted(guides.items()):
        boxes = filter_valid_multi_boxes(list(guide.boxes))
        if getattr(guide, "force_zero_pair", False):
            boxes = []
        path = _find_tif(key)
        if path is None:
            continue
        n_images += 1
        rgb, gray, vessels, _, scale = process_image(path, force_um_per_px=None)
        for v in vessels:
            v.pair_id = None
        h, w = gray.shape[:2]
        pairs = find_pairs_auto(
            vessels, scale["um_per_px"], path.name, img_w=w, img_h=h, rules=rules
        )
        auto_boxes = [_pair_aabb(p) for p in pairs]
        for p in pairs:
            tvw_vals.append(float(p.tvw_mean_um))

        matched_auto: set[int] = set()
        for bi, box in enumerate(boxes):
            n_human_boxes += 1
            hb = _box_aabb(box, pad=8.0)
            best_i, best_iou = -1, 0.0
            for ai, ab in enumerate(auto_boxes):
                if ai in matched_auto:
                    continue
                iou = _iou(hb, ab)
                if iou > best_iou:
                    best_iou, best_i = iou, ai
            if best_i >= 0 and best_iou >= iou_thr:
                hit += 1
                matched_auto.add(best_i)
                # 框内多余对
                extras = sum(
                    1
                    for ai, ab in enumerate(auto_boxes)
                    if ai not in matched_auto and _iou(hb, ab) >= iou_thr
                )
                if extras:
                    fp += extras
                    fp_types["误检_框内多余对(应只保留人工对)"] += extras
                    details.append(
                        {
                            "image": key,
                            "type": "误检_框内多余对(应只保留人工对)",
                            "extras": extras,
                        }
                    )
            else:
                miss += 1
                nv = _vessels_in_box(vessels, box)
                if nv < 2:
                    t = "漏检_框内腔不足2(分割/排除过严)"
                else:
                    t = "漏检_有腔未配上(共壁/TVW/排序挤掉)"
                miss_types[t] += 1
                details.append(
                    {
                        "image": key,
                        "type": t,
                        "human_box_i": bi,
                        "vessels_in_box": nv,
                        "auto_pairs": len(pairs),
                    }
                )

        # 框外假双导管
        for ai, ab in enumerate(auto_boxes):
            if ai in matched_auto:
                continue
            covered = any(_iou(ab, _box_aabb(b, pad=8.0)) >= iou_thr for b in boxes)
            if not covered:
                fp += 1
                fp_types["误检_框外假双导管"] += 1
                details.append(
                    {"image": key, "type": "误检_框外假双导管", "auto_pair_i": ai}
                )

    recall = hit / max(n_human_boxes, 1)
    precision = hit / max(hit + fp, 1)
    result = {
        "source": str(mark_root),
        "n_images": n_images,
        "n_human_boxes": n_human_boxes,
        "hit": hit,
        "miss": miss,
        "false_positive": fp,
        "recall": recall,
        "precision": precision,
        "miss_types": miss_types,
        "fp_types": fp_types,
        "tvw_um": {
            "n": len(tvw_vals),
            "mean": float(np.mean(tvw_vals)) if tvw_vals else None,
            "p50": float(np.median(tvw_vals)) if tvw_vals else None,
            "p90": float(np.percentile(tvw_vals, 90)) if tvw_vals else None,
        },
        "details": details,
        "prior_attribution": None,
    }
    if ATTR_JSON.exists():
        prior = json.loads(ATTR_JSON.read_text(encoding="utf-8"))
        result["prior_attribution"] = {
            "recall": prior.get("recall"),
            "precision": prior.get("precision"),
            "hit": prior.get("hit"),
            "miss": prior.get("miss"),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "baseline_metrics.json",
    )
    ap.add_argument("--mark-dir", type=Path, default=MARK_DIR)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    res = evaluate(args.mark_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"双导管框: recall={res['recall']:.3f} precision={res['precision']:.3f} "
        f"hit={res['hit']}/{res['n_human_boxes']} fp={res['false_positive']}"
    )
    print(f"漏检构成: {res['miss_types']}")
    print(f"已写: {out}")


if __name__ == "__main__":
    main()
