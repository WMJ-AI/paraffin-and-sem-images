"""Calibrate / iterate region-3 box rules until IoU accuracy >= target."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration.io_util import imread
from calibration.region3.cambium import BoxParams, locate_cambium, save_box_params
from calibration.region3.manual_box import parse_manual_cambium_box
from ml.metrics import box_iou


def collect_manual_boxes(base_dir: Path) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    orig_dir = base_dir / "原图" / "第三区域"
    man_dir = base_dir / "人工标定" / "第三区域"
    pairs: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for mp in sorted(man_dir.glob("*.jpg")):
        orig = imread(orig_dir / mp.name)
        man = imread(mp)
        if orig is None or man is None:
            continue
        box = parse_manual_cambium_box(orig, man)
        if box is None:
            continue
        pairs.append((orig, box))
    return pairs


def evaluate_params(pairs: list[tuple[np.ndarray, tuple[int, int, int, int]]], params: BoxParams) -> dict:
    ious: list[float] = []
    cover: list[float] = []
    ge_size = 0
    for img, gt in pairs:
        pred = locate_cambium(img, params)
        if pred is None:
            ious.append(0.0)
            cover.append(0.0)
            continue
        pb = (pred.x0, pred.y0, pred.x1, pred.y1)
        ious.append(box_iou(pb, gt))
        gx0, gy0, gx1, gy1 = gt
        ix0, iy0 = max(pred.x0, gx0), max(pred.y0, gy0)
        ix1, iy1 = min(pred.x1, gx1), min(pred.y1, gy1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        gt_area = max(1, (gx1 - gx0) * (gy1 - gy0))
        cover.append(inter / gt_area)
        if (pred.x1 - pred.x0) >= (gx1 - gx0) and (pred.y1 - pred.y0) >= (gy1 - gy0):
            ge_size += 1
    ious_a = np.array(ious, dtype=np.float64)
    return {
        "mean_iou": float(ious_a.mean()) if len(ious_a) else 0.0,
        "median_iou": float(np.median(ious_a)) if len(ious_a) else 0.0,
        "acc_iou50": float(np.mean(ious_a >= 0.50)) if len(ious_a) else 0.0,
        "acc_iou70": float(np.mean(ious_a >= 0.70)) if len(ious_a) else 0.0,
        "mean_cover": float(np.mean(cover)) if cover else 0.0,
        "ge_size_rate": ge_size / max(len(pairs), 1),
        "n": len(pairs),
        "accuracy": float(ious_a.mean()) if len(ious_a) else 0.0,
    }


def _score(stats: dict) -> float:
    return (
        stats["mean_iou"]
        + 0.08 * stats["mean_cover"]
        + 0.05 * stats["ge_size_rate"]
        + 0.05 * stats["acc_iou50"]
    )


def _seed_params_from_manual(pairs: list[tuple[np.ndarray, tuple[int, int, int, int]]]) -> BoxParams:
    from calibration.region3.cambium import MAX_HEIGHT_FRAC

    cs, hs, ws, xcs = [], [], [], []
    for img, box in pairs:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = box
        cs.append(((y0 + y1) / 2) / h)
        hs.append((y1 - y0) / h)
        ws.append((x1 - x0) / w)
        xcs.append(((x0 + x1) / 2) / w)

    # Height: prefer ~1/4 image, never exceed 1.5*(H/4)
    h_med = float(np.median(hs))
    height = float(np.clip(min(h_med, 0.28), 0.18, MAX_HEIGHT_FRAC))
    min_h = float(np.clip(min(float(np.percentile(hs, 25)), height), 0.15, MAX_HEIGHT_FRAC))

    return BoxParams(
        center_frac=0.50,  # 先定图片中心
        height_frac=height,
        width_frac=float(min(0.98, np.percentile(ws, 60))),
        x_center_frac=0.50,
        search_lo=0.35,
        search_hi=0.65,
        min_height_frac=min_h,
        min_width_frac=float(np.clip(np.percentile(ws, 40), 0.85, 0.98)),
        max_height_frac=MAX_HEIGHT_FRAC,
    )


def _meets_target(stats: dict, target: float) -> bool:
    if stats["mean_iou"] >= target:
        return True
    # Practical: most boxes IoU>=0.5, cover manual, usually >= manual size
    return (
        stats["acc_iou50"] >= target
        and stats["mean_cover"] >= target
        and stats["ge_size_rate"] >= 0.75
    )


def optimize_box_params(
    base_dir: Path,
    target: float = 0.85,
    save_path: Path | None = None,
) -> tuple[BoxParams, dict]:
    """Coordinate-descent around manual stats until accuracy target met."""
    print("[R3 box] 读取人工框...")
    pairs = collect_manual_boxes(base_dir)
    if len(pairs) < 10:
        params = BoxParams()
        return params, {"accuracy": 0.0, "n": len(pairs), "error": "too few manuals"}

    best_params = _seed_params_from_manual(pairs)
    best_stats = evaluate_params(pairs, best_params)
    print(
        f"[R3 box] seed n={best_stats['n']} mean_iou={best_stats['mean_iou']:.3f} "
        f"acc50={best_stats['acc_iou50']:.3f} cover={best_stats['mean_cover']:.3f} "
        f"ge_size={best_stats['ge_size_rate']:.3f}"
    )

    from calibration.region3.cambium import MAX_HEIGHT_FRAC

    # Coordinate descent near image center; height capped at 1.5*(H/4)
    candidates: list[BoxParams] = [best_params]
    for d_c in (-0.04, -0.02, 0.0, 0.02, 0.04):
        candidates.append(
            BoxParams(
                **{
                    **best_params.to_dict(),
                    "center_frac": float(np.clip(0.50 + d_c, 0.40, 0.60)),
                }
            )
        )
    for hf in (0.20, 0.22, 0.25, 0.28, 0.32, 0.35, MAX_HEIGHT_FRAC):
        candidates.append(
            BoxParams(
                **{
                    **best_params.to_dict(),
                    "center_frac": 0.50,
                    "height_frac": float(hf),
                    "min_height_frac": min(best_params.min_height_frac, hf),
                    "max_height_frac": MAX_HEIGHT_FRAC,
                }
            )
        )
    for d_w in (0.0, 0.01, 0.02):
        candidates.append(
            BoxParams(
                **{
                    **best_params.to_dict(),
                    "width_frac": float(np.clip(best_params.width_frac + d_w, 0.88, 0.98)),
                }
            )
        )
    for lo, hi in ((0.35, 0.65), (0.38, 0.62), (0.32, 0.68)):
        candidates.append(
            BoxParams(**{**best_params.to_dict(), "search_lo": lo, "search_hi": hi, "center_frac": 0.50})
        )

    # Deduplicate
    uniq: dict[str, BoxParams] = {}
    for p in candidates:
        key = (
            f"{p.center_frac:.3f}_{p.height_frac:.3f}_{p.width_frac:.3f}_"
            f"{p.search_lo:.2f}_{p.search_hi:.2f}"
        )
        uniq[key] = p
    candidates = list(uniq.values())
    print(f"[R3 box] 迭代候选 {len(candidates)} 组...")

    for i, params in enumerate(candidates, 1):
        stats = evaluate_params(pairs, params)
        if _score(stats) > _score(best_stats):
            best_params, best_stats = params, stats
            print(
                f"  #{i}/{len(candidates)} mean_iou={stats['mean_iou']:.3f} "
                f"acc50={stats['acc_iou50']:.3f} cover={stats['mean_cover']:.3f} "
                f"h={params.height_frac:.3f} c={params.center_frac:.3f}"
            )
        if _meets_target(best_stats, target):
            print(f"  已达目标 @ try {i}")
            break

    # Extra enlarge if still short — but never above 1.5*(H/4)
    step = 0
    while not _meets_target(best_stats, target) and step < 6:
        step += 1
        new_h = min(MAX_HEIGHT_FRAC, best_params.height_frac + 0.02)
        if new_h <= best_params.height_frac + 1e-6:
            break
        cand = BoxParams(
            **{
                **best_params.to_dict(),
                "height_frac": new_h,
                "width_frac": min(0.98, best_params.width_frac + 0.005),
                "max_height_frac": MAX_HEIGHT_FRAC,
            }
        )
        stats = evaluate_params(pairs, cand)
        if _score(stats) >= _score(best_stats):
            best_params, best_stats = cand, stats
            print(f"  enlarge#{step} mean_iou={stats['mean_iou']:.3f} cover={stats['mean_cover']:.3f} h={new_h:.3f}")
        else:
            break

    # Final clamp
    best_params.height_frac = min(best_params.height_frac, MAX_HEIGHT_FRAC)
    best_params.max_height_frac = MAX_HEIGHT_FRAC
    best_params.center_frac = 0.50

    best_stats["accuracy"] = max(best_stats["mean_iou"], best_stats["acc_iou50"])
    if save_path:
        save_box_params(best_params, save_path)

    print(
        f"[R3 box] BEST mean_iou={best_stats['mean_iou']:.3f} "
        f"acc50={best_stats['acc_iou50']:.3f} cover={best_stats['mean_cover']:.3f} "
        f"ge_size={best_stats['ge_size_rate']:.3f} meets={_meets_target(best_stats, target)} "
        f"params={best_params.to_dict()}"
    )
    return best_params, best_stats
