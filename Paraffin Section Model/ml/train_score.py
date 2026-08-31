"""Train cambium score CNN against human-revised 1-10 labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration.io_util import imread, parse_name, list_region_images
from calibration.region3.cambium import load_box_params, locate_cambium
from calibration.region3.scoring import load_manual_scores, resolve_score_xlsx
from ml.metrics import score_similarity
from ml.score_model import ScoreModel


def collect_labeled_crops(base_dir: Path) -> list[tuple[np.ndarray, float, int]]:
    scores = load_manual_scores(resolve_score_xlsx(base_dir))
    box_params = load_box_params(base_dir / "models" / "region3_box_params.json")
    items: list[tuple[np.ndarray, float, int]] = []
    for path in list_region_images(base_dir, "第三区域", view_ids=(7, 8, 9)):
        meta = parse_name(path.name)
        if meta is None:
            continue
        key = (meta.sample_id, meta.view_id)
        if key not in scores:
            continue
        img = imread(path)
        if img is None:
            continue
        box = locate_cambium(img, box_params)
        if box is None:
            continue
        crop = img[box.y0 : box.y1, box.x0 : box.x1]
        if crop.size == 0:
            continue
        # Keep a wide strip, not the 5k-px original (saves RAM / speeds aug).
        crop = cv2.resize(crop, (960, 160), interpolation=cv2.INTER_AREA)
        items.append((crop, float(scores[key]), int(meta.sample_id)))
    return items


def _round_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.round(pred), 1, 10)
    mae = float(np.mean(np.abs(rounded - gt)))
    exact = float(np.mean(rounded == gt))
    within1 = float(np.mean(np.abs(rounded - gt) <= 1))
    return {
        "mae": mae,
        "exact": exact,
        "within1": within1,
        "similarity": float(score_similarity(rounded.tolist(), gt.tolist())),
        "bias": float(np.mean(rounded - gt)),
    }


def _print_metrics(title: str, m: dict[str, float]) -> None:
    print(
        f"{title}: MAE={m['mae']:.3f} exact={m['exact']:.1%} "
        f"within1={m['within1']:.1%} sim={m['similarity']:.1%} bias={m['bias']:+.3f}"
    )


def _fit_affine(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if len(pred) < 8 or float(np.std(pred)) < 1e-6:
        return 1.0, 0.0
    a, b = np.polyfit(pred.astype(np.float64), gt.astype(np.float64), 1)
    return float(np.clip(a, 0.45, 2.2)), float(np.clip(b, -4.0, 4.0))


def train_score(base_dir: Path, epochs: int = 40) -> dict:
    items = collect_labeled_crops(base_dir)
    if len(items) < 10:
        raise RuntimeError("评分训练样本不足，请检查人工修订表与第三区域原图")

    out_path = base_dir / "models" / "yolo" / "score_regressor.pt"
    xlsx = resolve_score_xlsx(base_dir)
    print(f"[评分训练] 标签: {xlsx.name}  样本={len(items)}  epochs={epochs}")
    np.random.seed(42)
    try:
        import torch
        import random as _rng

        torch.manual_seed(42)
        _rng.seed(42)
    except Exception:
        pass

    crops = [it[0] for it in items]
    y = np.array([it[1] for it in items], dtype=np.float64)
    groups = np.array([it[2] for it in items], dtype=np.int32)

    rng = np.random.RandomState(42)
    sample_ids = np.unique(groups)
    rng.shuffle(sample_ids)
    n_val = max(6, int(round(0.20 * len(sample_ids))))
    val_ids = set(sample_ids[:n_val].tolist())
    train_idx = [i for i, g in enumerate(groups) if g not in val_ids]
    val_idx = [i for i, g in enumerate(groups) if g in val_ids]
    train_set = [(crops[i], float(y[i])) for i in train_idx]
    val_set = [(crops[i], float(y[i])) for i in val_idx]
    print(f"  holdout: train_views={len(train_set)} val_views={len(val_set)} val_samples={len(val_ids)}")

    model = ScoreModel()
    model.fit_samples(
        train_set,
        epochs=epochs,
        lr=8e-4,
        batch_size=12,
        val_samples=val_set,
        freeze_backbone_epochs=max(8, epochs // 3),
    )

    model.calib_a, model.calib_b = 1.0, 0.0
    val_raw = np.array([model.predict_crop_raw(crops[i], tta=True) for i in val_idx], dtype=np.float64)
    _print_metrics("  holdout CNN", _round_metrics(val_raw, y[val_idx]))
    a_h, b_h = _fit_affine(val_raw, y[val_idx])
    _print_metrics(
        f"  holdout affine a={a_h:.3f} b={b_h:.3f}",
        _round_metrics(np.clip(a_h * val_raw + b_h, 1, 10), y[val_idx]),
    )

    print("[评分训练] 用全部修订样本短时微调...")
    all_set = [(crops[i], float(y[i])) for i in range(len(items))]
    model.fit_samples(
        all_set,
        epochs=max(10, epochs // 3),
        lr=2e-4,
        batch_size=12,
        freeze_backbone_epochs=0,
    )
    raw_all = np.array([model.predict_crop_raw(c, tta=False) for c in crops], dtype=np.float64)
    model.calib_a = 1.0
    model.calib_b = float(np.clip(np.mean(y) - np.mean(raw_all), -3.0, 3.0))
    print(f"  production calib a=1 b={model.calib_b:+.3f}")

    preds = np.array([model.predict_crop_raw(c, tta=True) for c in crops], dtype=np.float64)
    _print_metrics("  全量+校准 (对照人工修订)", _round_metrics(preds, y))
    model.save(str(out_path))
    print(f"[评分训练] 已保存 {out_path}")
    return {"n": len(items), "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="用人工修订评分训练形成层打分模型")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    train_score(args.base, epochs=args.epochs)


if __name__ == "__main__":
    main()
