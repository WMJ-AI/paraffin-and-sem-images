"""Train region-1 YOLO-Pose only (pith/center + line keypoints). Region-3 left unchanged."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from calibration.io_util import imread
from calibration.geometry import parse_manual_geometry
from ml.export_datasets import export_region1_pose
from ml.metrics import region1_geometry_similarity
from ml.yolo_infer import YoloRegion1Model
from ml.yolo_train import train_yolo_pose


def evaluate_region1(base_dir: Path, weights: Path) -> dict:
    from calibration.stem import detect_pith, detect_stem_mask

    model = YoloRegion1Model(weights)
    manual_dir = base_dir / "人工标定" / "第一区域"
    orig_dir = base_dir / "原图" / "第一区域"
    sims = []
    pith_errs = []
    for mp in sorted(manual_dir.glob("*.jpg")):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            continue
        gt = parse_manual_geometry(manual)
        pred = model.predict(orig)
        if gt is None or pred is None:
            continue
        stem_info = detect_stem_mask(orig)
        if stem_info is not None:
            pith = detect_pith(orig, stem_info[0])
            gt.center = pith
            gt.green_start = pith
        h, w = orig.shape[:2]
        diag = (h * h + w * w) ** 0.5
        sims.append(region1_geometry_similarity(pred, gt, diag))
        pith_errs.append(
            ((pred.center[0] - gt.center[0]) ** 2 + (pred.center[1] - gt.center[1]) ** 2) ** 0.5 / diag
        )
    import numpy as np

    return {
        "n": len(sims),
        "mean_sim": float(np.mean(sims)) if sims else 0.0,
        "pith_err_norm": float(np.mean(pith_errs)) if pith_errs else 1.0,
        "pith_ok_5pct": float(np.mean(np.array(pith_errs) <= 0.05)) if pith_errs else 0.0,
    }


def train_region1_yolo(
    base_dir: Path,
    epochs: int = 80,
    imgsz: int = 640,
    threshold: float = 0.85,
    max_rounds: int = 5,
) -> Path:
    datasets_dir = base_dir / "datasets" / "region1_pose"
    models_dir = base_dir / "models" / "yolo"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = base_dir / "推理结果" / "iterate_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("[YOLO-R1] 从人工标定导出 Pose 数据集...")
    data_yaml = datasets_dir / "data.yaml"
    if data_yaml.exists() and (datasets_dir / "images" / "train").exists():
        n_train = len(list((datasets_dir / "images" / "train").glob("*.jpg")))
        n_val = len(list((datasets_dir / "images" / "val").glob("*.jpg")))
        n = n_train + n_val
        print(f"  复用已有数据集: train={n_train} val={n_val}")
    else:
        n = export_region1_pose(base_dir, datasets_dir)
        print(f"  有效标注样本: {n}")
    if n < 20:
        raise RuntimeError("第一区域有效人工标定过少，无法训练 YOLO")

    best_path = models_dir / "region1_pose_best.pt"
    history = []
    cur_epochs = epochs

    for round_idx in range(1, max_rounds + 1):
        print("=" * 60)
        print(f"[YOLO-R1] 第 {round_idx}/{max_rounds} 轮训练 epochs={cur_epochs}")
        print("=" * 60)
        weights = train_yolo_pose(
            datasets_dir / "data.yaml",
            models_dir,
            name=f"region1_pose_r{round_idx}",
            epochs=cur_epochs,
            imgsz=imgsz,
        )
        shutil.copy2(weights, best_path)

        stats = evaluate_region1(base_dir, best_path)
        print(
            f"  相似度 mean={stats['mean_sim']:.1%}  "
            f"髓心误差(相对对角线)={stats['pith_err_norm']:.3f}  "
            f"髓心≤5%对角线命中={stats['pith_ok_5pct']:.1%}  n={stats['n']}"
        )
        history.append({"round": round_idx, "epochs": cur_epochs, **stats, "ts": datetime.now().isoformat()})
        (logs_dir / "region1_yolo_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if stats["mean_sim"] >= threshold:
            print(f"[YOLO-R1] 已达 {threshold:.0%} 目标 -> {best_path}")
            break
        cur_epochs += 20
        print(f"  未达标，下一轮 epochs -> {cur_epochs}")
    else:
        print(f"[YOLO-R1] 达到最大轮数，使用当前最优模型: {best_path}")

    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="仅训练第一区域 YOLO-Pose（第三区域不改）")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()
    train_region1_yolo(args.base, args.epochs, args.imgsz, args.threshold, args.max_rounds)


if __name__ == "__main__":
    main()
