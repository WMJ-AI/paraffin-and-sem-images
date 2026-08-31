"""Iterative YOLO + score training until >=85% similarity with manual labels."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ml.evaluate import evaluate_models
from ml.export_datasets import export_all
from ml.score_model import ScoreModel
from ml.yolo_train import train_yolo_detect, train_yolo_pose


def train_iterate(
    base_dir: Path,
    threshold: float = 0.85,
    max_rounds: int = 20,
    base_epochs: int = 40,
    epoch_step: int = 20,
) -> dict:
    datasets_dir = base_dir / "datasets"
    models_dir = base_dir / "models" / "yolo"
    logs_dir = base_dir / "推理结果" / "iterate_logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("导出 YOLO 数据集（人工标定 -> 训练集）")
    counts = export_all(base_dir, datasets_dir)
    print(f"  样本数: {counts}")

    r1_best = models_dir / "region1_pose_best.pt"
    r3_best = models_dir / "region3_detect_best.pt"
    score_best = models_dir / "score_regressor.pt"

    history = []
    epochs = base_epochs

    for round_idx in range(1, max_rounds + 1):
        print("=" * 60)
        print(f"第 {round_idx}/{max_rounds} 轮训练  epochs={epochs}")
        print("=" * 60)

        # Region 1 YOLO Pose
        print("[训练] 第一区域 YOLO-Pose（6 关键点 = 5 条线端点）...")
        r1_weights = train_yolo_pose(
            datasets_dir / "region1_pose" / "data.yaml",
            models_dir,
            name=f"region1_r{round_idx}",
            epochs=epochs,
            imgsz=640,
        )
        shutil.copy2(r1_weights, r1_best)

        # Region 3 YOLO Detect
        print("[训练] 第三区域 YOLO-Detect（形成层矩形框）...")
        r3_weights = train_yolo_detect(
            datasets_dir / "region3_detect" / "data.yaml",
            models_dir,
            name=f"region3_r{round_idx}",
            epochs=epochs,
            imgsz=640,
        )
        shutil.copy2(r3_weights, r3_best)

        # Score regressor
        print("[训练] 形成层评分 CNN 回归（1-10）...")
        score_model = ScoreModel()
        losses = score_model.train_from_manifest(
            datasets_dir / "score_crops" / "manifest.csv",
            datasets_dir / "score_crops",
            epochs=max(20, epochs // 2),
        )
        score_model.save(str(score_best))
        print(f"  score loss={losses[-1]:.4f}" if losses else "  score: no samples")

        # Evaluate
        print("[评估] 与人工标定对比...")
        bundle = evaluate_models(base_dir, r1_best, r3_best, score_best)
        rep = bundle.report
        print(
            f"  第一区域相似度: {rep.region1:.1%}\n"
            f"  第三区域相似度: {rep.region3:.1%}\n"
            f"  评分相似度:     {rep.score:.1%}\n"
            f"  综合相似度:     {rep.overall:.1%}  (目标 {threshold:.0%})"
        )

        record = {
            "round": round_idx,
            "epochs": epochs,
            "region1": rep.region1,
            "region3": rep.region3,
            "score": rep.score,
            "overall": rep.overall,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(record)
        log_path = logs_dir / "history.json"
        log_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        if rep.meets(threshold):
            print("=" * 60)
            print(f"已达到 {threshold:.0%} 相似度目标，训练结束。")
            print(f"模型保存于: {models_dir}")
            print("=" * 60)
            break

        epochs += epoch_step
        print(f"未达标，下一轮增加 epochs -> {epochs}")
    else:
        print(f"已达最大轮数 {max_rounds}，请检查数据或提高 max_rounds。")

    return {"history": history, "models": str(models_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 迭代训练直至与人工标定相似度>=85%")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--base-epochs", type=int, default=40)
    parser.add_argument("--epoch-step", type=int, default=20)
    args = parser.parse_args()
    train_iterate(args.base, args.threshold, args.max_rounds, args.base_epochs, args.epoch_step)


if __name__ == "__main__":
    main()
