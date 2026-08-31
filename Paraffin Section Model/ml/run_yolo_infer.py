"""Run YOLO-based inference using trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calibration.geometry import draw_calibration_lines
from calibration.io_util import imread, imwrite, list_region_images
from calibration.merge import stitch_horizontal
from calibration.region3.cambium import draw_cambium_annotation
from ml.score_model import ScoreModel
from ml.yolo_infer import YoloRegion1Model, YoloRegion3Model


def run_yolo_inference(base_dir: Path) -> None:
    models = base_dir / "models" / "yolo"
    r1_w = models / "region1_pose_best.pt"
    r3_w = models / "region3_detect_best.pt"
    score_w = models / "score_regressor.pt"

    if not r1_w.exists() or not r3_w.exists():
        raise FileNotFoundError("请先运行 python -m ml.train_iterate 训练 YOLO 模型")

    r1_model = YoloRegion1Model(r1_w)
    r3_model = YoloRegion3Model(r3_w)
    score_model = ScoreModel(score_w) if score_w.exists() else None

    auto_r1 = base_dir / "自动标定" / "第一区域"
    auto_r3 = base_dir / "自动标定" / "第三区域"
    merge_r1 = base_dir / "合并标定" / "第一区域"
    merge_r3 = base_dir / "合并标定" / "第三区域"
    manual_r1 = base_dir / "人工标定" / "第一区域"
    manual_r3 = base_dir / "人工标定" / "第三区域"
    for d in (auto_r1, auto_r3, merge_r1, merge_r3):
        d.mkdir(parents=True, exist_ok=True)

    for path in list_region_images(base_dir, "第一区域", view_ids=(1, 2, 3)):
        img = imread(path)
        if img is None:
            continue
        geom = r1_model.predict(img)
        if geom is not None:
            auto_img = draw_calibration_lines(img, geom)
            imwrite(auto_r1 / path.name, auto_img)
            manual = imread(manual_r1 / path.name)
            if manual is not None:
                imwrite(merge_r1 / path.name, stitch_horizontal(manual, auto_img))

    for path in list_region_images(base_dir, "第三区域", view_ids=(7, 8, 9)):
        img = imread(path)
        if img is None:
            continue
        box = r3_model.predict(img)
        if box is None:
            continue
        score = score_model.predict_crop(img[box.y0 : box.y1, box.x0 : box.x1]) if score_model else None
        auto_img = draw_cambium_annotation(img, box, score=score)
        imwrite(auto_r3 / path.name, auto_img)
        manual = imread(manual_r3 / path.name)
        if manual is not None:
            imwrite(merge_r3 / path.name, stitch_horizontal(manual, auto_img))

    print(f"YOLO 推理完成 -> {auto_r1} / {auto_r3}")
    print(f"合并标定 -> {merge_r1} / {merge_r3}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    run_yolo_inference(args.base)


if __name__ == "__main__":
    main()
