"""Evaluate model predictions against manual ground truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import parse_manual_geometry
from calibration.io_util import imread, parse_name
from calibration.region3.scoring import load_manual_scores, resolve_score_xlsx
from ml.export_datasets import _manual_cambium_box
from ml.metrics import (
    SimilarityReport,
    combine_similarity,
    region1_geometry_similarity,
    region3_box_similarity,
    score_similarity,
)
from ml.score_model import ScoreModel
from ml.yolo_infer import YoloRegion1Model, YoloRegion3Model


@dataclass
class EvalBundle:
    report: SimilarityReport
    region1_scores: list[float]
    region3_scores: list[float]
    score_preds: list[float]
    score_gts: list[float]


def evaluate_models(
    base_dir: Path,
    region1_weights: Path | None,
    region3_weights: Path | None,
    score_weights: Path | None,
) -> EvalBundle:
    r1_sims: list[float] = []
    r3_sims: list[float] = []
    pred_scores: list[float] = []
    gt_scores: list[float] = []

    r1_model = YoloRegion1Model(region1_weights) if region1_weights and region1_weights.exists() else None
    r3_model = YoloRegion3Model(region3_weights) if region3_weights and region3_weights.exists() else None
    score_model = ScoreModel(score_weights) if score_weights and score_weights.exists() else None
    manual_scores = load_manual_scores(resolve_score_xlsx(base_dir))

    # Region 1
    orig_r1 = base_dir / "原图" / "第一区域"
    manual_r1 = base_dir / "人工标定" / "第一区域"
    if r1_model:
        for mp in sorted(manual_r1.glob("*.jpg")):
            op = orig_r1 / mp.name
            if not op.exists():
                continue
            orig = imread(op)
            manual = imread(mp)
            if orig is None or manual is None:
                continue
            gt = parse_manual_geometry(manual)
            pred = r1_model.predict(orig)
            if gt is None or pred is None:
                continue
            diag = math.hypot(orig.shape[1], orig.shape[0])
            r1_sims.append(region1_geometry_similarity(pred, gt, diag))

    # Region 3 boxes + scores
    orig_r3 = base_dir / "原图" / "第三区域"
    manual_r3 = base_dir / "人工标定" / "第三区域"
    if r3_model:
        for mp in sorted(manual_r3.glob("*.jpg")):
            meta = parse_name(mp.name)
            op = orig_r3 / mp.name
            if meta is None or not op.exists():
                continue
            orig = imread(op)
            manual = imread(mp)
            if orig is None or manual is None:
                continue
            gt_box = _manual_cambium_box(orig, manual)
            pred_box_obj = r3_model.predict(orig)
            if gt_box is None or pred_box_obj is None:
                continue
            pred_box = (pred_box_obj.x0, pred_box_obj.y0, pred_box_obj.x1, pred_box_obj.y1)
            r3_sims.append(region3_box_similarity(pred_box, gt_box))

            if score_model is not None:
                x0, y0, x1, y1 = pred_box
                crop = orig[y0:y1, x0:x1]
                if crop.size > 0:
                    key = (meta.sample_id, meta.view_id)
                    if key in manual_scores:
                        pred_scores.append(float(score_model.predict_crop(crop)))
                        gt_scores.append(float(manual_scores[key]))

    report = combine_similarity(
        float(np.mean(r1_sims)) if r1_sims else 0.0,
        float(np.mean(r3_sims)) if r3_sims else 0.0,
        score_similarity(pred_scores, gt_scores),
    )
    return EvalBundle(report, r1_sims, r3_sims, pred_scores, gt_scores)
