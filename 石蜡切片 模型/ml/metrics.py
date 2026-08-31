"""Similarity metrics: auto vs manual calibration (target >= 85%)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from calibration.geometry import LineGeometry


@dataclass
class SimilarityReport:
    region1: float
    region3: float
    score: float
    overall: float

    def meets(self, threshold: float = 0.85) -> bool:
        return self.overall >= threshold and min(self.region1, self.region3, self.score) >= threshold * 0.75


def _point_sim(pred: tuple[float, float], gt: tuple[float, float], norm: float) -> float:
    err = math.hypot(pred[0] - gt[0], pred[1] - gt[1]) / max(norm, 1.0)
    return max(0.0, 1.0 - err)


def region1_geometry_similarity(pred: LineGeometry, gt: LineGeometry, img_diag: float) -> float:
    norm = img_diag * 0.25
    pts_pred = [
        pred.center,
        pred.green_end,
        (pred.seg_x0, pred.seg_y0 or pred.seg_y),
        (pred.xylem_end, pred.xylem_end_y or pred.seg_y),
        (pred.phloem_end, pred.phloem_end_y or pred.seg_y),
        (pred.bark_end, pred.bark_end_y or pred.seg_y),
    ]
    pts_gt = [
        gt.center,
        gt.green_end,
        (gt.seg_x0, gt.seg_y0 or gt.seg_y),
        (gt.xylem_end, gt.xylem_end_y or gt.seg_y),
        (gt.phloem_end, gt.phloem_end_y or gt.seg_y),
        (gt.bark_end, gt.bark_end_y or gt.seg_y),
    ]
    sims = [_point_sim(p, g, norm) for p, g in zip(pts_pred, pts_gt)]
    angle_diff = abs(pred.green_angle_deg - gt.green_angle_deg) % 360
    angle_diff = min(angle_diff, 360 - angle_diff)
    angle_sim = max(0.0, 1.0 - angle_diff / 45.0)
    return float(0.85 * np.mean(sims) + 0.15 * angle_sim)


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def region3_box_similarity(pred_box: tuple[int, int, int, int], gt_box: tuple[int, int, int, int]) -> float:
    return box_iou(pred_box, gt_box)


def score_similarity(pred_scores: list[float], gt_scores: list[float]) -> float:
    if not pred_scores or not gt_scores:
        return 0.0
    pred = np.array(pred_scores, dtype=np.float64)
    gt = np.array(gt_scores, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - gt)))
    within1 = float(np.mean(np.abs(pred - gt) <= 1.0))
    mae_sim = max(0.0, 1.0 - mae / 4.0)
    return float(0.6 * mae_sim + 0.4 * within1)


def combine_similarity(r1: float, r3: float, sc: float) -> SimilarityReport:
    overall = 0.40 * r1 + 0.35 * r3 + 0.25 * sc
    return SimilarityReport(region1=r1, region3=r3, score=sc, overall=overall)
