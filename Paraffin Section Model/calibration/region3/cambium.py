"""Cambium box: image-center first, algorithm refine; outline only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration.region3.rules import CAMBIUM_COLOR, CAMBIUM_LINE_THICKNESS, DEFAULT_BOX_PARAMS


# Hard rule: box height <= 1.5 * (image_height / 4)
MAX_HEIGHT_FRAC = 1.5 * 0.25  # 0.375


@dataclass
class CambiumBox:
    x0: int
    y0: int
    x1: int
    y1: int
    center_row: float
    polyline: np.ndarray


@dataclass
class BoxParams:
    """Normalized box rules (center-first + size cap)."""

    center_frac: float = 0.50  # image center
    height_frac: float = 0.25  # start near 1/4 height
    width_frac: float = 0.95
    x_center_frac: float = 0.50
    search_lo: float = 0.35
    search_hi: float = 0.65
    min_height_frac: float = 0.18
    min_width_frac: float = 0.88
    max_height_frac: float = MAX_HEIGHT_FRAC

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "BoxParams":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def load_box_params(path: Path | None = None) -> BoxParams:
    if path and path.exists():
        p = BoxParams.from_dict(json.loads(path.read_text(encoding="utf-8")))
    else:
        p = BoxParams.from_dict(DEFAULT_BOX_PARAMS)
    # Always enforce height cap
    p.max_height_frac = min(float(getattr(p, "max_height_frac", MAX_HEIGHT_FRAC)), MAX_HEIGHT_FRAC)
    p.height_frac = min(p.height_frac, p.max_height_frac)
    p.min_height_frac = min(p.min_height_frac, p.max_height_frac)
    return p


def save_box_params(params: BoxParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")


def _smooth_1d(values: np.ndarray, k: int = 11) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float32)
    k = max(3, min(k | 1, len(values) - (1 - len(values) % 2)))
    if k % 2 == 0:
        k -= 1
    return np.convolve(values.astype(np.float32), np.ones(k, dtype=np.float32) / k, mode="same")


def _mid_interface_row(gray: np.ndarray, y_lo: int, y_hi: int, x0: int, x1: int) -> int:
    """Algorithm refine: strongest horizontal interface near image center."""
    roi = gray[y_lo:y_hi, x0:x1]
    if roi.size == 0 or roi.shape[0] < 12:
        return (y_lo + y_hi) // 2
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)
    row = _smooth_1d(np.mean(np.abs(gy), axis=1), k=9)
    mid = (len(row) - 1) / 2.0
    xs = np.arange(len(row), dtype=np.float32)
    prior = np.exp(-0.5 * ((xs - mid) / (0.30 * len(row) + 1e-6)) ** 2)
    score = row * (0.35 + 0.65 * prior)
    return y_lo + int(np.argmax(score))


def locate_cambium(img: np.ndarray, params: BoxParams | None = None) -> CambiumBox | None:
    """
    1) Rule: lock to image center
    2) Algorithm: fine-tune center in mid band
    3) Box height capped at 1.5 * (H/4) = 0.375H
    """
    params = params or load_box_params()
    h, w = img.shape[:2]
    max_h_frac = min(params.max_height_frac, MAX_HEIGHT_FRAC)

    scale = 4
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sh, _sw = gray.shape

    # Width around horizontal center
    bw = max(int(round(params.width_frac * w)), int(round(params.min_width_frac * w)))
    bw = min(bw, w - 2)
    cx = int(round(0.50 * w))  # horizontal center
    x0 = int(np.clip(cx - bw // 2, 0, w - bw - 1))
    x1 = x0 + bw

    # 1) Image vertical center first
    img_center_s = 0.50 * sh

    # 2) Algorithm calibrate near center only
    y_lo = int(sh * max(0.30, params.search_lo))
    y_hi = int(sh * min(0.70, params.search_hi))
    if y_hi - y_lo < 10:
        y_lo, y_hi = int(sh * 0.35), int(sh * 0.65)
    sx0 = int(x0 / scale)
    sx1 = max(sx0 + 1, int(x1 / scale))
    algo_s = _mid_interface_row(gray, y_lo, y_hi, sx0, sx1)

    # Keep mostly at image center; algorithm only nudges
    center_s = int(round(0.70 * img_center_s + 0.30 * algo_s))
    center_row = float(center_s * scale)

    # Height: calibrated but never exceed 1.5*(H/4)
    bh = max(int(round(params.height_frac * h)), int(round(params.min_height_frac * h)))
    bh = min(bh, int(round(max_h_frac * h)), h - 2)
    bh = max(bh, int(round(0.12 * h)))  # keep visible

    y0 = int(np.clip(int(round(center_row)) - bh // 2, 0, h - bh - 1))
    y1 = y0 + bh

    step = max(1, (x1 - x0) // 200)
    xs = np.arange(x0, x1 + 1, step)
    poly = np.column_stack([xs, np.full(len(xs), int(round(center_row)))]).astype(np.int32)
    return CambiumBox(x0=x0, y0=y0, x1=x1, y1=y1, center_row=center_row, polyline=poly)


def draw_cambium_annotation(img: np.ndarray, box: CambiumBox, score: int | None = None) -> np.ndarray:
    """Outline only — no fill color inside the box."""
    out = img.copy()
    cv2.rectangle(out, (box.x0, box.y0), (box.x1, box.y1), CAMBIUM_COLOR, CAMBIUM_LINE_THICKNESS, cv2.LINE_AA)
    if score is not None:
        label = f"{score}"
        y = max(box.y0 - 12, 40)
        cv2.putText(out, label, (box.x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
        cv2.putText(out, label, (box.x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, CAMBIUM_COLOR, 2)
    return out
