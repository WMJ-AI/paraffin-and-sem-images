"""YOLO pose: 6 keypoints (pith, green end, layer 3–6), then snapped by rules."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from calibration.region3.cambium import CambiumBox


class YoloRegion1Model:
    def __init__(self, weights: str | Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def keypoints(self, img: np.ndarray, conf: float = 0.12) -> np.ndarray | None:
        results = self.model.predict(img, verbose=False, conf=conf, imgsz=640)[0]
        if results.keypoints is None or len(results.keypoints) == 0:
            return None
        kpts = results.keypoints.xy[0].cpu().numpy()
        if kpts.shape[0] < 2:
            return None
        return kpts

    def hint(self, img: np.ndarray) -> tuple[tuple[float, float] | None, float | None]:
        """Return (pith_xy, green_angle_deg) or (None, None)."""
        kpts = self.keypoints(img)
        if kpts is None:
            return None, None
        pith = (float(kpts[0, 0]), float(kpts[0, 1]))
        g_end = (float(kpts[1, 0]), float(kpts[1, 1]))
        green_angle = math.degrees(math.atan2(-(g_end[1] - pith[1]), g_end[0] - pith[0]))
        return pith, green_angle

    def predict(self, img: np.ndarray, mag: int = 2, light: bool = False):
        """6-point geometry snapped to stem (or None)."""
        from calibration.region1_sixpoint import geom_from_keypoints, light_from_keypoints
        from calibration.stem import detect_stem_mask

        kpts = self.keypoints(img)
        if kpts is None:
            return None
        stem_info = detect_stem_mask(img)
        stem_mask = stem_info[0] if stem_info is not None else None
        if light:
            geom = light_from_keypoints(kpts, stem_mask, mag=mag)
            if geom is not None:
                return geom
        return geom_from_keypoints(kpts, img, stem_mask, mag=mag)


class YoloRegion3Model:
    def __init__(self, weights: str | Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def predict(self, img: np.ndarray) -> CambiumBox | None:
        results = self.model.predict(img, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return None
        box = results.boxes.xyxy[0].cpu().numpy()
        x0, y0, x1, y1 = map(int, box)
        poly = np.column_stack(
            [np.linspace(x0, x1, num=50), np.full(50, (y0 + y1) // 2)]
        ).astype(np.int32)
        return CambiumBox(x0=x0, y0=y0, x1=x1, y1=y1, center_row=(y0 + y1) / 2, polyline=poly)


class YoloRegion2Model:
    """Optional vessel detector. Boxes are only seeds; lumens are still CV-segmented."""

    def __init__(self, weights: str | Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def boxes(self, img: np.ndarray, conf: float = 0.20) -> list[tuple[int, int, int, int]]:
        results = self.model.predict(img, verbose=False, conf=conf, imgsz=1280)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return []
        out = []
        for box in results.boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = map(int, box)
            out.append((x0, y0, x1 - x0, y1 - y0))
        return out

