"""
YOLOv8-seg 推理 → Vessel 候选轮廓。

用法（库）:
  from infer_lumen_seg import predict_lumen_vessels, default_weights
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from segment import Vessel, tissue_height

ROOT = Path(__file__).resolve().parent
# 本地 --target 安装的 ultralytics
_pylibs = ROOT / ".pylibs"
if _pylibs.is_dir():
    import sys

    if str(_pylibs) not in sys.path:
        sys.path.insert(0, str(_pylibs))

import numpy as _np

if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid  # type: ignore[attr-defined]

DEFAULT_WEIGHTS = ROOT / "output" / "models" / "lumen_yolov8seg" / "weights" / "best.pt"


def default_weights() -> Path | None:
    for p in (
        DEFAULT_WEIGHTS,
        ROOT / "output" / "models" / "lumen_yolov8seg" / "weights" / "last.pt",
    ):
        if p.exists():
            return p
    return None


@lru_cache(maxsize=2)
def _load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def masks_to_vessels(
    masks: list[np.ndarray],
    image_name: str,
    um_per_px: float,
    confs: list[float] | None = None,
    class_ids: list[int] | None = None,
    lumen_class: int = 0,
) -> list[Vessel]:
    """二值 mask 列表 → Vessel（仅 lumen 类）。"""
    vessels: list[Vessel] = []
    m = re.match(r"(\d+)\s*\((\d+)\)", image_name)
    prefix = f"{m.group(1)}({m.group(2)})" if m else Path(image_name).stem
    idx = 0
    for i, mask in enumerate(masks):
        if class_ids is not None and int(class_ids[i]) != lumen_class:
            continue
        if mask is None or mask.size == 0:
            continue
        bin_m = (mask > 0.5).astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
        if bin_m.max() <= 1:
            bin_m = (bin_m > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < 50:
            continue
        peri = float(cv2.arcLength(cnt, True))
        mom = cv2.moments(cnt)
        if mom["m00"] == 0:
            continue
        cx = float(mom["m10"] / mom["m00"])
        cy = float(mom["m01"] / mom["m00"])
        x, y, bw, bh = cv2.boundingRect(cnt)
        filled = np.zeros_like(bin_m)
        cv2.drawContours(filled, [cnt], -1, 255, -1)
        idx += 1
        conf = float(confs[i]) if confs is not None else 1.0
        vessels.append(
            Vessel(
                vessel_id=f"{prefix}-DL{idx:02d}",
                image_name=image_name,
                contour=cnt,
                area_px=area,
                peri_px=peri,
                cx=cx,
                cy=cy,
                bbox=(x, y, bw, bh),
                status="primary",
                reason=f"yolo_seg;conf={conf:.3f}",
                mask=filled,
            )
        )
    return vessels


def predict_lumen_vessels(
    rgb_or_bgr: np.ndarray,
    image_name: str,
    um_per_px: float,
    weights: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
) -> list[Vessel]:
    """
    对整图跑 YOLO-seg，返回 lumen 类实例为 Vessel。
    输入可为 RGB 或 BGR（ultralytics 内部会处理路径/数组）。
    """
    wpath = Path(weights) if weights else default_weights()
    if wpath is None or not wpath.exists():
        return []

    model = _load_model(str(wpath.resolve()))
    # ultralytics 期望 BGR ndarray 时也可；统一转 BGR
    if rgb_or_bgr.ndim == 2:
        bgr = cv2.cvtColor(rgb_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = rgb_or_bgr
        # 若传入 RGB，色差不大对 SEM 影响有限；保持原样

    results = model.predict(
        source=bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        retina_masks=True,
    )
    if not results:
        return []
    r0 = results[0]
    if r0.masks is None or r0.boxes is None or len(r0.masks) == 0:
        return []

    h, w = bgr.shape[:2]
    masks_data = r0.masks.data.cpu().numpy()  # (N, mh, mw)
    cls = r0.boxes.cls.cpu().numpy().astype(int)
    confs = r0.boxes.conf.cpu().numpy().astype(float)
    masks: list[np.ndarray] = []
    for i in range(masks_data.shape[0]):
        m = masks_data[i]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        # 裁到组织区上方，抑制标尺区误检
        th = tissue_height(h)
        m = m.copy()
        m[th:, :] = 0
        masks.append(m)

    return masks_to_vessels(
        masks, image_name, um_per_px, confs=list(confs), class_ids=list(cls), lumen_class=0
    )


def contour_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    ma = np.zeros(shape, dtype=np.uint8)
    mb = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(ma, [a.astype(np.int32)], -1, 1, -1)
    cv2.drawContours(mb, [b.astype(np.int32)], -1, 1, -1)
    inter = int(np.logical_and(ma, mb).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(ma, mb).sum())
    return inter / max(union, 1)
