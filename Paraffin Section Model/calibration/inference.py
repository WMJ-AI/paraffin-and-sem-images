"""CV-first inference for stem layer boundary detection."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from calibration.geometry import LineGeometry
from calibration.region1_geom import build_straight_geometry, valid_angle_pair
from calibration.stem import (
    detect_inner_ring_radius,
    detect_pith,
    detect_stem_mask,
    estimate_green_angle,
    point_on_ray,
    _ray_boundary,
)


def _stem_radial_profile(
    gray: np.ndarray,
    stem_mask: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """Gray values from pith outward, stopping at stem boundary."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), -math.sin(rad)
    cx, cy = center
    h, w = gray.shape
    values: list[float] = []
    for t in range(1, int(max(h, w))):
        x = int(round(cx + t * dx))
        y = int(round(cy + t * dy))
        if x < 0 or y < 0 or x >= w or y >= h or stem_mask[y, x] == 0:
            break
        values.append(float(gray[y, x]))
    return np.array(values, dtype=np.float32)


def _find_layer_radii(profile: np.ndarray) -> tuple[float, float, float, float]:
    """Return (pith_edge, xylem_end, phloem_end, bark_end) distances along ray.

    Tissue layout from pith: xylem (bulk) → phloem (thin) → bark (outer).
    Boundaries are detected from gradient peaks in the outer half of the radius.
    """
    n = len(profile)
    if n < 40:
        return 3.0, max(n * 0.7, 10), max(n * 0.88, 12), float(max(n - 1, 15))

    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, 31), 0).flatten()
    grad = np.abs(np.diff(smooth.astype(np.float32)))
    bark_end = float(n - 1)

    # Outer half: phloem / bark / late xylem transitions
    search0 = int(n * 0.50)
    outer = grad[search0:]
    if len(outer) < 10:
        return 3.0, n * 0.78, n * 0.90, bark_end

    thr = float(np.percentile(outer, 72))
    candidates: list[tuple[int, float]] = []
    for i in range(2, len(outer) - 2):
        v = float(outer[i])
        if v < thr:
            continue
        if v >= outer[i - 1] and v >= outer[i + 1]:
            candidates.append((search0 + i, v))
    candidates.sort(key=lambda item: -item[1])

    min_gap = max(10, int(n * 0.02))
    chosen: list[int] = []
    for idx, _ in candidates:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
        if len(chosen) >= 3:
            break
    chosen = sorted(chosen)

    if len(chosen) >= 3:
        xylem_end, phloem_end = float(chosen[0]), float(chosen[1])
        bark_end = float(max(chosen[2], n - 3))
    elif len(chosen) == 2:
        xylem_end, phloem_end = float(chosen[0]), float(chosen[1])
        bark_end = float(n - 2)
    elif len(chosen) == 1:
        xylem_end = float(chosen[0])
        phloem_end = float(0.5 * (xylem_end + n))
        bark_end = float(n - 2)
    else:
        xylem_end, phloem_end, bark_end = n * 0.78, n * 0.90, float(n - 2)

    # Ensure ordered distances with visible segment lengths
    pith_edge = max(3.0, n * 0.02)
    xylem_end = max(xylem_end, pith_edge + n * 0.35)
    phloem_end = max(phloem_end, xylem_end + max(8.0, n * 0.02))
    bark_end = max(bark_end, phloem_end + max(6.0, n * 0.015))
    bark_end = min(bark_end, float(n - 1))
    phloem_end = min(phloem_end, bark_end - 4.0)
    xylem_end = min(xylem_end, phloem_end - 4.0)
    return pith_edge, xylem_end, phloem_end, bark_end


def _estimate_seg_angle(
    img: np.ndarray,
    stem_mask: np.ndarray,
    pith: tuple[float, float],
    green_angle_deg: float,
    min_sep_deg: float = 16.0,
) -> float:
    """Second radial ray for orange/blue/red layer line (manual style)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cx, cy = pith
    best_angle = green_angle_deg + 35.0
    best_score = -1.0

    for angle in np.arange(0, 360, 5):
        diff = abs(((float(angle) - green_angle_deg + 180) % 360) - 180)
        if diff < min_sep_deg or diff > 165.0:
            continue
        profile = _stem_radial_profile(gray, stem_mask, pith, float(angle))
        if len(profile) < 40:
            continue
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, 15), 0).flatten()
        grad = np.abs(np.diff(smooth))
        outer = grad[int(len(grad) * 0.5) :]
        if len(outer) < 5:
            continue
        score = float(np.percentile(outer, 90)) * math.log1p(len(profile))
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def infer_geometry_cv(
    img: np.ndarray,
    green_angle_deg: float | None = None,
) -> LineGeometry | None:
    stem_info = detect_stem_mask(img)
    if stem_info is None:
        return None
    stem_mask, _ = stem_info

    pith = detect_pith(img, stem_mask)
    if green_angle_deg is None:
        green_angle_deg = estimate_green_angle(img, stem_mask, pith)

    seg_angle = _estimate_seg_angle(img, stem_mask, pith, green_angle_deg)
    if not valid_angle_pair(green_angle_deg, seg_angle):
        seg_angle = None
    return build_straight_geometry(img, pith, green_angle_deg, seg_angle, stem_mask)


class BoundaryRegressor(nn.Module):
    def __init__(self, out_dim: int = 8) -> None:
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _geom_to_vector(geom: LineGeometry, w: int, h: int) -> np.ndarray:
    cx, cy = geom.center
    return np.array(
        [
            geom.green_end[0] / w,
            geom.green_end[1] / h,
            geom.seg_x0 / w,
            geom.seg_y0 / h,
            geom.xylem_end / w,
            geom.xylem_end_y / h,
            geom.phloem_end / w,
            geom.bark_end / w,
        ],
        dtype=np.float32,
    )


def _vector_to_geom(vec: np.ndarray, pith: tuple[float, float], w: int, h: int) -> LineGeometry:
    cx, cy = pith
    return LineGeometry(
        center=(cx, cy),
        green_start=(cx, cy),
        green_end=(float(vec[0] * w), float(vec[1] * h)),
        seg_x0=float(vec[2] * w),
        seg_y=float(vec[3] * h),
        xylem_end=float(vec[4] * w),
        phloem_end=float(vec[6] * w),
        bark_end=float(vec[7] * w),
        green_angle_deg=math.degrees(math.atan2(-(vec[1] * h - cy), vec[0] * w - cx)),
        seg_y0=float(vec[3] * h),
        xylem_end_y=float(vec[5] * h),
        phloem_end_y=float(vec[3] * h),
        bark_end_y=float(vec[3] * h),
    )


class CalibrationInference:
    """CV-first hybrid: neural refines CV baseline anchored at detected pith."""

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BoundaryRegressor(out_dim=8).to(self.device)
        self.model.eval()
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._has_model = False
        if model_path:
            try:
                self.load(model_path)
                self._has_model = True
            except Exception:
                self._has_model = False

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        if "head.4.weight" in state and state["head.4.weight"].shape[0] == 9:
            # legacy 9-dim checkpoint: skip incompatible model
            raise ValueError("legacy checkpoint")
        self.model.load_state_dict(state)
        self._has_model = True

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    @torch.no_grad()
    def predict(self, img: np.ndarray, fallback_angle: float | None = None) -> LineGeometry | None:
        stem_info = detect_stem_mask(img)
        if stem_info is None:
            return None
        stem_mask, _ = stem_info
        pith = detect_pith(img, stem_mask)

        cv_geom = infer_geometry_cv(img, green_angle_deg=fallback_angle)
        if cv_geom is None:
            return None

        if not self._has_model:
            return cv_geom

        h, w = img.shape[:2]
        cx, cy = pith
        pad = int(min(h, w) * 0.25)
        x0 = max(int(cx) - pad, 0)
        y0 = max(int(cy) - pad, 0)
        x1 = min(int(cx) + pad, w)
        y1 = min(int(cy) + pad, h)
        crop = img[y0:y1, x0:x1]

        tensor = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        pred = self.model(tensor).cpu().numpy()[0]
        if np.any(np.isnan(pred)) or np.max(np.abs(pred)) > 2:
            return cv_geom

        nn_geom = _vector_to_geom(pred, pith, w, h)
        nn_geom = clamp_geometry(nn_geom, stem_mask)

        # Blend: keep pith + green from CV, layer boundaries averaged
        blended = LineGeometry(
            center=pith,
            green_start=pith,
            green_end=cv_geom.green_end,
            seg_y=nn_geom.seg_y,
            seg_x0=nn_geom.seg_x0,
            xylem_end=(cv_geom.xylem_end * 0.4 + nn_geom.xylem_end * 0.6),
            phloem_end=(cv_geom.phloem_end * 0.4 + nn_geom.phloem_end * 0.6),
            bark_end=(cv_geom.bark_end * 0.4 + nn_geom.bark_end * 0.6),
            green_angle_deg=cv_geom.green_angle_deg,
            seg_angle_deg=cv_geom.seg_angle_deg,
            seg_y0=nn_geom.seg_y0,
            xylem_end_y=(cv_geom.xylem_end_y * 0.4 + nn_geom.xylem_end_y * 0.6),
            phloem_end_y=(cv_geom.phloem_end_y * 0.4 + nn_geom.phloem_end_y * 0.6),
            bark_end_y=(cv_geom.bark_end_y * 0.4 + nn_geom.bark_end_y * 0.6),
        )
        return clamp_geometry(blended, stem_mask)

    def train_from_manual(
        self,
        samples: list[tuple[np.ndarray, LineGeometry]],
        epochs: int = 40,
        lr: float = 1e-4,
    ) -> list[float]:
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        losses: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0
            indices = np.random.permutation(len(samples))
            for idx in indices:
                img, geom = samples[idx]
                stem_info = detect_stem_mask(img)
                if stem_info is None:
                    continue
                stem_mask, _ = stem_info
                pith = detect_pith(img, stem_mask)
                h, w = img.shape[:2]
                cx, cy = pith
                pad = int(min(h, w) * 0.25)
                x0 = max(int(cx) - pad, 0)
                y0 = max(int(cy) - pad, 0)
                x1 = min(int(cx) + pad, w)
                y1 = min(int(cy) + pad, h)
                crop = img[y0:y1, x0:x1]

                fixed = LineGeometry(
                    center=pith,
                    green_start=pith,
                    green_end=geom.green_end,
                    seg_y=geom.seg_y,
                    seg_x0=geom.seg_x0,
                    xylem_end=geom.xylem_end,
                    phloem_end=geom.phloem_end,
                    bark_end=geom.bark_end,
                    green_angle_deg=geom.green_angle_deg,
                    seg_y0=geom.seg_y0,
                    xylem_end_y=geom.xylem_end_y,
                    phloem_end_y=geom.phloem_end_y,
                    bark_end_y=geom.bark_end_y,
                )

                x = self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                y = torch.from_numpy(_geom_to_vector(fixed, w, h)).unsqueeze(0).to(self.device)

                pred = self.model(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                count += 1

            avg = epoch_loss / max(count, 1)
            losses.append(avg)
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1}/{epochs} loss={avg:.5f}")

        self.model.eval()
        self._has_model = True
        return losses
