"""Cambium continuity scorer: wide-strip CNN + 1D ridge profile.

The cambium crop is a long horizontal band (~5.7:1). Square 224x224 resize
destroys gap/continuity cues, so inputs stay wide (96x576). A cheap 1D
profile branch injects explicit horizontal-fault signal.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF


STRIP_H = 96
STRIP_W = 576
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HybridStripRegressor(nn.Module):
    """ImageNet MobileNet features on a wide strip + 1D continuity profile."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.profile = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        feat = self.pool(self.features(x))
        prof = self.profile(profile)
        return self.head(torch.cat([feat.flatten(1), prof.flatten(1)], dim=1))


def _column_profile(bgr_or_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2GRAY) if bgr_or_rgb.shape[2] == 3 else bgr_or_rgb
    gray = gray.astype(np.float32)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    strength = np.mean(np.abs(gy), axis=0)
    if strength.size != STRIP_W:
        strength = cv2.resize(strength.reshape(1, -1), (STRIP_W, 1), interpolation=cv2.INTER_AREA).ravel()
    strength = np.convolve(strength, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    strength = strength / (float(np.median(strength)) + 1e-6)
    return strength.astype(np.float32)


def _resize_strip(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.size == 0:
        return np.zeros((STRIP_H, STRIP_W, 3), dtype=np.uint8)
    return cv2.resize(crop_bgr, (STRIP_W, STRIP_H), interpolation=cv2.INTER_AREA)


def _augment_crop(crop_bgr: np.ndarray) -> np.ndarray:
    img = crop_bgr
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    fw = random.uniform(0.88, 1.0)
    fh = random.uniform(0.82, 1.0)
    nw, nh = max(16, int(w * fw)), max(8, int(h * fh))
    x0 = random.randint(0, max(0, w - nw))
    y0 = random.randint(0, max(0, h - nh))
    img = img[y0 : y0 + nh, x0 : x0 + nw]
    alpha = random.uniform(0.82, 1.18)
    beta = random.uniform(-18, 18)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        k = random.choice((3, 5))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def _to_tensor(strip_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


class ScoreModel:
    def __init__(self, weights: str | Path | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = HybridStripRegressor().to(self.device)
        self.model.eval()
        self.calib_a = 1.0
        self.calib_b = 0.0
        if weights and Path(weights).exists():
            self.load(str(weights))

    def load(self, path: str) -> None:
        raw = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            self.model.load_state_dict(raw["state_dict"])
            self.calib_a = float(raw.get("calib_a", 1.0))
            self.calib_b = float(raw.get("calib_b", 0.0))
        else:
            # Legacy square MobileNet checkpoint cannot load into hybrid strip head.
            try:
                self.model.load_state_dict(raw, strict=False)
            except Exception:
                pass
        self.model.eval()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": "hybrid_strip_v1",
                "state_dict": self.model.state_dict(),
                "input_hw": [STRIP_H, STRIP_W],
                "calib_a": self.calib_a,
                "calib_b": self.calib_b,
            },
            path,
        )

    def _predict_strip(self, strip_bgr: np.ndarray) -> float:
        x = _to_tensor(strip_bgr).unsqueeze(0).to(self.device)
        prof = torch.from_numpy(_column_profile(strip_bgr)).view(1, 1, -1).to(self.device)
        return float(self.model(x, prof).item())

    @torch.no_grad()
    def predict_crop_raw(self, crop_bgr: np.ndarray, tta: bool = True) -> float:
        if crop_bgr is None or crop_bgr.size == 0:
            return 1.0
        strip = _resize_strip(crop_bgr)
        pred = self._predict_strip(strip)
        if tta:
            pred = 0.5 * (pred + self._predict_strip(cv2.flip(strip, 1)))
        return float(np.clip(self.calib_a * pred + self.calib_b, 1.0, 10.0))

    @torch.no_grad()
    def predict_crop(self, crop_bgr: np.ndarray) -> int:
        return int(np.clip(round(self.predict_crop_raw(crop_bgr)), 1, 10))

    def _batch_tensors(
        self, crops: list[np.ndarray], scores: list[float], augment: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, ps, ys = [], [], []
        for crop, score in zip(crops, scores):
            img = _augment_crop(crop) if augment else crop
            strip = _resize_strip(img)
            xs.append(_to_tensor(strip))
            ps.append(torch.from_numpy(_column_profile(strip)).view(1, -1))
            ys.append(score)
        x = torch.stack(xs).to(self.device)
        p = torch.stack(ps).to(self.device)
        y = torch.tensor(ys, dtype=torch.float32, device=self.device).unsqueeze(1)
        return x, p, y

    def fit_samples(
        self,
        samples: list[tuple[np.ndarray, float]],
        epochs: int = 60,
        lr: float = 8e-4,
        batch_size: int = 12,
        val_samples: list[tuple[np.ndarray, float]] | None = None,
        sample_weights: list[float] | None = None,
        freeze_backbone_epochs: int = 12,
    ) -> list[float]:
        if not samples:
            return []

        if sample_weights is None:
            counts = Counter(int(np.clip(round(s), 1, 10)) for _, s in samples)
            sample_weights = [1.0 / np.sqrt(counts[int(np.clip(round(s), 1, 10))]) for _, s in samples]

        def _set_backbone_trainable(trainable: bool) -> None:
            for param in self.model.features.parameters():
                param.requires_grad = trainable

        def _make_opt(learning_rate: float) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=learning_rate,
                weight_decay=1e-4,
            )

        self.model.train()
        _set_backbone_trainable(freeze_backbone_epochs <= 0)
        opt = _make_opt(lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        loss_fn = nn.SmoothL1Loss(reduction="none")
        losses: list[float] = []
        best_state = None
        best_val = 1e9

        for epoch in range(epochs):
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
                _set_backbone_trainable(True)
                opt = _make_opt(lr * 0.4)
                remain = max(epochs - epoch, 1)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain)

            order = np.random.permutation(len(samples))
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(samples), batch_size):
                idx = order[start : start + batch_size]
                batch = [samples[int(i)] for i in idx]
                weights = torch.tensor(
                    [sample_weights[int(i)] for i in idx],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(1)
                crops = [item[0] for item in batch]
                scores = [item[1] for item in batch]
                x, p, y = self._batch_tensors(crops, scores, augment=True)
                pred = self.model(x, p)
                loss = (loss_fn(pred, y) * weights).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            sched.step()
            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            do_val = val_samples is not None and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs)
            if do_val:
                val_mae = self._eval_mae(val_samples)
                print(f"    epoch {epoch + 1:3d}/{epochs} loss={avg:.4f} val_mae={val_mae:.3f}", flush=True)
                if val_mae < best_val:
                    best_val = val_mae
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            elif (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1:3d}/{epochs} loss={avg:.4f}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return losses

    @torch.no_grad()
    def _eval_mae(self, samples: list[tuple[np.ndarray, float]]) -> float:
        self.model.eval()
        errs = []
        for crop, score in samples:
            pred = float(self.predict_crop_raw(crop, tta=False))
            errs.append(abs(pred - score))
        self.model.train()
        return float(np.mean(errs)) if errs else 1e9

    def train_from_manifest(
        self,
        manifest_path: Path,
        crops_root: Path,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 16,
    ) -> list[float]:
        rows = manifest_path.read_text(encoding="utf-8").strip().splitlines()[1:]
        samples: list[tuple[np.ndarray, float]] = []
        for row in rows:
            path, score = row.split(",")
            if not path.startswith("train/"):
                continue
            img = cv2.imdecode(np.fromfile(str(crops_root / path), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                samples.append((img, float(score)))
        return self.fit_samples(samples, epochs=epochs, lr=lr, batch_size=batch_size)
