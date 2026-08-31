"""Prototype richer cambium continuity features vs human-revised scores."""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import cv2
import openpyxl
from collections import defaultdict

from calibration.io_util import imread, parse_name, list_region_images
from calibration.region3.cambium import load_box_params, locate_cambium
from calibration.region3.scoring import load_manual_scores, extract_continuity_features, _composite

base = Path(r"d:\BaiduNetdiskDownload\shibie")


def find_xlsx(folder: Path, *needles: str) -> Path:
    for p in folder.iterdir():
        if p.suffix.lower() != ".xlsx":
            continue
        if all(n in p.name for n in needles):
            return p
    raise FileNotFoundError(needles)


def extract_rich(img: np.ndarray, box) -> dict:
    patch = img[box.y0 : box.y1, box.x0 : box.x1]
    if patch.size == 0:
        return {k: 0.0 for k in [
            "coverage","gap_ratio","break_count","row_jitter","comp",
            "cov30","cov50","cov70","max_gap","mean_gap","n_gaps",
            "strength_cv","low20","ridge_spread","vert_conc",
            "period","tex_energy","contrast","align_mad",
        ]}

    target_w = 1024
    scale = max(1.0, patch.shape[1] / target_w)
    small = cv2.resize(
        patch,
        (target_w, max(32, int(patch.shape[0] / scale))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape

    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_e = np.mean(np.abs(gy), axis=1)
    ridge = int(np.argmax(row_e))
    band_h = max(6, h // 5)
    y0b = max(0, ridge - band_h)
    y1b = min(h, ridge + band_h + 1)
    band = gray[y0b:y1b]
    gy_band = np.abs(gy)[y0b:y1b]
    gx_band = np.abs(gx)[y0b:y1b]

    strength = np.mean(gy_band, axis=0)
    strength = np.convolve(strength, np.ones(9) / 9, mode="same")
    tex = np.mean(gy_band + gx_band, axis=0)
    tex = np.convolve(tex, np.ones(9) / 9, mode="same")
    contrast = np.ptp(band, axis=0)
    contrast = np.convolve(contrast, np.ones(9) / 9, mode="same")

    med = float(np.median(strength)) + 1e-6
    def cov_at(frac):
        return float((strength >= frac * med).mean())

    def gaps_at(frac, min_gap):
        active = strength >= frac * med
        gaps = []
        run = 0
        for v in active:
            if v:
                if run >= min_gap:
                    gaps.append(run)
                run = 0
            else:
                run += 1
        if run >= min_gap:
            gaps.append(run)
        return gaps

    min_gap = max(4, int(w * 0.01))
    g55 = gaps_at(0.55, min_gap)
    g70 = gaps_at(0.70, min_gap)

    peaks = np.array([int(np.argmax(gray[:, x])) for x in range(0, w, 2)], dtype=np.float32)
    peaks_s = np.convolve(peaks, np.ones(7) / 7, mode="same")
    jitter = float(np.std(peaks))
    align_mad = float(np.median(np.abs(peaks - np.median(peaks))))
    ridge_spread = float(np.std(peaks_s))

    # vertical concentration: energy in +/- 15% of height around ridge
    row_e_n = row_e / (row_e.sum() + 1e-6)
    win = max(2, int(0.15 * h))
    vert_conc = float(row_e_n[max(0, ridge - win) : min(h, ridge + win + 1)].sum())

    # periodicity via autocorr of strength
    s = strength - strength.mean()
    if s.std() > 1e-6:
        ac = np.correlate(s, s, mode="full")[len(s) - 1 :]
        ac = ac / (ac[0] + 1e-6)
        # skip lag 0-8, look for first peak
        search = ac[8: min(80, len(ac))]
        period = float(np.max(search)) if len(search) else 0.0
    else:
        period = 0.0

    low20 = float((strength <= np.percentile(strength, 20)).mean())
    strength_cv = float(strength.std() / (strength.mean() + 1e-6))

    feat_old = extract_continuity_features(img, box)
    return {
        "coverage": feat_old.coverage,
        "gap_ratio": feat_old.gap_ratio,
        "break_count": float(feat_old.break_count),
        "row_jitter": feat_old.row_jitter,
        "comp": _composite(feat_old),
        "cov30": cov_at(0.30),
        "cov50": cov_at(0.50),
        "cov70": cov_at(0.70),
        "max_gap": float(max(g55) / w) if g55 else 0.0,
        "mean_gap": float(np.mean(g55) / w) if g55 else 0.0,
        "n_gaps": float(len(g55)),
        "n_gaps70": float(len(g70)),
        "max_gap70": float(max(g70) / w) if g70 else 0.0,
        "strength_cv": strength_cv,
        "low20": low20,
        "ridge_spread": ridge_spread,
        "vert_conc": vert_conc,
        "period": period,
        "tex_energy": float(tex.mean()),
        "contrast": float(contrast.mean()),
        "align_mad": align_mad,
        "jitter": jitter,
        "tex_cv": float(tex.std() / (tex.mean() + 1e-6)),
        "contrast_cv": float(contrast.std() / (contrast.mean() + 1e-6)),
    }


revised = load_manual_scores(find_xlsx(base / "推理结果", "人工"))
box_params = load_box_params(base / "models" / "region3_box_params.json")
files = list_region_images(base, "第三区域", view_ids=(7, 8, 9))

rows = []
for path in files:
    meta = parse_name(path.name)
    if meta is None:
        continue
    key = (meta.sample_id, meta.view_id)
    if key not in revised:
        continue
    img = imread(path)
    if img is None:
        continue
    box = locate_cambium(img, box_params)
    if box is None:
        continue
    feat = extract_rich(img, box)
    feat["y"] = revised[key]
    feat["sample"] = meta.sample_id
    feat["view"] = meta.view_id
    rows.append(feat)
    if len(rows) % 40 == 0:
        print(f"  extracted {len(rows)}")

print(f"n={len(rows)}")
names = [k for k in rows[0] if k not in ("y", "sample", "view")]
y = np.array([r["y"] for r in rows], dtype=float)
print("\n=== Pearson corr vs revised ===")
corrs = []
for n in names:
    x = np.array([r[n] for r in rows], dtype=float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        c = 0.0
    else:
        c = float(np.corrcoef(x, y)[0, 1])
    corrs.append((abs(c), c, n))
for abs_c, c, n in sorted(corrs, reverse=True):
    print(f"  {n:14s} {c:+.3f}")

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

X = np.array([[r[n] for n in names] for r in rows], dtype=float)
groups = np.array([r["sample"] for r in rows])


def metrics(pred, gt):
    pred = np.clip(np.round(pred), 1, 10)
    mae = float(np.mean(np.abs(pred - gt)))
    exact = float(np.mean(pred == gt))
    w1 = float(np.mean(np.abs(pred - gt) <= 1))
    mae_sim = max(0.0, 1.0 - mae / 4.0)
    sim = 0.6 * mae_sim + 0.4 * w1
    return mae, exact, w1, sim


def cv_eval(model_factory, label):
    gkf = GroupKFold(n_splits=5)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    mae, exact, w1, sim = metrics(preds, y)
    print(f"{label:28s} MAE={mae:.3f} exact={exact:.1%} within1={w1:.1%} sim={sim:.1%}")
    return preds


print("\n=== GroupKFold by sample ===")
cv_eval(lambda: make_pipeline(StandardScaler(), Ridge(alpha=2.0)), "Ridge")
cv_eval(lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)), "ElasticNet")
cv_eval(lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0), "GBR")
cv_eval(lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0), "HGBR")

# CNN raw from csv as a feature
csv_path = base / "推理结果" / "region3_scores.csv"
cnn = {}
with csv_path.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        cnn[(int(row["sample_id"]), int(row["view_id"]))] = float(row["raw"])

raw = np.array([cnn[(r["sample"], r["view"])] for r in rows])
print("\nCNN raw only (rounded):", metrics(raw, y))

# isotonic on CNN
iso_preds = np.zeros(len(y))
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(raw.reshape(-1,1), y, groups):
    iso = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip")
    iso.fit(raw[tr], y[tr])
    iso_preds[te] = iso.predict(raw[te])
print("CNN + isotonic CV:", metrics(iso_preds, y))

# Ridge on features + CNN raw
X2 = np.hstack([X, raw.reshape(-1, 1)])
def cv_eval2(model_factory, label):
    gkf = GroupKFold(n_splits=5)
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X2, y, groups):
        m = model_factory()
        m.fit(X2[tr], y[tr])
        preds[te] = m.predict(X2[te])
    mae, exact, w1, sim = metrics(preds, y)
    print(f"{label:28s} MAE={mae:.3f} exact={exact:.1%} within1={w1:.1%} sim={sim:.1%}")
    return preds

print("\n=== features + CNN raw ===")
cv_eval2(lambda: make_pipeline(StandardScaler(), Ridge(alpha=2.0)), "Ridge+CNN")
cv_eval2(lambda: GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=0), "GBR+CNN")
cv_eval2(lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=250, random_state=0), "HGBR+CNN")

# Fit on all, report in-sample (upper bound)
gbr = GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.85, random_state=0)
gbr.fit(X2, y)
print("GBR+CNN in-sample:", metrics(gbr.predict(X2), y))
print("feature importances (top):")
imp = list(zip(names + ["cnn_raw"], gbr.feature_importances_))
for n, v in sorted(imp, key=lambda t: -t[1])[:12]:
    print(f"  {n:14s} {v:.3f}")
