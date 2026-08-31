"""Detect 局部韧皮部 / 局部髓 by an inner tissue line, then boxes inside it.

Column classifier decides which side is phloem vs pith. The inner line follows
the extra/xylem boundary (snapped off xylem vessels). Area is the pixels
inside that line; drawing is the line plus a few stacked rectangles.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

BINS = 48
BLOCK = 48
MAX_SIDE_FRAC = 0.28
MIN_SIDE_FRAC = 0.018
MIN_AREA_FRAC = 0.006
SMOOTH_K = 51


def _column_features(img: np.ndarray, bins: int = BINS) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(bins * 4, w // 4), max(64, h // 4)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local = cv2.blur(blur, (31, 31))
    blob = ((blur.astype(np.int16) - local.astype(np.int16) > 10) & (blur > 140)).astype(np.uint8) * 255
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_small = np.zeros(bins, np.float32)
    n_large = np.zeros(bins, np.float32)
    for c in cnts:
        area = cv2.contourArea(c)
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        bi = int(np.clip((m["m10"] / m["m00"]) / sw * bins, 0, bins - 1))
        if area >= 280:
            n_large[bi] += 1
        elif area >= 30:
            n_small[bi] += 1

    feats = np.zeros((bins, 11), np.float32)
    bw = sw / bins
    bh = float(sh)
    for i in range(bins):
        x0, x1 = int(i * bw), int((i + 1) * bw)
        roi_l = L[:, x0:x1]
        roi_a = a[:, x0:x1]
        roi_b = b[:, x0:x1]
        roi_e = edges[:, x0:x1]
        xf = (i + 0.5) / bins
        feats[i] = [
            float(roi_l.mean()),
            float(roi_l.std()),
            float(roi_a.mean()),
            float(roi_b.mean()),
            float(roi_e.mean()),
            n_small[i],
            n_large[i],
            xf,
            abs(xf - 0.5),
            1.0 if xf < 0.22 or xf > 0.78 else 0.0,
            bh,
        ]
    return feats


def _labels_from_masks(phloem: np.ndarray, pith: np.ndarray, bins: int = BINS) -> np.ndarray:
    h, w = phloem.shape
    lab = np.zeros(bins, np.int32)
    for i in range(bins):
        x0, x1 = int(i / bins * w), int((i + 1) / bins * w)
        p = int((phloem[:, x0:x1] > 0).mean() * 1000)
        m = int((pith[:, x0:x1] > 0).mean() * 1000)
        if p > 80 and p >= m:
            lab[i] = 1
        elif m > 80:
            lab[i] = 2
    return lab


def _blob_counts(img: np.ndarray, gh: int, gw: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local = cv2.blur(blur, (31, 31))
    blob = ((blur.astype(np.int16) - local.astype(np.int16) > 10) & (blur > 140)).astype(np.uint8) * 255
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_small = np.zeros((gh, gw), np.float32)
    n_large = np.zeros((gh, gw), np.float32)
    for c in cnts:
        area = cv2.contourArea(c)
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        r = int(np.clip((m["m01"] / m["m00"]) / block, 0, gh - 1))
        cix = int(np.clip((m["m10"] / m["m00"]) / block, 0, gw - 1))
        if area >= 280:
            n_large[r, cix] += 1
        elif area >= 30:
            n_small[r, cix] += 1
    return n_small, n_large


def _block_features(img: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    gh, gw = h // block, w // block
    crop = img[: gh * block, : gw * block]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    def bmean(ch: np.ndarray) -> np.ndarray:
        return ch.reshape(gh, block, gw, block).mean(axis=(1, 3)).astype(np.float32)

    def bstd(ch: np.ndarray) -> np.ndarray:
        return ch.reshape(gh, block, gw, block).std(axis=(1, 3)).astype(np.float32)

    n_small, n_large = _blob_counts(crop, gh, gw, block)
    yy, xx = np.indices((gh, gw), dtype=np.float32)
    xf = (xx + 0.5) / max(gw, 1)
    yf = (yy + 0.5) / max(gh, 1)
    feats = np.stack(
        [
            bmean(L),
            bstd(L),
            bmean(a),
            bmean(b),
            bmean(edges),
            n_small,
            n_large,
            xf,
            yf,
            np.abs(xf - 0.5),
            (xf < 0.22).astype(np.float32),
            (xf > 0.78).astype(np.float32),
            np.minimum(xf, 1.0 - xf),
        ],
        axis=-1,
    )
    return feats, gh, gw


def _block_labels(phloem: np.ndarray, pith: np.ndarray, gh: int, gw: int, block: int) -> np.ndarray:
    ph = (phloem[: gh * block, : gw * block] > 0).reshape(gh, block, gw, block).mean(axis=(1, 3))
    pi = (pith[: gh * block, : gw * block] > 0).reshape(gh, block, gw, block).mean(axis=(1, 3))
    lab = np.zeros((gh, gw), np.int32)
    lab[ph >= 0.35] = 1
    lab[(pi >= 0.35) & (pi >= ph)] = 2
    return lab


def train_area_model(base_dir: Path, save_path: Path | None = None):
    from sklearn.ensemble import RandomForestClassifier

    from calibration.io_util import imread
    from calibration.region2.manual_areas import parse_manual_areas

    orig_dir = base_dir / "原图" / "第二区域"
    man_dir = base_dir / "人工标定" / "第二区域"
    col_x, col_y = [], []
    blk_x, blk_y = [], []
    rng = np.random.RandomState(0)
    for mp in sorted(man_dir.glob("*.jpg")):
        orig = imread(orig_dir / mp.name)
        manual = imread(mp)
        if orig is None or manual is None:
            continue
        phloem, pith = parse_manual_areas(orig, manual)
        col_x.append(_column_features(orig))
        col_y.append(_labels_from_masks(phloem, pith))
        feats, gh, gw = _block_features(orig, BLOCK)
        labels = _block_labels(phloem, pith, gh, gw, BLOCK)
        xf = feats[..., 7]
        side = (xf < 0.30) | (xf > 0.70)
        keep = (labels > 0) | side | (rng.rand(gh, gw) < 0.10)
        blk_x.append(feats[keep])
        blk_y.append(labels[keep])
    col_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )
    col_clf.fit(np.vstack(col_x), np.concatenate(col_y))
    blk_clf = RandomForestClassifier(
        n_estimators=140,
        max_depth=12,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )
    bx = np.vstack(blk_x)
    by = np.concatenate(blk_y)
    blk_clf.fit(bx, by)
    col_acc = float((col_clf.predict(np.vstack(col_x)) == np.concatenate(col_y)).mean())
    blk_acc = float((blk_clf.predict(bx) == by).mean())
    if save_path is None:
        save_path = base_dir / "models" / "region2_area_clf.joblib"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(
        {
            "version": 2,
            "clf": col_clf,
            "col_clf": col_clf,
            "blk_clf": blk_clf,
            "bins": BINS,
            "block": BLOCK,
            "col_acc": col_acc,
            "blk_acc": blk_acc,
        },
        save_path,
    )
    return {"col_clf": col_clf, "blk_clf": blk_clf, "block": BLOCK}, (col_acc, blk_acc), save_path


def load_area_model(path: Path):
    import joblib

    model = joblib.load(path)
    if isinstance(model, dict):
        for key in ("clf", "col_clf", "blk_clf"):
            est = model.get(key)
            if est is not None and hasattr(est, "n_jobs"):
                est.n_jobs = 1
    elif hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def _model_parts(model):
    if isinstance(model, dict):
        col = model.get("col_clf") or model.get("clf")
        blk = model.get("blk_clf")
        block = int(model.get("block") or BLOCK)
        return col, blk, block
    return model, None, BLOCK


def _side_run(pred: np.ndarray, proba: np.ndarray, from_left: bool) -> tuple[int, int, int] | None:
    n = len(pred)
    max_n = int(MAX_SIDE_FRAC * n)
    min_n = max(1, int(MIN_SIDE_FRAC * n))
    idxs = range(n) if from_left else range(n - 1, -1, -1)
    run: list[int] = []
    for i in idxs:
        if len(run) >= max_n:
            break
        extra = pred[i] in (1, 2) and float(proba[i, 1] + proba[i, 2]) >= 0.42
        if extra:
            run.append(i)
        else:
            break
    if len(run) < min_n:
        return None
    lo, hi = min(run), max(run)
    votes = pred[lo : hi + 1]
    kind = 1 if int((votes == 1).sum()) >= int((votes == 2).sum()) else 2
    return lo, hi + 1, kind


def _column_sides(img: np.ndarray, col_clf) -> list[tuple[bool, int, int, int]]:
    """[(from_left, x0, x1, kind), ...] in pixel coords."""
    h, w = img.shape[:2]
    feats = _column_features(img)
    raw = col_clf.predict_proba(feats)
    classes = list(col_clf.classes_)
    full = np.zeros((len(feats), 3), np.float32)
    for j, c in enumerate(classes):
        full[:, int(c)] = raw[:, j]
    pred = np.argmax(full, axis=1)
    out = []
    for from_left in (True, False):
        span = _side_run(pred, full, from_left)
        if span is None:
            continue
        b0, b1, kind = span
        x0, x1 = int(b0 / BINS * w), int(b1 / BINS * w)
        x0 = max(0, x0)
        x1 = min(w, max(x1, x0 + 1))
        out.append((from_left, x0, x1, kind))
    return out


def _block_extra_mask(
    img: np.ndarray,
    blk_clf,
    block: int,
    from_left: bool,
    x0: int,
    x1: int,
    kind: int,
) -> np.ndarray:
    h, w = img.shape[:2]
    feats, gh, gw = _block_features(img, block)
    proba = blk_clf.predict_proba(feats.reshape(-1, feats.shape[-1]))
    classes = list(blk_clf.classes_)
    full = np.zeros((proba.shape[0], 3), np.float32)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    extra_p = full[:, int(kind)].reshape(gh, gw)
    extra_p = cv2.GaussianBlur(extra_p, (3, 5), 0)
    grid = np.zeros((gh, gw), np.uint8)
    c_lo = max(0, x0 // block)
    c_hi = min(gw, (x1 + block - 1) // block)
    min_run = 3
    for r in range(gh):
        if from_left:
            run = -1
            miss = 0
            for c in range(0, c_hi):
                if extra_p[r, c] >= 0.34:
                    run = c
                    miss = 0
                elif extra_p[r, c] >= 0.20 and run >= 0:
                    run = c
                    miss = 0
                else:
                    miss += 1
                    if miss >= 2:
                        break
            if run + 1 >= min_run:
                grid[r, : run + 1] = 255
        else:
            run = gw
            miss = 0
            for c in range(gw - 1, c_lo - 1, -1):
                if extra_p[r, c] >= 0.34:
                    run = c
                    miss = 0
                elif extra_p[r, c] >= 0.20 and run < gw:
                    run = c
                    miss = 0
                else:
                    miss += 1
                    if miss >= 2:
                        break
            if run < gw and (gw - run) >= min_run:
                grid[r, run:] = 255
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    has = grid.max(axis=1) > 0
    best = (0, 0)
    i = 0
    while i < gh:
        if not has[i]:
            i += 1
            continue
        j = i
        while j < gh and has[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    keep = np.zeros_like(grid)
    if best[1] - best[0] >= 4:
        keep[best[0] : best[1]] = grid[best[0] : best[1]]
    mask = cv2.resize(keep, (gw * block, gh * block), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w), np.uint8)
    mh, mw = mask.shape
    out[:mh, :mw] = mask
    if from_left:
        out[:, x1:] = 0
    else:
        out[:, :x0] = 0
        if mw < w:
            out[:mh, mw:] = np.repeat(out[:mh, mw - 1 : mw], w - mw, axis=1)
    return out


def _inner_polyline(mask: np.ndarray, from_left: bool) -> tuple[np.ndarray, np.ndarray]:
    """Inner x per row of a side band. Vectorized — the old per-pixel loop was ~10M Python steps."""
    h, w = mask.shape
    max_w = min(w, int(MAX_SIDE_FRAC * w) + 8)
    gap = 24
    if from_left:
        band = (mask[:, :max_w] > 0).astype(np.uint8)
    else:
        band = (mask[:, w - max_w :] > 0).astype(np.uint8)[:, ::-1]

    closed = cv2.morphologyEx(band * 255, cv2.MORPH_CLOSE, np.ones((1, 2 * gap + 1), np.uint8))
    stopped = closed == 0
    has_stop = stopped.any(axis=1)
    first_stop = np.argmax(stopped, axis=1)
    first_stop = np.where(has_stop, first_stop, band.shape[1])
    cols = np.arange(band.shape[1], dtype=np.int32)[None, :]
    kept = (band > 0) & (cols < first_stop[:, None])
    any_true = kept.any(axis=1)
    last_from_right = np.argmax(kept[:, ::-1], axis=1)
    run = np.where(any_true, band.shape[1] - last_from_right, 0).astype(np.int32)
    valid = (run >= 16).astype(np.uint8)
    inner = np.full(h, 0 if from_left else w, np.int32)
    if from_left:
        inner[valid > 0] = run[valid > 0]
    else:
        inner[valid > 0] = w - run[valid > 0]
    if int(valid.sum()) < h * 0.08:
        return inner, valid
    idx = np.arange(h)
    good = valid > 0
    filled = inner.astype(np.float32)
    filled[~good] = np.interp(idx[~good], idx[good], inner[good])
    k = SMOOTH_K if SMOOTH_K % 2 == 1 else SMOOTH_K + 1
    pad = np.pad(filled, k // 2, mode="edge")
    smooth = np.convolve(pad, np.ones(k, np.float32) / k, mode="valid")
    # bridge small holes only; do not invent extra at the top/bottom
    valid_img = (valid * 255).reshape(h, 1)
    closed = cv2.morphologyEx(valid_img, cv2.MORPH_CLOSE, np.ones((121, 1), np.uint8))
    valid2 = (closed[:, 0] > 0).astype(np.uint8)
    return np.clip(np.round(smooth), 0, w).astype(np.int32), valid2


def _fill_polyline(h: int, w: int, inner: np.ndarray, valid: np.ndarray, from_left: bool) -> np.ndarray:
    xs = np.arange(w, dtype=np.int32)
    if from_left:
        mask = (xs[None, :] < inner[:, None]) & (valid[:, None] > 0)
    else:
        mask = (xs[None, :] >= inner[:, None]) & (valid[:, None] > 0)
    out = mask.astype(np.uint8) * 255
    out[int(h * 0.90) :, int(w * 0.68) :] = 0
    return out


def polyline_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Inner tissue line (x per row) and whether the band sits on the left."""
    if cv2.countNonZero(mask) == 0:
        return None
    h, w = mask.shape
    xs = np.where(mask > 0)[1]
    from_left = float(xs.mean()) < 0.5 * w
    inner, valid = _inner_polyline(mask, from_left)
    if int(valid.sum()) < 16:
        return None
    return inner, valid, from_left


def _clip_inner_spikes(inner: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Drop one-row spikes so a box top/bottom edge cannot yank the line inward."""
    ys = np.where(valid > 0)[0]
    if ys.size < 12:
        return inner
    step = 8
    sampled = inner[::step].astype(np.float32)
    k = 11
    pad = np.pad(sampled, k // 2, mode="edge")
    lo = np.empty_like(sampled)
    hi = np.empty_like(sampled)
    for i in range(len(sampled)):
        win = pad[i : i + k]
        lo[i] = float(np.percentile(win, 20))
        hi[i] = float(np.percentile(win, 80))
    clipped = np.clip(sampled, lo, hi)
    idx = np.arange(len(inner), dtype=np.float32)
    xp = (np.arange(len(clipped)) * step).astype(np.float32)
    xp[-1] = max(xp[-1], idx[-1])
    return np.interp(idx, xp, clipped).astype(np.int32)


def refine_side_mask(mask: np.ndarray) -> np.ndarray:
    """Rebuild a side band from a despiked inner line (no fat bounding rect)."""
    parsed = polyline_from_mask(mask)
    if parsed is None:
        return mask
    inner, valid, from_left = parsed
    inner = _clip_inner_spikes(inner, valid)
    h, w = mask.shape
    return _fill_polyline(h, w, inner, valid, from_left)


def mask_to_hrects(mask: np.ndarray, merge_tol: int = 28) -> list[tuple[int, int, int, int]]:
    """Split an outlined region into stacked horizontal rectangles (x0,y0,x1,y1)."""
    return [
        (x, y, x + bw, y + bh)
        for x, y, bw, bh in stack_band_boxes(mask)
    ]


def rects_inside_polyline(
    inner: np.ndarray,
    valid: np.ndarray,
    from_left: bool,
    w: int,
    n_bands: int = 10,
    max_w_frac: float = 0.30,
) -> list[tuple[int, int, int, int]]:
    """Stacked rectangles that stay *inside* the dividing line."""
    ys = np.where(valid > 0)[0]
    if ys.size < 32:
        return []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    span = y1 - y0
    n = int(n_bands) if n_bands > 0 else 10
    n = int(np.clip(round(span / 280.0), 3, min(12, n)))
    max_w = int(max_w_frac * w)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(n):
        ya = y0 + int(round(i * span / n))
        yb = y0 + int(round((i + 1) * span / n)) if i < n - 1 else y1
        ok = valid[ya:yb] > 0
        if int(ok.sum()) < 8:
            continue
        sl = inner[ya:yb][ok]
        if from_left:
            xb = int(np.clip(np.percentile(sl, 20), 12, max_w))
            xa = 0
        else:
            xa = int(np.clip(np.percentile(sl, 80), w - max_w, w - 12))
            xb = w
        ww, hh = xb - xa, yb - ya
        if ww >= 12 and hh >= 24:
            boxes.append((xa, ya, ww, hh))
    return boxes


def stack_band_boxes(
    mask: np.ndarray,
    n_bands: int = 10,
    max_w_frac: float = 0.30,
) -> list[tuple[int, int, int, int]]:
    """Cover a line-partitioned side region with stacked rectangles (x, y, w, h).

    The inner tissue line is the partition. Rectangles use a low/high percentile
    of that line in each band so they stay inside and do not become one fat column.
    """
    parsed = polyline_from_mask(mask)
    if parsed is None:
        return []
    inner, valid, from_left = parsed
    inner = _clip_inner_spikes(inner, valid)
    return rects_inside_polyline(
        inner, valid, from_left, mask.shape[1], n_bands=n_bands, max_w_frac=max_w_frac
    )


def band_rects(mask: np.ndarray, n_bands: int = 14) -> list[tuple[int, int, int, int]]:
    """Turn a partitioned side region into stacked ImageJ-style rectangles."""
    return stack_band_boxes(mask, n_bands=n_bands)


def side_edge_boxes(
    boxes: list[tuple[int, int, int, int]], w: int, h: int
) -> list[tuple[int, int, int, int]]:
    """Keep only 韧皮部/髓 boxes that sit on the left or right image edge."""
    edge = int(0.045 * w)
    out: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh in boxes:
        if bw < 12 or bh < 16 or bw > 0.36 * w:
            continue
        cx = x + bw * 0.5
        on_left = x <= edge
        on_right = x + bw >= w - edge
        if not (on_left or on_right):
            continue
        if 0.34 * w < cx < 0.66 * w:
            continue
        out.append((x, y, bw, bh))
    return out


def hrects_area_px(rects: list[tuple[int, int, int, int]]) -> int:
    return int(sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects))


def _vessel_front(img: np.ndarray, from_left: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-row x of the xylem mass facing that edge (empty round lumens).

    Walks a side band in short horizontal strips so the front can curve
    (髓 is often thicker in the middle). Returns (front_x, ok).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 4
    sw, sh = max(8, w // scale), max(8, h // scale)
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    local = cv2.blur(blur, (25, 25))
    ves = ((blur.astype(np.int16) - local.astype(np.int16) > 14) & (blur > 165)).astype(np.uint8)
    ves = cv2.morphologyEx(
        ves * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ves, connectivity=8)
    clean = np.zeros((sh, sw), np.uint8)
    for i in range(1, n):
        _x, _y, bw, bh, area = stats[i]
        if area < 12 or area > 2800:
            continue
        if max(int(bw), int(bh)) / max(min(int(bw), int(bh)), 1) > 2.8:
            continue
        clean[labels == i] = 1

    strip = max(8, 48 // scale)
    search = min(sw, int((MAX_SIDE_FRAC + 0.08) * sw) + 4)
    n_strips = max(1, sh // strip)
    front_s = np.full(n_strips, search if from_left else sw - search, np.float32)
    ok_s = np.zeros(n_strips, np.uint8)
    y_s = np.zeros(n_strips, np.float32)
    for i in range(n_strips):
        y0 = i * strip
        y1 = sh if i == n_strips - 1 else (i + 1) * strip
        y_s[i] = 0.5 * (y0 + y1)
        occ = clean[y0:y1].mean(axis=0)
        if from_left:
            xs = np.where(occ[:search] > 0.035)[0]
            if xs.size:
                front_s[i] = float(xs.min())
                ok_s[i] = 1
        else:
            xs = np.where(occ[sw - search :] > 0.035)[0]
            if xs.size:
                front_s[i] = float(sw - search + int(xs.max()))
                ok_s[i] = 1
    front = np.full(h, w if from_left else 0, np.int32)
    ok = np.zeros(h, np.uint8)
    if int(ok_s.sum()) < 2:
        return front, ok
    good = ok_s > 0
    yy = (y_s * scale).astype(np.float32)
    fy = np.arange(h, dtype=np.float32)
    interp = np.interp(fy, yy[good], front_s[good] * scale)
    k = 51
    pad = np.pad(interp, k // 2, mode="edge")
    smooth = np.convolve(pad, np.ones(k, np.float32) / k, mode="valid")
    front[:] = np.clip(np.round(smooth), 0, w).astype(np.int32)
    # rows covered by strips that actually saw vessels, plus a small close
    ok_img = np.zeros(sh, np.uint8)
    for i in range(n_strips):
        if ok_s[i]:
            y0 = i * strip
            y1 = sh if i == n_strips - 1 else (i + 1) * strip
            ok_img[y0:y1] = 1
    ok_img = cv2.morphologyEx(ok_img.reshape(sh, 1), cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8))[:, 0]
    ok = cv2.resize(ok_img, (1, h), interpolation=cv2.INTER_NEAREST).ravel()
    return front, (ok > 0).astype(np.uint8)


def _snap_inner_to_vessels(
    inner: np.ndarray,
    valid: np.ndarray,
    front: np.ndarray,
    front_ok: np.ndarray,
    from_left: bool,
    pad: int,
    w: int,
) -> np.ndarray:
    """Pull the tissue line back so it does not swallow xylem vessels."""
    out = inner.copy()
    take = (valid > 0) & (front_ok > 0)
    if from_left:
        limit = front - pad
        hit = take & (limit < out)
        out[hit] = np.clip(limit[hit], 8, w)
    else:
        limit = front + pad
        hit = take & (limit > out)
        out[hit] = np.clip(limit[hit], 0, w - 8)
    return out


def _pith_from_texture(img: np.ndarray, from_left: bool) -> np.ndarray:
    """Side band of large parenchyma when the RF misses 髓 / 随心部.

    Prefer the curved xylem-vessel front (髓 is thicker in the middle). Fall
    back to a gray-std walk if that front is missing.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pad = max(16, int(0.006 * w))
    front, ok = _vessel_front(img, from_left)
    if int(ok.sum()) >= int(h * 0.12):
        inner = np.full(h, 0 if from_left else w, np.int32)
        valid = np.zeros(h, np.uint8)
        max_w = int(MAX_SIDE_FRAC * w)
        if from_left:
            width = np.clip(front - pad, 8, max_w)
            keep = (ok > 0) & (width >= 16)
            inner[keep] = width[keep]
            valid[keep] = 1
        else:
            x0 = np.clip(front + pad, w - max_w, w - 8)
            keep = (ok > 0) & ((w - x0) >= 16)
            inner[keep] = x0[keep]
            valid[keep] = 1
        inner = _clip_inner_spikes(inner, valid)
        out = _fill_polyline(h, w, inner, valid, from_left)
        out[int(h * 0.90) :, int(w * 0.68) :] = 0
        out[gray > 245] = 0
        if cv2.countNonZero(out) >= h * w * MIN_AREA_FRAC:
            return out

    max_w = int(MAX_SIDE_FRAC * w)
    col_w = max(20, w // 90)
    xref0, xref1 = int(0.40 * w), int(0.60 * w)
    ref_std = float(gray[:, xref0:xref1].std())
    n = max(4, max_w // col_w)
    inner_x = 0 if from_left else w
    n_keep = 0
    for i in range(n):
        if from_left:
            x0, x1 = i * col_w, min(max_w, (i + 1) * col_w)
        else:
            x1 = w - i * col_w
            x0 = max(w - max_w, x1 - col_w)
        std = float(gray[:, x0:x1].std())
        slack = 5.0 if n_keep == 0 else 2.5
        if std < ref_std + slack:
            break
        n_keep += 1
        inner_x = x1 if from_left else x0
    if n_keep < 2:
        return np.zeros((h, w), np.uint8)
    out = np.zeros((h, w), np.uint8)
    if from_left:
        out[:, :inner_x] = 255
    else:
        out[:, inner_x:] = 255
    out[int(h * 0.90) :, int(w * 0.68) :] = 0
    out[gray > 245] = 0
    if cv2.countNonZero(out) < h * w * MIN_AREA_FRAC:
        return np.zeros((h, w), np.uint8)
    return out


def _keep_side_band(mask: np.ndarray, from_left: bool, max_frac: float = MAX_SIDE_FRAC) -> np.ndarray:
    """Clip a side mask to at most max_frac of the width from that edge."""
    if cv2.countNonZero(mask) == 0:
        return mask
    h, w = mask.shape
    limit = int(max_frac * w)
    out = mask.copy()
    if from_left:
        out[:, limit:] = 0
    else:
        out[:, : w - limit] = 0
    return out


def detect_phloem_pith(img: np.ndarray, clf, vessels=None) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    phloem = np.zeros((h, w), np.uint8)
    pith = np.zeros((h, w), np.uint8)
    col_clf, blk_clf, block = _model_parts(clf)
    if col_clf is None:
        return phloem, pith
    sides = _column_sides(img, col_clf)

    for from_left, x0, x1, kind in sides:
        slack = block * 2
        if from_left:
            x1 = min(w, x1 + slack)
        else:
            x0 = max(0, x0 - slack)
        if blk_clf is not None:
            raw = _block_extra_mask(img, blk_clf, block, from_left, x0, x1, kind)
        else:
            raw = np.zeros((h, w), np.uint8)
            raw[:, x0:x1] = 255
        inner, valid = _inner_polyline(raw, from_left)
        front, front_ok = _vessel_front(img, from_left)
        inner = _snap_inner_to_vessels(
            inner, valid, front, front_ok, from_left, max(16, int(0.006 * w)), w
        )
        inner = _clip_inner_spikes(inner, valid)
        refined = _fill_polyline(h, w, inner, valid, from_left)
        refined = _keep_side_band(refined, from_left)
        if cv2.countNonZero(refined) < h * w * MIN_AREA_FRAC:
            continue
        target = phloem if kind == 1 else pith
        target[refined > 0] = 255

    # Opposite the phloem: 随心部 is often large parenchyma the RF calls xylem.
    # Do not paint 导管 there.
    if cv2.countNonZero(phloem) > 0 and cv2.countNonZero(pith) == 0:
        ys, xs = np.where(phloem > 0)
        ph_cx = float(xs.mean())
        pith_from_left = ph_cx > 0.5 * w
        guessed = _pith_from_texture(img, pith_from_left)
        if cv2.countNonZero(guessed) > 0:
            guessed = cv2.subtract(guessed, phloem)
            pith = guessed

    # Final side clip — never let 韧皮部/髓 spill into the mid FOV.
    if cv2.countNonZero(phloem) > 0:
        ys, xs = np.where(phloem > 0)
        phloem = _keep_side_band(phloem, float(xs.mean()) < 0.5 * w)
        phloem = refine_side_mask(phloem)
    if cv2.countNonZero(pith) > 0:
        ys, xs = np.where(pith > 0)
        pith = _keep_side_band(pith, float(xs.mean()) < 0.5 * w)
        pith = refine_side_mask(pith)
    return phloem, pith
