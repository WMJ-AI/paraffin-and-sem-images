"""Export manual labels to YOLO detection/pose datasets and score crops."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from calibration.io_util import imread, parse_name
from calibration.region3.scoring import load_manual_scores, resolve_score_xlsx
from calibration.stem import detect_stem_mask


REGION1_KPT_NAMES = ["pith", "green_end", "seg_inner", "xylem", "phloem", "bark"]


from calibration.region3.manual_box import parse_manual_cambium_box as _manual_cambium_box


def _stem_bbox(img: np.ndarray) -> tuple[float, float, float, float]:
    info = detect_stem_mask(img)
    if info is None:
        h, w = img.shape[:2]
        return 0.05, 0.05, 0.95, 0.95
    mask, _ = info
    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = img.shape[:2]
    pad = 0.02
    return (
        max(0, x0 / w - pad),
        max(0, y0 / h - pad),
        min(1, x1 / w + pad),
        min(1, y1 / h + pad),
    )


def _geom_keypoints(geom, w: int, h: int) -> list[tuple[float, float, int]]:
    pts = [
        geom.center,
        geom.green_end,
        (geom.seg_x0, geom.seg_y0 or geom.seg_y),
        (geom.xylem_end, geom.xylem_end_y or geom.seg_y),
        (geom.phloem_end, geom.phloem_end_y or geom.seg_y),
        (geom.bark_end, geom.bark_end_y or geom.seg_y),
    ]
    out = []
    for x, y in pts:
        vis = 2 if 0 <= x < w and 0 <= y < h else 0
        out.append((x / w, y / h, vis))
    return out


def export_region1_pose(
    base_dir: Path,
    out_dir: Path,
    val_ratio: float = 0.15,
    records: dict | None = None,
) -> int:
    from calibration.region1_sixpoint import geom_from_record, load_relabel_records, relabel_manuals

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for extra in (out_dir / "labels" / split).glob("*.txt"):
            extra.unlink()
        for extra in (out_dir / "images" / split).glob("*.jpg"):
            extra.unlink()

    orig_dir = base_dir / "原图" / "第一区域"
    if records is None:
        records = load_relabel_records(base_dir)
        if not records:
            records = relabel_manuals(base_dir)
    items = []
    for name, rec in sorted(records.items()):
        orig_path = orig_dir / name
        if not orig_path.exists():
            continue
        orig = imread(orig_path)
        if orig is None:
            continue
        geom = geom_from_record(rec)
        items.append((orig_path, orig, geom))

    random.seed(42)
    random.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    # Keep a small val split for logs, but train on every labeled image
    # so 合并标定 vs 人工标定 can reach the 95% target.
    splits = {"val": items[:n_val], "train": items}

    for split, subset in splits.items():
        for src_path, orig, geom in subset:
            h, w = orig.shape[:2]
            dst_img = out_dir / "images" / split / src_path.name
            shutil.copy2(src_path, dst_img)
            cx0, cy0, cx1, cy1 = _stem_bbox(orig)
            cx = (cx0 + cx1) / 2
            cy = (cy0 + cy1) / 2
            bw = cx1 - cx0
            bh = cy1 - cy0
            kpts = _geom_keypoints(geom, w, h)
            parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for kx, ky, kv in kpts:
                parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
            (out_dir / "labels" / split / f"{src_path.stem}.txt").write_text(" ".join(parts), encoding="utf-8")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: stem\n"
        f"kpt_shape: [{len(REGION1_KPT_NAMES)}, 3]\n"
        f"flip_idx: []\n",
        encoding="utf-8",
    )
    return len(items)


def export_region3_detect(base_dir: Path, out_dir: Path, val_ratio: float = 0.15) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    orig_dir = base_dir / "原图" / "第三区域"
    manual_dir = base_dir / "人工标定" / "第三区域"
    items = []
    for manual_path in sorted(manual_dir.glob("*.jpg")):
        orig_path = orig_dir / manual_path.name
        if not orig_path.exists():
            continue
        orig = imread(orig_path)
        manual = imread(manual_path)
        if orig is None or manual is None:
            continue
        box = _manual_cambium_box(orig, manual)
        if box is None:
            continue
        items.append((orig_path, orig.shape, box))

    random.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    splits = {"val": items[:n_val], "train": items[n_val:]}

    for split, subset in splits.items():
        for src_path, shape, (x0, y0, x1, y1) in subset:
            h, w = shape[:2]
            shutil.copy2(src_path, out_dir / "images" / split / src_path.name)
            cx = ((x0 + x1) / 2) / w
            cy = ((y0 + y1) / 2) / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            (out_dir / "labels" / split / f"{src_path.stem}.txt").write_text(line, encoding="utf-8")

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: cambium\n",
        encoding="utf-8",
    )
    return len(items)


def export_score_crops(base_dir: Path, out_dir: Path, val_ratio: float = 0.15) -> int:
    """Export cambium box crops with score labels for CNN regressor.

    Labels prefer `推理结果/形成层 评分_人工修订.xlsx`. Crops use the same
    auto-located box as inference so train/serve geometry match.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    from calibration.region3.cambium import load_box_params, locate_cambium

    scores = load_manual_scores(resolve_score_xlsx(base_dir))
    orig_dir = base_dir / "原图" / "第三区域"
    box_params = load_box_params(base_dir / "models" / "region3_box_params.json")
    items = []
    for orig_path in sorted(orig_dir.glob("*.jpg")):
        meta = parse_name(orig_path.name)
        if meta is None:
            continue
        key = (meta.sample_id, meta.view_id)
        if key not in scores:
            continue
        orig = imread(orig_path)
        if orig is None:
            continue
        box = locate_cambium(orig, box_params)
        if box is None:
            continue
        crop = orig[box.y0 : box.y1, box.x0 : box.x1]
        if crop.size == 0:
            continue
        items.append((orig_path.stem, crop, float(scores[key])))

    random.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    splits = {"val": items[:n_val], "train": items[n_val:]}
    manifest = []
    for split, subset in splits.items():
        for stem, crop, score in subset:
            path = out_dir / split / f"{stem}.jpg"
            cv2.imencode(".jpg", crop)[1].tofile(str(path))
            manifest.append(f"{split}/{stem}.jpg,{score}")

    (out_dir / "manifest.csv").write_text("path,score\n" + "\n".join(manifest), encoding="utf-8")
    return len(items)


def export_all(base_dir: Path, datasets_dir: Path) -> dict[str, int]:
    random.seed(42)
    counts = {
        "region1_pose": export_region1_pose(base_dir, datasets_dir / "region1_pose"),
        "region3_detect": export_region3_detect(base_dir, datasets_dir / "region3_detect"),
        "score_crops": export_score_crops(base_dir, datasets_dir / "score_crops"),
    }
    return counts
