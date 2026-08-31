"""
从样品 1–40 原图重跑管线，导出 YOLO-seg 训练标签（腔 polygon + 双导管区框）。

默认全部写入 ok 清单；错误归因中漏检/误检图写入 fix_or_partial。

用法:
  python export_review_masks.py
  python export_review_masks.py --lo 1 --hi 40 --out output/dataset_lumen
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from batch_io import BATCH, natural_key  # noqa: E402
from compare_panel import imwrite_unicode  # noqa: E402
from human_rules import (  # noqa: E402
    filter_vessels_by_min_ai,
    find_pairs_auto,
    load_learned_rules,
    select_single_vessels,
)
from learn_from_review import image_key_from_stem  # noqa: E402
from run_sample1 import process_image  # noqa: E402

CLASSES = ["lumen", "multi_vessel_region"]
ATTR_JSON = ROOT / "output" / "错误归因_双导管.json"

# 人工「无双导管区域」：蓝圈小腔 → 训练标签也不写双导管区框
FORCE_SINGLES = {
    "4 (8).tif",
    "6 (13).tif",
    "7 (2).tif",
    "7 (5).tif",
    "8 (4).tif",
    "8 (6).tif",
    "9 (9).tif",
    "11 (4).tif",
    "13 (1).tif",
    "14 (6).tif",
    "17 (4).tif",
    "18 (8).tif",
}


def _sample_id(key: str) -> int | None:
    m = re.match(r"(\d+)\s*\(", key)
    return int(m.group(1)) if m else None


def _stem_key(key: str) -> str:
    """'28 (6).tif' -> '28_(06)'"""
    m = re.match(r"(\d+)\s*\((\d+)\)", key)
    if not m:
        return key.replace(".tif", "").replace(" ", "_")
    return f"{int(m.group(1))}_{int(m.group(2)):02d}"


def _contour_to_yolo(contour: np.ndarray, w: int, h: int, class_id: int) -> str | None:
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return None
    peri = cv2.arcLength(pts.astype(np.float32), True)
    approx = cv2.approxPolyDP(pts.astype(np.float32), max(1.5, 0.005 * peri), True)
    pts = approx.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        pts = contour.reshape(-1, 2).astype(np.float64)
    xs = np.clip(pts[:, 0] / w, 0.0, 1.0)
    ys = np.clip(pts[:, 1] / h, 0.0, 1.0)
    if abs(float(xs.max() - xs.min())) < 1e-4 or abs(float(ys.max() - ys.min())) < 1e-4:
        return None
    body = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{class_id} {body}"


def _pair_box_yolo(v1, v2, w: int, h: int, pad: int = 18) -> str | None:
    x1, y1, bw1, bh1 = cv2.boundingRect(v1.contour)
    x2, y2, bw2, bh2 = cv2.boundingRect(v2.contour)
    x0 = max(0, min(x1, x2) - pad)
    y0 = max(0, min(y1, y2) - pad)
    x1b = min(w - 1, max(x1 + bw1, x2 + bw2) + pad)
    y1b = min(h - 1, max(y1 + bh1, y2 + bh2) + pad)
    corners = np.array([[x0, y0], [x1b, y0], [x1b, y1b], [x0, y1b]], dtype=np.float64)
    return _contour_to_yolo(corners, w, h, class_id=1)


def _load_fix_keys() -> set[str]:
    keys: set[str] = set()
    if not ATTR_JSON.exists():
        return keys
    data = json.loads(ATTR_JSON.read_text(encoding="utf-8"))
    for d in data.get("details", []):
        img = d.get("image") or ""
        if img:
            keys.add(image_key_from_stem(Path(img).stem))
    return keys


def _counted_vessels(vessels, pairs, singles):
    seen: set[str] = set()
    out = []
    for p in pairs:
        for v in (p.v1, p.v2):
            if v.vessel_id not in seen:
                out.append(v)
                seen.add(v.vessel_id)
    for v in singles:
        if v.vessel_id not in seen:
            out.append(v)
            seen.add(v.vessel_id)
    return out


def export_dataset(lo: int, hi: int, out_dir: Path, wipe: bool = False) -> dict:
    if wipe and out_dir.exists():
        shutil.rmtree(out_dir)
    img_train = out_dir / "images" / "train"
    img_val = out_dir / "images" / "val"
    lbl_train = out_dir / "labels" / "train"
    lbl_val = out_dir / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    fix_keys = _load_fix_keys()
    rules = load_learned_rules()
    ok_list: list[str] = []
    fix_list: list[str] = []
    rows: list[dict] = []

    tifs = []
    for f in BATCH.glob("*.tif"):
        key = image_key_from_stem(f.stem)
        sid = _sample_id(key)
        if sid is not None and lo <= sid <= hi:
            tifs.append((key, f))
    tifs.sort(key=lambda kv: natural_key(kv[1]))

    # 按样品号划分：前 80% train，后 20% val（避免同一样品泄漏）
    sample_ids = sorted({_sample_id(k) for k, _ in tifs if _sample_id(k) is not None})
    n_val = max(1, int(round(len(sample_ids) * 0.2)))
    val_samples = set(sample_ids[-n_val:])

    print(f"导出 {lo}-{hi}: {len(tifs)} 张 | val samples={sorted(val_samples)}")
    print(f"输出: {out_dir}")

    for key, path in tifs:
        sid = _sample_id(key)
        split = "val" if sid in val_samples else "train"
        stem = _stem_key(key)

        rgb, gray, vessels, _, scale = process_image(
            path, force_um_per_px=None, hybrid=True
        )
        vessels = filter_vessels_by_min_ai(
            vessels, rules, gray=gray, um_per_px=scale["um_per_px"]
        )
        for v in vessels:
            v.pair_id = None
        h, w = gray.shape[:2]
        if key in FORCE_SINGLES:
            pairs = []
        else:
            pairs = find_pairs_auto(
                vessels,
                scale["um_per_px"],
                path.name,
                img_w=w,
                img_h=h,
                rules=rules,
                gray=gray,
            )
        singles = select_single_vessels(vessels, pairs, w, h, rules=rules)
        counted = _counted_vessels(vessels, pairs, singles)

        lines: list[str] = []
        for v in counted:
            line = _contour_to_yolo(v.contour, w, h, class_id=0)
            if line:
                lines.append(line)
        for p in pairs:
            line = _pair_box_yolo(p.v1, p.v2, w, h)
            if line:
                lines.append(line)

        img_out = (img_train if split == "train" else img_val) / f"{stem}.png"
        lbl_out = (lbl_train if split == "train" else lbl_val) / f"{stem}.txt"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Windows 中文路径下 cv2.imwrite 会静默失败 → 用 imencode+write_bytes
        imwrite_unicode(img_out, bgr)
        lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # 实例 mask 审计目录
        inst_dir = out_dir / "instances" / stem
        inst_dir.mkdir(parents=True, exist_ok=True)
        for i, v in enumerate(counted):
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m, [v.contour], -1, 255, -1)
            imwrite_unicode(inst_dir / f"lumen_{i:02d}.png", m)

        quality = "fix_or_partial" if key in fix_keys else "ok"
        (ok_list if quality == "ok" else fix_list).append(stem)
        rows.append(
            {
                "stem": stem,
                "key": key,
                "split": split,
                "quality": quality,
                "n_lumen": len(counted),
                "n_pairs": len(pairs),
            }
        )
        print(f"  [{split}/{quality}] {stem}: lumen={len(counted)} pairs={len(pairs)}")

    yaml_text = (
        f"# vessel lumen YOLO-seg\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {CLASSES[0]}\n"
        f"  1: {CLASSES[1]}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    manifest = {
        "lo": lo,
        "hi": hi,
        "n_images": len(rows),
        "n_ok": len(ok_list),
        "n_fix_or_partial": len(fix_list),
        "val_samples": sorted(val_samples),
        "ok": ok_list,
        "fix_or_partial": fix_list,
        "rows": rows,
        "classes": CLASSES,
        "note": "ok=默认可用种子；fix_or_partial=错误归因命中图，训练时可降权或人工 scrub",
    }
    (out_dir / "quality_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n完成: {len(rows)} 图 | ok={len(ok_list)} fix={len(fix_list)} | data.yaml 已写"
    )
    return manifest


def main():
    ap = argparse.ArgumentParser(description="导出复核范围 YOLO-seg 数据集")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "dataset_lumen",
    )
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    export_dataset(args.lo, args.hi, out, wipe=args.wipe)


if __name__ == "__main__":
    main()
