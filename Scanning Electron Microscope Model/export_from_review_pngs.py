"""
从「复核_1-40」右侧标注图解析绿腔 / 橙框，导出 YOLO-seg 训练集并可供训练。

图像用原图 TIF（无标注），标签来自复核 PNG：
  class 0 lumen            ← 绿色轮廓（拟合椭圆/填洞）
  class 1 multi_vessel_region ← 橙色框

用法:
  python export_from_review_pngs.py --review 复核_1-40 --wipe
  python train_lumen_seg.py --epochs 20 --model output/models/lumen_yolov8seg/weights/best.pt
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
_pylibs = ROOT / ".pylibs"
if _pylibs.is_dir():
    sys.path.insert(0, str(_pylibs))

from annotate_style import ORANGE  # noqa: E402
from batch_io import BATCH  # noqa: E402
from compare_panel import LABEL_H, imwrite_unicode  # noqa: E402
from learn_from_review import _imread_unicode, image_key_from_stem  # noqa: E402
from run_from_human import _find_tif, natural_key_name  # noqa: E402
from run_sample1 import process_image  # noqa: E402

CLASSES = ["lumen", "multi_vessel_region"]


def _stem_key(key: str) -> str:
    m = re.match(r"(\d+)\s*\((\d+)\)", key)
    if not m:
        return key.replace(".tif", "").replace(" ", "_")
    return f"{int(m.group(1))}_{int(m.group(2)):02d}"


def _sample_id(key: str) -> int | None:
    m = re.match(r"(\d+)\s*\(", key)
    return int(m.group(1)) if m else None


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


def _right_body(compare_bgr: np.ndarray) -> np.ndarray:
    """对照图右半内容区（去掉顶栏），与原图像素对齐。"""
    h, w = compare_bgr.shape[:2]
    right = compare_bgr[:, w // 2 :]
    if right.shape[0] > LABEL_H and float(right[:LABEL_H].mean()) < 60:
        return right[LABEL_H:]
    return right


def _green_mask(body_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (40, 70, 70), (95, 255, 255))
    rgb = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
    m2 = ((g > 140) & (g > r + 35) & (g > b + 35) & (r < 160)).astype(np.uint8) * 255
    # 去掉偏蓝的 TVW / 紫箭
    blueish = ((b > g) & (b > r + 20)).astype(np.uint8) * 255
    m = cv2.bitwise_or(m1, m2)
    m = cv2.bitwise_and(m, cv2.bitwise_not(blueish))
    return m


def _lumen_contours_from_green(body_bgr: np.ndarray) -> list[np.ndarray]:
    mask = _green_mask(body_bgr)
    if int((mask > 0).sum()) < 30:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    out: list[np.ndarray] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 80 or min(bw, bh) < 12:
            continue
        if max(bw, bh) > max(w, h) * 0.7:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        ys, xs = np.where(comp > 0)
        if len(xs) < 20:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        contour = None
        if len(pts) >= 5:
            try:
                (ex, ey), (maj, mino), ang = cv2.fitEllipse(pts)
                if maj >= 10 and mino >= 10 and maj < max(w, h) * 0.8:
                    # 实心椭圆 mask → 外轮廓
                    ell = np.zeros((h, w), np.uint8)
                    cv2.ellipse(
                        ell,
                        (int(ex), int(ey)),
                        (max(1, int(maj * 0.5)), max(1, int(mino * 0.5))),
                        float(ang),
                        0,
                        360,
                        255,
                        -1,
                    )
                    cnts, _ = cv2.findContours(ell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if cnts:
                        contour = max(cnts, key=cv2.contourArea)
            except Exception:
                contour = None
        if contour is None:
            # 回退：笔画填洞
            inv = cv2.bitwise_not(comp)
            ff = inv.copy()
            flood = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, flood, (0, 0), 0)
            interior = (ff > 0).astype(np.uint8) * 255
            if int((interior > 0).sum()) < 50:
                # 用外接椭圆
                (cx, cy), rad = cv2.minEnclosingCircle(pts.reshape(-1, 1, 2))
                if rad < 6:
                    continue
                ell = np.zeros((h, w), np.uint8)
                cv2.circle(ell, (int(cx), int(cy)), int(rad), 255, -1)
                interior = ell
            cnts, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour) < 80:
            continue
        out.append(contour)
    # 去重：中心过近保留较大
    kept: list[np.ndarray] = []
    metas: list[tuple[float, float, float]] = []
    for c in sorted(out, key=cv2.contourArea, reverse=True):
        M = cv2.moments(c)
        if M["m00"] < 1:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        a = float(cv2.contourArea(c))
        if any(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < 28 for ox, oy, _ in metas):
            continue
        kept.append(c)
        metas.append((cx, cy, a))
    return kept


def _orange_box_contours(body_bgr: np.ndarray) -> list[np.ndarray]:
    rgb = cv2.cvtColor(body_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    dist = np.linalg.norm(rgb - np.array(ORANGE, dtype=np.float32), axis=2)
    mask = (dist < 55).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    h, w = mask.shape
    inv = cv2.bitwise_not(mask)
    ff = inv.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 0)
    interior = (ff > 0).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(interior, 8)
    boxes: list[np.ndarray] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 1500 or min(bw, bh) < 40:
            continue
        if max(bw, bh) > max(w, h) * 0.95:
            continue
        corners = np.array(
            [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.int32
        ).reshape(-1, 1, 2)
        boxes.append(corners)
    return boxes


def export_from_review(
    review_dir: Path,
    out_dir: Path,
    *,
    wipe: bool = False,
) -> dict:
    if wipe and out_dir.exists():
        shutil.rmtree(out_dir)
    img_train = out_dir / "images" / "train"
    img_val = out_dir / "images" / "val"
    lbl_train = out_dir / "labels" / "train"
    lbl_val = out_dir / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    pngs = sorted(
        review_dir.glob("*_左原图_右自动.png"),
        key=lambda p: natural_key_name(image_key_from_stem(p.stem)),
    )
    if not pngs:
        raise SystemExit(f"复核目录无对照图: {review_dir}")

    sample_ids = sorted(
        {
            s
            for p in pngs
            if (s := _sample_id(image_key_from_stem(p.stem))) is not None
        }
    )
    n_val = max(1, int(round(len(sample_ids) * 0.2)))
    val_samples = set(sample_ids[-n_val:])

    print(f"从复核导出: {review_dir}")
    print(f"共 {len(pngs)} 张 | val samples={sorted(val_samples)}")
    print(f"输出: {out_dir}\n")

    rows: list[dict] = []
    n_skip = 0
    for png in pngs:
        key = image_key_from_stem(png.stem)
        sid = _sample_id(key)
        split = "val" if sid in val_samples else "train"
        stem = _stem_key(key)

        tif = _find_tif(key)
        if tif is None:
            print(f"  skip no tif: {key}")
            n_skip += 1
            continue

        compare = _imread_unicode(png)
        if compare is None:
            n_skip += 1
            continue
        body = _right_body(compare)
        lumens = _lumen_contours_from_green(body)
        boxes = _orange_box_contours(body)

        # 原图（无标注）作训练图
        rgb, gray, _, _, _ = process_image(tif, hybrid=False)
        h, w = gray.shape[:2]
        # 若复核图与原图尺寸略差，按比例缩放轮廓
        bh, bw = body.shape[:2]
        sx, sy = w / bw, h / bh

        lines: list[str] = []
        for c in lumens:
            cc = c.astype(np.float64)
            cc[:, 0, 0] *= sx
            cc[:, 0, 1] *= sy
            line = _contour_to_yolo(cc.astype(np.int32), w, h, 0)
            if line:
                lines.append(line)
        for c in boxes:
            cc = c.astype(np.float64)
            cc[:, 0, 0] *= sx
            cc[:, 0, 1] *= sy
            line = _contour_to_yolo(cc.astype(np.int32), w, h, 1)
            if line:
                lines.append(line)

        img_out = (img_train if split == "train" else img_val) / f"{stem}.png"
        lbl_out = (lbl_train if split == "train" else lbl_val) / f"{stem}.txt"
        imwrite_unicode(img_out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        rows.append(
            {
                "stem": stem,
                "key": key,
                "split": split,
                "n_lumen": len(lumens),
                "n_pairs": len(boxes),
                "n_label_lines": len(lines),
            }
        )
        print(
            f"  [{split}] {stem}: lumen={len(lumens)} orange={len(boxes)} labels={len(lines)}"
        )

    yaml_text = (
        f"# vessel lumen YOLO-seg (from reviewed PNGs)\n"
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {CLASSES[0]}\n"
        f"  1: {CLASSES[1]}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    manifest = {
        "source": str(review_dir.resolve()),
        "n_images": len(rows),
        "n_skip": n_skip,
        "val_samples": sorted(val_samples),
        "rows": rows,
        "classes": CLASSES,
        "note": "labels parsed from reviewed right-panel green/orange marks; images from TIF",
    }
    (out_dir / "quality_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成: {len(rows)} 图 | skip={n_skip} | data.yaml 已写")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="从复核对照图导出 YOLO-seg 数据集")
    ap.add_argument("--review", type=str, default="复核_1-40")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "dataset_lumen")
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()
    review = Path(args.review)
    if not review.is_absolute():
        review = ROOT / "output" / args.review
    # 容错：按名称匹配复核目录
    if not review.exists():
        outp = ROOT / "output"
        cand = next(
            (p for p in outp.iterdir() if p.is_dir() and "复核" in p.name and "1-40" in p.name),
            None,
        )
        if cand is None:
            raise SystemExit(f"找不到复核目录: {review}")
        review = cand
    out = args.out if args.out.is_absolute() else ROOT / args.out
    export_from_review(review, out, wipe=args.wipe)


if __name__ == "__main__":
    main()
