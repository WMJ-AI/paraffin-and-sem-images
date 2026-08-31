"""Fresh Region-1 pipeline: LabelMe JSON -> YOLO-Pose train -> re-label originals.

Discards previous region-1 training/inference artifacts, then:
  1. Parse 原图/第一区域/*.json (labels 1-6 + line 10)
  2. Export YOLO pose dataset
  3. Train from yolo11n-pose (not old region1 weights)
  4. Infer with YOLO and write 自动标定/第一区域
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from calibration.geometry import draw_calibration_lines
from calibration.io_util import imread, imwrite, list_region_images, parse_name
from calibration.metrics import measure_region1
from calibration.region1_sixpoint import (
    geom_from_record,
    geom_json_starts_predicted_rays,
    load_labelme_records,
)
from calibration.scale import get_scale_info
from calibration.stem import detect_stem_mask
from ml.export_datasets import export_region1_pose
from ml.metrics import region1_geometry_similarity
from ml.yolo_infer import YoloRegion1Model
from ml.yolo_train import train_yolo_pose


REGION = "第一区域"
RUN_NAME = "region1_pose_labelme"


def wipe_region1_artifacts(base_dir: Path) -> None:
    """Remove prior region-1 training / inference products (keep LabelMe + originals)."""
    targets: list[Path] = [
        base_dir / "datasets" / "region1_pose",
        base_dir / "models" / "region1_boundary.pt",
        base_dir / "自动标定" / REGION,
        base_dir / "合并标定" / REGION,
        base_dir / "推理结果" / "coords" / REGION,
        base_dir / "推理结果" / "dbg_sixpoint",
        base_dir / "推理结果" / "dbg_starts",
        base_dir / "推理结果" / "iterate_logs",
    ]
    for p in targets:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"  已删除目录 {p.relative_to(base_dir)}")
        elif p.is_file():
            p.unlink(missing_ok=True)
            print(f"  已删除文件 {p.relative_to(base_dir)}")

    yolo_dir = base_dir / "models" / "yolo"
    if yolo_dir.exists():
        for child in list(yolo_dir.iterdir()):
            name = child.name
            if name.startswith("region1_pose") or name.startswith("region1_"):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                print(f"  已删除 {child.relative_to(base_dir)}")

    infer = base_dir / "推理结果"
    if infer.exists():
        for pat in (
            "region1_*.json",
            "region1_*.csv",
            "labelme_first30_raw.txt",
            "dbg_r1_*.jpg",
            "debug_fixed_*.jpg",
        ):
            for f in infer.glob(pat):
                f.unlink(missing_ok=True)
                print(f"  已删除 {f.relative_to(base_dir)}")


def resolve_pretrained(models_dir: Path) -> Path | str:
    """Use local yolo11n-pose.pt if present; else ultralytics name (may download)."""
    local = models_dir / "yolo11n-pose.pt"
    if local.exists():
        return local
    cwd = Path("yolo11n-pose.pt")
    if cwd.exists():
        shutil.copy2(cwd, local)
        return local
    return "yolo11n-pose.pt"


def train_from_labelme(base_dir: Path, epochs: int, imgsz: int, batch: int) -> Path:
    records = load_labelme_records(base_dir)
    if len(records) < 10:
        raise RuntimeError(f"LabelMe 有效样本过少: {len(records)}")

    datasets_dir = base_dir / "datasets" / "region1_pose"
    models_dir = base_dir / "models" / "yolo"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[第一区域] 导出 YOLO Pose 数据集...")
    n = export_region1_pose(base_dir, datasets_dir, records=records)
    print(f"  有效样本: {n}")

    start = resolve_pretrained(models_dir)
    print(f"[第一区域] 训练 YOLO name={RUN_NAME} epochs={epochs} start={start}")
    weights = train_yolo_pose(
        datasets_dir / "data.yaml",
        models_dir,
        name=RUN_NAME,
        epochs=epochs,
        imgsz=imgsz,
        weights=start,
        batch=batch,
    )
    best = models_dir / "region1_pose_best.pt"
    shutil.copy2(weights, best)
    print(f"[第一区域] 权重 -> {best}")
    return best


def relabel_originals(base_dir: Path, weights: Path, light: bool = True) -> None:
    """Predict on all 原图/第一区域 and write 自动标定.

    Same-stem LabelMe JSON present → guided geometry; otherwise pure YOLO.
    """
    auto_dir = base_dir / "自动标定" / REGION
    merge_dir = base_dir / "合并标定" / REGION
    metrics_csv = base_dir / "推理结果" / "region1_metrics.csv"
    auto_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    for extra in auto_dir.glob("*.jpg"):
        extra.unlink()
    for extra in merge_dir.glob("*.jpg"):
        extra.unlink()

    # Prefer on-disk sidecar JSON next to each image; also accept aggregated records.
    gt = load_labelme_records(base_dir)
    model = YoloRegion1Model(weights)
    print(f"[第一区域] YOLO 重标: {weights}")

    files = list_region_images(base_dir, REGION, view_ids=(1, 2, 3))
    print(f"[第一区域] 推理 {len(files)} 张")

    rows = []
    sims: list[float] = []
    ok = 0
    for i, path in enumerate(files):
        meta = parse_name(path.name)
        orig = imread(path)
        if orig is None:
            continue
        mag = meta.magnification if meta else 2
        stem_info = detect_stem_mask(orig)
        stem_mask = stem_info[0] if stem_info is not None else None

        yolo_geom = None
        try:
            yolo_geom = model.predict(orig, mag=mag, light=light)
        except Exception as exc:
            print(f"  失败 {path.name}: {exc}")
            continue

        has_json = path.name in gt
        if has_json:
            geom = geom_json_starts_predicted_rays(
                orig, gt[path.name], yolo_geom, stem_mask, mag=mag
            )
        else:
            geom = yolo_geom

        if geom is None:
            print(f"  跳过(无预测): {path.name}")
            continue

        auto_img = draw_calibration_lines(orig, geom, stem_mask, numbered=True)
        imwrite(auto_dir / path.name, auto_img)

        sim = None
        if has_json:
            gt_geom = geom_from_record(gt[path.name])
            h, w = orig.shape[:2]
            diag = (h * h + w * w) ** 0.5
            sim = region1_geometry_similarity(geom, gt_geom, diag)
            sims.append(float(sim))
            from calibration.merge import stitch_horizontal

            gt_img = draw_calibration_lines(orig, gt_geom, stem_mask, numbered=True)
            imwrite(merge_dir / path.name, stitch_horizontal(gt_img, auto_img))

        meas = None
        scale = get_scale_info(orig, mag)
        if scale is not None:
            meas = measure_region1(geom, scale)

        rows.append(
            {
                "filename": path.name,
                "sample_id": meta.sample_id if meta else "",
                "view_id": meta.view_id if meta else "",
                "magnification": mag,
                "radius_um": meas.radius_um if meas else "",
                "xylem_um": meas.xylem_um if meas else "",
                "phloem_um": meas.phloem_um if meas else "",
                "bark_um": meas.bark_um if meas else "",
                "backend": "yolo",
                "sim_vs_labelme": "" if sim is None else f"{sim:.4f}",
            }
        )
        ok += 1
        if (i + 1) % 10 == 0 or i == 0:
            extra = f" sim={sim:.1%}" if sim is not None else ""
            print(f"  {i + 1}/{len(files)}: {path.name}{extra}")

    if rows:
        with metrics_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[第一区域] 指标 CSV: {metrics_csv}")

    if sims:
        import numpy as np

        arr = np.array(sims, dtype=np.float64)
        print(
            f"[第一区域] vs LabelMe mean={arr.mean():.1%}  "
            f">=95%={float((arr >= 0.95).mean()):.1%}  "
            f">=85%={float((arr >= 0.85).mean()):.1%}  n={len(arr)}"
        )
    print(f"[第一区域] 自动标定 {ok} 张 -> {auto_dir}")
    print(f"          对照图 -> {merge_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="第一区域：LabelMe 重新训练并重标原图")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--no-wipe", action="store_true", help="不清理旧第一区域训练产物")
    parser.add_argument("--no-train", action="store_true", help="跳过训练，只用现有 region1_pose_best.pt")
    parser.add_argument("--no-infer", action="store_true", help="只训练不重标")
    parser.add_argument("--no-light", action="store_true", help="推理时做完整规则扣齐（默认 light 更贴近 JSON）")
    args = parser.parse_args()

    base = args.base
    if not args.no_wipe:
        print("[第一区域] 清理旧训练/推理产物...")
        wipe_region1_artifacts(base)

    weights = base / "models" / "yolo" / "region1_pose_best.pt"
    if not args.no_train:
        weights = train_from_labelme(base, args.epochs, args.imgsz, args.batch)
    elif not weights.exists():
        raise SystemExit(f"未找到权重 {weights}，请去掉 --no-train")

    if not args.no_infer:
        relabel_originals(base, weights, light=not args.no_light)


if __name__ == "__main__":
    main()
