"""Region-2 pipeline: mark all vessel lumens + xylem-area metrics."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import cv2

cv2.setNumThreads(4)

from calibration.excel_export import (
    SampleRecord,
    empty_records,
    records_from_region2_csv,
    view_index,
    write_region2_excel,
)
from calibration.io_util import imread, imwrite, list_region_images, parse_name
from calibration.merge import stitch_horizontal
from calibration.region2.draw import draw_region2
from calibration.region2.vessels import analyze_region2


def _name_key(filename: str) -> tuple[int, int, str]:
    meta = parse_name(filename)
    if meta is None:
        return (10**9, 9, filename)
    return (meta.sample_id, meta.view_id, filename)


def _in_name_range(filename: str, from_name: str | None, to_name: str | None) -> bool:
    key = _name_key(filename)
    if from_name and key < _name_key(from_name):
        return False
    if to_name and key > _name_key(to_name):
        return False
    return True


def _flush_metrics(metrics_csv: Path, csv_rows: list[dict]) -> list[dict]:
    if not csv_rows:
        return []
    merged: dict[str, dict] = {}
    if metrics_csv.exists():
        with metrics_csv.open(newline="", encoding="utf-8-sig") as f:
            merged = {row["filename"]: row for row in csv.DictReader(f)}
    for row in csv_rows:
        merged[row["filename"]] = row
    fieldnames = list(csv_rows[0].keys())
    ordered = sorted(merged.values(), key=lambda r: _name_key(r["filename"]))
    with metrics_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return ordered


def _load_or_train_areas(base_dir: Path, retrain: bool = False):
    from calibration.region2.areas import load_area_model, train_area_model

    path = base_dir / "models" / "region2_area_clf.joblib"
    if retrain or not path.exists():
        print("[第二区域] 从人工标定学习韧皮部/髓面积框...")
        clf, acc, path = train_area_model(base_dir, path)
        if isinstance(acc, tuple):
            print(f"  列分类={acc[0]:.1%} 块分类={acc[1]:.1%} -> {path}")
        else:
            print(f"  列分类训练准确率={acc:.1%} -> {path}")
        return clf
    print(f"[第二区域] 加载面积分类器: {path}", flush=True)
    return load_area_model(path)


def run_region2(
    base_dir: Path,
    records: dict[int, SampleRecord] | None = None,
    from_name: str | None = None,
    to_name: str | None = None,
    after_name: str | None = None,
    retrain_areas: bool = False,
) -> dict[int, SampleRecord]:
    region = "第二区域"
    orig_dir = base_dir / "原图" / region
    manual_dir = base_dir / "人工标定" / region
    auto_dir = base_dir / "自动标定" / region
    merge_dir = base_dir / "合并标定" / region
    metrics_csv = base_dir / "推理结果" / "region2_metrics.csv"

    auto_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = empty_records()

    from calibration.region2.rules import MIN_AREA_UM2, MIN_DIAM_UM

    print(f"[第二区域] 最小导管: 直径 {MIN_DIAM_UM:.0f} um / {MIN_AREA_UM2:.0f} um2 (贴近人工黄框里的腔)", flush=True)

    manual_names = {p.name for p in manual_dir.glob("*.jpg")}
    for extra in merge_dir.glob("*.jpg"):
        if extra.name not in manual_names:
            extra.unlink()

    area_clf = _load_or_train_areas(base_dir, retrain=retrain_areas)
    files = [
        p
        for p in list_region_images(base_dir, region, view_ids=(4, 5, 6))
        if _in_name_range(p.name, from_name, to_name)
    ]
    if after_name:
        names = [p.name for p in files]
        if after_name in names:
            files = files[names.index(after_name) + 1 :]
            print(f"[第二区域] 从 {after_name} 之后继续，剩余 {len(files)} 张", flush=True)
    print(f"[第二区域] 标导管腔 + 韧皮部/髓 {len(files)} 张图...", flush=True)

    csv_rows: list[dict] = []
    for i, orig_path in enumerate(files):
        meta = parse_name(orig_path.name)
        if meta is None:
            continue
        img = imread(orig_path)
        if img is None:
            continue
        manual_path = manual_dir / orig_path.name
        manual = imread(manual_path) if orig_path.name in manual_names else None
        result = analyze_region2(
            img, area_clf=area_clf, manual=manual, image_name=orig_path.name
        )
        annotated = draw_region2(img, result)
        imwrite(auto_dir / orig_path.name, annotated)

        if manual is not None:
            imwrite(merge_dir / orig_path.name, stitch_horizontal(manual, annotated))

        idx = view_index(meta.view_id, region)
        if idx is not None and meta.sample_id in records:
            rec = records[meta.sample_id]
            rec.vessel_count[idx] = float(result.count)
            rec.vessel_area[idx] = float(result.lumen_area_um2)
            rec.xylem_area[idx] = float(result.xylem_area_um2)
            rec.phloem_area[idx] = float(result.phloem_area_um2)
            rec.pith_area[idx] = float(result.pith_area_um2)

        csv_rows.append(
            {
                "filename": orig_path.name,
                "sample_id": meta.sample_id,
                "view_id": meta.view_id,
                "vessel_count": result.count,
                "lumen_area_um2": round(result.lumen_area_um2, 1),
                "xylem_area_um2": round(result.xylem_area_um2, 1),
                "phloem_area_um2": round(result.phloem_area_um2, 1),
                "pith_area_um2": round(result.pith_area_um2, 1),
                "um_per_pixel": round(result.scale.um_per_pixel, 6),
                "bar_px": round(result.scale.bar_pixels, 1),
                "backend": "manual-areas+rules" if manual is not None else "rules",
            }
        )
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(files):
            print(
                f"  {i + 1}/{len(files)}: {orig_path.name}  "
                f"导管={result.count}  腔面积={result.lumen_area_um2:.0f}um2  "
                f"木质部={result.xylem_area_um2:.0f} 韧皮部={result.phloem_area_um2:.0f} "
                f"髓={result.pith_area_um2:.0f}",
                flush=True,
            )
        if (i + 1) % 10 == 0:
            _flush_metrics(metrics_csv, csv_rows)
        del img, annotated, result, manual
        gc.collect()

    if csv_rows:
        csv_rows = _flush_metrics(metrics_csv, csv_rows)
        print(f"[第二区域] 指标 CSV: {metrics_csv}")
        records = records_from_region2_csv(metrics_csv, records)

    template = base_dir / "结果呈现表.xlsx"
    infer_xlsx = base_dir / "结果呈现表_推理.xlsx"
    xlsx = write_region2_excel(template, infer_xlsx, records, detail_rows=csv_rows)
    write_region2_excel(template, template, records, detail_rows=csv_rows)
    print(f"[第二区域] Excel(必须): {template}")
    print(f"          副本: {xlsx}")
    print("          已填 导管总数量 / 导管（腔）总面积 / 局部木质部面积(整图)")
    print("          以及 局部韧皮部面积 / 局部髓面积")
    from calibration.excel_export_new import fill_new_result_workbook

    fill_new_result_workbook(base_dir, records=records)

    print(f"[第二区域] 完成 -> {auto_dir}")
    print(f"          合并标定 -> {merge_dir} (左:人工标定 右:自动标定)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="第二区域：标导管腔并统计个数/面积，写入 Excel")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="兼容参数：导管始终用 CV 规则抠腔，不加载 YOLO",
    )
    parser.add_argument("--retrain-areas", action="store_true", help="重新从人工图训练韧皮部/髓分类器")
    parser.add_argument("--from-name", type=str, default=None)
    parser.add_argument("--to-name", type=str, default=None)
    parser.add_argument("--after-name", type=str, default=None, help="从该文件名之后继续（按目录顺序）")
    parser.add_argument(
        "--excel-only",
        action="store_true",
        help="不重新推理，只用已有 CSV 填写 Excel",
    )
    args = parser.parse_args()
    if args.excel_only:
        csv_path = args.base / "推理结果" / "region2_metrics.csv"
        records = records_from_region2_csv(csv_path)
        rows = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        infer_path = write_region2_excel(
            args.base / "结果呈现表.xlsx",
            args.base / "结果呈现表_推理.xlsx",
            records,
            detail_rows=rows,
        )
        official = write_region2_excel(
            args.base / "结果呈现表.xlsx",
            args.base / "结果呈现表.xlsx",
            records,
            detail_rows=rows,
        )
        print(f"[第二区域] Excel 已写入: {official}")
        if infer_path != official:
            print(f"          副本: {infer_path}")
        from calibration.excel_export_new import fill_new_result_workbook

        fill_new_result_workbook(args.base, records=records)
        return
    run_region2(
        args.base,
        from_name=args.from_name,
        to_name=args.to_name,
        after_name=args.after_name,
        retrain_areas=args.retrain_areas,
    )


if __name__ == "__main__":
    main()
