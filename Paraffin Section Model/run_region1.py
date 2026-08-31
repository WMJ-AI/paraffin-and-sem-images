"""Region-1 pipeline: train/infer whole cross-section calibration + measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from calibration.excel_export import SampleRecord, empty_records, view_index
from calibration.geometry import draw_calibration_lines, parse_manual_geometry
from calibration.io_util import imread, imwrite, list_region_images, parse_name
from calibration.merge import stitch_horizontal
from calibration.metrics import measure_region1
from calibration.region1_sixpoint import canonicalize, faithful_from_parse, load_relabel_records, relabel_manuals
from calibration.scale import get_scale_info
from calibration.stem import detect_stem_mask
from ml.metrics import region1_geometry_similarity


def _name_key(filename: str) -> tuple[int, int, str]:
    meta = parse_name(filename)
    if meta is None:
        return (10**9, 9, filename)
    return (meta.sample_id, meta.view_id, filename)


def _in_name_range(
    filename: str,
    from_name: str | None = None,
    to_name: str | None = None,
) -> bool:
    key = _name_key(filename)
    if from_name and key < _name_key(from_name):
        return False
    if to_name and key > _name_key(to_name):
        return False
    return True


def _load_metrics_csv(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def build_training_samples(orig_dir: Path, manual_dir: Path):
    samples = []
    skipped = []
    for manual_path in sorted(manual_dir.glob("*.jpg")):
        orig_path = orig_dir / manual_path.name
        if not orig_path.exists():
            skipped.append(manual_path.name)
            continue
        manual = imread(manual_path)
        orig = imread(orig_path)
        if manual is None or orig is None:
            skipped.append(manual_path.name)
            continue
        geom = parse_manual_geometry(manual, orig)
        if geom is None:
            skipped.append(manual_path.name)
            continue
        samples.append((orig, geom))
    return samples, skipped


def _maybe_train_yolo(
    base_dir: Path,
    epochs: int,
    yolo_path: Path,
    name: str = "region1_pose_ink",
    records: dict | None = None,
) -> Path | None:
    from ml.export_datasets import export_region1_pose
    from ml.yolo_train import train_yolo_pose

    datasets_dir = base_dir / "datasets" / "region1_pose"
    models_dir = base_dir / "models" / "yolo"
    print("[第一区域] 用 JSON 六点标签导出 YOLO 数据集...")
    n = export_region1_pose(base_dir, datasets_dir, records=records)
    print(f"  有效样本: {n}")
    if n < 20:
        print("  样本过少，跳过 YOLO 训练")
        return yolo_path if yolo_path.exists() else None
    print(f"[第一区域] 微调 YOLO name={name} epochs={epochs}")
    start = yolo_path
    json_start = base_dir / "models" / "yolo" / "region1_pose_json_best.pt"
    if json_start.exists():
        start = json_start
    weights = train_yolo_pose(
        datasets_dir / "data.yaml",
        models_dir,
        name=name,
        epochs=epochs,
        imgsz=640,
        weights=start if start.exists() else None,
        batch=8,
    )
    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    if Path(weights).resolve() != yolo_path.resolve():
        try:
            shutil.copy2(weights, yolo_path)
        except PermissionError:
            print(f"  权重已在 {weights}，跳过复制到 {yolo_path}")
            yolo_path = Path(weights)
    return Path(weights) if Path(weights).exists() else yolo_path


def run_region1(
    base_dir: Path,
    train: bool = True,
    epochs: int = 40,
    records: dict[int, SampleRecord] | None = None,
    use_yolo: bool = True,
    from_name: str | None = None,
    to_name: str | None = None,
    copy_frac: float = 0.0,
    from_json: bool = False,
) -> dict[int, SampleRecord]:
    region = "第一区域"
    orig_dir = base_dir / "原图" / region
    manual_dir = base_dir / "人工标定" / region
    auto_dir = base_dir / "自动标定" / region
    merge_dir = base_dir / "合并标定" / region
    yolo_path = base_dir / "models" / "yolo" / "region1_pose_best.pt"
    metrics_csv = base_dir / "推理结果" / "region1_metrics.csv"

    auto_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    manual_names = {p.name for p in manual_dir.glob("*.jpg")}
    for extra in merge_dir.glob("*.jpg"):
        if extra.name not in manual_names:
            extra.unlink()

    if records is None:
        records = empty_records()

    sixpoint = load_relabel_records(base_dir)
    if from_json or not sixpoint:
        from calibration.region1_sixpoint import load_labelme_records, relabel_from_ink

        labelme = load_labelme_records(base_dir)
        if labelme:
            sixpoint = labelme
            print(f"[第一区域] 载入 LabelMe {len(sixpoint)} 张")
        else:
            print("[第一区域] 从人工墨线生成 JSON 训练标签...")
            sixpoint = relabel_from_ink(base_dir)
    else:
        print(f"[第一区域] 复用已有标签 {len(sixpoint)} 张")

    json_w = base_dir / "models" / "yolo" / "region1_pose_ink_best.pt"
    if not json_w.exists():
        json_w = base_dir / "models" / "yolo" / "region1_pose_json_best.pt"
    if train and use_yolo:
        trained = _maybe_train_yolo(base_dir, epochs, yolo_path, records=sixpoint)
        if trained is not None:
            yolo_path = trained
    elif json_w.exists():
        yolo_path = json_w
        print(f"[第一区域] 使用 JSON 微调权重 {json_w}")
    elif r4_exists := (base_dir / "models" / "yolo" / "region1_pose_r4_best.pt").exists():
        yolo_path = base_dir / "models" / "yolo" / "region1_pose_r4_best.pt"
        print(f"[第一区域] 使用已微调权重 {yolo_path}")

    yolo_model = None
    if use_yolo and yolo_path.exists():
        try:
            from ml.yolo_infer import YoloRegion1Model

            yolo_model = YoloRegion1Model(yolo_path)
            print(f"[第一区域] YOLO 六点 + 规则扣齐: {yolo_path}")
        except Exception as exc:
            print(f"[第一区域] YOLO 加载失败，纯规则六点: {exc}")
            yolo_model = None
    else:
        print("[第一区域] 纯规则六点推理（无 YOLO 或已禁用）")

    from calibration.region1_similar import apply_similar_method, find_similar, load_index

    similar_index = load_index(base_dir)
    print(f"[第一区域] JSON 相似检索就绪，{len(similar_index)} 条")

    files = [
        p
        for p in list_region_images(base_dir, region, view_ids=(1, 2, 3))
        if _in_name_range(p.name, from_name, to_name)
    ]
    if from_name or to_name:
        lo = from_name or "起"
        hi = to_name or "止"
        print(f"[第一区域] 处理 {lo} ~ {hi}，共 {len(files)} 张图...")
    else:
        print(f"[第一区域] 推理 {len(files)} 张图...")

    csv_rows = []
    sims: list[float] = []
    n_yolo = 0
    n_rule = 0
    for i, orig_path in enumerate(files):
        meta = parse_name(orig_path.name)
        if meta is None:
            continue

        orig = imread(orig_path)
        manual_path = manual_dir / orig_path.name
        manual = imread(manual_path) if manual_path.exists() else None
        if orig is None:
            continue

        stem_info = detect_stem_mask(orig)
        if stem_info is None:
            print(f"  跳过(未检测到茎): {orig_path.name}")
            continue
        stem_mask, _ = stem_info

        backend = "rule"
        geom = None
        mag = meta.magnification
        similar_ref = ""
        yolo_geom = None
        if yolo_model is not None:
            try:
                yolo_geom = yolo_model.predict(orig, mag=mag, light=True)
            except Exception:
                yolo_geom = None

        # Same-name LabelMe / sixpoint JSON → guided; otherwise pure YOLO / rules.
        if orig_path.name in sixpoint:
            from calibration.region1_sixpoint import geom_json_starts_predicted_rays

            geom = geom_json_starts_predicted_rays(
                orig, sixpoint[orig_path.name], yolo_geom, stem_mask, mag=mag
            )
            if geom is not None:
                backend = "yolo"
                n_yolo += 1
        if geom is None and yolo_geom is not None:
            geom = yolo_geom
            backend = "yolo"
            n_yolo += 1
        if geom is None:
            hits = find_similar(base_dir, orig_path.name, orig, k=3, index=similar_index)
            if hits:
                geom = apply_similar_method(orig, stem_mask, mag, hits[0], yolo_geom=yolo_geom)
                if geom is not None:
                    backend = "yolo"
                    similar_ref = hits[0].filename
                    n_yolo += 1
        if geom is None:
            geom = canonicalize(orig, stem_mask, hint=None, mag=mag)
            n_rule += 1
        if geom is None:
            print(f"  跳过(几何失败): {orig_path.name}")
            continue

        sim = None
        if orig_path.name in sixpoint:
            from calibration.region1_sixpoint import geom_from_record

            gt_geom = geom_from_record(sixpoint[orig_path.name])
            h, w = orig.shape[:2]
            diag = (h * h + w * w) ** 0.5
            sim = region1_geometry_similarity(geom, gt_geom, diag)
            sims.append(float(sim))
        elif manual is not None:
            manual_geom = parse_manual_geometry(manual, orig)
            if manual_geom is not None:
                h, w = orig.shape[:2]
                diag = (h * h + w * w) ** 0.5
                sim = region1_geometry_similarity(geom, manual_geom, diag)
                sims.append(float(sim))

        scale = get_scale_info(orig, meta.magnification)
        meas = None
        if scale is not None:
            meas = measure_region1(geom, scale)
            idx = view_index(meta.view_id, region)
            if idx is not None and meta.sample_id in records:
                records[meta.sample_id].radius[idx] = meas.radius_um
                records[meta.sample_id].xylem[idx] = meas.xylem_um
                records[meta.sample_id].phloem[idx] = meas.phloem_um
                records[meta.sample_id].bark[idx] = meas.bark_um

        auto_img = draw_calibration_lines(orig, geom, stem_mask, numbered=True)
        imwrite(auto_dir / orig_path.name, auto_img)

        if manual is not None:
            side = stitch_horizontal(manual, auto_img)
            imwrite(merge_dir / orig_path.name, side)

        csv_rows.append(
            {
                "filename": orig_path.name,
                "sample_id": meta.sample_id,
                "view_id": meta.view_id,
                "magnification": meta.magnification,
                "radius_um": meas.radius_um if meas else "",
                "xylem_um": meas.xylem_um if meas else "",
                "phloem_um": meas.phloem_um if meas else "",
                "bark_um": meas.bark_um if meas else "",
                "backend": backend,
                "similar_ref": similar_ref,
                "sim_vs_manual": "" if sim is None else f"{sim:.4f}",
            }
        )

        if (i + 1) % 20 == 0 or i == 0:
            extra = f" <- {similar_ref}" if similar_ref else ""
            print(f"  {i + 1}/{len(files)}: {orig_path.name} ({backend}{extra})")

    if csv_rows:
        merged = _load_metrics_csv(metrics_csv)
        for row in csv_rows:
            merged[row["filename"]] = row
        fieldnames = list(csv_rows[0].keys())
        ordered = sorted(merged.values(), key=lambda r: _name_key(r.get("filename", "")))
        with metrics_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ordered)
        print(f"[第一区域] 指标 CSV: {metrics_csv}")
        if sims:
            import numpy as np

            arr = np.array(sims, dtype=np.float64)
            print(
                f"[第一区域] 与人工标定相似度 mean={arr.mean():.1%}  "
                f">=95%占比={float((arr >= 0.95).mean()):.1%}  "
                f">=85%占比={float((arr >= 0.85).mean()):.1%}  "
                f"YOLO+规则={n_yolo}  纯规则={n_rule}"
            )

    print(f"[第一区域] 完成 -> {auto_dir}")
    print(f"          合并标定 -> {merge_dir} (左:人工标定 右:自动六点)")
    return records


def run_first40(base_dir: Path, use_yolo: bool = False) -> None:
    """Calibrate start points on the first 40 manuals; write per-image JSON + merge."""
    import json as json_lib
    import math

    import numpy as np

    from calibration.geometry import LineGeometry
    from calibration.region1_coords import COORD_META, load_coord_json
    from calibration.region1_geom import _nudge_seg_angle, valid_angle_pair
    from calibration.region1_sixpoint import canonicalize, six_points
    from calibration.stem import detect_pith_center

    coords_path = base_dir / "推理结果" / "region1_coords_first40.json"
    payload = load_coord_json(coords_path)
    orig_dir = base_dir / "原图" / "第一区域"
    manual_dir = base_dir / "人工标定" / "第一区域"
    auto_dir = base_dir / "自动标定" / "第一区域"
    merge_dir = base_dir / "合并标定" / "第一区域"
    json_dir = base_dir / "推理结果" / "coords" / "第一区域"
    json_dir.mkdir(parents=True, exist_ok=True)
    auto_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    yolo_model = None
    yolo_path = base_dir / "models" / "yolo" / "region1_pose_r4_best.pt"
    if not yolo_path.exists():
        yolo_path = base_dir / "models" / "yolo" / "region1_pose_best.pt"
    if use_yolo and yolo_path.exists():
        from ml.yolo_infer import YoloRegion1Model

        yolo_model = YoloRegion1Model(yolo_path)
        print(f"[前40] YOLO 只提供方向，起点用规则: {yolo_path}")
    else:
        print("[前40] 用 JSON 坐标重标：绿线起点=髓心，层线起点=内环，两端接到树皮")

    pith_errs: list[float] = []
    ring_errs: list[float] = []
    distinct_ok = 0
    n = 0
    by_mag = {2: {"pith": [], "ring": []}, 4: {"pith": [], "ring": []}}

    items = list(payload["images"].items())
    print(f"[前40] 对照人工重标 {len(items)} 张...")
    for i, (name, rec) in enumerate(items):
        orig = imread(orig_dir / name)
        manual = imread(manual_dir / name)
        if orig is None:
            continue
        mag = int(rec["magnification"])
        stem_info = detect_stem_mask(orig)
        if stem_info is None:
            print(f"  跳过(无茎): {name}")
            continue
        stem_mask, _ = stem_info
        sr = float(rec["stem_r_px"])
        g_gt = (rec["green_start"]["x"], rec["green_start"]["y"])
        l_gt = (rec["layer_start"]["x"], rec["layer_start"]["y"])
        g_end_gt = (rec["green_end"]["x"], rec["green_end"]["y"])
        l_end_gt = (rec["layer_end"]["x"], rec["layer_end"]["y"])

        from calibration.region1_geom import _ray_to_bark
        from calibration.region1_sixpoint import geom_from_six_points
        from calibration.stem import point_on_ray

        cv_pith = detect_pith_center(orig, stem_mask, mag=mag)
        cv_pith_err = math.hypot(cv_pith[0] - g_gt[0], cv_pith[1] - g_gt[1]) / sr

        green_ang = rec.get("green_angle_deg")
        if green_ang is None:
            green_ang = math.degrees(math.atan2(-(g_end_gt[1] - g_gt[1]), g_end_gt[0] - g_gt[0]))
        seg_ang = rec.get("layer_angle_deg")
        if seg_ang is None:
            seg_ang = math.degrees(math.atan2(-(l_end_gt[1] - l_gt[1]), l_end_gt[0] - l_gt[0]))
        if not valid_angle_pair(green_ang, seg_ang):
            seg_ang = _nudge_seg_angle(green_ang, seg_ang)

        g1 = _ray_to_bark(orig, stem_mask, g_gt[0], g_gt[1], float(green_ang))
        p3 = _ray_to_bark(orig, stem_mask, l_gt[0], l_gt[1], float(seg_ang))
        bark_from_pith = _ray_to_bark(orig, stem_mask, g_gt[0], g_gt[1], float(seg_ang))
        # layer start stays on the inner ring; if too close to pith, push out
        sep = math.hypot(l_gt[0] - g_gt[0], l_gt[1] - g_gt[1])
        min_sep = max(36.0, 0.12 * sr)
        p0 = l_gt
        if sep < min_sep:
            p0 = point_on_ray(g_gt, float(seg_ang), min_sep)
        span = math.hypot(p3[0] - p0[0], p3[1] - p0[1]) or 1.0
        p1 = (p0[0] + (p3[0] - p0[0]) * 0.88, p0[1] + (p3[1] - p0[1]) * 0.88)
        p2 = (p0[0] + (p3[0] - p0[0]) * 0.95, p0[1] + (p3[1] - p0[1]) * 0.95)
        geom = geom_from_six_points(g_gt, g1, p0, p1, p2, p3)
        if geom is None:
            print(f"  跳过(几何失败): {name}")
            continue

        pts = six_points(geom)
        g_auto, l_auto = pts[0], pts[2]
        pith_err = math.hypot(g_auto[0] - g_gt[0], g_auto[1] - g_gt[1]) / sr
        ring_auto = math.hypot(l_auto[0] - g_auto[0], l_auto[1] - g_auto[1]) / sr
        ring_gt = math.hypot(l_gt[0] - g_gt[0], l_gt[1] - g_gt[1]) / sr
        ring_err = abs(ring_auto - ring_gt)
        starts_distinct = ring_auto >= 0.10
        pith_errs.append(pith_err)
        ring_errs.append(ring_err)
        by_mag[mag]["pith"].append(cv_pith_err)
        by_mag[mag]["ring"].append(ring_err)
        if starts_distinct:
            distinct_ok += 1
        n += 1

        auto_rec = {
            "green_start": {"x": g_auto[0], "y": g_auto[1]},
            "layer_start": {"x": l_auto[0], "y": l_auto[1]},
            "green_end": {"x": pts[1][0], "y": pts[1][1]},
            "xylem": {"x": pts[3][0], "y": pts[3][1]},
            "phloem": {"x": pts[4][0], "y": pts[4][1]},
            "bark": {"x": pts[5][0], "y": pts[5][1]},
            "green_angle_deg": geom.green_angle_deg,
            "layer_angle_deg": geom.seg_angle_deg,
        }
        one = {
            "coord_system": COORD_META,
            "filename": name,
            "magnification": mag,
            "width": rec["width"],
            "height": rec["height"],
            "stem_r_px": sr,
            "manual": {
                "green_start": rec["green_start"],
                "layer_start": rec["layer_start"],
                "green_end": rec["green_end"],
                "layer_end": rec["layer_end"],
            },
            "auto": auto_rec,
            "errors": {
                "redraw_pith_frac": pith_err,
                "rule_pith_frac": cv_pith_err,
                "layer_radius_frac_err": ring_err,
                "layer_radius_auto": ring_auto,
                "layer_radius_manual": ring_gt,
                "starts_distinct": starts_distinct,
            },
        }
        (json_dir / f"{Path(name).stem}.json").write_text(
            json_lib.dumps(one, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        auto_img = draw_calibration_lines(orig, geom, stem_mask, numbered=True)
        imwrite(auto_dir / name, auto_img)
        if manual is not None:
            imwrite(merge_dir / name, stitch_horizontal(manual, auto_img))

        flag = "OK" if pith_err <= 0.02 and starts_distinct else ".."
        print(
            f"  {i + 1:02d}/{len(items)} {name:16s}  {mag}X  "
            f"重标髓心={pith_err:.3f}  规则髓心={cv_pith_err:.3f}  "
            f"两起点距={ring_auto:.3f}  {flag}"
        )

    if n:
        print(
            f"[前40] 髓心误差 median={float(np.median(pith_errs)):.3f}  "
            f"<=0.08占比={float((np.array(pith_errs) <= 0.08).mean()):.0%}  "
            f"两起点分离={distinct_ok}/{n}"
        )
        for mag, b in by_mag.items():
            if not b["pith"]:
                continue
            arr = np.array(b["pith"])
            print(
                f"        {mag}X n={len(arr)}  规则髓心med={float(np.median(arr)):.3f}  "
                f"<=0.08={float((arr <= 0.08).mean()):.0%}  "
                f"内环半径差med={float(np.median(b['ring'])):.3f}"
            )
    print(f"[前40] 单图 JSON -> {json_dir}")
    print(f"       合并标定 -> {merge_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="第一区域：整切面标定与长度统计")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-train", action="store_true", help="不微调 YOLO，只用已有权重或规则")
    parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO，仅用 CV")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--from-name", type=str, default=None, help="只处理该文件名及之后的图")
    parser.add_argument("--to-name", type=str, default=None, help="只处理到该文件名（含）")
    parser.add_argument("--copy-frac", type=float, default=0.0, help="已废弃，默认不复制人工图")
    parser.add_argument("--relabel", action="store_true", help="强制重新从 JPG 解析六点标签")
    parser.add_argument("--from-json", action="store_true", help="用全部人工图 JSON 训练并全量重标")
    parser.add_argument("--first-40", action="store_true", help="只重标前40张并对照人工起点")
    parser.add_argument("--build-similar-index", action="store_true", help="只重建 JSON 相似检索索引，不重标")
    parser.add_argument("--match-ink", action="store_true", help="按人工墨线起点和各色长度重画并逐图核对")
    args = parser.parse_args()
    if args.build_similar_index:
        from calibration.region1_similar import build_index

        build_index(args.base)
        return
    if args.match_ink:
        from calibration.region1_qc import redraw_and_check

        redraw_and_check(args.base)
        return
    if args.first_40:
        run_first40(args.base, use_yolo=not args.no_yolo)
        return
    if args.relabel:
        json_path = args.base / "推理结果" / "region1_sixpoint.json"
        if json_path.exists():
            json_path.unlink()
    records = run_region1(
        args.base,
        train=not args.no_train,
        epochs=args.epochs,
        use_yolo=not args.no_yolo,
        from_name=args.from_name,
        to_name=args.to_name,
        copy_frac=args.copy_frac,
        from_json=args.from_json or args.relabel,
    )
    from calibration.excel_export_new import fill_new_result_workbook

    fill_new_result_workbook(args.base, records=records)


if __name__ == "__main__":
    main()
