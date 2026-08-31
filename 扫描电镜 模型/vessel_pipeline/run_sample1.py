"""Run automatic analysis on the golden example set and compare to Excel."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from io_report import (  # noqa: E402
    compare_report,
    load_golden,
    match_vessels,
    write_sample_excel,
)
from pair_tvw import find_pairs  # noqa: E402
from scale_bar import detect_scale_um_per_px, load_rgb_gray  # noqa: E402
from segment import attach_metrics, segment_dark_lumens, segment_hybrid  # noqa: E402
from annotate_style import annotate_pairs_human_style  # noqa: E402
from compare_panel import imwrite_unicode, make_compare_bgr  # noqa: E402

DATA = Path(r"H:\尉明杰\扫描电镜 模型")
_GOLDEN_ROOT = next(
    (p for p in DATA.iterdir() if p.is_dir() and "例子" in p.name and "模拟" in p.name),
    None,
)
GOLDEN_IMG_DIR = (
    next((p for p in _GOLDEN_ROOT.iterdir() if p.is_dir() and "原图" in p.name), None)
    if _GOLDEN_ROOT is not None
    else None
)
GOLDEN_XLSX = (
    next(
        (
            p
            for p in _GOLDEN_ROOT.glob("*.xlsx")
            if "Wand" in p.name or "扫描电镜" in p.name
        ),
        None,
    )
    if _GOLDEN_ROOT is not None
    else None
)
OUT_DIR = ROOT / "output"


def process_image(
    path: Path,
    force_um_per_px: float | None = None,
    *,
    hybrid: bool = False,
    dl_weights: Path | str | None = None,
    dl_conf: float = 0.25,
):
    rgb, gray = load_rgb_gray(path)
    scale = detect_scale_um_per_px(rgb)
    um = force_um_per_px if force_um_per_px is not None else scale["um_per_px"]
    if hybrid:
        vessels = segment_hybrid(
            gray,
            um_per_px=um,
            image_name=path.name,
            rgb=rgb,
            weights=dl_weights,
            dl_conf=dl_conf,
        )
    else:
        vessels = segment_dark_lumens(gray, um_per_px=um, image_name=path.name)
    vessels = attach_metrics(vessels, um)
    pairs = find_pairs(vessels, um_per_px=um, image_name=path.name)
    return rgb, gray, vessels, pairs, {**scale, "um_per_px": um}


def main():
    if _GOLDEN_ROOT is None:
        raise SystemExit(f"未找到黄金例资料目录（名称含「例子」「模拟」）: {DATA}")
    if GOLDEN_IMG_DIR is None or GOLDEN_XLSX is None:
        raise SystemExit(f"黄金例原图或 Excel 缺失: {_GOLDEN_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use golden Excel calibration for fair comparison; also report auto scale.
    FORCE_UM = 0.2233

    paths = sorted(GOLDEN_IMG_DIR.glob("*.tif"))
    if not paths:
        raise SystemExit(f"No tifs in {GOLDEN_IMG_DIR}")

    all_vessels = []
    all_pairs = []
    scale_methods = []
    cmp_dir = OUT_DIR / "黄金例_对照"
    if cmp_dir.exists():
        import shutil

        shutil.rmtree(cmp_dir)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        rgb, gray, vessels, pairs, scale = process_image(path, force_um_per_px=FORCE_UM)
        print(
            f"{path.name}: vessels={len(vessels)} pairs={len(pairs)} "
            f"scale_auto={scale.get('bar_width_px')}px -> would be "
            f"{20/scale['bar_width_px']:.4f} if used; forced={FORCE_UM}"
            if scale.get("bar_width_px")
            else f"{path.name}: vessels={len(vessels)} pairs={len(pairs)}"
        )
        all_vessels.extend(vessels)
        all_pairs.extend(pairs)
        scale_methods.append(scale["method"])
        auto = annotate_pairs_human_style(rgb, pairs, vessels=vessels)
        stem = path.stem.replace(" ", "_")
        imwrite_unicode(
            cmp_dir / f"{stem}_左原图_右自动.png",
            make_compare_bgr(rgb, auto, path.name),
        )

    xlsx_out = OUT_DIR / "黄金例_自动分析结果.xlsx"
    summary = write_sample_excel(
        xlsx_out,
        "黄金例",
        all_vessels,
        all_pairs,
        um_per_px=FORCE_UM,
        scale_method=f"forced_golden_excel; auto_methods={set(scale_methods)}",
    )

    golden = load_golden(GOLDEN_XLSX)
    matches, ua, ug = match_vessels(all_vessels, golden["vessels"])
    report = compare_report(summary, golden, matches, ua, ug)
    report_path = OUT_DIR / "golden_compare_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {xlsx_out}")
    print(f"Wrote {report_path}")
    print(f"对照图: {cmp_dir}")


if __name__ == "__main__":
    main()
