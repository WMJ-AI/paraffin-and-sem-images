"""
按通用规则批量生成复核对照图：识别 boxes → 自动画 TVW lines。
不读、不写按文件名的坐标缓存。

用法:
  python run_review_batch.py 1 63
  python run_review_batch.py 1 20 --out-dir 复核_1-40
  python run_review_batch.py 1 20 --out-dir 复核_1-40 --singles-only 1(1),2(3)
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from annotate_style import annotate_pairs_human_style
from compare_panel import imwrite_unicode, make_compare_bgr
from human_rules import filter_vessels_by_min_ai, find_pairs_auto, load_learned_rules
from learn_from_review import image_key_from_stem
from run_from_human import natural_key_name
from run_sample1 import process_image

DATA = Path(r"D:\BaiduNetdiskDownload\资料\资料")
BATCH = next(p for p in DATA.iterdir() if p.is_dir() and "63" in p.name and "526" in p.name)


def _sample_id(key: str) -> int | None:
    m = re.match(r"(\d+)\s*\(", key)
    return int(m.group(1)) if m else None


def out_name(key: str) -> str:
    m = re.match(r"(\d+)\s*\((\d+)\)", key)
    if m:
        return f"{int(m.group(1)):02d}_({int(m.group(2)):02d})_左原图_右自动.png"
    return key.replace(" ", "_").replace(".tif", "") + "_左原图_右自动.png"


def _norm_key(s: str) -> str:
    """1(1) / 1 (1) / 01_(01) -> '1 (1)'"""
    s = s.strip().replace(".tif", "").replace("_", " ")
    m = re.match(r"(\d+)\s*\(\s*(\d+)\s*\)", s)
    if m:
        return f"{int(m.group(1))} ({int(m.group(2))})"
    return s


def _parse_key_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {_norm_key(x) for x in raw.split(",") if x.strip()}


def run_batch(
    lo: int,
    hi: int,
    *,
    out_dir: Path | None = None,
    wipe: bool = False,
    singles_only: set[str] | None = None,
    hybrid: bool = False,
    dl_weights: Path | str | None = None,
) -> None:
    """
    singles_only: 运行时名单（仅本次命令），强制这些图只画单导管绿轮廓、不画橙框/TVW。
    不写入规则文件、不含坐标。
    hybrid: 启用 YOLO-seg + 经典分割并集（需已训练权重）。
    """
    singles_only = singles_only or set()
    out_dir = out_dir or (ROOT / "output" / f"复核_{lo}-{hi}")
    if not out_dir.is_absolute():
        out_dir = ROOT / "output" / out_dir

    tifs = []
    for f in BATCH.glob("*.tif"):
        key = image_key_from_stem(f.stem)
        sid = _sample_id(key)
        if sid is not None and lo <= sid <= hi:
            tifs.append((key, f))
    tifs.sort(key=lambda kv: natural_key_name(kv[0]))

    if wipe and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    rows = []
    mode = "混合(DL+规则)" if hybrid else "通用规则（识框+自动画线）"
    print(f"范围 {lo}-{hi}: 原图 {len(tifs)} | 模式={mode}")
    if singles_only:
        print(f"本次强制仅单导管: {sorted(singles_only)}")
    print(f"输出: {out_dir}\n")

    for key, path in tifs:
        rgb, gray, vessels, _, scale = process_image(
            path,
            force_um_per_px=None,
            hybrid=hybrid,
            dl_weights=dl_weights,
        )
        rules = load_learned_rules()
        vessels = filter_vessels_by_min_ai(
            vessels, rules, gray=gray, um_per_px=scale["um_per_px"]
        )
        for v in vessels:
            v.pair_id = None
        h, w = gray.shape[:2]
        stem = _norm_key(key.replace(".tif", ""))
        if stem in singles_only:
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
        auto = annotate_pairs_human_style(rgb, pairs, vessels=vessels, guide_boxes=None)
        name = out_name(key)
        imwrite_unicode(out_dir / name, make_compare_bgr(rgb, auto, path.name))
        n_ok += 1
        tag = "singles-only" if stem in singles_only else f"p={len(pairs)}"
        print(f"{path.name}: v={len(vessels)} {tag} -> {name}")
        rows.append(f"{key}\t{tag}\tv={len(vessels)}\t{name}")

    readme = out_dir / "_README.txt"
    prev = readme.read_text(encoding="utf-8") if readme.exists() else ""
    note = [
        f"范围 {lo}–{hi} 自动对照图（请人工复核）",
        "=" * 40,
        f"模式: {mode}",
        "规则文件: output/learned_rules.json（无逐图坐标）",
        "左：原图 | 右：自动标注",
        "绿轮廓=双导管腔+其外单导管；橙框=双导管区；紫十字=Ai；蓝线=TVW",
        "顶栏追加在图像上方，不遮挡标注",
        "",
        f"本批写入 {n_ok} 张",
        "",
        "明细：",
        *rows,
        "",
    ]
    if prev and f"范围 {lo}–{hi}" not in prev[:80]:
        note = note + ["---", "此前说明保留：", prev]
    readme.write_text("\n".join(note), encoding="utf-8")
    print(f"\n完成: {n_ok} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="通用规则批量生成复核对照图")
    ap.add_argument("lo", type=int, nargs="?", default=1)
    ap.add_argument("hi", type=int, nargs="?", default=None)
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录名或路径（默认 output/复核_{lo}-{hi}）；可指定 复核_1-40 以覆盖写入",
    )
    ap.add_argument(
        "--wipe",
        action="store_true",
        help="先清空输出目录（默认否，便于只更新部分样本）",
    )
    ap.add_argument(
        "--singles-only",
        type=str,
        default="",
        help="逗号分隔图号，本次强制只画单导管（例: 1(1),2(3)）；不落盘、无坐标",
    )
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="启用 YOLO-seg + 经典分割混合腔检出（需 output/models/.../best.pt）",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=None,
        help="可选 YOLO-seg 权重路径",
    )
    args = ap.parse_args()
    lo = args.lo
    hi = args.hi if args.hi is not None else lo
    run_batch(
        lo,
        hi,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        wipe=args.wipe,
        singles_only=_parse_key_list(args.singles_only),
        hybrid=args.hybrid,
        dl_weights=args.weights,
    )


if __name__ == "__main__":
    main()
