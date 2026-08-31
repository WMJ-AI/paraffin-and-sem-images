"""
内部对照：读人工标定笔画图，按框复现对照图（不写任何按文件名的 JSON 缓存）。
归纳通用规则请用: python learn_general_rules.py
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from annotate_style import annotate_pairs_human_style  # noqa: E402
from compare_panel import imwrite_unicode, make_compare_bgr  # noqa: E402
from human_rules import find_pairs_by_human_rules  # noqa: E402
from learn_from_review import image_key_from_stem, load_human_mark_guides  # noqa: E402
from run_sample1 import process_image  # noqa: E402

DATA = Path(r"H:\尉明杰\扫描电镜 模型")
BATCH = next(p for p in DATA.iterdir() if p.is_dir() and "63" in p.name and "526" in p.name)
MARK_DIR = ROOT / "output" / "人工标定"
OUT_DIR = ROOT / "output" / "对照图"


def _find_tif(key: str) -> Path | None:
    p = BATCH / key
    if p.exists():
        return p
    m = re.match(r"(\d+)\s*\((\d+)\)", key)
    if not m:
        return None
    for f in BATCH.glob(f"{m.group(1)}*.tif"):
        if image_key_from_stem(f.stem) == key:
            return f
    return None


def natural_key_name(name: str):
    m = re.match(r"(\d+)\s*\((\d+)\)", name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (9999, name)


def main():
    MARK_DIR.mkdir(parents=True, exist_ok=True)
    marks = [f for f in MARK_DIR.rglob("*.png") if f.is_file()]
    if not marks:
        print(f"请把人工标定图放入：{MARK_DIR}")
        return

    guides = load_human_mark_guides(MARK_DIR)
    for f in MARK_DIR.rglob("*_无对.txt"):
        key = image_key_from_stem(f.stem.replace("_无对", ""))
        if key in guides:
            guides[key].force_zero_pair = True
    if not any(g.boxes or g.lines or g.force_zero_pair for g in guides.values()):
        print("未解析到蓝/黄框或壁厚线，请检查标定颜色。")
        return

    n_box = sum(1 for g in guides.values() if g.boxes)
    n_line = sum(1 for g in guides.values() if g.lines)
    print(f"解析人工标定笔画: {len(guides)} 张 (框={n_box} 线={n_line})（不写文件名缓存）")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, guide in sorted(guides.items(), key=lambda kv: natural_key_name(kv[0])):
        path = _find_tif(key)
        if path is None:
            print(f"跳过（无原图TIF）: {key}")
            continue
        rgb, gray, vessels, _, scale = process_image(path, force_um_per_px=None)
        for v in vessels:
            v.pair_id = None
        h, w = gray.shape[:2]
        pairs = find_pairs_by_human_rules(
            vessels,
            scale["um_per_px"],
            path.name,
            img_w=w,
            img_h=h,
            yellow_guide=guide,
            gray=gray,
        )
        auto = annotate_pairs_human_style(
            rgb, pairs, vessels=vessels, guide_boxes=guide.boxes
        )
        sid = re.match(r"(\d+)", path.stem).group(1)
        stem = path.stem.replace(" ", "_")
        out = OUT_DIR / f"组{sid}" / f"{stem}_左原图_右自动.png"
        imwrite_unicode(out, make_compare_bgr(rgb, auto, path.name))
        print(f"{path.name}: v={len(vessels)} p={len(pairs)} -> {out.name}")

    print(f"\n对照图目录: {OUT_DIR}")
    print("归纳通用规则: python learn_general_rules.py")


if __name__ == "__main__":
    main()
