"""
在已复核好的双导管对照图上，只叠加单导管绿轮廓；不重画橙框/紫箭/TVW。

用法:
  python overlay_singles_frozen.py              # 整理冻结目录并全量叠加
  python overlay_singles_frozen.py --repair 1 10  # 修复已有复核图样本 1–10 的单导管坐标
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from annotate_style import GREEN, draw_green_contour
from compare_panel import imwrite_unicode
from exclusion_rules import get_exclusion_config
from human_rules import load_learned_rules
from learn_from_review import _imread_unicode
from run_from_human import _find_tif, natural_key_name
from run_sample1 import process_image
from segment import tissue_height

OUT = ROOT / "output"
SRC_140 = OUT / "复核_样品1-40"
SRC_4163 = OUT / "复核_样品41-63"
DST_140 = OUT / "复核_1-40"
DST_4161 = OUT / "复核_41-61"

# 冻结图中橙框颜色（annotate_style.ORANGE）
ORANGE_RGB = np.array([248, 104, 40], dtype=np.float32)
# panel_label 覆盖顶部条带高度（不增高画布）；检测橙框时跳过该区以免误匹配
LABEL_H = 36


def _parse_name(png: Path) -> tuple[int, int] | None:
    m = re.match(r"(\d+)_?\((\d+)\)", png.stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def detect_orange_rois(right_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """从冻结右图检测橙框 ROI（xyxy，与原图像素同一坐标系）。"""
    body = right_rgb[LABEL_H:, :, :].astype(np.float32)
    dist = np.linalg.norm(body - ORANGE_RGB[None, None, :], axis=2)
    mask = (dist < 55).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rois = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 800 or bw < 40 or bh < 40:
            continue
        # body 相对全图下移 LABEL_H；转回与 vessel 一致的全图坐标
        rois.append((int(x), int(y + LABEL_H), int(x + bw), int(y + LABEL_H + bh)))
    return rois


def vessel_in_rois(v, rois, pad: float = 25.0) -> bool:
    for x0, y0, x1, y1 in rois:
        if (x0 - pad) <= v.cx <= (x1 + pad) and (y0 - pad) <= v.cy <= (y1 + pad):
            return True
    return False


def select_singles_outside_frozen(
    vessels,
    rois,
    img_w: int,
    img_h: int,
) -> list:
    rules = load_learned_rules()
    xcfg = get_exclusion_config()
    min_ai = max(rules.min_single_ai_um2, xcfg.min_single_ai_um2)
    th = tissue_height(img_h)
    cands = []
    for v in vessels:
        if vessel_in_rois(v, rois, pad=30.0):
            continue
        if (v.ai_um2 or 0) < min_ai:
            continue
        if (
            v.cx < rules.edge_margin_px
            or v.cy < rules.edge_margin_px
            or v.cx > img_w - rules.edge_margin_px
            or v.cy > th - rules.edge_margin_px
        ):
            continue
        cands.append(v)
    cands.sort(key=lambda z: z.ai_um2 or 0, reverse=True)
    return cands[: rules.max_singles]


def _green_stroke_mask(right_rgb: np.ndarray) -> np.ndarray:
    r, g, b = right_rgb[:, :, 0], right_rgb[:, :, 1], right_rgb[:, :, 2]
    return ((g > 140) & (r < 100) & (b < 100)).astype(np.uint8) * 255


def restore_single_greens_from_left(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    rois: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """
    用左图像素擦掉右图「橙框外」的绿色笔画（错误/旧单导管），保留框内双导管绿线。
    panel_label 只覆盖顶栏、不平移内容，故左右同坐标可直接替换。
    """
    out = right_rgb.copy()
    h, w = out.shape[:2]
    gmask = _green_stroke_mask(out)
    # 橙框内双导管绿轮廓保留
    for x0, y0, x1, y1 in rois:
        pad = 8
        xa, ya = max(0, x0 - pad), max(0, y0 - pad)
        xb, yb = min(w, x1 + pad), min(h, y1 + pad)
        gmask[ya:yb, xa:xb] = 0
    # 略膨胀，清掉既有线宽
    gmask = cv2.dilate(gmask, np.ones((5, 5), np.uint8), iterations=1)
    sel = gmask > 0
    if sel.any():
        out[sel] = left_rgb[sel]
    return out


def overlay_one(png: Path, *, repair: bool = False) -> int:
    parsed = _parse_name(png)
    if parsed is None:
        return 0
    sid, iid = parsed
    key = f"{sid} ({iid})"
    tif = _find_tif(f"{sid} ({iid}).tif") or _find_tif(key)
    if tif is None:
        print(f"  skip no tif: {png.name}")
        return 0

    compare = _imread_unicode(str(png))
    if compare is None:
        print(f"  skip read fail: {png.name}")
        return 0
    # BGR -> work in RGB for color match / PIL
    compare_rgb = cv2.cvtColor(compare, cv2.COLOR_BGR2RGB)
    h, w = compare_rgb.shape[:2]
    mid = w // 2
    left = compare_rgb[:, :mid].copy()
    right = compare_rgb[:, mid:].copy()

    rgb, gray, vessels, _, scale = process_image(tif)
    ih, iw = gray.shape[:2]
    # 新对照图：panel_label 追加顶栏 → 内容整体下移 LABEL_H；旧图覆盖顶栏则 y_off=0
    y_off = LABEL_H if right.shape[0] == ih + LABEL_H else 0
    rois = detect_orange_rois(right)
    if repair:
        right = restore_single_greens_from_left(left, right, rois)
        rois = detect_orange_rois(right)
    rois_img = [
        (x0, max(0, y0 - y_off), x1, max(0, y1 - y_off)) for x0, y0, x1, y1 in rois
    ]
    singles = select_singles_outside_frozen(vessels, rois_img, iw, ih)

    if singles:
        base = Image.fromarray(right).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for v in singles:
            cnt = v.contour.copy().astype(np.float32)
            if y_off:
                cnt[:, 0, 1] += y_off
            draw_green_contour(draw, cnt.astype(np.int32), GREEN, width=2)
        right = np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))

    out_rgb = np.concatenate([left, right], axis=1)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    imwrite_unicode(png, out_bgr)
    return len(singles)


def organize_folders() -> None:
    if DST_140.exists():
        shutil.rmtree(DST_140)
    if DST_4161.exists():
        shutil.rmtree(DST_4161)

    if not SRC_140.exists():
        raise SystemExit(f"缺少恢复目录: {SRC_140}")
    shutil.copytree(SRC_140, DST_140)

    DST_4161.mkdir(parents=True, exist_ok=True)
    if not SRC_4163.exists():
        raise SystemExit(f"缺少恢复目录: {SRC_4163}")
    for png in sorted(SRC_4163.glob("*.png")):
        parsed = _parse_name(png)
        if parsed is None:
            continue
        sid, _ = parsed
        if 41 <= sid <= 61:
            shutil.copy2(png, DST_4161 / png.name)
        # 62–63 丢弃

    # 清理样品命名目录与 zip
    for p in [SRC_140, SRC_4163]:
        if p.exists():
            shutil.rmtree(p)
    for z in OUT.glob("*.zip"):
        if "样品" in z.name or "复核" in z.name:
            z.unlink()
            print("removed zip", z.name)


def _jobs_for_samples(lo: int, hi: int) -> list[Path]:
    jobs: list[Path] = []
    for d in (DST_140, DST_4161):
        if not d.exists():
            continue
        for png in sorted(d.glob("*.png"), key=lambda p: natural_key_name(p.stem)):
            parsed = _parse_name(png)
            if parsed is None:
                continue
            sid, _ = parsed
            if lo <= sid <= hi:
                jobs.append(png)
    return jobs


def repair_samples(lo: int, hi: int) -> None:
    jobs = _jobs_for_samples(lo, hi)
    if not jobs:
        raise SystemExit(f"未找到样本 {lo}–{hi} 的复核图（查 {DST_140} / {DST_4161}）")
    print(f"修复单导管坐标: 样本 {lo}–{hi}，共 {len(jobs)} 张（先擦旧绿线再按原图坐标重画）\n")
    n_single = 0
    for i, png in enumerate(jobs, 1):
        k = overlay_one(png, repair=True)
        n_single += k
        print(f"[{i}/{len(jobs)}] {png.name}: +{k} singles")
    print(f"\n完成: 样本 {lo}–{hi} 单导管合计 {n_single} 个腔")


def main() -> None:
    ap = argparse.ArgumentParser(description="冻结双导管图上叠加/修复单导管绿轮廓")
    ap.add_argument(
        "--repair",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        help="修复已有复核图中样本 LO–HI 的单导管（不重做目录整理）",
    )
    args = ap.parse_args()
    if args.repair:
        repair_samples(args.repair[0], args.repair[1])
        return

    print("整理冻结复核目录 → 复核_1-40 / 复核_41-61 …")
    organize_folders()
    jobs = []
    for d in (DST_140, DST_4161):
        for png in sorted(d.glob("*.png"), key=lambda p: natural_key_name(p.stem)):
            jobs.append(png)
    print(f"叠加单导管: {len(jobs)} 张（双导管标注保持不动）\n")
    n_single = 0
    for i, png in enumerate(jobs, 1):
        k = overlay_one(png)
        n_single += k
        if i % 20 == 0 or k:
            print(f"[{i}/{len(jobs)}] {png.name}: +{k} singles")
    # README
    for d, lo, hi in ((DST_140, 1, 40), (DST_4161, 41, 61)):
        (d / "_README.txt").write_text(
            "\n".join(
                [
                    f"复核 {lo}–{hi}（冻结双导管 + 叠加单导管）",
                    "=" * 40,
                    "双导管：沿用此前复核好的橙框/紫箭/TVW，未重算、未重画",
                    "单导管：按通用 exclusion / min_single_ai 规则识别后仅补绿轮廓",
                    "坐标：与原图一致（panel_label 仅覆盖顶栏，不平移）",
                    f"共 {len(list(d.glob('*.png')))} 张",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    # 确认无其它复核目录
    for p in list(OUT.iterdir()):
        if p.is_dir() and "复核" in p.name and p.name not in ("复核_1-40", "复核_41-61"):
            shutil.rmtree(p)
            print("removed extra", p.name)
    print(f"\n完成: 单导管叠加合计 {n_single} 个腔")
    print(f"  {DST_140}  png={len(list(DST_140.glob('*.png')))}")
    print(f"  {DST_4161} 附图={len(list(DST_4161.glob('*.png')))}")


if __name__ == "__main__":
    main()
