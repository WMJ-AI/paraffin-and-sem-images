"""
把平铺的复核对照图按文件名自动匹配，写入 人工标定/。

用法：
  1) 把复核好的 PNG 全部丢进 output/人工标定/（根目录即可，不必分子文件夹）
     也可用 --from-review 从 output/复核_* 目录收集
  2) python ingest_review_marks.py
  3) 再跑 python run_review_batch.py 1 63 或 python run_from_human.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from learn_from_review import (  # noqa: E402
    _human_mark_mask,
    _imread_unicode,
    _panel_for_marks,
    image_key_from_stem,
)

OUT = ROOT / "output"
MARK = OUT / "人工标定"


def _review_dirs() -> list[Path]:
    if not OUT.exists():
        return []
    return sorted(
        p for p in OUT.iterdir() if p.is_dir() and p.name.startswith("复核_")
    )


def _key_ok(key: str) -> bool:
    return bool(re.match(r"\d+ \(\d+\)\.tif$", key))


def _dest_name(key: str) -> str:
    m = re.match(r"(\d+) \((\d+)\)\.tif$", key)
    assert m
    return f"{m.group(1)}_({m.group(2)})_左原图_右自动.png"


def collect_candidates(extra_dirs: list[Path]) -> list[Path]:
    seen: set[str] = set()
    files: list[Path] = []
    roots = [MARK, *extra_dirs]
    for root in roots:
        if root is None or not root.exists():
            continue
        for f in root.rglob("*.png"):
            if f.name.startswith("_"):
                continue
            if "对照图" in str(f) and "人工标定" not in str(f):
                continue
            key = image_key_from_stem(f.stem)
            if not _key_ok(key):
                continue
            rp = str(f.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            files.append(f)
    return files


def has_human_paint(path: Path) -> bool:
    """左半有蓝/黄框或短线则视为复核过；仅刻度条伪彩色不算。"""
    bgr = _imread_unicode(path)
    if bgr is None:
        return False
    left = _panel_for_marks(bgr, path)
    y = _human_mark_mask(left)
    pix = int((y > 0).sum())
    from learn_from_review import _boxes_from_yellow, _lines_from_yellow

    boxes = _boxes_from_yellow(y)
    lines = _lines_from_yellow(y)
    if boxes or lines:
        return True
    return pix >= 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-review",
        action="store_true",
        help="同时从 output/复核_* 目录收集",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="不过滤是否画过标记，凡能匹配文件名的都入库",
    )
    ap.add_argument(
        "--dir",
        type=str,
        default="",
        help="额外平铺目录",
    )
    args = ap.parse_args()

    extras: list[Path] = []
    if args.from_review:
        extras.extend(_review_dirs())
    if args.dir:
        extras.append(Path(args.dir))

    cands = collect_candidates(extras)
    best: dict[str, Path] = {}
    for f in cands:
        key = image_key_from_stem(f.stem)
        prev = best.get(key)
        if prev is None or f.stat().st_mtime >= prev.stat().st_mtime:
            best[key] = f

    MARK.mkdir(parents=True, exist_ok=True)
    n_copy = n_skip = n_paint = 0
    for key, src in sorted(
        best.items(),
        key=lambda kv: (
            int(kv[0].split()[0]),
            int(kv[0].split("(")[1].split(")")[0]),
        ),
    ):
        painted = has_human_paint(src)
        if painted:
            n_paint += 1
        if not args.all and not painted:
            n_skip += 1
            continue
        sid = int(key.split()[0])
        dest_dir = MARK / f"组{sid}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / _dest_name(key)
        if src.resolve() == dest.resolve():
            print(f"已在位: {dest.relative_to(OUT)}")
            continue
        shutil.copy2(src, dest)
        n_copy += 1
        tag = "有标定" if painted else "无笔画"
        print(f"{tag} {src.name} -> 组{sid}/{dest.name}")

    print(
        f"\n完成: 入库 {n_copy} | 跳过无笔画 {n_skip} | 检测到有标定 {n_paint} | "
        f"候选键 {len(best)}"
    )
    print(f"目录: {MARK}")
    print("下一步: python run_review_batch.py 1 63   或   python run_from_human.py")


if __name__ == "__main__":
    main()
