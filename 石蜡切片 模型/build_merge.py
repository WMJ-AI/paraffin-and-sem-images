"""Rebuild 合并标定 from existing 人工标定 + 自动标定 (no re-inference)."""

from __future__ import annotations

import argparse
from pathlib import Path

from calibration.io_util import imread, imwrite
from calibration.merge import stitch_horizontal


def rebuild_merge(base_dir: Path, region: str) -> int:
    manual_dir = base_dir / "人工标定" / region
    auto_dir = base_dir / "自动标定" / region
    merge_dir = base_dir / "合并标定" / region
    merge_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    manuals = {p.name for p in manual_dir.glob("*.jpg")}
    for extra in merge_dir.glob("*.jpg"):
        if extra.name not in manuals:
            extra.unlink()
    for auto_path in sorted(auto_dir.glob("*.jpg")):
        if auto_path.name not in manuals:
            continue
        manual_path = manual_dir / auto_path.name
        auto = imread(auto_path)
        manual = imread(manual_path)
        if auto is None or manual is None:
            continue
        imwrite(merge_dir / auto_path.name, stitch_horizontal(manual, auto))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="重建合并标定图：左人工标定，右自动标定")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--region", choices=["第一区域", "第二区域", "第三区域", "all"], default="all")
    args = parser.parse_args()

    regions = ["第一区域", "第二区域", "第三区域"] if args.region == "all" else [args.region]
    for region in regions:
        n = rebuild_merge(args.base, region)
        print(f"{region}: 已生成 {n} 张 -> {args.base / '合并标定' / region}")


if __name__ == "__main__":
    main()
