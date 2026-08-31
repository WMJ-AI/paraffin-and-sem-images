"""Run region-1 / region-2 / region-3 inference pipelines and export Excel summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from calibration.excel_export import empty_records, export_results
from calibration.excel_export_new import fill_new_result_workbook
from run_region1 import run_region1
from run_region2 import run_region2
from run_region3 import run_region3


def main() -> None:
    parser = argparse.ArgumentParser(description="三个区域联合推理与 Excel 导出")
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-train", action="store_true", help="第一区域不训练，仅加载已有模型")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--calibrate", action="store_true", help="第三区域重新校准评分权重")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 Excel 路径，默认 结果呈现表_推理.xlsx",
    )
    args = parser.parse_args()

    base = args.base
    template = base / "结果呈现表.xlsx"
    output = args.output or base / "结果呈现表_推理.xlsx"

    records = empty_records(max_sample=63)

    print("=" * 60)
    print("开始第一区域推理（整切面四层长度）")
    print("=" * 60)
    records = run_region1(base, train=not args.no_train, epochs=args.epochs, records=records)

    print()
    print("=" * 60)
    print("开始第二区域推理（导管腔标定 + 面积）")
    print("=" * 60)
    records = run_region2(base, records=records)

    print()
    print("=" * 60)
    print("开始第三区域推理（形成层标定 + 评分）")
    print("=" * 60)
    records = run_region3(base, calibrate=args.calibrate, records=records)

    print()
    print("=" * 60)
    print("导出 Excel 汇总")
    print("=" * 60)
    export_results(template, output, records)
    print(f"Excel 已写入: {output}")
    fill_new_result_workbook(base, records=records)
    print("完成。")


if __name__ == "__main__":
    main()
