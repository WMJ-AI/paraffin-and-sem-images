"""
新图继续标定闭环：自动预标 →（人工改）→ ingest → 可选增量训练 / 重学规则。

用法:
  # 1) 对新样品范围出复核对照图（混合模式若有权重）
  python continue_annotate_loop.py preview --lo 41 --hi 45

  # 2) 人工修订后收集到 人工标定/
  python continue_annotate_loop.py ingest --from-review 复核_41-45

  # 3) 重学几何规则 + 增量导出训练集 + 微调 DL
  python continue_annotate_loop.py learn
  python continue_annotate_loop.py finetune --epochs 20

  # 4) 回归双导管框基线
  python continue_annotate_loop.py eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_preview(lo: int, hi: int, out_dir: str | None, hybrid: bool) -> None:
    args = [
        sys.executable,
        "run_review_batch.py",
        str(lo),
        str(hi),
    ]
    if out_dir:
        args += ["--out-dir", out_dir]
    else:
        args += ["--out-dir", f"复核_{lo}-{hi}"]
    if hybrid:
        args += ["--hybrid"]
    raise SystemExit(_run(args))


def cmd_ingest(from_review: str | None) -> None:
    args = [sys.executable, "ingest_review_marks.py", "--from-review"]
    if from_review:
        # 指定某一复核目录：用 --dir 传入
        rev = Path(from_review)
        if not rev.is_absolute():
            rev = ROOT / "output" / from_review
        args += ["--dir", str(rev)]
    raise SystemExit(_run(args))


def cmd_learn() -> None:
    code = _run([sys.executable, "learn_general_rules.py"])
    if code != 0:
        raise SystemExit(code)
    # 追加导出（全 1-40 + 新图若在 BATCH 内）
    raise SystemExit(
        _run([sys.executable, "export_review_masks.py", "--lo", "1", "--hi", "63"])
    )


def cmd_finetune(epochs: int, model: str | None) -> None:
    args = [
        sys.executable,
        "train_lumen_seg.py",
        "--epochs",
        str(epochs),
        "--data",
        "output/dataset_lumen/data.yaml",
    ]
    if model:
        args += ["--model", model]
    else:
        best = ROOT / "output" / "models" / "lumen_yolov8seg" / "weights" / "best.pt"
        if best.exists():
            args += ["--model", str(best)]
    raise SystemExit(_run(args))


def cmd_eval() -> None:
    raise SystemExit(_run([sys.executable, "baseline_metrics.py"]))


def main():
    ap = argparse.ArgumentParser(description="新图标定闭环")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("preview", help="批量预标定复核图")
    p0.add_argument("--lo", type=int, required=True)
    p0.add_argument("--hi", type=int, required=True)
    p0.add_argument("--out-dir", type=str, default=None)
    p0.add_argument("--hybrid", action="store_true", default=True)
    p0.add_argument("--no-hybrid", action="store_true")

    p1 = sub.add_parser("ingest", help="收集人工修订到人工标定/")
    p1.add_argument("--from-review", type=str, default=None)

    sub.add_parser("learn", help="重学规则并刷新 YOLO 数据集")

    p3 = sub.add_parser("finetune", help="增量微调 YOLO-seg")
    p3.add_argument("--epochs", type=int, default=20)
    p3.add_argument("--model", type=str, default=None)

    sub.add_parser("eval", help="回归双导管框基线指标")

    # 打印闭环说明
    sub.add_parser("help-loop", help="打印推荐步骤")

    args = ap.parse_args()
    if args.cmd == "preview":
        hybrid = not args.no_hybrid
        cmd_preview(args.lo, args.hi, args.out_dir, hybrid=hybrid)
    elif args.cmd == "ingest":
        cmd_ingest(args.from_review)
    elif args.cmd == "learn":
        cmd_learn()
    elif args.cmd == "finetune":
        cmd_finetune(args.epochs, args.model)
    elif args.cmd == "eval":
        cmd_eval()
    elif args.cmd == "help-loop":
        print(
            json.dumps(
                {
                    "steps": [
                        "python continue_annotate_loop.py preview --lo 41 --hi 45",
                        "人工只改错的复核 PNG",
                        "python continue_annotate_loop.py ingest --from-review 复核_41-45",
                        "python continue_annotate_loop.py learn",
                        "python continue_annotate_loop.py finetune --epochs 20",
                        "python continue_annotate_loop.py eval",
                    ]
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
