"""
训练 YOLOv8-seg 导管腔实例分割。

用法:
  python train_lumen_seg.py
  python train_lumen_seg.py --data output/dataset_lumen/data.yaml --epochs 40 --model yolov8n-seg.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# 本地 --target 安装的 ultralytics（无写 conda 权限时）
_pylibs = ROOT / ".pylibs"
if _pylibs.is_dir():
    sys.path.insert(0, str(_pylibs))

# NumPy 2.x 移除了 np.trapz；ultralytics 8.3 仍调用它
import numpy as _np

if not hasattr(_np, "trapz"):
    _np.trapz = _np.trapezoid  # type: ignore[attr-defined]

DEFAULT_DATA = ROOT / "output" / "dataset_lumen" / "data.yaml"
DEFAULT_OUT = ROOT / "output" / "models" / "lumen_yolov8seg"


def train(
    data_yaml: Path,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 40,
    imgsz: int = 640,
    batch: int = 4,
    out_dir: Path = DEFAULT_OUT,
    device: str | None = None,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "需要 ultralytics：pip install ultralytics\n" + str(e)
        ) from e

    if not data_yaml.exists():
        raise SystemExit(f"缺少数据集: {data_yaml}\n请先运行: python export_review_masks.py")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 预训练权重不可用（GitHub 受限）时，从 yaml 从头训（nano 尺度）
    model_path = Path(model_name)
    if model_name.endswith(".pt") and not model_path.exists():
        local_pt = ROOT / "yolov8n-seg.pt"
        if local_pt.exists() and local_pt.stat().st_size > 1_000_000:
            model_name = str(local_pt)
        else:
            print("[train] 无预训练 .pt，使用 yolov8n-seg.yaml 从头训练（nano）")
            model_name = "yolov8n-seg.yaml"

    model = YOLO(model_name)
    kwargs = dict(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(out_dir.parent.resolve()),
        name=out_dir.name,
        exist_ok=True,
        patience=15,
        degrees=10.0,
        translate=0.08,
        scale=0.25,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.4,
        mosaic=0.8,
        close_mosaic=10,
        workers=0,
        verbose=True,
    )
    if device is not None:
        kwargs["device"] = device
    results = model.train(**kwargs)

    best = out_dir / "weights" / "best.pt"
    last = out_dir / "weights" / "last.pt"
    meta = {
        "data": str(data_yaml.resolve()),
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "best": str(best) if best.exists() else str(last),
        "save_dir": str(out_dir.resolve()),
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"训练完成 best={meta['best']}")
    return Path(meta["best"])


def main():
    ap = argparse.ArgumentParser(description="训练导管腔 YOLOv8-seg")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model", type=str, default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", type=str, default=None, help="如 0 或 cpu")
    args = ap.parse_args()
    data = args.data if args.data.is_absolute() else ROOT / args.data
    out = args.out if args.out.is_absolute() else ROOT / args.out
    train(
        data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        out_dir=out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
