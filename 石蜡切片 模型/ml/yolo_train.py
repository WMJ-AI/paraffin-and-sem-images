"""Train YOLO models and score regressor."""

from __future__ import annotations

import shutil
from pathlib import Path


def _load_pose_model(weights: Path | str | None, project: Path):
    from ultralytics import YOLO

    candidates: list[Path | str] = []
    if weights is not None:
        candidates.append(weights)
    candidates.extend(
        [
            project / "yolo11n-pose.pt",
            Path("yolo11n-pose.pt"),
            Path(__file__).resolve().parent.parent / "models" / "yolo" / "yolo11n-pose.pt",
            "yolo11n-pose.pt",
        ]
    )

    for cand in candidates:
        if isinstance(cand, Path):
            if cand.exists():
                print(f"  加载权重: {cand}")
                return YOLO(str(cand)), True
            continue
        # string: existing path or ultralytics downloadable name
        p = Path(cand)
        if p.exists():
            print(f"  加载权重: {p}")
            return YOLO(str(p)), True
        if cand.endswith(".pt") and "/" not in cand and "\\" not in cand:
            try:
                print(f"  加载/下载权重: {cand}")
                return YOLO(cand), True
            except Exception as exc:
                print(f"  无法加载 {cand}: {exc}")
                continue

    print("  使用 yolo11n-pose.yaml 从零训练")
    try:
        return YOLO("yolo11n-pose.yaml"), False
    except Exception:
        return YOLO("yolo11-pose.yaml"), False


def train_yolo_pose(
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    imgsz: int = 640,
    weights: Path | str | None = None,
    batch: int = 8,
) -> Path:
    model, pretrained = _load_pose_model(weights, project)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
        pretrained=pretrained,
        batch=batch,
        mosaic=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        close_mosaic=0,
        pose=40.0,
        patience=0,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = project / f"{name}_best.pt"
    shutil.copy2(best, dest)
    return dest


def train_yolo_detect(data_yaml: Path, project: Path, name: str, epochs: int, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=20,
        verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = project / f"{name}_best.pt"
    shutil.copy2(best, dest)
    return dest
