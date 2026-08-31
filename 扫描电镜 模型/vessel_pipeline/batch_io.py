"""Shared batch image listing (526 TIFs)."""
from __future__ import annotations

import re
from pathlib import Path

DATA = Path(r"H:\尉明杰\扫描电镜 模型")
BATCH = next(p for p in DATA.iterdir() if p.is_dir() and "63" in p.name and "526" in p.name)


def natural_key(p: Path):
    m = re.match(r"(\d+)\s*\((\d+)\)", p.stem)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (9999, p.name)


def list_unique_images() -> list[Path]:
    files = sorted(BATCH.glob("*.tif"), key=natural_key)
    seen, uniq = set(), []
    for f in files:
        k = f.name.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)
    return uniq


def sample_id(path: Path) -> str:
    m = re.match(r"(\d+)\s*\((\d+)\)", path.stem)
    return m.group(1) if m else "other"
