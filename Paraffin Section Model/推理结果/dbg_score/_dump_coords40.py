import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from calibration.region1_coords import build_coord_json, save_coord_json

base = ROOT
payload = build_coord_json(base, 40)
out = base / "推理结果" / "region1_coords_first40.json"
save_coord_json(payload, out)
print("n=", payload["n"], "->", out)
for mag, b in payload["summary"].items():
    print("--- mag", mag, "count", b["count"])
    for k in (
        "green_start_to_stem_c_frac",
        "layer_start_to_stem_c_frac",
        "layer_minus_green_frac",
    ):
        st = b[k]
        print(
            f"  {k}: med={st['median']:.3f} p25={st['p25']:.3f} p75={st['p75']:.3f}"
        )
    print("  starts_distinct_rate", round(b["starts_distinct_rate"], 3))
