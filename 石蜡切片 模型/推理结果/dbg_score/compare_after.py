from pathlib import Path
import numpy as np
from collections import Counter
from calibration.region3.scoring import load_manual_scores, resolve_score_xlsx
from ml.metrics import score_similarity

base = Path(r"d:\BaiduNetdiskDownload\shibie")

def find_xlsx(folder, *needles):
    for p in folder.iterdir():
        if p.suffix.lower()==".xlsx" and all(n in p.name for n in needles):
            return p
    raise FileNotFoundError(needles)

rev = load_manual_scores(resolve_score_xlsx(base))
auto = load_manual_scores(find_xlsx(base/"推理结果", "自动"))
keys = sorted(set(rev)&set(auto))
p = np.array([auto[k] for k in keys])
g = np.array([rev[k] for k in keys])
d = p-g
print("n", len(keys))
print("auto mean", p.mean(), "rev mean", g.mean())
print("MAE", np.mean(np.abs(d)), "RMSE", np.sqrt(np.mean(d**2)), "bias", d.mean())
print("exact", np.mean(d==0), "within1", np.mean(np.abs(d)<=1), "within2", np.mean(np.abs(d)<=2))
print("similarity", score_similarity(p.tolist(), g.tolist()))
print("auto dist", dict(sorted(Counter(int(round(v)) for v in p).items())))
print("rev  dist", dict(sorted(Counter(int(round(v)) for v in g).items())))
print("worst:")
rows = sorted(((abs(auto[k]-rev[k]), k[0], k[1], auto[k], rev[k]) for k in keys), reverse=True)
for a,s,v,au,r in rows[:15]:
    print(f"  {s:3d}-{v} auto={au:.0f} rev={r:.0f} |d|={a:.0f}")
