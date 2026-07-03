#!/usr/bin/env python3
"""
Export mean ± std (ddof=1) for FID and CMMD per (group, model).
Output: summary_export.csv
"""

import csv
import json
import math
import pathlib
import re
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent
OUT  = ROOT / "summary_export.csv"

# collect values keyed by (group, model)
data: dict[tuple, dict[str, list]] = defaultdict(lambda: {"fid": [], "cmmd": []})

for mpath in sorted(ROOT.rglob("metrics.json")):
    run_dir   = mpath.parent
    group_dir = run_dir.parent
    group = group_dir.name if group_dir != ROOT else "top_level"

    m = re.search(r"_(dm|dmf|ofm)_(\d+)$", run_dir.name)
    if not m:
        continue
    model = m.group(1)

    with open(mpath) as f:
        d = json.load(f)
    data[(group, model)]["fid"].append(d["fid"])
    data[(group, model)]["cmmd"].append(d["cmmd"])

def mean_std(values):
    n = len(values)
    m = sum(values) / n
    std = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))  # ddof=1
    return m, std

rows = []
for (group, model), vals in sorted(data.items()):
    fids  = data[(group, model)]["fid"]
    cmmds = data[(group, model)]["cmmd"]
    fid_mean,  fid_std  = mean_std(fids)
    cmmd_mean, cmmd_std = mean_std(cmmds)
    rows.append({
        "group":     group,
        "model":     model,
        "n_seeds":   len(fids),
        "fid_mean":  fid_mean,
        "fid_std":   fid_std,
        "cmmd_mean": cmmd_mean,
        "cmmd_std":  cmmd_std,
    })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group", "model", "n_seeds",
                                           "fid_mean", "fid_std",
                                           "cmmd_mean", "cmmd_std"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows -> {OUT}")
for r in rows:
    print(f"  {r['group']:30s}  {r['model']:4s}  FID {r['fid_mean']:.4f} ± {r['fid_std']:.4f}"
          f"  CMMD {r['cmmd_mean']:.6f} ± {r['cmmd_std']:.6f}  (n={r['n_seeds']})")
