#!/usr/bin/env python3
"""
Export per-run metrics to CSV for experiment_1.
Infers model, seed, and run-group from directory structure.
Output: metrics_export.csv
"""

import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).parent
OUT  = ROOT / "metrics_export.csv"

rows = []
for mpath in sorted(ROOT.rglob("metrics.json")):
    run_dir   = mpath.parent
    group_dir = run_dir.parent

    # run-group: name of the subdirectory containing this run (or "." for top-level)
    group = group_dir.name if group_dir != ROOT else "top_level"

    # parse model and seed from the run directory name, e.g. "22-04-2026_dm_42"
    m = re.search(r"_(dm|dmf|ofm)_(\d+)$", run_dir.name)
    if not m:
        print(f"WARNING: skipping unrecognised directory {run_dir.name}")
        continue
    model = m.group(1)
    seed  = int(m.group(2))

    with open(mpath) as f:
        d = json.load(f)

    rows.append({
        "group":         group,
        "model":         model,
        "seed":          seed,
        "run_dir":       run_dir.name,
        "fid":           d.get("fid"),
        "cmmd":          d.get("cmmd"),
        "training_time": d.get("training_time"),
    })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["group", "model", "seed", "run_dir",
                                           "fid", "cmmd", "training_time"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows -> {OUT}")
