#!/usr/bin/env python
"""
Friction γ-schedule sweep: 9 configs × 3 seeds = 27 runs (~7 h GPU).

Gamma schedules
---------------
A0 none           — γ = 0 (no friction, baseline)
A1 const_0.25     — γ = 0.25
A2 const_0.5      — γ = 0.5
A3 const_0.75     — γ = 0.75
A4 linear         — γ ramps linearly 0→1
A5 quadratic      — γ = (t/T)²
A6 sqrt           — γ = √(t/T)
A7 sin            — γ = sin(π/2 · t/T)
A8 delayed_linear — γ=0 for first 50%, then 0→1

Criterion (reviewer §4 Q5): A4 (linear) in top-3 by FID AND beats A0 on all metrics.

Usage
-----
# Full sweep (default 10 001 steps, 3 seeds):
    python run_friction_sweep.py

# Quick smoke-test (100 steps, no metrics):
    python run_friction_sweep.py --max_steps 100 --no_metrics

# Run only selected schedules:
    python run_friction_sweep.py --schedules none linear quadratic
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmf_alae_gamma import train, GAMMA_SCHEDULES

SEEDS = [42, 123, 777]

SCHEDULE_LABELS = {
    "none":           "A0",
    "const_0.25":     "A1",
    "const_0.5":      "A2",
    "const_0.75":     "A3",
    "linear":         "A4",
    "quadratic":      "A5",
    "sqrt":           "A6",
    "sin":            "A7",
    "delayed_linear": "A8",
}


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(results_by_schedule):
    import numpy as np
    summary = {}
    for sched, runs in results_by_schedule.items():
        fids  = [r["fid"]  for r in runs if "fid"  in r]
        cmmds = [r["cmmd"] for r in runs if "cmmd" in r]
        summary[sched] = {
            "fid_mean":  float(np.mean(fids))  if fids  else None,
            "fid_std":   float(np.std(fids))   if fids  else None,
            "cmmd_mean": float(np.mean(cmmds)) if cmmds else None,
            "cmmd_std":  float(np.std(cmmds))  if cmmds else None,
            "n_seeds":   len(runs),
        }
    return summary


def print_summary(summary, schedules):
    col = f"{'Schedule':<16} {'Label':<6} {'FID mean':>10} {'FID std':>9} {'CMMD mean':>11} {'CMMD std':>10} {'n':>3}"
    sep = "=" * len(col)
    print(f"\n{sep}\n{col}\n{sep}")
    for sched in schedules:
        if sched not in summary:
            continue
        s = summary[sched]
        label  = SCHEDULE_LABELS[sched]
        fid_m  = f"{s['fid_mean']:.4f}"  if s["fid_mean"]  is not None else "N/A"
        fid_s  = f"{s['fid_std']:.4f}"   if s["fid_std"]   is not None else "N/A"
        cmmd_m = f"{s['cmmd_mean']:.4f}" if s["cmmd_mean"] is not None else "N/A"
        cmmd_s = f"{s['cmmd_std']:.4f}"  if s["cmmd_std"]  is not None else "N/A"
        print(f"{sched:<16} {label:<6} {fid_m:>10} {fid_s:>9} {cmmd_m:>11} {cmmd_s:>10} {s['n_seeds']:>3}")
    print(sep)


# ── Criterion check ───────────────────────────────────────────────────────────

def check_criterion(summary, schedules):
    if "none" not in summary or "linear" not in summary:
        print("\nCriterion check skipped — A0/A4 not in completed runs.")
        return False

    a0 = summary["none"]
    a4 = summary["linear"]
    print("\n--- Criterion check (reviewer §4 Q5) ---")

    beats_a0 = True
    for metric, label in [("fid_mean", "FID"), ("cmmd_mean", "CMMD")]:
        v0, v4 = a0[metric], a4[metric]
        if v0 is None or v4 is None:
            print(f"  A4 vs A0 {label}: MISSING DATA")
            beats_a0 = False
            continue
        ok = v4 < v0
        print(f"  A4 {label} {v4:.4f} < A0 {v0:.4f}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            beats_a0 = False

    ranked = sorted(
        [(s, summary[s]["fid_mean"]) for s in schedules
         if s in summary and summary[s]["fid_mean"] is not None],
        key=lambda x: x[1],
    )
    rank_a4 = next((i + 1 for i, (s, _) in enumerate(ranked) if s == "linear"), None)
    in_top3 = rank_a4 is not None and rank_a4 <= 3
    print(f"  A4 FID rank (lower is better): {rank_a4} — top-3 required: {'PASS' if in_top3 else 'FAIL'}")

    passed = beats_a0 and in_top3
    print(f"\n  Overall: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Friction γ-schedule sweep: 9 configs × 3 seeds."
    )
    parser.add_argument("--max_steps",   type=int, default=30001)
    parser.add_argument("--no_metrics",  action="store_true")
    parser.add_argument("--base_output", type=str,
                        default="./scripts/runs/friction_sweep")
    parser.add_argument("--schedules",   nargs="+", default=GAMMA_SCHEDULES,
                        choices=GAMMA_SCHEDULES,
                        help="Subset of gamma schedules to run (default: all 9)")
    parser.add_argument("--seeds",       nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    sweep_dir = os.path.join(args.base_output, timestamp)
    os.makedirs(sweep_dir, exist_ok=True)

    results_by_schedule = {s: [] for s in args.schedules}
    all_results = []
    total_runs = len(args.schedules) * len(args.seeds)
    run_idx = 0

    for gamma_schedule in args.schedules:
        for seed in args.seeds:
            run_idx += 1
            label = SCHEDULE_LABELS[gamma_schedule]
            print(f"\n{'='*80}")
            print(f"[{run_idx}/{total_runs}] {label} | gamma={gamma_schedule} | seed={seed}")
            print(f"{'='*80}")

            out_dir = os.path.join(sweep_dir, f"{label}_{gamma_schedule}_seed{seed}")
            try:
                metrics = train(
                    max_steps=args.max_steps,
                    compute_metrics=not args.no_metrics,
                    gamma_schedule=gamma_schedule,
                    seed=seed,
                    output_path=out_dir,
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                metrics = {
                    "gamma_schedule": gamma_schedule,
                    "seed": seed,
                    "label": label,
                    "error": str(exc),
                }

            metrics.setdefault("gamma_schedule", gamma_schedule)
            metrics.setdefault("seed", seed)
            metrics["label"] = label

            results_by_schedule[gamma_schedule].append(metrics)
            all_results.append(metrics)

            with open(os.path.join(sweep_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

    summary = aggregate(results_by_schedule)
    print_summary(summary, args.schedules)
    check_criterion(summary, args.schedules)

    with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to: {sweep_dir}")


if __name__ == "__main__":
    main()
