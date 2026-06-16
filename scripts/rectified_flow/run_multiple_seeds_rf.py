import argparse
import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from scripts.rectified_flow.run_rectified_flow_alae import train as train_rf


def main():
    parser = argparse.ArgumentParser(description="Run Rectified Flow across multiple seeds and report mean/std metrics.")
    parser.add_argument("--max_steps", type=int, default=1001, help="Max steps for each run.")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_ode_steps", type=int, default=20)
    parser.add_argument("--model", type=str, default="LatentRFUNet", choices=["LatentRFUNet", "TimeConditionedMLP"])
    parser.add_argument("--reflow_checkpoint", type=str, default=None, help="Path to checkpoint for reflow.")
    args = parser.parse_args()

    seeds = [42, 123, 777]
    fid_scores = []
    cmmd_scores = []

    shared_stats_dir = os.path.join("scripts", "runs", "rf_target_stats_cache")
    os.makedirs(shared_stats_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"RUNNING Rectified Flow | SEED: {seed}")
        print(f"{'='*80}")

        try:
            metrics = train_rf(
                max_steps=args.max_steps,
                compute_metrics=not args.no_metrics,
                batch_size=args.batch_size,
                lr=args.lr,
                num_ode_steps=args.num_ode_steps,
                model_name=args.model,
                reflow_from_checkpoint=args.reflow_checkpoint,
                seed=seed,
                target_stats_path=shared_stats_dir,
            )
            if metrics and "fid" in metrics:
                fid_scores.append(metrics["fid"])
            if metrics and "cmmd" in metrics:
                cmmd_scores.append(metrics["cmmd"])
        except Exception as e:
            print(f"Error running seed {seed}: {e}")

    print(f"\n{'='*80}")
    print("MULTI-SEED SUMMARY — Rectified Flow")
    print(f"Seeds: {seeds}")
    if fid_scores:
        print(f"FID  : {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}  (per seed: {[f'{v:.4f}' for v in fid_scores]})")
    if cmmd_scores:
        print(f"CMMD : {np.mean(cmmd_scores):.4f} ± {np.std(cmmd_scores):.4f}  (per seed: {[f'{v:.4f}' for v in cmmd_scores]})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
