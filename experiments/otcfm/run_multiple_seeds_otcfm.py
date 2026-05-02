import argparse
import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from experiments.otcfm.otcfm_alae import train as train_otcfm


def main():
    parser = argparse.ArgumentParser(description="Run OT-CFM across multiple seeds and report mean/std metrics.")
    parser.add_argument("--max_steps",    type=int,   default=5000)
    parser.add_argument("--no_metrics",   action="store_true", help="Disable FID/CMMD evaluation.")
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_ode_steps",type=int,   default=20)
    parser.add_argument("--model",        type=str,   default="OTCFMMLP",
                        choices=["OTCFMMLP", "LatentRFUNet", "TimeConditionedMLP"])
    parser.add_argument("--sigma",        type=float, default=0.1,
                        help="Gaussian noise std on interpolation path (paper E.7 best: σ=0.1).")
    args = parser.parse_args()

    seeds = [42, 123, 777]
    fid_scores  = []
    cmmd_scores = []

    shared_stats_dir = os.path.join("experiments", "runs", "otcfm_target_stats_cache")
    os.makedirs(shared_stats_dir, exist_ok=True)

    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"RUNNING OT-CFM | SEED: {seed} | sigma={args.sigma}")
        print(f"{'='*80}")

        try:
            metrics = train_otcfm(
                max_steps=args.max_steps,
                compute_metrics=not args.no_metrics,
                batch_size=args.batch_size,
                lr=args.lr,
                num_ode_steps=args.num_ode_steps,
                model_name=args.model,
                sigma=args.sigma,
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
    print("MULTI-SEED SUMMARY — OT-CFM")
    print(f"Seeds: {seeds}  sigma={args.sigma}")
    if fid_scores:
        print(f"FID  : {np.mean(fid_scores):.4f} ± {np.std(fid_scores):.4f}  (per seed: {[f'{v:.4f}' for v in fid_scores]})")
    if cmmd_scores:
        print(f"CMMD : {np.mean(cmmd_scores):.4f} ± {np.std(cmmd_scores):.4f}  (per seed: {[f'{v:.4f}' for v in cmmd_scores]})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
