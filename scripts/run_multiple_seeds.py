import argparse
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from ofm_alae import train as train_ofm
from dm_alae import train as train_dm
from dmf_alae import train as train_dmf

def main():
    parser = argparse.ArgumentParser(description="Run all models across multiple seeds.")
    parser.add_argument("--max_steps", type=int, default=30001, help="Max steps for each run.")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation.")
    args = parser.parse_args()

    seeds = [42, 123, 777, 1234, 2024]
    models = [
        #("OFM", train_ofm)
        ("DM", train_dm),
        ("DMF", train_dmf)
    ]

    for model_name, train_fn in models:
        for seed in seeds:
            print(f"\n{'='*80}")
            print(f"RUNNING MODEL: {model_name} | SEED: {seed}")
            print(f"{'='*80}")
            
            try:
                train_fn(
                    max_steps=args.max_steps,
                    compute_metrics=not args.no_metrics,
                    seed=seed
                    # TODO Add arguments
                )
            except Exception as e:
                print(f"Error running {model_name} with seed {seed}: {e}")

if __name__ == "__main__":
    main()
