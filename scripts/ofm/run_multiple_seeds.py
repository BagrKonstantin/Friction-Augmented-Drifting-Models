import argparse
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from scripts.ofm.train import train

def main():
    parser = argparse.ArgumentParser(description="Run OFM model (Adult to Child) across multiple seeds.")
    parser.add_argument("--max_steps", type=int, default=10002, help="Max steps for each run.")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation.")
    args = parser.parse_args()

    seeds = [42, 123, 777, 1234, 2024]
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"RUNNING OFM (Adult to Child) | SEED: {seed}")
        print(f"{'='*80}")
        
        try:
            train(
                max_steps=args.max_steps,
                compute_metrics=not args.no_metrics,
                seed=seed,
                input_data="ADULT",
                target_data="CHILDREN",
                method_name="ofm",
                data_loader="alae"
            )
        except Exception as e:
            print(f"Error running OFM with seed {seed}: {e}")

if __name__ == "__main__":
    main()
