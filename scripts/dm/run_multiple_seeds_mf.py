import argparse
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from scripts.dm.train import train

def main():
    parser = argparse.ArgumentParser(description="Run DM model (Male to Female) across multiple seeds.")
    parser.add_argument("--max_steps", type=int, default=10001, help="Max steps for each run.")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation.")
    args = parser.parse_args()

    seeds = [42, 123, 777]
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"RUNNING DM (Male to Female) | SEED: {seed}")
        print(f"{'='*80}")
        
        try:
            train(
                max_steps=args.max_steps,
                compute_metrics=not args.no_metrics,
                seed=seed,
                input_data="MAN",
                target_data="WOMAN",
                method_name="dm_mf",
                data_loader="gender"
            )
        except Exception as e:
            print(f"Error running DM MF with seed {seed}: {e}")

if __name__ == "__main__":
    main()
