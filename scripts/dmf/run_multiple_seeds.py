import argparse
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from scripts.dmf.train import train

def main():
    parser = argparse.ArgumentParser(description="Run DMF model (Adult to Child) across multiple seeds.")
    parser.add_argument("--max_steps", type=int, default=10001, help="Max steps for each run.")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation.")
    parser.add_argument("--gamma_schedule", type=str, default="linear", help="Friction gamma schedule.")
    args = parser.parse_args()

    seeds = [42, 123, 777, 1234, 2024]
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"RUNNING DMF (Adult to Child) | SEED: {seed}")
        print(f"{'='*80}")
        
        try:
            train(
                max_steps=args.max_steps,
                compute_metrics=not args.no_metrics,
                seed=seed,
                gamma_schedule=args.gamma_schedule,
                input_data="ADULT",
                target_data="CHILDREN",
                method_name="dmf",
                data_loader="alae"
            )
        except Exception as e:
            print(f"Error running DMF with seed {seed}: {e}")

if __name__ == "__main__":
    main()
