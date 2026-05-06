import argparse
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.getcwd())

from scripts.dm import train

def main():
    parser = argparse.ArgumentParser(description="Run DM model Adult to Child.")
    parser.add_argument("--max_steps", type=int, default=10001, help="Max steps for training.")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation.")
    parser.add_argument("--batch_size", type=int, default=515, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0xBADBEEF, help="Random seed.")
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        input_data="ADULT",
        target_data="CHILDREN",
        method_name="dm",
        data_loader="alae"
    )

if __name__ == "__main__":
    main()
