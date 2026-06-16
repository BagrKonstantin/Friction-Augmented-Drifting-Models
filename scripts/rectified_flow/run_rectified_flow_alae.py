#!/usr/bin/env python
"""
Rectified Flow implementation for ALAE Adult/Children transformation
Based on: Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow
Paper: https://arxiv.org/abs/2209.03003

Supports both 1-RF and 2-RF (reflow):
  
  1-RF: Train on random pairs (X₀, X₁) from source and target distributions
        python rectified_flow_alae.py --max_steps 10000
  
  2-RF: Generate paired data from 1-RF, then train on straighter trajectories
        # First train 1-RF
        python rectified_flow_alae.py --max_steps 10000 --output_path ./1_rf
        # Then reflow for 2-RF
        python rectified_flow_alae.py --max_steps 10000 --reflow_from_checkpoint ./1_rf/checkpoint_final.pth --output_path ./2_rf
"""
import argparse
from scripts.exp_utils.exp_utils import setup_paths
setup_paths()
from train import train



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Rectified Flow for ALAE Adult/Children transformation")
    parser.add_argument("--max_steps", type=int, default=10001, help="Number of training steps")
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--reflow_batch_size", type=int, default=512, help="Batch size for ODE evaluation during reflow data generation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default="LatentRFUNet", choices=["LatentRFUNet", "TimeConditionedMLP"], help="Model architecture to use")
    parser.add_argument("--num_ode_steps", type=int, default=20, help="Number of ODE integration steps")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of MLP layers")
    parser.add_argument("--activation", type=str, default='tanh', choices=['tanh', 'relu'], help="Activation function")
    parser.add_argument("--reflow_from_checkpoint", type=str, default=None, help="Path to 1-RF checkpoint for reflow (2-RF)")
    parser.add_argument("--seed", type=int, default=0xBADBEEF, help="Random seed")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        batch_size=args.batch_size,
        reflow_batch_size=args.reflow_batch_size,
        lr=args.lr,
        num_ode_steps=args.num_ode_steps,
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        reflow_from_checkpoint=args.reflow_from_checkpoint,
        seed=args.seed,
        output_path=args.output_path
    )
