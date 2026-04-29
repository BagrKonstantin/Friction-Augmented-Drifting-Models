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
import os, sys
import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import torch.nn.functional as F

from experiments.exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics, load_or_compute_target_stats
setup_paths()

from experiments.data_loader import download_data, download_alae_model, load_alae_data
from experiments.eval_metrics import MetricsEvaluator
from experiments.visualization import save_comparison_plot

# OFM Specific Imports
from OFM.ofmsrc.tools import EMA
from OFM.ofmsrc.alae_distributions import TensorSampler

from experiments.rectified_flow.rectified_flow_model import RFUNet


class LatentRFUNet(nn.Module):
    """Wraps RFUNet to operate on flat latent vectors [B, latent_dim]."""

    def __init__(self, latent_dim=512, spatial_ch=2, spatial_size=16):
        super().__init__()
        assert spatial_ch * spatial_size * spatial_size == latent_dim, \
            f"spatial_ch * spatial_size^2 must equal latent_dim ({latent_dim})"
        self.spatial_ch = spatial_ch
        self.spatial_size = spatial_size
        self.net = RFUNet(in_channels=spatial_ch)

    def forward(self, z, t):
        B = z.shape[0]
        if t.dim() == 2:
            t = t.squeeze(1)                    # [B]
        t_scaled = (t * 999).long()             # float [0,1] → int [0,999]
        x = z.view(B, self.spatial_ch, self.spatial_size, self.spatial_size)
        out = self.net(x, t_scaled)
        return out.view(B, -1)


class TimeConditionedMLP(nn.Module):
    """
    Time-conditioned MLP for Rectified Flow velocity field v(z, t).
    Architecture from Rectified Flow paper for latent space transformation.
    
    Input: z (latent) concatenated with t (time)
    Output: velocity field v(z,t) with same dimension as z
    """
    def __init__(self, dim=512, hidden_dim=1024, num_layers=3, activation='tanh'):
        super().__init__()
        self.dim = dim
        
        layers = []
        # First layer: (dim + 1) -> hidden_dim (the +1 is for time t)
        layers.append(nn.Linear(dim + 1, hidden_dim))
        
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers.append(act_fn())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
        
        # Output layer: hidden_dim -> dim
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, t):
        """
        Args:
            z: [batch_size, dim] - latent vectors
            t: [batch_size, 1] or [batch_size] - time values in [0, 1]
        Returns:
            velocity: [batch_size, dim] - velocity field v(z,t)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size, 1]
        
        # Concatenate z and t
        zt = torch.cat([z, t], dim=1)  # [batch_size, dim + 1]
        
        # Predict velocity
        velocity = self.net(zt)
        
        return velocity

class RectifiedFlow:
    """
    Rectified Flow model for learning optimal transport between two distributions.
    
    The key idea: learn velocity field v(z,t) such that solving ODE:
        dZ/dt = v(Z_t, t)
    transports samples from π₀ to π₁.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def get_train_tuple(self, z0, z1):
        """
        Generate training data for rectified flow.
        
        Sample random time t ∈ [0,1], compute:
        - z_t = t·z1 + (1-t)·z0  (linear interpolation)
        - target = z1 - z0  (velocity of straight line)
        
        Args:
            z0: [batch_size, dim] - source samples
            z1: [batch_size, dim] - target samples
        Returns:
            z_t: [batch_size, dim] - interpolated samples
            t: [batch_size, 1] - random times
            target: [batch_size, dim] - target velocity
        """
        batch_size = z0.shape[0]
        
        # Sample random time t ∈ [0, 1]
        t = torch.rand((batch_size, 1), device=z0.device)
        
        # Linear interpolation: z_t = t·z1 + (1-t)·z0
        z_t = t * z1 + (1 - t) * z0
        
        # Target velocity (constant along straight line)
        target = z1 - z0
        
        return z_t, t, target
    
    @torch.no_grad()
    def sample_ode(self, z0, num_steps=100):
        """
        Sample from the learned flow using Euler method.
        
        Solve ODE: dZ/dt = v(Z_t, t) from t=0 to t=1
        
        Args:
            z0: [batch_size, dim] - initial samples from π₀
            num_steps: number of discretization steps
        Returns:
            z1: [batch_size, dim] - transported samples ≈ samples from π₁
        """
        batch_size = z0.shape[0]
        dt = 1.0 / num_steps
        
        z = z0.clone()
        
        for step in tqdm(range(num_steps), desc="Sampling ODE"):
            t = torch.ones((batch_size, 1), device=z.device) * (step / num_steps)
            
            # Predict velocity
            velocity = self.model(z, t)
            
            # Euler step
            z = z + velocity * dt
        
        return z


def train(
    max_steps=10001,
    compute_metrics=True,
    batch_size=512,
    reflow_batch_size=512,
    lr=1e-3,
    model_name="LatentRFUNet",
    num_ode_steps=100,
    hidden_dim=1024,
    num_layers=3,
    activation='tanh',
    reflow_from_checkpoint=None,
    seed=0xBADBEEF,
    input_data="ADULT",
    target_data="CHILDREN",
    output_path=None,
    target_stats_path=None,
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        rf_name = "2_rectified_flow" if reflow_from_checkpoint else "1_rectified_flow"
        output_path = get_run_dir(rf_name, seed)
    os.makedirs(output_path, exist_ok=True)

    # Save config
    config = {
        "method": "rectified_flow",
        "max_steps": max_steps,
        "batch_size": batch_size,
        "reflow_batch_size": reflow_batch_size,
        "lr": lr,
        "num_ode_steps": num_ode_steps,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "activation": activation,
        "reflow_from_checkpoint": reflow_from_checkpoint,
        "seed": seed,
        "input_data": input_data,
        "target_data": target_data,
    }
    save_config(config, output_path)

    # ── Data Loading ──────────────────────────────────────────────────────
    DATA_PATH = "../baselines/OFM/data"
    try:
        download_data(DATA_PATH)
        download_alae_model()
    except Exception as e:
        print(f"Download failed: {e}")

    data = load_alae_data(DATA_PATH, input_data, target_data, device)
    X_sampler = data["X_sampler"]
    Y_sampler = data["Y_sampler"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    x_inds_test = data["x_inds_test"]
    test_inp_images = data["test_inp_images"]

    # ── Reflow Data Generation (for 2-RF) ─────────────────────────────────
    if reflow_from_checkpoint:
        print(f"\n=== REFLOW MODE: Generating paired data from 1-RF ===")
        print(f"Loading 1-RF model from: {reflow_from_checkpoint}")
        
        # Load 1-RF model
        checkpoint = torch.load(reflow_from_checkpoint, map_location=device)
        if model_name == "LatentRFUNet":
            prev_model = LatentRFUNet().to(device)
        elif model_name == "TimeConditionedMLP":
            prev_model = TimeConditionedMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation).to(device)

        prev_model.load_state_dict(checkpoint['model'])
        prev_model.eval()
        prev_flow = RectifiedFlow(prev_model, device=device)
        
        # Generate paired data by running 1-RF on source distribution
        print("Generating deterministic pairs (Z₀, Z₁) from 1-RF...")
        X_train = data["X_train"]

        Z0_chunks, Z1_chunks = [], []
        n = len(X_train)
        with torch.no_grad():
            for start in tqdm(range(0, n, reflow_batch_size), desc="Reflow ODE batches"):
                z0_batch = X_train[start:start + reflow_batch_size].to(device)
                z1_batch = prev_flow.sample_ode(z0_batch, num_steps=num_ode_steps)
                Z0_chunks.append(z0_batch.cpu())
                Z1_chunks.append(z1_batch.cpu())

        Z0 = torch.cat(Z0_chunks, dim=0)
        Z1 = torch.cat(Z1_chunks, dim=0)

        # Update samplers with new paired data
        X_sampler = TensorSampler(Z0, device=device)
        Y_sampler = TensorSampler(Z1, device=device)
        
        print(f"Generated {len(Z0)} paired samples for 2-RF training")
        print("Training 2-RF will learn straighter trajectories!")
        
        del prev_model, prev_flow
        torch.cuda.empty_cache() if device == 'cuda' else None

    # ── Model Initialisation ──────────────────────────────────────────────
    if model_name == "LatentRFUNet":    
        model = LatentRFUNet().to(device)
    elif model_name == "TimeConditionedMLP":
        model = TimeConditionedMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation).to(device)

    # For 2-RF, optionally initialize with 1-RF weights for faster training
    if reflow_from_checkpoint:
        try:
            checkpoint = torch.load(reflow_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print("Initialized 2-RF with 1-RF weights for faster convergence")
        except Exception as e:
            print(f"Could not load weights for initialization: {e}")
            print("Training from scratch")
    
    rectified_flow = RectifiedFlow(model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    ema = EMA(0, betas=[0.999, 0.99])

    print(f"\nModel: LatentRFUNet (RFUNet U-Net, latent 512 → 2×16×16)")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Eval Metrics Setup ───────────────────────────────────────────────
    evaluator = MetricsEvaluator(device, compute_metrics=compute_metrics)

    mu_target, sigma_target = None, None
    target_cmmd_act = None

    if compute_metrics:
        stats_dir = target_stats_path if target_stats_path is not None else output_path
        target_cmmd_act, target_fid_act, mu_target, sigma_target = load_or_compute_target_stats(
            stats_dir, evaluator, Y_test, target_data
        )

    # ── Training Loop ──────────────────────────────────────────────────────
    rf_version = "2-RF (Reflow)" if reflow_from_checkpoint else "1-RF"
    print(f"\nStarting {rf_version} training: {input_data} → {target_data}")
    print(f"Training objective: min ∫₀¹ E[||X₁ - X₀ - v(X_t, t)||²] dt")
    start_time = time.perf_counter()
    
    with tqdm(range(max_steps)) as tbar:
        for step in tbar:
            model.train()
            
            # Sample batch from source and target distributions (random coupling)
            z0 = X_sampler.sample(batch_size)  # X₀ ~ π₀
            z1 = Y_sampler.sample(batch_size)  # X₁ ~ π₁
            
            # Get training tuple: (z_t, t, target_velocity)
            z_t, t, target_velocity = rectified_flow.get_train_tuple(z0, z1)
            
            # Predict velocity
            pred_velocity = model(z_t, t)
            
            # Rectified Flow loss: ||target_velocity - pred_velocity||²
            loss = F.mse_loss(pred_velocity, target_velocity)
            
            # Optimization step
            optim.zero_grad()
            loss.backward()
            optim.step()
            ema(model)
            
            # Logging
            tbar.set_postfix(
                loss=loss.item(),
                v_std=pred_velocity.std().item()
            )
            
            # Periodic checkpointing
            if step > 0 and step % 2000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                }, os.path.join(output_path, f'checkpoint_{step}.pth'))

    # Save final model
    torch.save({
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'step': max_steps,
        'config': config
    }, os.path.join(output_path, 'checkpoint_final.pth'))
    
    end_time = time.perf_counter()
    total_train_time = end_time - start_time
    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")

    # ── Final Evaluation ───────────────────────────────────────────────────
    final_metrics = {"training_time": total_train_time}
    
    if compute_metrics:
        print(f"\nComputing final metrics with {num_ode_steps} ODE steps...")
        model.eval()
        
        with torch.no_grad():
            # Generate samples by solving ODE
            mapped_test = rectified_flow.sample_ode(X_test.to(device), num_steps=num_ode_steps)
            
            # Compute FID and CMMD
            gen_fid_act, gen_cmmd_act = evaluator.get_eval_features(mapped_test)
            mu_gen, sigma_gen = evaluator.get_stats(gen_fid_act)
            fid_score = evaluator.calculate_fid(mu_gen, sigma_gen, mu_target, sigma_target)
            cmmd_score = evaluator.calculate_cmmd(target_cmmd_act, gen_cmmd_act)

            print(f"\nFinal FID over {len(X_test)} samples = {fid_score:.4f}")
            print(f"Final CMMD over {len(X_test)} samples = {cmmd_score:.4f}")
            
            final_metrics.update({
                "fid": float(fid_score),
                "cmmd": float(cmmd_score)
            })
        
        model.train()

    save_metrics(final_metrics, output_path)

    # ── Results Visualization ──────────────────────────────────────────────
    if evaluator.alae_model:
        print("\nGenerating visualization...")
        model.eval()
        with torch.no_grad():
            mapped_test_vis = rectified_flow.sample_ode(X_test.to(device), num_steps=num_ode_steps)

        save_comparison_plot(
            lambda x: rectified_flow.sample_ode(x, num_steps=num_ode_steps),
            X_test,
            x_inds_test,
            test_inp_images,
            evaluator.alae_model,
            device,
            os.path.join(output_path, 'rectified_flow_transfer.png'),
            f"Rectified Flow ({num_ode_steps} steps)"
        )

    rf_label = "2-RF" if reflow_from_checkpoint else "1-RF"
    print(f"\n✓ {rf_label} training complete! Results saved to: {output_path}")

    return final_metrics


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
