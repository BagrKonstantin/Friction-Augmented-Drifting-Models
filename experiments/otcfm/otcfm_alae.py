#!/usr/bin/env python
"""
OT-CFM (Optimal Transport Conditional Flow Matching) for ALAE Adult/Children transformation
Based on: Flow Matching for Generative Modeling (Lipman et al., 2022)
          Improving and Generalizing Flow-Matching (Tong et al., 2023) — arXiv:2302.00482

Key differences vs Rectified Flow:
  1. Mini-batch OT coupling: pairs (x₀, x₁) via POT earth mover's distance instead of random
  2. Gaussian noise on path: x_t ~ N(t·x₁ + (1-t)·x₀, σ²I)  [σ=0 → identical to RF]

Training objective: min E[||v_θ(t, x_t) − (x₁ − x₀)||²]
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
import torch.nn.functional as F
import ot

from experiments.exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics, load_or_compute_target_stats
setup_paths()

from experiments.data_loader import download_data, download_alae_model, load_alae_data
from experiments.eval_metrics import MetricsEvaluator
from experiments.visualization import save_comparison_plot

from OFM.ofmsrc.tools import EMA
from OFM.ofmsrc.alae_distributions import TensorSampler

from experiments.rectified_flow.rectified_flow_alae import LatentRFUNet, TimeConditionedMLP


class OTCFMMLP(nn.Module):
    """
    MLP velocity field per paper Appendix E.7 (unsupervised latent-space translation):
      4 hidden layers × 512 units, leaky ReLU, input = [x; t] (dim + 1).
    """
    def __init__(self, dim=512, hidden_dim=512, num_layers=4):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden_dim), nn.LeakyReLU(0.01)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.01)]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.net(torch.cat([z, t], dim=1))


class OTCFM:
    """
    OT-CFM model: learns velocity field v(x_t, t) that transports π₀ → π₁.

    Compared to Rectified Flow, this class:
      - Re-couples each mini-batch via exact earth mover's distance (POT).
      - Adds isotropic Gaussian noise σ to the interpolated position.
    """

    def __init__(self, model, sigma=0.0, device='cuda'):
        self.model = model
        self.sigma = sigma
        self.device = device

    def _ot_coupling(self, z0, z1):
        """Return (z0, z1_permuted) where z1 is re-indexed by the OT plan."""
        n = z0.shape[0]
        x0_np = z0.detach().cpu().numpy().astype(np.float64)
        x1_np = z1.detach().cpu().numpy().astype(np.float64)

        M = ot.dist(x0_np, x1_np)  # squared Euclidean cost [n×n]
        M /= M.max() + 1e-8         # normalize to [0,1] for numerical stability

        a = np.ones(n) / n
        b = np.ones(n) / n
        T = ot.emd(a, b, M)         # exact OT plan [n×n], rows sum to 1/n

        # For each z0[i] sample a matched z1[j] from row T[i] (re-normalize for float errors)
        row_sums = T.sum(axis=1, keepdims=True)
        T_normalized = T / (row_sums + 1e-12)
        idx = np.array([np.random.choice(n, p=T_normalized[i]) for i in range(n)])

        return z0, z1[torch.from_numpy(idx).long().to(z1.device)]

    def get_train_tuple(self, z0, z1):
        """
        OT-coupled training tuple.

        Steps:
          1. Re-couple (z0, z1) via mini-batch OT.
          2. Sample t ~ U(0,1).
          3. Interpolate with noise: x_t ~ N(t·z1 + (1-t)·z0, σ²I).
          4. Target velocity = z1 − z0.
        """
        z0_c, z1_c = self._ot_coupling(z0, z1)

        batch_size = z0_c.shape[0]
        t = torch.rand((batch_size, 1), device=z0_c.device)

        mu_t = t * z1_c + (1 - t) * z0_c
        x_t = mu_t + self.sigma * torch.randn_like(mu_t)

        target = z1_c - z0_c
        return x_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0, num_steps=100):
        """Euler integration of the learned velocity field from t=0 to t=1."""
        batch_size = z0.shape[0]
        dt = 1.0 / num_steps
        z = z0.clone()

        for step in tqdm(range(num_steps), desc="Sampling ODE"):
            t = torch.ones((batch_size, 1), device=z.device) * (step / num_steps)
            velocity = self.model(z, t)
            z = z + velocity * dt

        return z


def train(
    max_steps=5000,
    compute_metrics=True,
    batch_size=256,
    lr=1e-3,
    model_name="OTCFMMLP",
    num_ode_steps=100,
    hidden_dim=512,
    num_layers=4,
    activation='leaky_relu',
    sigma=0.1,
    seed=0xBADBEEF,
    input_data="ADULT",
    target_data="CHILDREN",
    output_path=None,
    target_stats_path=None,
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        output_path = get_run_dir("otcfm", seed)
    os.makedirs(output_path, exist_ok=True)

    config = {
        "method": "otcfm",
        "max_steps": max_steps,
        "batch_size": batch_size,
        "lr": lr,
        "num_ode_steps": num_ode_steps,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "activation": activation,
        "sigma": sigma,
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
    X_test    = data["X_test"]
    Y_test    = data["Y_test"]
    x_inds_test    = data["x_inds_test"]
    test_inp_images = data["test_inp_images"]

    # ── Model Initialisation ──────────────────────────────────────────────
    if model_name == "OTCFMMLP":
        model = OTCFMMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    elif model_name == "LatentRFUNet":
        model = LatentRFUNet().to(device)
    elif model_name == "TimeConditionedMLP":
        model = TimeConditionedMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    otcfm = OTCFM(model, sigma=sigma, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(0, betas=[0.999, 0.99])

    print(f"\nModel: {model_name}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  sigma={sigma}  (noise on interpolation path)")

    # ── Eval Metrics Setup ────────────────────────────────────────────────
    evaluator = MetricsEvaluator(device, compute_metrics=compute_metrics)

    mu_target, sigma_target = None, None
    target_cmmd_act = None

    if compute_metrics:
        stats_dir = target_stats_path if target_stats_path is not None else output_path
        target_cmmd_act, target_fid_act, mu_target, sigma_target = load_or_compute_target_stats(
            stats_dir, evaluator, Y_test, target_data
        )

    # ── Training Loop ─────────────────────────────────────────────────────
    print(f"\nStarting OT-CFM training: {input_data} → {target_data}")
    print(f"Training objective: min E[||v_θ(t, x_t) − (x₁ − x₀)||²]  (OT-coupled mini-batches)")
    start_time = time.perf_counter()

    with tqdm(range(max_steps)) as tbar:
        for step in tbar:
            model.train()

            z0 = X_sampler.sample(batch_size)  # X₀ ~ π₀
            z1 = Y_sampler.sample(batch_size)  # X₁ ~ π₁

            x_t, t, target_velocity = otcfm.get_train_tuple(z0, z1)

            pred_velocity = model(x_t, t)
            loss = F.mse_loss(pred_velocity, target_velocity)

            optim.zero_grad()
            loss.backward()
            optim.step()
            ema(model)

            tbar.set_postfix(loss=loss.item(), v_std=pred_velocity.std().item())

            if step > 0 and step % 2000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                }, os.path.join(output_path, f'checkpoint_{step}.pth'))

    torch.save({
        'model': model.state_dict(),
        'optimizer': optim.state_dict(),
        'step': max_steps,
        'config': config,
    }, os.path.join(output_path, 'checkpoint_final.pth'))

    end_time = time.perf_counter()
    total_train_time = end_time - start_time
    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")

    # ── Final Evaluation ──────────────────────────────────────────────────
    final_metrics = {"training_time": total_train_time}

    if compute_metrics:
        print(f"\nComputing final metrics with {num_ode_steps} ODE steps...")
        model.eval()

        with torch.no_grad():
            mapped_test = otcfm.sample_ode(X_test.to(device), num_steps=num_ode_steps)

            gen_fid_act, gen_cmmd_act = evaluator.get_eval_features(mapped_test)
            mu_gen, sigma_gen = evaluator.get_stats(gen_fid_act)
            fid_score  = evaluator.calculate_fid(mu_gen, sigma_gen, mu_target, sigma_target)
            cmmd_score = evaluator.calculate_cmmd(target_cmmd_act, gen_cmmd_act)

            print(f"\nFinal FID  over {len(X_test)} samples = {fid_score:.4f}")
            print(f"Final CMMD over {len(X_test)} samples = {cmmd_score:.4f}")

            final_metrics.update({"fid": float(fid_score), "cmmd": float(cmmd_score)})

        model.train()

    save_metrics(final_metrics, output_path)

    # ── Visualization ─────────────────────────────────────────────────────
    if evaluator.alae_model:
        print("\nGenerating visualization...")
        model.eval()
        save_comparison_plot(
            lambda x: otcfm.sample_ode(x, num_steps=num_ode_steps),
            X_test,
            x_inds_test,
            test_inp_images,
            evaluator.alae_model,
            device,
            os.path.join(output_path, 'otcfm_transfer.png'),
            f"OT-CFM ({num_ode_steps} steps, σ={sigma})"
        )

    print(f"\n✓ OT-CFM training complete! Results saved to: {output_path}")
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OT-CFM for ALAE Adult/Children transformation")
    parser.add_argument("--max_steps",    type=int,   default=5000)
    parser.add_argument("--no_metrics",   action="store_true", help="Disable FID/CMMD evaluation")
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--model",        type=str,   default="OTCFMMLP",
                        choices=["OTCFMMLP", "LatentRFUNet", "TimeConditionedMLP"])
    parser.add_argument("--num_ode_steps",type=int,   default=20)
    parser.add_argument("--hidden_dim",   type=int,   default=1024)
    parser.add_argument("--num_layers",   type=int,   default=3)
    parser.add_argument("--activation",   type=str,   default='tanh', choices=['tanh', 'relu'])
    parser.add_argument("--sigma",        type=float, default=0.1,
                        help="Std of Gaussian noise added to interpolation path (paper E.7 best: σ=0.1)")
    parser.add_argument("--seed",         type=int,   default=0xBADBEEF)
    parser.add_argument("--output_path",  type=str,   default=None)
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model,
        num_ode_steps=args.num_ode_steps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        sigma=args.sigma,
        seed=args.seed,
        output_path=args.output_path,
    )
