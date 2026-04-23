#!/usr/bin/env python
import os, sys
import torch
import argparse
import time
from tqdm import tqdm

from exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics
setup_paths()

from data_loader import download_data, download_alae_model, load_gender_transfer_data
from eval_metrics import MetricsEvaluator
from models_and_losses import MLP, drift_loss_pytorch
from visualization import save_comparison_plot

# OFM Specific Imports
from OFM.ofmsrc.tools import EMA

def train(
    max_steps=10001,
    compute_metrics=True,
    batch_size=515,
    lr=5e-4,
    r_list=(0.005, 0.01, 0.02),
    seed=0xBADBEEF,
    input_data="MAN",
    target_data="WOMAN",
    output_path=None
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        output_path = get_run_dir("dm_mf", seed)
    os.makedirs(output_path, exist_ok=True)

    # Save config
    config = {
        "method": "dm_mf",
        "max_steps": max_steps,
        "batch_size": batch_size,
        "lr": lr,
        "r_list": list(r_list),
        "seed": seed,
        "input_data": input_data,
        "target_data": target_data,
    }
    save_config(config, output_path)

    # ── Data Loading ──────────────────────────────────────────────────────
    DATA_PATH = "../OFM/data"
    try:
        download_data(DATA_PATH)
        download_alae_model()
    except Exception as e:
        print(f"Download failed: {e}")

    data = load_gender_transfer_data(DATA_PATH, input_data, target_data, device)
    X_sampler = data["X_sampler"]
    Y_sampler = data["Y_sampler"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    x_inds_test = data["x_inds_test"]
    test_inp_images = data["test_inp_images"]

    # ── Model Initialisation ──────────────────────────────────────────────
    DIM = 512
    model = MLP(dim=DIM, dimh=1024, num_hidden_layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(0, betas=[0.999, 0.99])

    # ── Eval Metrics Setup ───────────────────────────────────────────────
    evaluator = MetricsEvaluator(device, compute_metrics=compute_metrics)

    mu_target, sigma_target = None, None
    target_cmmd_act = None

    if compute_metrics:
        print(f"Precomputing FID & CMMD stats for {target_data} on the entire test set...")
        target_fid_act, target_cmmd_act = evaluator.get_eval_features(Y_test)
        mu_target, sigma_target = evaluator.get_stats(target_fid_act)
        print(f"Target stats computed on {len(target_fid_act)} samples.")

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"Starting training: dm_ALAE_{input_data}_TO_{target_data}")
    start_time = time.perf_counter()
    with tqdm(range(max_steps)) as tbar:
        for step in tbar:
            model.train()
            X, Y = X_sampler.sample(batch_size), Y_sampler.sample(batch_size)

            Y_gen = model(X)

            # Compute drift loss (friction_coeff=0 for standard DM)
            loss, info = drift_loss_pytorch(
                Y_gen.unsqueeze(0),
                Y.unsqueeze(0),
                R_list=r_list,
                friction_coeff=0.0
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
            ema(model)

            tbar.set_postfix(
                loss=loss.item(),
                scale=info["scale"].item(),
                f_norm=info["force_norm"].item(),
                y_std=Y_gen.std().item()
            )

            if step > 0 and step % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(output_path, 'checkpoint.pth'))

    torch.save(model.state_dict(), os.path.join(output_path, 'checkpoint_final.pth'))
    end_time = time.perf_counter()
    total_train_time = end_time - start_time
    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")

    # ── Final Evaluation ───────────────────────────────────────────────────
    final_metrics = {"training_time": total_train_time}
    if compute_metrics:
        print("Computing final FID and CMMD on the entire test set...")
        model.eval()
        with torch.no_grad():
            mapped_test = model(X_test.to(device))
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
        save_comparison_plot(model, X_test, x_inds_test, test_inp_images, evaluator.alae_model, device, os.path.join(output_path, 'dm_mf_transfer_faces.png'), "DM Mapped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10001)
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation")
    parser.add_argument("--batch_size", type=int, default=515)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output_path", type=str, default='./experiments/results/dm_alae_mf')
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output_path
    )
