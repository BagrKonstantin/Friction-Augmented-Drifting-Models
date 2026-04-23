#!/usr/bin/env python
import os, sys
import torch
import argparse
import time
from tqdm import tqdm

from exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics
setup_paths()

from data_loader import download_data, download_alae_model, load_alae_data
from eval_metrics import MetricsEvaluator
from models_and_losses import MLP # DenseICNN is imported below
from visualization import save_comparison_plot

# OFM Specific Imports
from OFM.ofmsrc.tools import EMA
from OFM.ofmsrc.icnn import DenseICNN
from OFM.ofmsrc.model_tools import ofm_loss, TorchlbfgsInvParams, BruteforceIHVParams
from OFM.ofmsrc.discrete_ot import OTPlanSampler

def train(
        max_steps=10002,
        compute_metrics=True,
        batch_size=128,
        lr=1e-3,
        inv_tolerance=1e-2,
        seed=0xBADBEEF,
        input_data="ADULT",
        target_data="CHILDREN",
        output_path=None
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        output_path = get_run_dir("ofm", seed)
    os.makedirs(output_path, exist_ok=True)

    # Save config
    config = {
        "method": "ofm",
        "max_steps": max_steps,
        "batch_size": batch_size,
        "lr": lr,
        "inv_tolerance": inv_tolerance,
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

    data = load_alae_data(DATA_PATH, input_data, target_data, device)
    X_sampler = data["X_sampler"]
    Y_sampler = data["Y_sampler"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    x_inds_test = data["x_inds_test"]
    test_inp_images = data["test_inp_images"]

    # ── Model Initialisation ──────────────────────────────────────────────
    DIM = 512
    D = DenseICNN(DIM, [1024, 1024]).to(device)
    D.convexify()
    D_opt = torch.optim.Adam(D.parameters(), lr=lr)
    ema = EMA(0, betas=[0.999, 0.99])

    # ── OT Sampling Logic ────────────────────────────────────────────────
    ot_plan_sampler = OTPlanSampler('exact')

    def sampling_fn(bs):
        x_samples = X_sampler.sample(bs)
        y_samples = Y_sampler.sample(bs)
        return ot_plan_sampler.sample_plan(x_samples, y_samples)

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
    print(f"Starting training: ofm_ALAE_{input_data}_TO_{target_data}")
    start_time = time.perf_counter()
    with tqdm(range(max_steps)) as tbar:
        for step in tbar:
            D.train()
            X, Y = sampling_fn(batch_size)
            t = (torch.rand(batch_size) + 1e-8).to(device)

            current_metrics = {}
            loss, true_loss = ofm_loss(
                D, X, Y, t,
                TorchlbfgsInvParams(lbfgs_params=dict(tolerance_grad=inv_tolerance), max_iter=10),
                BruteforceIHVParams(),
                tol_inverse_border=0.5,
                stats=current_metrics
            )

            D_opt.zero_grad()
            loss.backward()
            D_opt.step()
            D.convexify()
            ema(D)

            tbar.set_postfix(
                loss=loss.item(),
                true_loss=true_loss.item(),
                y_std=Y.std().item()
            )


            if compute_metrics and step > 0 and step % 2000 == 0:
                D.eval()
                X_test_input = X_test.to(device).float().requires_grad_(True)
                mapped_test = D.push_nograd(X_test_input)
                with torch.no_grad():
                    gen_fid_act, _ = evaluator.get_eval_features(mapped_test.cpu(), batch_size=64)
                    mu_gen, sigma_gen = evaluator.get_stats(gen_fid_act)
                    fid_score = evaluator.calculate_fid(mu_gen, sigma_gen, mu_target, sigma_target)
                    print(f"\nStep {step} | FID: {fid_score:.4f}")
                D.train()

            if step > 0 and step % 2000 == 0:
                torch.save(D.state_dict(), os.path.join(output_path, 'checkpoint.pth'))

    torch.save(D.state_dict(), os.path.join(output_path, 'checkpoint_final.pth'))
    end_time = time.perf_counter()
    total_train_time = end_time - start_time
    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")

    # ── Final Evaluation ───────────────────────────────────────────────────
    final_metrics = {"training_time": total_train_time}
    if compute_metrics:
        print("Computing final FID and CMMD on the entire test set...")
        D.eval()
        X_test_input = X_test.to(device).float().requires_grad_(True)
        mapped_test = D.push_nograd(X_test_input)
        with torch.no_grad():
            gen_fid_act, gen_cmmd_act = evaluator.get_eval_features(mapped_test.cpu())

            mu_gen, sigma_gen = evaluator.get_stats(gen_fid_act)
            fid_score = evaluator.calculate_fid(mu_gen, sigma_gen, mu_target, sigma_target)
            cmmd_score = evaluator.calculate_cmmd(target_cmmd_act, gen_cmmd_act)

            print(f"\nFinal FID over {len(X_test)} samples = {fid_score:.4f}")
            print(f"Final CMMD over {len(X_test)} samples = {cmmd_score:.4f}")

            final_metrics.update({
                "fid": float(fid_score),
                "cmmd": float(cmmd_score)
            })
        D.train()

    save_metrics(final_metrics, output_path)

    # ── Results Visualization ──────────────────────────────────────────────
    if evaluator.alae_model:
        save_comparison_plot(D, X_test, x_inds_test, test_inp_images, evaluator.alae_model, device, os.path.join(output_path, 'ofm_transfer_faces.png'), "OFM Mapped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10002)
    parser.add_argument("--no_metrics", action="store_true", help="Disable FID/CMMD evaluation")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_path", type=str, default='./experiments/results/ofm_alae')
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output_path
    )
