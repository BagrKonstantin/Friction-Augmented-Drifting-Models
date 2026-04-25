#!/usr/bin/env python
"""
DMF-ALAE with configurable friction γ-schedule.

Gamma schedules
---------------
none           — A0: γ = 0 always (no friction, baseline)
const_0.25     — A1: γ = 0.25 constant
const_0.5      — A2: γ = 0.5 constant
const_0.75     — A3: γ = 0.75 constant
linear         — A4: γ ramps linearly 0→1 over training
quadratic      — A5: γ = (t/T)²
sqrt           — A6: γ = √(t/T)
sin            — A7: γ = sin(π/2 · t/T)
delayed_linear — A8: γ=0 for first 50% of steps, then ramps 0→1

Reviewer §4 Q5 requirement: A4 (linear) must be top-3 and beat A0 on all metrics.
"""
import math
import os
import sys
import time
import argparse

import torch
from tqdm import tqdm

from exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics
setup_paths()

from data_loader import download_data, download_alae_model, load_alae_data
from eval_metrics import MetricsEvaluator
from models_and_losses import MLP, drift_loss_pytorch
from visualization import save_comparison_plot
from OFM.ofmsrc.tools import EMA

GAMMA_SCHEDULES = [
    "none",
    "const_0.25",
    "const_0.5",
    "const_0.75",
    "linear",
    "quadratic",
    "sqrt",
    "sin",
    "delayed_linear",
]


def compute_friction(step: int, max_steps: int, gamma_schedule: str) -> float:
    t = step / (max_steps - 1) if max_steps > 1 else 1.0
    if gamma_schedule == "none":
        return 0.0
    elif gamma_schedule == "const_0.25":
        return 0.25
    elif gamma_schedule == "const_0.5":
        return 0.5
    elif gamma_schedule == "const_0.75":
        return 0.75
    elif gamma_schedule == "linear":
        return t
    elif gamma_schedule == "quadratic":
        return t ** 2
    elif gamma_schedule == "sqrt":
        return math.sqrt(t)
    elif gamma_schedule == "sin":
        return math.sin(math.pi / 2 * t)
    elif gamma_schedule == "delayed_linear":
        # Zero for first half, then linear ramp 0→1 over second half
        return max(0.0, (t - 0.5) * 2.0)
    else:
        raise ValueError(f"Unknown gamma_schedule: {gamma_schedule!r}. "
                         f"Choose from: {GAMMA_SCHEDULES}")


def train(
    max_steps=10001,
    compute_metrics=True,
    gamma_schedule="linear",
    batch_size=515,
    lr=5e-4,
    r_list=(0.005, 0.01, 0.02),
    seed=0xBADBEEF,
    input_data="ADULT",
    target_data="CHILDREN",
    output_path=None,
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        output_path = get_run_dir(f"dmf_{gamma_schedule}", seed)
    os.makedirs(output_path, exist_ok=True)

    config = {
        "method": "dmf_gamma",
        "max_steps": max_steps,
        "gamma_schedule": gamma_schedule,
        "batch_size": batch_size,
        "lr": lr,
        "r_list": list(r_list),
        "seed": seed,
        "input_data": input_data,
        "target_data": target_data,
    }
    save_config(config, output_path)

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

    DIM = 512
    model = MLP(dim=DIM, dimh=1024, num_hidden_layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(0, betas=[0.999, 0.99])

    evaluator = MetricsEvaluator(device, compute_metrics=compute_metrics)
    mu_target, sigma_target, target_cmmd_act = None, None, None

    if compute_metrics:
        print(f"Precomputing FID & CMMD stats for {target_data}...")
        target_fid_act, target_cmmd_act = evaluator.get_eval_features(Y_test)
        mu_target, sigma_target = evaluator.get_stats(target_fid_act)
        print(f"Target stats computed on {len(target_fid_act)} samples.")

    print(f"Starting: DMF | gamma_schedule={gamma_schedule} | seed={seed}")
    start_time = time.perf_counter()
    with tqdm(range(max_steps)) as tbar:
        for step in tbar:
            model.train()
            X, Y = X_sampler.sample(batch_size), Y_sampler.sample(batch_size)

            Y_gen = model(X)
            friction_coeff = compute_friction(step, max_steps, gamma_schedule)

            loss, info = drift_loss_pytorch(
                Y_gen.unsqueeze(0),
                Y.unsqueeze(0),
                R_list=r_list,
                friction_coeff=friction_coeff,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
            ema(model)

            tbar.set_postfix(
                loss=loss.item(),
                scale=info["scale"].item(),
                friction=f"{friction_coeff:.3f}",
                y_std=Y_gen.std().item(),
            )

            if step > 0 and step % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(output_path, "checkpoint.pth"))

    torch.save(model.state_dict(), os.path.join(output_path, "checkpoint_final.pth"))
    total_train_time = time.perf_counter() - start_time
    print(f"\nTotal Training Time: {total_train_time:.2f} s")

    final_metrics = {
        "training_time": total_train_time,
        "gamma_schedule": gamma_schedule,
        "seed": seed,
    }

    if compute_metrics:
        print("Computing final FID and CMMD on the entire test set...")
        model.eval()
        with torch.no_grad():
            mapped_test = model(X_test.to(device))
            gen_fid_act, gen_cmmd_act = evaluator.get_eval_features(mapped_test)
            mu_gen, sigma_gen = evaluator.get_stats(gen_fid_act)
            fid_score = evaluator.calculate_fid(mu_gen, sigma_gen, mu_target, sigma_target)
            cmmd_score = evaluator.calculate_cmmd(target_cmmd_act, gen_cmmd_act)

        print(f"Final FID  = {fid_score:.4f}")
        print(f"Final CMMD = {cmmd_score:.4f}")
        final_metrics["fid"] = float(fid_score)
        final_metrics["cmmd"] = float(cmmd_score)
        model.train()

    save_metrics(final_metrics, output_path)

    if evaluator.alae_model:
        save_comparison_plot(
            model, X_test, x_inds_test, test_inp_images, evaluator.alae_model, device,
            os.path.join(output_path, f"dm_transfer_faces_{gamma_schedule}.png"),
            f"DMF {gamma_schedule}",
        )

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10001)
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--gamma_schedule", type=str, default="linear",
                        choices=GAMMA_SCHEDULES)
    parser.add_argument("--batch_size", type=int, default=515)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0xBADBEEF)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        gamma_schedule=args.gamma_schedule,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_path=args.output_path,
    )
