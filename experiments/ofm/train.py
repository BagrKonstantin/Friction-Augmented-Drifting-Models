#!/usr/bin/env python
import os, sys
import torch
import argparse
import time
from tqdm import tqdm

# Add the root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from experiments.exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics
setup_paths()

from experiments.data_loader import download_data, download_alae_model, load_alae_data, load_gender_transfer_data
from experiments.eval_metrics import MetricsEvaluator
from experiments.visualization import save_comparison_plot

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
    output_path=None,
    method_name="ofm",
    data_loader="alae"
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        output_path = get_run_dir(method_name, seed)
    os.makedirs(output_path, exist_ok=True)

    # Save config
    config = {
        "method": method_name,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "lr": lr,
        "inv_tolerance": inv_tolerance,
        "seed": seed,
        "input_data": input_data,
        "target_data": target_data,
        "data_loader": data_loader,
    }
    save_config(config, output_path)

    # ── Data Loading ──────────────────────────────────────────────────────
    root = os.path.dirname(os.path.dirname(setup_paths()))
    DATA_PATH = os.path.join(root, "OFM", "data")
    try:
        download_data(DATA_PATH)
        download_alae_model()
    except Exception as e:
        print(f"Download failed: {e}")

    if data_loader == "alae":
        data = load_alae_data(DATA_PATH, input_data, target_data, device)
    elif data_loader == "gender":
        data = load_gender_transfer_data(DATA_PATH, input_data, target_data, device)
    else:
        raise ValueError(f"Unknown data_loader: {data_loader}")

    X_sampler, Y_sampler = data["X_sampler"], data["Y_sampler"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    x_inds_test, test_inp_images = data["x_inds_test"], data["test_inp_images"]

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
    mu_target, sigma_target, target_cmmd_act = None, None, None

    if compute_metrics:
        print(f"Precomputing FID & CMMD stats for {target_data} on the entire test set...")
        target_fid_act, target_cmmd_act = evaluator.get_eval_features(Y_test)
        mu_target, sigma_target = evaluator.get_stats(target_fid_act)
        print(f"Target stats computed on {len(target_fid_act)} samples.")

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"Starting training: {method_name}_ALAE_{input_data}_TO_{target_data}")
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
    total_train_time = time.perf_counter() - start_time
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
            final_metrics.update({"fid": float(fid_score), "cmmd": float(cmmd_score)})
        D.train()
    
    save_metrics(final_metrics, output_path)

    # ── Results Visualization ──────────────────────────────────────────────
    if evaluator.alae_model:
        plot_name = f'{method_name}_transfer_faces.png'
        save_comparison_plot(D, X_test, x_inds_test, test_inp_images, evaluator.alae_model, device, os.path.join(output_path, plot_name), "OFM Mapped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10002)
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--method_name", type=str, default="ofm")
    parser.add_argument("--data_loader", type=str, default="alae", choices=["alae", "gender"])
    parser.add_argument("--input_data", type=str, default="ADULT")
    parser.add_argument("--target_data", type=str, default="CHILDREN")
    parser.add_argument("--seed", type=int, default=0xBADBEEF)
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output_path,
        method_name=args.method_name,
        data_loader=args.data_loader,
        input_data=args.input_data,
        target_data=args.target_data,
        seed=args.seed
    )
