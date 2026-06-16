#!/usr/bin/env python
import os, sys
from PIL import Image
import numpy as np
import torch
import argparse
import time
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add the root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from OFM.ofmsrc.alae_distributions import TensorSampler
from scripts.exp_utils.exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics
setup_paths()

from scripts.exp_utils.data_loader import download_data, download_alae_model, load_alae_data, load_gender_transfer_data
from scripts.exp_utils.eval_metrics import MetricsEvaluator
from scripts.dmf_gamma.models_and_losses import MLP, drift_loss_pytorch
from scripts.visualization import decode_latent_to_image, save_comparison_plot, write_image_to_tensorboard
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
        return max(0.0, (t - 0.5) * 2.0)
    else:
        raise ValueError(f"Unknown gamma_schedule: {gamma_schedule}")

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
    method_name="dmf",
    data_loader="alae"
):
    device = get_device()
    seed_all(seed)

    if output_path is None:
        output_path = get_run_dir(method_name, seed)
    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path, flush_secs=30)

    # Save config
    config = {
        "method": method_name,
        "max_steps": max_steps,
        "gamma_schedule": gamma_schedule,
        "batch_size": batch_size,
        "lr": lr,
        "r_list": list(r_list),
        "seed": seed,
        "input_data": input_data,
        "target_data": target_data,
        "data_loader": data_loader,
    }
    save_config(config, output_path)
    writer.add_text("method", method_name)
    writer.add_text("output_path", output_path)
    writer.add_scalar("max_steps", max_steps)
    writer.add_scalar("batch_size", batch_size)
    writer.add_scalar("seed", seed)
    validation_period = int(max_steps / 50)
    print(f"validating each {validation_period} step")

    # ── Data Loading ──────────────────────────────────────────────────────
    # Fixed DATA_PATH to be relative to the root or absolute
    root = setup_paths()
    DATA_PATH = os.path.join(root, "data")
    
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
    X_valid_sampler: TensorSampler = data["X_valid_sampler"]
    Y_valid_sampler: TensorSampler = data["Y_valid_sampler"]
    X_valid_tensorboard_indices = np.random.choice(len(X_valid_sampler), size=5, replace=False)
    Y_valid_tensorboard_indices = np.random.choice(len(Y_valid_sampler), size=5, replace=False)
    X_test, Y_test = data["X_test"], data["Y_test"]
    x_inds_test, test_inp_images = data["x_inds_test"], data["test_inp_images"]

    # ── Model Initialisation ──────────────────────────────────────────────
    DIM = 512
    model = MLP(dim=DIM, dimh=1024, num_hidden_layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(0, betas=[0.999, 0.99])

    # ── Eval Metrics Setup ───────────────────────────────────────────────
    evaluator = MetricsEvaluator(device, compute_metrics=compute_metrics)
    mu_target, sigma_target, target_cmmd_act = None, None, None

    if compute_metrics:
        print(f"Precomputing FID & CMMD stats for {target_data} on the entire test set...")
        target_fid_act, target_cmmd_act = evaluator.get_eval_features(Y_test)
        mu_target, sigma_target = evaluator.get_stats(target_fid_act)
        print(f"Target stats computed on {len(target_fid_act)} samples.")

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"Starting training: {method_name}_ALAE_{input_data}_TO_{target_data}_friction")
    start_time = time.perf_counter()
    with tqdm(range(max_steps)) as tbar:
        for step in tbar:
            model.train()
            X, Y = X_sampler.sample(batch_size), Y_sampler.sample(batch_size)
            Y_gen = model(X)

            friction_coeff = compute_friction(step, max_steps, gamma_schedule)

            # Compute drift loss
            loss, info = drift_loss_pytorch(
                Y_gen.unsqueeze(0),
                Y.unsqueeze(0),
                R_list=r_list,
                friction_coeff=friction_coeff
            )

            optim.zero_grad()
            loss.backward()
            optim.step()
            ema(model)

            tbar.set_postfix(
                loss=loss.item(),
                scale=info["scale"].item(),
                friction=f"{friction_coeff:.3f}",
                y_std=Y_gen.std().item()
            )

            if step > 0 and step % 2000 == 0:
                torch.save(model.state_dict(), os.path.join(output_path, 'checkpoint.pth'))
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/scale", info["scale"].item(), step)
            writer.add_scalar("train/force_norm", info["force_norm"].item(), step)
            for r in r_list:
                writer.add_scalar(f"train/loss_{r}", info[f"loss_{r}"].item(), step)
            writer.add_scalar("train/friction_coeff", friction_coeff, step)
            writer.add_scalar("train/lr", optim.param_groups[0]['lr'], step)
           
            if step > 0 and step % validation_period == 0:
                # validation
                model.eval()
                with torch.no_grad():
                    
                    X = X_valid_sampler.sample(batch_size)
                    Y = Y_valid_sampler.sample(batch_size)
                    Y_pred = model(X.to(device))
                    loss, info = drift_loss_pytorch(
                        Y_pred.unsqueeze(0),
                        Y.unsqueeze(0),
                        R_list=r_list,
                        friction_coeff=friction_coeff
                    )

                    writer.add_scalar("val/loss", loss.item(), step)
                    writer.add_scalar("val/scale", info["scale"].item(), step)
                    writer.add_scalar("val/force_norm", info["force_norm"].item(), step)
                    for r in r_list:
                        writer.add_scalar(f"val/loss_{r}", info[f"loss_{r}"].item(), step)
                    writer.add_scalar("val/friction_coeff", friction_coeff, step)
                    writer.add_scalar("val/lr", optim.param_groups[0]['lr'], step)
                    
                    # writing images to tensorboard
                    X_tb = X_valid_sampler[X_valid_tensorboard_indices]
                    Y_tb = Y_valid_sampler[Y_valid_tensorboard_indices]
                    Y_tb_pred = model(X.to(device))
                    
                    for image_index, image in enumerate(decode_latent_to_image(evaluator.alae_model, X_tb)):
                        write_image_to_tensorboard(writer, step, image_index, image, split='val', stage='source')
                    for image_index, image in enumerate(decode_latent_to_image(evaluator.alae_model, Y_tb)):
                        write_image_to_tensorboard(writer, step, image_index, image, split='val', stage='target')
                    for image_index, image in enumerate(decode_latent_to_image(evaluator.alae_model, Y_tb_pred)):
                        write_image_to_tensorboard(writer, step, image_index, image, split='val', stage='prediction')
            writer.flush()
        
                    

    torch.save(model.state_dict(), os.path.join(output_path, 'checkpoint_final.pth'))
    total_train_time = time.perf_counter() - start_time
    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")

    # ── Final Evaluation ───────────────────────────────────────────────────
    final_metrics = {"training_time": total_train_time, "gamma_schedule": gamma_schedule, "seed": seed}
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
            final_metrics.update({"fid": float(fid_score), "cmmd": float(cmmd_score)})
            writer.add_scalar("test/fid", final_metrics['fid'])
            writer.add_scalar("test/cmmd", final_metrics['cmmd'])
        model.train()

    save_metrics(final_metrics, output_path)

    # ── Results Visualization ──────────────────────────────────────────────
    if evaluator.alae_model:
        plot_name = f'{method_name}_transfer_faces.png'
        saved_comparison_path = save_comparison_plot(model, X_test, x_inds_test, test_inp_images, evaluator.alae_model, device, os.path.join(output_path, plot_name), "DMF Mapped")
        with open(saved_comparison_path, 'rb') as f:
            image = np.array(Image.open(f))
            image = np.transpose(image, [2, 0, 1])
            writer.add_image(f"comparison", image, step)
            writer.flush()
    writer.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10001)
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--gamma_schedule", type=str, default="linear", choices=GAMMA_SCHEDULES)
    parser.add_argument("--batch_size", type=int, default=515)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--method_name", type=str, default="dmf")
    parser.add_argument("--data_loader", type=str, default="alae", choices=["alae", "gender"])
    parser.add_argument("--input_data", type=str, default="ADULT")
    parser.add_argument("--target_data", type=str, default="CHILDREN")
    parser.add_argument("--seed", type=int, default=0xBADBEEF)
    args = parser.parse_args()

    train(
        max_steps=args.max_steps,
        compute_metrics=not args.no_metrics,
        gamma_schedule=args.gamma_schedule,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output_path,
        method_name=args.method_name,
        data_loader=args.data_loader,
        input_data=args.input_data,
        target_data=args.target_data,
        seed=args.seed
    )
