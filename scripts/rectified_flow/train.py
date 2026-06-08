
import os
import numpy as np
import time
import pathlib
import torch
from scripts.rectified_flow.rf_model import RectifiedFlow, LatentRFUNet, TimeConditionedMLP
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from scripts.exp_utils.data_loader import download_data, download_alae_model, load_alae_data
from scripts.exp_utils.eval_metrics import MetricsEvaluator
from scripts.visualization import save_comparison_plot, decode_latent_to_image, write_image_to_tensorboard

from scripts.exp_utils.exp_utils import seed_all, get_device, setup_paths, get_run_dir, save_config, save_metrics, load_or_compute_target_stats
# OFM Specific Imports
from OFM.ofmsrc.tools import EMA
from OFM.ofmsrc.alae_distributions import TensorSampler


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
    writer = SummaryWriter(output_path, flush_secs=30)


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
    writer.add_text("method", config['method'])
    writer.add_text("output_path", output_path)
    writer.add_scalar("max_steps", max_steps)
    writer.add_scalar("batch_size", batch_size)
    writer.add_scalar("seed", seed)
    save_config(config, output_path)

    # ── Data Loading ──────────────────────────────────────────────────────
    DATA_PATH = os.path.join(setup_paths(), "data")
    try:
        download_data(DATA_PATH)
        download_alae_model()
    except Exception as e:
        print(f"Download failed: {e}")

    data = load_alae_data(DATA_PATH, input_data, target_data, device)
    X_sampler = data["X_sampler"]
    Y_sampler = data["Y_sampler"]
    X_valid_sampler: TensorSampler = data["X_valid_sampler"]
    Y_valid_sampler: TensorSampler = data["Y_valid_sampler"]
    X_valid_tensorboard_indices = np.random.choice(len(X_valid_sampler), size=5, replace=False)
    Y_valid_tensorboard_indices = np.random.choice(len(Y_valid_sampler), size=5, replace=False)
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
            for start in tqdm(range(0, n, batch_size*3), desc="Reflow ODE batches"):
                z0_batch = X_train[start:start + batch_size*3].to(device)
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
        stats_dir = target_stats_path if target_stats_path is not None else pathlib.Path(output_path).parent
        target_cmmd_act, target_fid_act, mu_target, sigma_target = load_or_compute_target_stats(
            str(stats_dir.absolute()), evaluator, Y_test, target_data
        )

    # ── Training Loop ──────────────────────────────────────────────────────
    rf_version = "2-RF (Reflow)" if reflow_from_checkpoint else "1-RF"
    print(f"\nStarting {rf_version} training: {input_data} → {target_data}")
    print(f"Training objective: min ∫₀¹ E[||X₁ - X₀ - v(X_t, t)||²] dt")
    start_time = time.perf_counter()
    validation_period = int(max_steps / 50)
    print(f"validating each {validation_period} step")
    
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
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/pred_velocity_std", pred_velocity.std().item(), step)
            # writer.add_scalar("train/ema_decay", ema.betas[0], step)
            writer.add_scalar("train/lr", optim.param_groups[0]['lr'], step)
            writer.add_histogram("train/pred_velocity", pred_velocity.cpu().detach().numpy(), step)
            
            if step > 0 and step % validation_period == 0:
                # validation
                model.eval()
                with torch.no_grad():
                    X = X_valid_sampler.sample(reflow_batch_size)
                    Y = Y_valid_sampler.sample(reflow_batch_size)
                    z_t_val, t_val, target_velocity_val = rectified_flow.get_train_tuple(X, Y)
                    pred_velocity_val: torch.Tensor = model(z_t_val, t_val)
                    val_loss = F.mse_loss(pred_velocity_val, target_velocity_val)
                    writer.add_scalar("val/loss", val_loss.item(), step)
                    writer.add_scalar("val/pred_velocity_std", pred_velocity_val.std().item(), step)
                    writer.add_histogram("val/pred_velocity", pred_velocity_val.cpu().detach().numpy(), step)
                    
                    # Log sample reconstructions for qualitative monitoring
                    X_tb = X_valid_sampler[X_valid_tensorboard_indices]
                    Y_tb = Y_valid_sampler[Y_valid_tensorboard_indices]
                    z_t_val, t_val, target_velocity_val = rectified_flow.get_train_tuple(X_tb, Y_tb)
                    pred_velocity_val: torch.Tensor = model(z_t_val, t_val)
                    Y_pred = rectified_flow.sample_ode(X_tb, num_steps=num_ode_steps, mute_tqdm=True)
                    for image_index, image in enumerate(decode_latent_to_image(evaluator.alae_model, X_tb)):
                        # import pdb; pdb.set_trace()
                        write_image_to_tensorboard(writer, step, image_index, image, split='val', stage='source')
                    for image_index, image in enumerate(decode_latent_to_image(evaluator.alae_model, Y_tb)):
                        write_image_to_tensorboard(writer, step, image_index, image, split='val', stage='target')
                    for image_index, image in enumerate(decode_latent_to_image(evaluator.alae_model, Y_pred)):
                        write_image_to_tensorboard(writer, step, image_index, image, split='val', stage='prediction')
                    
                model.train()
            # Periodic checkpointing
            if step > 0 and step % 2000 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'step': step,
                }, os.path.join(output_path, f'checkpoint_{step}.pth'))
            writer.flush()

    # Save final model
    writer.close()
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
            writer.add_scalar("test/fid", final_metrics['fid'])
            writer.add_scalar("test/cmmd", final_metrics['cmmd'])

    save_metrics(final_metrics, output_path)

    # ── Results Visualization ──────────────────────────────────────────────
    if evaluator.alae_model:
        print("\nGenerating visualization...")
        model.eval()

        saved_comparison_path = save_comparison_plot(
            lambda x: rectified_flow.sample_ode(x, num_steps=num_ode_steps),
            X_test,
            x_inds_test,
            test_inp_images,
            evaluator.alae_model,
            device,
            os.path.join(output_path, 'rectified_flow_transfer.png'),
            f"Rectified Flow ({num_ode_steps} steps)"
        )
        with open(saved_comparison_path, 'rb') as f:
            image = np.array(Image.open(f))
            image = np.transpose(image, [2, 0, 1])
            writer.add_image(f"comparison", image, step)
            writer.flush()
        

    rf_label = "2-RF" if reflow_from_checkpoint else "1-RF"
    print(f"\n✓ {rf_label} training complete! Results saved to: {output_path}")

    return final_metrics
