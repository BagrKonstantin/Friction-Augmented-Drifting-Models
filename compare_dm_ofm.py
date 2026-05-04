#!/usr/bin/env python
import os
import sys
import random
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
sys.path.append(".")
sys.path.append("OFM")
sys.path.append("OFM/ALAE")

# ── Imports ────────────────────────────────────────────────────────────
try:
    from OFM.ALAE.alae_ffhq_inference import load_model, decode
    from OFM.ofmsrc.icnn import DenseICNN
except ImportError:
    print("Error: Required modules not found. Ensure OFM/ALAE and OFM/ofmsrc exist.")
    sys.exit(1)

# np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────────────────
DEVICE = 'cpu'
DIM = 512
DATA_PATH = "OFM/data"
RUNS_DIR = "experiment42"


# ── Helper for dynamic path discovery ──────────────────────────────────
def get_model_path(model_type, seed):
    """
    Finds the model path in experiments/runs based on model_type and seed.
    Pattern: experiments/runs/*_{model_type}_{seed}/checkpoint_final.pth
    """
    pattern = os.path.join(RUNS_DIR, f"*_{model_type}_{seed}")
    dirs = glob.glob(pattern)
    if not dirs:
        print(f"Error: No run directory found for model_type='{model_type}' and seed='{seed}' in {RUNS_DIR}")
        sys.exit(1)

    run_dir = dirs[0]
    model_path = os.path.join(run_dir, "checkpoint_final.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    return model_path


# ── DM Model Logic (Redefined as in dm_alae.py) ──────────────────────────
class MLP(nn.Module):
    def __init__(self, dim=512, dimh=1024, num_hidden_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, dimh))
        layers.append(nn.LayerNorm(dimh))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(dimh, dimh))
            layers.append(nn.LayerNorm(dimh))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dimh, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


def load_dm_model(path):
    if os.path.exists(path):
        print(f"Loading Model weights from {path}...")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            model = MLP(dim=DIM, dimh=1024, num_hidden_layers=3).to(DEVICE)
            model.load_state_dict(ckpt)
            model.eval()
            return model
        else:
            ckpt.to(DEVICE)
            ckpt.eval()
            return ckpt
    else:
        print(f"Error: Model not found at {path}")
        sys.exit(1)


def load_ofm_model(path):
    if os.path.exists(path):
        print(f"Loading OFM Model weights from {path}...")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict):
            model = DenseICNN(DIM, [1024, 1024]).to(DEVICE)
            model.load_state_dict(ckpt)
            model.eval()
            return model
        else:
            ckpt.to(DEVICE)
            ckpt.eval()
            return ckpt
    else:
        print(f"Error: OFM model not found at {path}")
        sys.exit(1)


def load_cfm_model(path, model_type):
    if not os.path.exists(path):
        print(f"Error: {model_type} model not found at {path}")
        sys.exit(1)

    print(f"Loading {model_type} weights from {path}...")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # These checkpoints are dicts: {'model': state_dict, 'config': {...}}
    cfg = ckpt.get('config', {})
    hidden_dim = cfg.get('hidden_dim', 1024)
    num_layers = cfg.get('num_layers', 3)
    activation = cfg.get('activation', 'tanh')

    if model_type == "otcfm":
        from experiments.otcfm.otcfm_alae import OTCFMMLP
        model = OTCFMMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    elif model_type == "rectified_flow":
        from experiments.rectified_flow.rectified_flow_alae import LatentRFUNet, TimeConditionedMLP
        # Determine architecture from config or default to LatentRFUNet
        m_name = cfg.get('model', 'LatentRFUNet')
        if m_name == "LatentRFUNet":
            model = LatentRFUNet().to(DEVICE)
        else:
            model = TimeConditionedMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation).to(
                DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def sample_ode(model, z0, num_steps=20):
    batch_size = z0.shape[0]
    dt = 1.0 / num_steps
    z = z0.clone()
    for step in range(num_steps):
        t = torch.ones((batch_size, 1), device=z.device) * (step / num_steps)
        velocity = model(z, t)
        z = z + velocity * dt
    return z


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare DM, DMF, and OFM models for a given seed.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed to use (e.g., 42, 777, 123)")
    parser.add_argument("--test_images", action="store_true",
                        help="Use specific predefined test image indices for testing purposes")
    args = parser.parse_args()

    seed = args.seed
    print(f"--- Comparing models for seed: {seed} ---")

    # Discover paths
    dm_model_path = get_model_path("dm", seed)
    dmf_model_path = get_model_path("dmf", seed)
    ofm_model_path = get_model_path("ofm", seed)
    otcfm_model_path = get_model_path("otcfm", seed)
    rf_model_path = get_model_path("rectified_flow", seed)

    # 1. Load Data
    print("Loading test dataset...")
    try:
        train_size = 60000
        latents = np.load(os.path.join(DATA_PATH, "latents.npy"))
        age = np.load(os.path.join(DATA_PATH, "age.npy"))
        test_inp_images = np.load(os.path.join(DATA_PATH, "test_images.npy"))

        test_latents = latents[train_size:]
        test_age = age[train_size:]

        # Filter for ADULT (age >= 18)
        adult_test_inds = np.where((test_age >= 18).reshape(-1) * (test_age != -1).reshape(-1))[0]

        # Only take those that have corresponding images in test_images.npy
        available_inds = [i for i in adult_test_inds if i < len(test_inp_images)]

        if not available_inds:
            print("No adult samples found in test set.")
            return

        print(f"Found {len(available_inds)} adult samples in test set.")

        if args.test_images:
            selected_inds = [47, 65, 119, 188, 205, 281]
            # Ensure indices are valid
            selected_inds = [i for i in selected_inds if i < len(test_inp_images) and i < len(test_latents)]
        else:
            # Randomly select 5
            selected_inds = np.random.choice(available_inds, size=5, replace=False)

        X_test_selected = torch.tensor(test_latents[selected_inds]).to(DEVICE)
        images_selected = test_inp_images[selected_inds]

    except Exception as e:
        print(f"Data loading failed: {e}")
        sys.exit(1)

    # 2. Load Models
    print("Loading ALAE model...")
    alae_model = load_model("OFM/ALAE/configs/ffhq.yaml", training_artifacts_dir="OFM/ALAE/training_artifacts/ffhq/")
    alae_model.to(DEVICE)
    alae_model.eval()

    print("Loading DM, DMF (friction), OFM, OTCFM, and Rectified Flow models...")
    dm_model = load_dm_model(dm_model_path)
    dmf_model = load_dm_model(dmf_model_path)
    ofm_model = load_ofm_model(ofm_model_path)
    otcfm_model = load_cfm_model(otcfm_model_path, "otcfm")
    rf_model = load_cfm_model(rf_model_path, "rectified_flow")

    # 3. Inference
    print("Running inference...")

    # DM and DMF Mapping needs no grad
    with torch.no_grad():
        dm_mapped = dm_model(X_test_selected)
        dmf_mapped = dmf_model(X_test_selected)

    # OFM Mapping
    ofm_mapped = ofm_model.push_nograd(X_test_selected)

    # OTCFM and RF Mapping (ODE)
    with torch.no_grad():
        otcfm_mapped = sample_ode(otcfm_model, X_test_selected, num_steps=20)
        rf_mapped = sample_ode(rf_model, X_test_selected, num_steps=20)

    with torch.no_grad():
        # Decode all
        decoded_original = decode(alae_model, X_test_selected)
        decoded_dm = decode(alae_model, dm_mapped)
        decoded_dmf = decode(alae_model, dmf_mapped)
        decoded_ofm = decode(alae_model, ofm_mapped)
        decoded_otcfm = decode(alae_model, otcfm_mapped)
        decoded_rf = decode(alae_model, rf_mapped)

        # Post-process to [0, 255] uint8 numpy
        def to_numpy(tensor):
            return ((tensor * 0.5 + 0.5) * 255).clamp(0, 255).cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

        orig_imgs = to_numpy(decoded_original)
        dm_imgs = to_numpy(decoded_dm)
        dmf_imgs = to_numpy(decoded_dmf)
        ofm_imgs = to_numpy(decoded_ofm)
        otcfm_imgs = to_numpy(decoded_otcfm)
        rf_imgs = to_numpy(decoded_rf)

    # 4. Visualization
    output_filename = f'comparison_seed_{seed}.png'
    if args.test_images:
        output_filename = f'comparison_seed_{seed}_test_images.png'
    print(f"Saving comparison to {output_filename}...")
    n = len(selected_inds)
    fig, axes = plt.subplots(n, 7, figsize=(21, 3 * n), dpi=150)

    # Handle the case where n=1 by ensuring axes is 2D
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        # Column 0: Original/Adult
        axes[i, 0].imshow(images_selected[i])
        if i == 0: axes[i, 0].set_title("Adult (Dataset)")
        axes[i, 0].axis('off')

        # Column 1: ALAE Reconstructed
        axes[i, 1].imshow(orig_imgs[i])
        if i == 0: axes[i, 1].set_title("ALAE Reconstructed")
        axes[i, 1].axis('off')

        # Column 2: Rectified Flow Child
        axes[i, 2].imshow(rf_imgs[i])
        if i == 0: axes[i, 2].set_title("Rectified Flow")
        axes[i, 2].axis('off')

        # Column 3: OTCFM Child
        axes[i, 3].imshow(otcfm_imgs[i])
        if i == 0: axes[i, 3].set_title("OT-CFM")
        axes[i, 3].axis('off')

        # Column 4: OFM Child
        axes[i, 4].imshow(ofm_imgs[i])
        if i == 0: axes[i, 4].set_title("OFM")
        axes[i, 4].axis('off')

        # Column 5: DM Child
        axes[i, 5].imshow(dm_imgs[i])
        if i == 0: axes[i, 5].set_title("DM")
        axes[i, 5].axis('off')

        # Column 6: DM friction Child
        axes[i, 6].imshow(dmf_imgs[i])
        if i == 0: axes[i, 6].set_title("DMF (our model)")
        axes[i, 6].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    print("Done!")


if __name__ == "__main__":
    main()
