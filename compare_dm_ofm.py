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
sys.path.append("OFM")
sys.path.append("OFM/ALAE")

# ── Imports ────────────────────────────────────────────────────────────
try:
    from OFM.ALAE.alae_ffhq_inference import load_model, decode
    from OFM.ofmsrc.icnn import DenseICNN
except ImportError:
    print("Error: Required modules not found. Ensure OFM/ALAE and OFM/ofmsrc exist.")
    sys.exit(1)

# ── Parameters ──────────────────────────────────────────────────────────
DEVICE = 'cpu'
DIM = 512
DATA_PATH = "OFM/data"
RUNS_DIR = "experiments/runs"

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

# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare DM, DMF, and OFM models for a given seed.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed to use (e.g., 42, 777, 123)")
    args = parser.parse_args()
    
    seed = args.seed
    print(f"--- Comparing models for seed: {seed} ---")

    # Discover paths
    dm_model_path = get_model_path("dm", seed)
    dmf_model_path = get_model_path("dmf", seed)
    ofm_model_path = get_model_path("ofm", seed)

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

    print("Loading DM, DMF (friction), and OFM models...")
    dm_model = load_dm_model(dm_model_path)
    dmf_model = load_dm_model(dmf_model_path)
    ofm_model = load_ofm_model(ofm_model_path)

    # 3. Inference
    print("Running inference...")
    
    # DM and DMF Mapping needs no grad
    with torch.no_grad():
        dm_mapped = dm_model(X_test_selected)
        dmf_mapped = dmf_model(X_test_selected)
    
    # OFM Mapping
    ofm_mapped = ofm_model.push_nograd(X_test_selected)
    
    with torch.no_grad():
        # Decode all
        decoded_original = decode(alae_model, X_test_selected)
        decoded_dm = decode(alae_model, dm_mapped)
        decoded_dmf = decode(alae_model, dmf_mapped)
        decoded_ofm = decode(alae_model, ofm_mapped)
        
        # Post-process to [0, 255] uint8 numpy
        def to_numpy(tensor):
            return ((tensor * 0.5 + 0.5) * 255).clamp(0, 255).cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
        
        orig_imgs = to_numpy(decoded_original)
        dm_imgs = to_numpy(decoded_dm)
        dmf_imgs = to_numpy(decoded_dmf)
        ofm_imgs = to_numpy(decoded_ofm)

    # 4. Visualization
    output_filename = f'comparison_seed_{seed}.png'
    print(f"Saving comparison to {output_filename}...")
    n = len(selected_inds)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), dpi=150)
    
    for i in range(n):
        # Column 0: Original/Adult
        axes[i, 0].imshow(orig_imgs[i])
        if i == 0: axes[i, 0].set_title("Adult (Dataset)")
        axes[i, 0].axis('off')

        # Column 1: DM Child
        axes[i, 1].imshow(dm_imgs[i])
        if i == 0: axes[i, 1].set_title("DM (Child)")
        axes[i, 1].axis('off')

        # Column 2: DMF Child
        axes[i, 2].imshow(dmf_imgs[i])
        if i == 0: axes[i, 2].set_title("DM Friction (Child)")
        axes[i, 2].axis('off')

        # Column 3: OFM Child
        axes[i, 3].imshow(ofm_imgs[i])
        if i == 0: axes[i, 3].set_title("OFM (Child)")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename)
    print("Done!")

if __name__ == "__main__":
    main()
