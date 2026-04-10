#!/usr/bin/env python
# coding: utf-8

import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from OFM.ofmsrc.alae_distributions import TensorSampler, NormalSampler
from OFM.ofmsrc.tools import EMA

# Set paths for ALAE imports (adjust if needed)
sys.path.append("OFM")
sys.path.append("OFM/ALAE")

# Import ALAE inference tools
# Note: These require the ALAE model checkpoints to be present in OFM/ALAE/training_artifacts/ffhq/
from OFM.ALAE.alae_ffhq_inference import load_model, encode, decode
from OFM.ofmsrc.inception import InceptionV3
from OFM.ofmsrc.fid_score import calculate_frechet_distance
from scipy import linalg

# try:
#     from OFM.ALAE.alae_ffhq_inference import load_model, encode, decode
# except ImportError:
#     print("Warning: alae_ffhq_inference not found. Plotting will be disabled.")
#     load_model = None

# ── Parameters ──────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.set_default_device(DEVICE)
DIM = 512
INPUT_DATA = "ADULT"
TARGET_DATA = "CHILDREN"


BATCH_SIZE = 515
D_LR = 5e-4
MAX_STEPS = 20001
r_list = (0.005, 0.01, 0.02)


OUTPUT_SEED = 0xBADBEEF

def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_all(OUTPUT_SEED)

# ── Data Loading ──────────────────────────────────────────────────────
# Path to data relative to this script
DATA_PATH = "OFM/data"

def download_data():
    import gdown
    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    urls = {
        os.path.join(DATA_PATH, "age.npy"): "https://drive.google.com/uc?id=1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
        os.path.join(DATA_PATH, "gender.npy"): "https://drive.google.com/uc?id=1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf",
        os.path.join(DATA_PATH, "latents.npy"): "https://drive.google.com/uc?id=1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
        os.path.join(DATA_PATH, "test_images.npy"): "https://drive.google.com/uc?id=1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i",
    }

    for name, url in urls.items():
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            gdown.download(url, name, quiet=False)

# Try to download if missing
try:
    download_data()
except Exception as e:
    print(f"Data download failed or gdown not installed: {e}")

def download_alae_model():
    import torch.hub
    model_dir = "OFM/ALAE/training_artifacts/ffhq/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_157.pth")
    last_checkpoint_path = os.path.join(model_dir, "last_checkpoint")

    if not os.path.exists(model_path):
        url = "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth"
        print(f"Downloading ALAE model weights to {model_path}...")
        torch.hub.download_url_to_file(url, model_path)

    if not os.path.exists(last_checkpoint_path):
        with open(last_checkpoint_path, "w") as f:
            f.write("training_artifacts/ffhq/model_157.pth")

try:
    download_alae_model()
except Exception as e:
    print(f"ALAE model download failed: {e}")

# Load data
train_size = 60000
test_size = 10000

try:
    latents = np.load(os.path.join(DATA_PATH, "latents.npy"))
    gender = np.load(os.path.join(DATA_PATH, "gender.npy"))
    age = np.load(os.path.join(DATA_PATH, "age.npy"))
    test_inp_images = np.load(os.path.join(DATA_PATH, "test_images.npy"))

    train_latents, test_latents = latents[:train_size], latents[train_size:]
    train_gender, test_gender = gender[:train_size], gender[train_size:]
    train_age, test_age = age[:train_size], age[train_size:]

    def get_indices(data_type, age_arr, gen_arr, size):
        if data_type == "MAN":
            return np.arange(size)[(gen_arr == "male").reshape(-1)]
        elif data_type == "WOMAN":
            return np.arange(size)[(gen_arr == "female").reshape(-1)]
        elif data_type == "ADULT":
            return np.arange(size)[(age_arr >= 18).reshape(-1) * (age_arr != -1).reshape(-1)]
        elif data_type == "CHILDREN":
            return np.arange(size)[(age_arr < 18).reshape(-1) * (age_arr != -1).reshape(-1)]
        return np.arange(size)

    x_inds_train = get_indices(INPUT_DATA, train_age, train_gender, train_size)
    x_inds_test = get_indices(INPUT_DATA, test_age, test_gender, test_size)
    y_inds_train = get_indices(TARGET_DATA, train_age, train_gender, train_size)
    y_inds_test = get_indices(TARGET_DATA, test_age, test_gender, test_size)

    X_train = torch.tensor(train_latents[x_inds_train])
    Y_train = torch.tensor(train_latents[y_inds_train])

    X_test = torch.tensor(test_latents[x_inds_test])
    Y_test = torch.tensor(test_latents[y_inds_test])

    X_sampler = TensorSampler(X_train, device=DEVICE)
    Y_sampler = TensorSampler(Y_train, device=DEVICE)
    print(f"Loaded {len(X_train)} samples for {INPUT_DATA} and {len(Y_train)} samples for {TARGET_DATA}")

    print(f"Latent stats:")
    print(f"  {INPUT_DATA}  - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    print(f"  {TARGET_DATA} - Mean: {Y_train.mean():.4f}, Std: {Y_train.std():.4f}")
except FileNotFoundError:
    print("Error: Required .npy files not found in OFM/data. Please download them first.")
    sys.exit(1)

# ── Drifting Model Logic ───────────────────────────────────────────────

def cdist_pytorch(x, y, eps=1e-8):
    # [B, N, D] x [B, M, D] -> [B, N, M]
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))

def drift_loss_pytorch(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
):
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = torch.zeros((B, 0, S), device=gen.device, dtype=gen.dtype)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = torch.ones((B, C_g), device=gen.device, dtype=gen.dtype)
    if weight_pos is None:
        weight_pos = torch.ones((B, C_p), device=gen.device, dtype=gen.dtype)
    if weight_neg is None:
        weight_neg = torch.ones((B, C_n), device=gen.device, dtype=gen.dtype)

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    with torch.no_grad():
        dist = cdist_pytorch(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean(dim=(1, 2)) / targets_w.mean(dim=1) # [B]

        info = {"scale": scale.mean()}

        scale_inputs = torch.clamp(scale / np.sqrt(S), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs[:, None, None]
        targets_scaled = targets / scale_inputs[:, None, None]

        dist_normed = dist / torch.clamp(scale[:, None, None], min=1e-3)

        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))
        block_mask = block_mask.unsqueeze(0)
        dist_normed = dist_normed + block_mask * mask_val

        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R
            affinity = F.softmax(logits, dim=-1)
            aff_transpose = F.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6))
            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = torch.sum(aff_pos, dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_pos_for_ref = torch.sum(aff_pos, dim=-1, keepdim=True) # Unused but keeping structure

            sum_neg = torch.sum(aff_neg, dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)
            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean(dim=(1, 2))

            info[f"loss_{R}"] = f_norm_val.mean()

            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale[:, None, None]

        info["force_norm"] = torch.sqrt((force_across_R**2).sum(dim=-1)).mean()

        goal_scaled = old_gen_scaled + force_across_R

    gen_scaled = gen / scale_inputs[:, None, None]
    diff = gen_scaled - goal_scaled
    loss = torch.mean(diff ** 2)

    return loss, info

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

# ── Model Initialisation ──────────────────────────────────────────────

model = MLP(dim=DIM, dimh=1024, num_hidden_layers=3).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=D_LR)
ema = EMA(0, betas=[0.999, 0.99])

EXP_NAME = f'dm_ALAE_{INPUT_DATA}_TO_{TARGET_DATA}'
OUTPUT_PATH = f'./checkpoints/{EXP_NAME}'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── Eval Metrics Setup (FID & CMMD) ────────────────────────────────────────

print("Setting up Evaluation models (FID & CMMD)...")
alae_model = load_model("OFM/ALAE/configs/ffhq.yaml", training_artifacts_dir="OFM/ALAE/training_artifacts/ffhq/").to(DEVICE)
inception_model = InceptionV3([3]).to(DEVICE)
inception_model.eval()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmmd_pytorch/src"))
from cmmd_pytorch.embedding import ClipEmbeddingModel
from cmmd_pytorch.distance import mmd

clip_model = ClipEmbeddingModel()

def get_eval_features(latents, alae, inception, clip_model, batch_size=32):
    fid_act = []
    cmmd_act = []
    with torch.no_grad():
        for i in tqdm(range(0, len(latents), batch_size), desc="Extracting features"):
            batch = latents[i:i+batch_size].to(DEVICE)

            # ALAE Decode (to [-1, 1])
            imgs = decode(alae, batch)
            # Normalize to [0, 1]
            imgs_01 = (imgs * 0.5 + 0.5).clamp(0, 1)

            # --- FID features ---
            feat_fid = inception(imgs_01)[0]
            if feat_fid.size(2) != 1 or feat_fid.size(3) != 1:
                feat_fid = F.adaptive_avg_pool2d(feat_fid, output_size=(1, 1))
            fid_act.append(feat_fid.view(feat_fid.size(0), -1).cpu().numpy())

            # --- CMMD features ---
            imgs_nhwc = imgs_01.permute(0, 2, 3, 1).cpu().numpy()
            feat_cmmd = clip_model.embed(imgs_nhwc).numpy()
            cmmd_act.append(feat_cmmd)

    return np.concatenate(fid_act, axis=0), np.concatenate(cmmd_act, axis=0)

def get_stats(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

# Precompute target stats
print(f"Precomputing FID & CMMD stats for {TARGET_DATA} on the entire test set...")
target_fid_act, target_cmmd_act = get_eval_features(Y_test, alae_model, inception_model, clip_model)
mu_target, sigma_target = get_stats(target_fid_act)
print(f"Target stats computed on {len(target_fid_act)} samples.")

# ── Training Loop ──────────────────────────────────────────────────────

print(f"Starting training: {EXP_NAME}")
with tqdm(range(MAX_STEPS)) as tbar:
    for step in tbar:
        model.train()
        X, Y = X_sampler.sample(BATCH_SIZE), Y_sampler.sample(BATCH_SIZE)

        # Generator samples
        Y_gen = model(X)

        # Compute drift loss
        loss, info = drift_loss_pytorch(
            Y_gen.unsqueeze(0),
            Y.unsqueeze(0),
            R_list=r_list
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

        if step % 5000 == 0:
            torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'model.pth'))

# ── Final Evaluation ───────────────────────────────────────────────────
print("Computing final FID and CMMD on the entire test set...")
model.eval()
with torch.no_grad():
    mapped_test = model(X_test.to(DEVICE))
    gen_fid_act, gen_cmmd_act = get_eval_features(mapped_test, alae_model, inception_model, clip_model)

    mu_gen, sigma_gen = get_stats(gen_fid_act)
    fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_target, sigma_target)

    cmmd_score = mmd(target_cmmd_act, gen_cmmd_act).item()

    print(f"\nFinal FID over {len(X_test)} samples = {fid_score:.4f}")
    print(f"Final CMMD over {len(X_test)} samples = {cmmd_score:.4f}")
model.train()

# ── Results Visualization ──────────────────────────────────────────────

if load_model is not None:
    print("Generating visual results...")
    try:
        # alae_model already loaded
        model.eval()

        # Select some test samples that have images
        available_test_inds = [i for i, idx in enumerate(x_inds_test) if idx < len(test_inp_images)]
        inds_to_map = np.random.choice(available_test_inds, size=5, replace=False)
        latent_to_map = X_test[inds_to_map].to(DEVICE)
        inp_images = test_inp_images[x_inds_test[inds_to_map]]

        with torch.no_grad():
            mapped = model(latent_to_map)

            # Use EMA model if available for better results
            # mapped = ema.get_model(0.999)(latent_to_map)

            decoded_input = decode(alae_model, latent_to_map)
            decoded_input = ((decoded_input * 0.5 + 0.5) * 255).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()

            decoded_output = decode(alae_model, mapped)
            decoded_output = ((decoded_output * 0.5 + 0.5) * 255).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()

        fig, axes = plt.subplots(len(inds_to_map), 2, figsize=(4, 2 * len(inds_to_map)), dpi=150)
        for i in range(len(inds_to_map)):
            axes[i, 0].imshow(decoded_input[i])
            axes[i, 0].set_title("Adult (Input)")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(decoded_output[i])
            axes[i, 1].set_title("Kid (Mapped)")
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig('dm_transfer_faces.png')
        print("Comparison saved to dm_transfer_faces.png")
    except Exception as e:
        print(f"Visual results generation failed: {e}")
else:
    print("Skipping visual results because ALAE model tools are not available.")
