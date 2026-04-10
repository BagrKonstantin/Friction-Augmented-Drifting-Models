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
import argparse

# Set paths for ALAE and OFM imports
sys.path.append("OFM")
sys.path.append("OFM/ALAE")

# OFM Imports
from OFM.ofmsrc.alae_distributions import TensorSampler, NormalSampler
from OFM.ofmsrc.tools import EMA
from OFM.ofmsrc.icnn import DenseICNN
from OFM.ofmsrc.model_tools import ofm_loss, TorchlbfgsInvParams, BruteforceIHVParams
from OFM.ofmsrc.discrete_ot import OTPlanSampler
from OFM.ofmsrc.inception import InceptionV3
from OFM.ofmsrc.fid_score import calculate_frechet_distance
from scipy import linalg

# ALAE Imports
from OFM.ALAE.alae_ffhq_inference import load_model, encode, decode

# ── Parameters ──────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.set_default_device(DEVICE)
print(DEVICE)
DIM = 512
INPUT_DATA = "ADULT"
TARGET_DATA = "CHILDREN"

parser = argparse.ArgumentParser()
parser.add_argument("--max_steps", type=int, default=10002)
args = parser.parse_args()

BATCH_SIZE = 128 # Matching alae_mb128.ipynb
D_LR = 1e-3
MAX_STEPS = args.max_steps
INV_TOLERANCE = 1e-2

OUTPUT_SEED = 0xBADBEEF

def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_all(OUTPUT_SEED)

# ── Data Loading ──────────────────────────────────────────────────────
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

# Try to download if missing
try:
    download_data()
    download_alae_model()
except Exception as e:
    print(f"Download failed: {e}")

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
except FileNotFoundError:
    print("Error: Required .npy files not found in OFM/data.")
    sys.exit(1)

# ── Model Initialisation ──────────────────────────────────────────────
# Using DenseICNN as per the OFM ALAE notebook
D = DenseICNN(DIM, [1024, 1024]).to(DEVICE)
D.convexify() # Initialize ICNN constraints
D_opt = torch.optim.Adam(D.parameters(), lr=D_LR)
ema = EMA(0, betas=[0.999, 0.99])

EXP_NAME = f'ofm_ALAE_{INPUT_DATA}_TO_{TARGET_DATA}'
OUTPUT_PATH = f'./checkpoints/{EXP_NAME}'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── OT Sampling Logic ────────────────────────────────────────────────
ot_plan_sampler = OTPlanSampler('exact')

def sampling_fn(batch_size):
    x_samples = X_sampler.sample(batch_size)
    y_samples = Y_sampler.sample(batch_size)
    # Couple samples using OT
    return ot_plan_sampler.sample_plan(x_samples, y_samples)

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
        D.train()
        X, Y = sampling_fn(BATCH_SIZE)
        t = (torch.rand(BATCH_SIZE) + 1e-8).to(DEVICE)

        # OFM Loss calculation
        current_metrics = {}
        loss, true_loss = ofm_loss(
            D, X, Y, t,
            TorchlbfgsInvParams(lbfgs_params=dict(tolerance_grad=INV_TOLERANCE), max_iter=10),
            BruteforceIHVParams(),
            tol_inverse_border=0.5,
            stats=current_metrics
        )

        D_opt.zero_grad()
        loss.backward()
        D_opt.step()
        D.convexify() # Maintain ICNN convexity
        ema(D)

        tbar.set_postfix(
            loss=loss.item(),
            true_loss=true_loss.item(),
            y_std=Y.std().item()
        )

        if step > 0 and step % 2000 == 0:
            # Save checkpoint
            torch.save(D.state_dict(), os.path.join(OUTPUT_PATH, 'D_latest.pth'))
            # Evaluation
            D.eval()
            with torch.no_grad():
                mapped_test = D.push_nograd(X_test.to(DEVICE))
                gen_fid_act, _ = get_eval_features(mapped_test.cpu(), alae_model, inception_model, clip_model, batch_size=64)
                mu_gen, sigma_gen = get_stats(gen_fid_act)
                fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_target, sigma_target)
                print(f"\nStep {step} | FID: {fid_score:.4f}")
            D.train()

# ── Final Evaluation ───────────────────────────────────────────────────
print("Computing final FID and CMMD on the entire test set...")
D.eval()
with torch.no_grad():
    mapped_test = D.push_nograd(X_test.to(DEVICE))
    gen_fid_act, gen_cmmd_act = get_eval_features(mapped_test.cpu(), alae_model, inception_model, clip_model)

    mu_gen, sigma_gen = get_stats(gen_fid_act)
    fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_target, sigma_target)

    cmmd_score = mmd(target_cmmd_act, gen_cmmd_act).item()

    print(f"\nFinal FID over {len(X_test)} samples = {fid_score:.4f}")
    print(f"Final CMMD over {len(X_test)} samples = {cmmd_score:.4f}")
D.train()

# ── Results Visualization ──────────────────────────────────────────────
print("Generating visualization...")
try:
    available_test_inds = [i for i, idx in enumerate(x_inds_test) if idx < len(test_inp_images)]
    inds_to_map = np.random.choice(available_test_inds, size=5, replace=False)
    latent_to_map = X_test[inds_to_map].to(DEVICE)
    
    # D.push_nograd internally requires gradients to compute the potential gradient
    mapped = D.push_nograd(latent_to_map)
    
    with torch.no_grad():
        decoded_input = decode(alae_model, latent_to_map)
        decoded_input = ((decoded_input * 0.5 + 0.5) * 255).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()

        decoded_output = decode(alae_model, mapped)
        decoded_output = ((decoded_output * 0.5 + 0.5) * 255).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(len(inds_to_map), 2, figsize=(6, 3 * len(inds_to_map)), dpi=150)
    for i in range(len(inds_to_map)):
        axes[i, 0].imshow(decoded_input[i])
        axes[i, 0].set_title("Adult (Original)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(decoded_output[i])
        axes[i, 1].set_title("Kid (OFM Mapped)")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('ofm_transfer_faces.png')
    print("Comparison saved to ofm_transfer_faces.png")
except Exception as e:
    print(f"Visualization failed: {e}")
