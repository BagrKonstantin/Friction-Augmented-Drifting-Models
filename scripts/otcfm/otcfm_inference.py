#!/usr/bin/env python
"""
Inference script for OT-CFM model.
Encodes an input face image via ALAE, transports the latent with the trained flow, decodes back.

Usage:
    # Single image
    python scripts/otcfm/otcfm_inference.py --checkpoint scripts/runs/.../checkpoint_final.pth --input_image face.jpg

    # Directory of images
    python scripts/otcfm/otcfm_inference.py --checkpoint ... --input_dir ./my_faces/

    # Default test data (no --input_image / --input_dir)
    python scripts/otcfm/otcfm_inference.py --checkpoint ...
"""
import os
import argparse
import numpy as np
from glob import glob
from PIL import Image

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

from scripts.exp_utils.exp_utils import get_device, setup_paths
setup_paths()

from scripts.exp_utils.data_loader import download_data, download_alae_model, load_alae_data
from scripts.otcfm.otcfm_alae import OTCFMMLP, OTCFM
from scripts.rectified_flow.rectified_flow_alae import LatentRFUNet, TimeConditionedMLP


# ── Model loading ──────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get('config', {})

    model_name = cfg.get('model_name', 'OTCFMMLP')
    hidden_dim = cfg.get('hidden_dim', 512)
    num_layers = cfg.get('num_layers', 4)
    activation = cfg.get('activation', 'leaky_relu')
    sigma      = cfg.get('sigma', 0.1)

    if model_name == 'OTCFMMLP':
        model = OTCFMMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    elif model_name == 'LatentRFUNet':
        model = LatentRFUNet().to(device)
    elif model_name == 'TimeConditionedMLP':
        model = TimeConditionedMLP(dim=512, hidden_dim=hidden_dim, num_layers=num_layers, activation=activation).to(device)
    else:
        raise ValueError(f"Unknown model_name in checkpoint: {model_name}")

    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  model={model_name}  step={ckpt.get('step','?')}  sigma={sigma}")
    return model, sigma, cfg


# ── ALAE helpers ───────────────────────────────────────────────────────────────

def load_alae():
    try:
        from OFM.ALAE.alae_ffhq_inference import load_model, encode, decode
        alae = load_model(
            "OFM/ALAE/configs/ffhq.yaml",
            training_artifacts_dir="data/alae_training_artifacts/ffhq/"
        )
        alae.eval()
        return alae, encode, decode
    except Exception as e:
        raise RuntimeError(f"Could not load ALAE: {e}")


def encode_images(alae, encode_fn, image_tensors, device, batch_size=8):
    """Encode [N,3,H,W] tensors → [N,512] ALAE latents."""
    latents = []
    alae.eval()
    with torch.no_grad():
        for i in range(0, len(image_tensors), batch_size):
            batch = image_tensors[i:i+batch_size].to(device)
            if batch.shape[0] == 1:          # InstanceNorm needs >1 sample
                batch = batch.repeat(2, 1, 1, 1)
                z = encode_fn(alae, batch)[:1]
            else:
                z = encode_fn(alae, batch)
            if z.dim() == 3:                 # [B, layers, 512] → [B, 512]
                z = z[:, 0, :]
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def decode_latents(alae, decode_fn, latents, device, batch_size=8):
    """Decode [N,512] latents → [N,3,H,W] image tensors."""
    images = []
    with torch.no_grad():
        for i in range(0, len(latents), batch_size):
            batch = latents[i:i+batch_size].to(device)
            images.append(decode_fn(alae, batch).cpu())
    return torch.cat(images, dim=0)


# ── Image I/O ──────────────────────────────────────────────────────────────────

def load_image(path, size=1024):
    img = Image.open(path).convert('RGB')
    tf  = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return tf(img)


def collect_paths(input_image, input_dir):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    if input_image:
        if not os.path.exists(input_image):
            raise FileNotFoundError(input_image)
        return [input_image]
    if input_dir:
        paths = []
        for e in exts:
            paths.extend(glob(os.path.join(input_dir, e)))
        if not paths:
            raise ValueError(f"No images in {input_dir}")
        return sorted(paths)
    return None


def save_comparison(originals, transformed, output_path, num_show=5):
    n = min(num_show, len(originals))
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    if n == 1:
        axes = axes.reshape(1, 2)
    axes[0, 0].set_title("Input",       fontsize=10)
    axes[0, 1].set_title("Transformed", fontsize=10)

    for i in range(n):
        for col, img_t in enumerate([originals[i], transformed[i]]):
            img = img_t.numpy() if isinstance(img_t, torch.Tensor) else img_t
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = np.clip((img + 1) / 2, 0, 1) if img.min() < 0 else np.clip(img, 0, 1)
            axes[i, col].imshow(img)
            axes[i, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison → {output_path}")


# ── Main inference ─────────────────────────────────────────────────────────────

def inference(
    checkpoint_path,
    input_image=None,
    input_dir=None,
    output_dir="./otcfm_inference_results",
    num_ode_steps=100,
    num_images=10,
    input_data="ADULT",
    target_data="CHILDREN",
):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    model, sigma, cfg = load_checkpoint(checkpoint_path, device)
    flow = OTCFM(model, sigma=0.0, device=device)   # σ=0 at inference (deterministic)

    alae, encode_fn, decode_fn = load_alae()
    alae = alae.to(device)

    # ── Source latents ─────────────────────────────────────────────────────
    paths = collect_paths(input_image, input_dir)

    if paths is not None:
        print(f"Loading {len(paths)} image(s)...")
        raw_tensors = []
        for p in tqdm(paths, desc="Reading"):
            raw_tensors.append(load_image(p))
        raw_tensors = torch.stack(raw_tensors)

        latents     = encode_images(alae, encode_fn, raw_tensors, device)
        src_images  = raw_tensors
    else:
        print("Using default test data...")
        DATA_PATH = os.path.join(setup_paths(), "data")
        try:
            download_data(DATA_PATH)
            download_alae_model()
        except Exception as e:
            print(f"Download warning: {e}")
        data        = load_alae_data(DATA_PATH, input_data, target_data, device)
        latents     = data["X_test"][:num_images].cpu()
        src_images  = torch.tensor(data["test_inp_images"][:num_images]).permute(0, 3, 1, 2) \
                      if data["test_inp_images"] is not None else None

    # ── Transport latents ──────────────────────────────────────────────────
    print(f"Running ODE ({num_ode_steps} steps) on {len(latents)} sample(s)...")
    model.eval()
    with torch.no_grad():
        transported = flow.sample_ode(latents.to(device), num_steps=num_ode_steps).cpu()

    np.save(os.path.join(output_dir, "transported_latents.npy"), transported.numpy())

    # ── Decode & save ──────────────────────────────────────────────────────
    print("Decoding latents → images...")
    out_images = decode_latents(alae, decode_fn, transported, device)
    np.save(os.path.join(output_dir, "transported_images.npy"), out_images.numpy())

    if src_images is not None:
        save_comparison(src_images, out_images,
                        os.path.join(output_dir, "comparison.png"))

    print(f"\n✓ Done. Results in: {output_dir}")
    return transported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OT-CFM inference: transform face image(s)")
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--input_image",   type=str, default=None,  help="Path to a single image")
    parser.add_argument("--input_dir",     type=str, default=None,  help="Directory of images")
    parser.add_argument("--output_dir",    type=str, default="./otcfm_inference_results")
    parser.add_argument("--num_ode_steps", type=int, default=100)
    parser.add_argument("--num_images",    type=int, default=10,    help="# test images if no input given")
    parser.add_argument("--input_data",    type=str, default="ADULT")
    parser.add_argument("--target_data",   type=str, default="CHILDREN")
    args = parser.parse_args()

    inference(
        checkpoint_path=args.checkpoint,
        input_image=args.input_image,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_ode_steps=args.num_ode_steps,
        num_images=args.num_images,
        input_data=args.input_data,
        target_data=args.target_data,
    )
