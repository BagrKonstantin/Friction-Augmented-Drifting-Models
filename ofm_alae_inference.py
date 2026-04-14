#!/usr/bin/env python
import os, sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import argparse

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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIM = 512
EXP_NAME = 'ofm_ALAE_ADULT_TO_CHILDREN'
OFM_MODEL_DIR = f'./checkpoints/{EXP_NAME}'

OFM_MODEL_PATH = os.path.join(OFM_MODEL_DIR, 'D_latest.pth')

# ── Load OFM Model (DenseICNN) ──────────────────────────────────────────
def load_ofm_model(path):
    # Using the same architecture as in ofm_alae.py
    model = DenseICNN(DIM, [1024, 1024]).to(DEVICE)
    if os.path.exists(path):
        print(f"Loading OFM Model weights from {path}...")
        # map_location handles loading onto CPU if GPU is not available
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        print(f"Error: Trained model not found at {path}")
        print("Please ensure you have run ofm_alae.py to train the model first.")
        sys.exit(1)
    model.eval()
    return model

# ── Image to Latent (W) ────────────────────────────────────────────────
def encode_single_image(alae_model, img_path):
    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1024, 1024), Image.LANCZOS)
    img_np = np.asarray(img).astype(np.float32) / 127.5 - 1.0 # [-1, 1]
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # ALAE encode
    with torch.no_grad():
        # model.encode returns (Z, _) where Z is the compressed latent (W)
        # Note: alae_ffhq_inference used layer_count=8 for 1024x1024 (log2(1024)-2)
        Z, _ = alae_model.encode(img_tensor, 8, 1)
        
        # In ALAE, the same Z is repeated across layers for decoding
        # But for the OFM model, we just need the single 512D latent Z
    return Z

# ── Process Single Image ─────────────────────────────────────────────
def process_image(img_path, output_path, alae_model, ofm_model):
    print(f"Processing {img_path}...")
    try:
        # 1. Encode to latent W (1, 512)
        latent_512 = encode_single_image(alae_model, img_path)
        
        # 2. Map through OFM Model
        # push_nograd internally requires gradients to compute the potential gradient
        # but we don't update weights here
        mapped_512 = ofm_model.push_nograd(latent_512)
            
        with torch.no_grad():
            # 3. Decode to image
            # alae_ffhq_inference.decode handles expansion if given [B, 512]
            output_tensor = decode(alae_model, mapped_512)
            
            # 4. Save output
            output_image = ((output_tensor[0] * 0.5 + 0.5) * 255).clamp(0, 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            Image.fromarray(output_image).save(output_path)
            print(f"  Successfully saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {img_path}: {e}")
        import traceback
        traceback.print_exc()

# ── Main ─────────────────────────────────────────────────────────────
def main():
    # 1. Load ALAE model
    print("Loading ALAE model...")
    # Load with training_artifacts_dir to find the checkpoint
    alae_model = load_model("OFM/ALAE/configs/ffhq.yaml", training_artifacts_dir="OFM/ALAE/training_artifacts/ffhq/")
    alae_model.to(DEVICE)
    alae_model.eval()

    # 2. Load OFM model
    ofm_model = load_ofm_model(OFM_MODEL_PATH)

    # 3. Setup Directories
    input_dir = "pics_in"
    output_dir = "pics_out"
    
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' not found.")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    # 4. Batch Process
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(files)} images. Starting conversion and mapping...")
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        # Prepend 'ofm_child_' to output filename
        output_name = "ofm_child_" + os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_name)
        
        process_image(input_path, output_path, alae_model, ofm_model)
    
    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()
