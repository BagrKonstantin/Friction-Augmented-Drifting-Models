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
    # The original encode in alae_ffhq_inference.py is a bit specialized, let's redefine it here correctly for 1 image
    from OFM.ALAE.model import Model
except ImportError:
    print("Error: ALAE modules not found. Ensure OFM/ALAE exists and sys.path is correct.")
    sys.exit(1)

# ── Parameters ──────────────────────────────────────────────────────────
DEVICE = 'cpu'
DIM = 512
EXP_NAME = 'dm_ALAE_ADULT_TO_CHILDREN'
DM_MODEL_PATH = f'./checkpoints/{EXP_NAME}/model.pth'

# ── MLP Model (ResNet style) ──────────────────────────────────────────
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
    model = MLP(dim=DIM, dimh=1024, num_hidden_layers=3).to(DEVICE)
    if os.path.exists(path):
        print(f"Loading Drifting Model weights from {path}...")
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        print(f"Error: Trained model not found at {path}")
        sys.exit(1)
    model.eval()
    return model

# ── Image to Latent ──────────────────────────────────────────────────
def encode_single_image(model, img_path):
    # Load and preprocess
    img = Image.open(img_path).convert('RGB')
    img = img.resize((1024, 1024), Image.LANCZOS)
    img_np = np.asarray(img).astype(np.float32) / 127.5 - 1.0 # [-1, 1]
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # ALAE encode
    with torch.no_grad():
        # model.encode returns (Z, _) where Z is the compressed latent (W)
        # Note: alae_ffhq_inference used layer_count=9 for 1024x1024
        Z, _ = model.encode(img_tensor, 8, 1) # 8 = log2(1024) - 2
        
        # In ALAE, the same Z is repeated for across layers (StyleSpace)
        # Check mapping_f to see num_layers
        Z = Z.repeat(1, model.mapping_f.num_layers, 1)
    return Z

# ── Process Single Image ─────────────────────────────────────────────
def process_image(img_path, output_path, alae_model, dm_model):
    print(f"Processing {img_path}...")
    try:
        # 1. Encode to latent W
        latent_w = encode_single_image(alae_model, img_path)
        
        # 2. Extract 512D latent (StyleSpace contains 18 layers of same 512D if just encoded)
        latent_512 = latent_w[:, 0, :]
        
        with torch.no_grad():
            # 3. Map through Drifting Model
            mapped_512 = dm_model(latent_512)
            
            # Decode to image
            # Note: alae_ffhq_inference.decode handles expansion if given [B, 512]
            output_tensor = decode(alae_model, mapped_512)
            
            # 5. Save output
            output_image = ((output_tensor[0] * 0.5 + 0.5) * 255).clamp(0, 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            Image.fromarray(output_image).save(output_path)
            print(f"  Successfully saved to {output_path}")

    except Exception as e:
        print(f"  Failed to process {img_path}: {e}")

# ── Main ─────────────────────────────────────────────────────────────
def main():
    # parser = argparse.ArgumentParser(description="Batch map Adult faces to Kid faces using Drifting Model")
    # parser.add_argument("--input_dir", default="pics_in", help="Directory containing input adult images")
    # parser.add_argument("--output_dir", default="child_pics", help="Directory to save output kid images")
    # args = parser.parse_args()

    # 1. Load ALAE model
    print("Loading ALAE model...")
    alae_model = load_model("OFM/ALAE/configs/ffhq.yaml", training_artifacts_dir="OFM/ALAE/training_artifacts/ffhq/")
    alae_model.to(DEVICE)
    alae_model.eval()

    # 2. Load DM model
    dm_model = load_dm_model(DM_MODEL_PATH)

    # 3. Setup Directories
    if not os.path.isdir("pics_in"):
        sys.exit(1)
    os.makedirs("pics_out", exist_ok=True)

    # 4. Batch Process
    files = [f for f in os.listdir("pics_in") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return

    print(f"Found {len(files)} images. Starting conversion...")
    for filename in files:
        input_path = os.path.join("pics_in", filename)
        # Prepend 'child_' to output filename
        output_name = "child_" + os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join("pics_out", output_name)
        
        process_image(input_path, output_path, alae_model, dm_model)
    
    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()
