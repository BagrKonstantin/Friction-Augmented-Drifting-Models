#!/usr/bin/env python
"""
Inference script for Rectified Flow model
Transforms images using a trained RF checkpoint

Usage:
    # Transform test images
    python rectified_flow_inference.py --checkpoint ./1_rf/checkpoint_final.pth

    # Transform a single image
    python rectified_flow_inference.py --checkpoint ./1_rf/checkpoint_final.pth --input_image face.jpg

    # Transform multiple images from a directory
    python rectified_flow_inference.py --checkpoint ./1_rf/checkpoint_final.pth --input_dir ./my_images/
    
    # Use specific number of ODE steps
    python rectified_flow_inference.py --checkpoint ./1_rf/checkpoint_final.pth --input_image face.png --num_ode_steps 50
"""
import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import torchvision.transforms as transforms

from scripts.exp_utils.exp_utils import setup_paths, get_device
setup_paths()

from scripts.exp_utils.data_loader import download_data, download_alae_model, load_alae_data


class TimeConditionedMLP(nn.Module):
    """Time-conditioned MLP for Rectified Flow velocity field v(z, t)."""
    def __init__(self, dim=512, hidden_dim=1024, num_layers=3, activation='tanh'):
        super().__init__()
        self.dim = dim
        
        layers = []
        layers.append(nn.Linear(dim + 1, hidden_dim))
        
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers.append(act_fn())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
        
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        zt = torch.cat([z, t], dim=1)
        velocity = self.net(zt)
        return velocity


class RectifiedFlow:
    """Rectified Flow model for inference."""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def sample_ode(self, z0, num_steps=100):
        """
        Sample from the learned flow using Euler method.
        
        Args:
            z0: [batch_size, dim] - initial samples from π₀
            num_steps: number of discretization steps
        Returns:
            z1: [batch_size, dim] - transported samples ≈ samples from π₁
        """
        batch_size = z0.shape[0]
        dt = 1.0 / num_steps
        
        z = z0.clone()
        
        for step in range(num_steps):
            t = torch.ones((batch_size, 1), device=z.device) * (step / num_steps)
            velocity = self.model(z, t)
            z = z + velocity * dt
        
        return z
    
    @torch.no_grad()
    def sample_ode_trajectory(self, z0, num_steps=100):
        """
        Sample trajectory from the learned flow.
        
        Returns:
            trajectory: [num_steps+1, batch_size, dim] - full trajectory from z0 to z1
        """
        batch_size = z0.shape[0]
        dt = 1.0 / num_steps
        
        trajectory = [z0.cpu()]
        z = z0.clone()
        
        for step in range(num_steps):
            t = torch.ones((batch_size, 1), device=z.device) * (step / num_steps)
            velocity = self.model(z, t)
            z = z + velocity * dt
            trajectory.append(z.cpu())
        
        return torch.stack(trajectory, dim=0)


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load a trained RF model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 1024)
    num_layers = config.get('num_layers', 3)
    activation = config.get('activation', 'tanh')
    
    print(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}, activation={activation}")
    
    # Create model
    model = TimeConditionedMLP(
        dim=512,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    
    return model, config


def load_alae_encoder_decoder():
    """Load ALAE model for encoding/decoding images."""
    try:
        from baselines.OFM.ALAE.alae_ffhq_inference import load_model, encode, decode
        
        print("Loading ALAE model...")
        alae_model = load_model(
            "OFM/ALAE/configs/ffhq.yaml",
            training_artifacts_dir="data/alae_training_artifacts/ffhq/"
        )
        # CRITICAL: Set model to eval mode to fix InstanceNorm issues with batch_size=1
        alae_model.eval()
        print("ALAE model loaded successfully")
        
        return alae_model, encode, decode
    except Exception as e:
        print(f"Warning: Could not load ALAE model: {e}")
        return None, None, None


def load_image(image_path, target_size=1024):
    """
    Load an image from file and convert to tensor.
    
    Args:
        image_path: Path to image file (jpg, png, etc.)
        target_size: Size to resize image to (default 1024 for ALAE FFHQ model)
    
    Returns:
        image_tensor: [3, H, W] tensor normalized to [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    return transform(img)


def load_images_from_paths(image_paths, target_size=1024):
    """Load multiple images from file paths."""
    images = []
    valid_paths = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img_tensor = load_image(path, target_size)
            images.append(img_tensor)
            valid_paths.append(path)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    if len(images) == 0:
        raise ValueError("No valid images found")
    
    return torch.stack(images), valid_paths


def get_image_paths(input_image=None, input_dir=None):
    """
    Get list of image paths from either single image or directory.
    
    Args:
        input_image: Path to single image file
        input_dir: Path to directory containing images
    
    Returns:
        List of image paths or None
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    if input_image is not None:
        if not os.path.exists(input_image):
            raise FileNotFoundError(f"Image not found: {input_image}")
        return [input_image]
    
    elif input_dir is not None:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Directory not found: {input_dir}")
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(input_dir, ext)))
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in directory: {input_dir}")
        
        return sorted(image_paths)
    
    else:
        return None


def save_image_grid(images, output_path, nrow=8, title="Generated Images"):
    """Save a grid of images."""
    n = len(images)
    ncol = nrow
    nrow = (n + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1 or ncol == 1:
        axes = axes.reshape(nrow, ncol)
    
    for idx in range(nrow * ncol):
        row = idx // ncol
        col = idx % ncol
        ax = axes[row, col]
        
        if idx < n:
            img = images[idx]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Normalize to [0, 1]
            if img.max() > 1.0 or img.min() < 0.0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            # Handle different image formats
            if img.shape[0] == 3:  # CHW format
                img = img.transpose(1, 2, 0)
            
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved image grid to: {output_path}")


def inference(
    checkpoint_path,
    input_image=None,
    input_dir=None,
    output_dir="./rf_inference_results",
    num_ode_steps=100,
    batch_size=32,
    num_images=10,
    save_trajectory=False,
    input_data="ADULT",
    target_data="CHILDREN"
):
    """Run inference using trained RF model."""
    
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, config = load_checkpoint(checkpoint_path, device)
    rectified_flow = RectifiedFlow(model, device=device)
    
    # Load ALAE for image encoding/decoding
    alae_model, encode_fn, decode_fn = load_alae_encoder_decoder()
    
    if alae_model is None:
        raise RuntimeError("ALAE model is required for encoding images. Please ensure it's properly installed.")
    
    # Move ALAE model to device and set to eval mode
    alae_model = alae_model.to(device)
    alae_model.eval()
    # Forcefully set all modules to eval mode
    for module in alae_model.modules():
        module.eval()
    
    # Load input data
    image_paths = get_image_paths(input_image, input_dir)
    
    if image_paths is not None:
        print(f"Loading {len(image_paths)} image(s) from file(s)...")
        
        # Load images as tensors
        image_tensors, valid_paths = load_images_from_paths(image_paths)
        
        # Save original images for comparison
        save_image_grid(
            image_tensors,
            os.path.join(output_dir, "original_images.png"),
            title="Original Images"
        )
        
        # Encode images to latent space using ALAE
        print("Encoding images to latent space...")
        latents = []
        # Force model to eval mode with context manager
        with torch.no_grad():
            # Temporarily store training state and force eval
            was_training = alae_model.training
            alae_model = alae_model.eval()
            
            for i in tqdm(range(0, len(image_tensors), batch_size), desc="Encoding"):
                batch = image_tensors[i:i+batch_size].to(device)
                # Ensure batch has at least 2 samples to avoid instance norm issues
                # If single image, duplicate it temporarily
                if batch.shape[0] == 1:
                    batch = batch.repeat(2, 1, 1, 1)
                    latent_batch = encode_fn(alae_model, batch)
                    latent_batch = latent_batch[:1]  # Take only first
                else:
                    latent_batch = encode_fn(alae_model, batch)
                
                # ALAE encode returns [batch, num_layers, 512], we need [batch, 512]
                # Take only the first layer (they're all the same, just repeated)
                if latent_batch.dim() == 3:
                    latent_batch = latent_batch[:, 0, :]  # [batch, 512]
                
                latents.append(latent_batch.cpu())
            
            # Restore training state
            if was_training:
                alae_model.train()
        
        latents = torch.cat(latents, dim=0)
        test_inp_images = image_tensors
        
        print(f"Encoded {len(latents)} images to latent vectors of shape {latents.shape}")
    else:
        # Load default test data
        print("Loading default test data...")
        DATA_PATH = os.path.join(setup_paths(), "data")
        try:
            download_data(DATA_PATH)
            download_alae_model()
        except Exception as e:
            print(f"Download warning: {e}")
        
        data = load_alae_data(DATA_PATH, input_data, target_data, device)
        latents = data["X_test"][:num_images]
        test_inp_images = data["test_inp_images"][:num_images]
        
        # Save original images if available
        if alae_model is not None and test_inp_images is not None:
            save_image_grid(
                test_inp_images,
                os.path.join(output_dir, "original_images.png"),
                title=f"Original {input_data} Images"
            )
    
    print(f"\nRunning inference on {len(latents)} samples with {num_ode_steps} ODE steps...")
    
    # Process in batches
    all_transformed = []
    
    for i in tqdm(range(0, len(latents), batch_size), desc="Transforming"):
        batch = latents[i:i+batch_size].to(device)
        
        if save_trajectory:
            # Get full trajectory
            trajectory = rectified_flow.sample_ode_trajectory(batch, num_steps=num_ode_steps)
            transformed = trajectory[-1]
            
            # Save trajectory for first sample
            if i == 0 and len(trajectory) > 0:
                traj_path = os.path.join(output_dir, "trajectory.npy")
                np.save(traj_path, trajectory[:, 0].cpu().numpy())
                print(f"Saved trajectory to: {traj_path}")
        else:
            # Just get final output
            transformed = rectified_flow.sample_ode(batch, num_steps=num_ode_steps)
        
        all_transformed.append(transformed.cpu())
    
    all_transformed = torch.cat(all_transformed, dim=0)
    
    # Save latents
    latents_path = os.path.join(output_dir, "transformed_latents.npy")
    np.save(latents_path, all_transformed.numpy())
    print(f"Saved transformed latents to: {latents_path}")
    
    # Decode to images if ALAE is available
    if alae_model is not None and decode_fn is not None:
        print("Decoding transformed latents to images...")
        transformed_images = []
        
        for i in range(0, len(all_transformed), batch_size):
            batch = all_transformed[i:i+batch_size].to(device)
            with torch.no_grad():
                decoded = decode_fn(alae_model, batch)
            transformed_images.append(decoded)
        
        transformed_images = torch.cat(transformed_images, dim=0)
        
        # Save transformed images
        save_image_grid(
            transformed_images,
            os.path.join(output_dir, "transformed_images.png"),
            title=f"Transformed to {target_data} ({num_ode_steps} steps)"
        )
        
        # Save as numpy array
        images_path = os.path.join(output_dir, "transformed_images.npy")
        np.save(images_path, transformed_images.cpu().numpy())
        print(f"Saved transformed images to: {images_path}")
        
        # Create side-by-side comparison if original images available
        if test_inp_images is not None:
            fig, axes = plt.subplots(min(5, len(latents)), 2, figsize=(8, min(5, len(latents)) * 2))
            if len(latents) == 1:
                axes = axes.reshape(1, 2)
            
            for idx in range(min(5, len(latents))):
                # Original
                orig_img = test_inp_images[idx]
                if isinstance(orig_img, torch.Tensor):
                    orig_img = orig_img.cpu().numpy()
                if orig_img.shape[0] == 3:
                    orig_img = orig_img.transpose(1, 2, 0)
                axes[idx, 0].imshow(orig_img)
                axes[idx, 0].set_title("Original" if idx == 0 else "")
                axes[idx, 0].axis('off')
                
                # Transformed
                trans_img = transformed_images[idx].cpu().numpy()
                if trans_img.shape[0] == 3:
                    trans_img = trans_img.transpose(1, 2, 0)
                axes[idx, 1].imshow(trans_img)
                axes[idx, 1].set_title("Transformed" if idx == 0 else "")
                axes[idx, 1].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, "comparison.png")
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison to: {comparison_path}")
    
    print(f"\n✓ Inference complete! Results saved to: {output_dir}")
    return all_transformed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with trained Rectified Flow model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to trained model checkpoint")
    parser.add_argument("--input_image", type=str, default=None,
                        help="Path to single input image (jpg, png, etc.)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to directory containing input images. If neither input_image nor input_dir is provided, uses test data")
    parser.add_argument("--output_dir", type=str, default="./rf_inference_results",
                        help="Output directory for results")
    parser.add_argument("--num_ode_steps", type=int, default=100,
                        help="Number of ODE integration steps (fewer = faster but lower quality)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of test images to process (if using default data)")
    parser.add_argument("--save_trajectory", action="store_true",
                        help="Save the full ODE trajectory for visualization")
    parser.add_argument("--input_data", type=str, default="ADULT",
                        help="Source domain (ADULT or CHILDREN) - used only with default test data")
    parser.add_argument("--target_data", type=str, default="CHILDREN",
                        help="Target domain (ADULT or CHILDREN) - used only with default test data")
    
    args = parser.parse_args()
    
    inference(
        checkpoint_path=args.checkpoint,
        input_image=args.input_image,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_ode_steps=args.num_ode_steps,
        batch_size=args.batch_size,
        num_images=args.num_images,
        save_trajectory=args.save_trajectory,
        input_data=args.input_data,
        target_data=args.target_data
    )
