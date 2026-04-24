import torch
import numpy as np
import matplotlib.pyplot as plt
from OFM.ALAE.alae_ffhq_inference import decode

def save_comparison_plot(model, x_test, x_inds_test, test_inp_images, alae_model, device, save_path, title_prefix="Mapped"):
    # model can be MLP or DenseICNN. DenseICNN has push_nograd, MLP just forward.
    is_icnn = hasattr(model, 'push_nograd')
    
    available_test_inds = [i for i, idx in enumerate(x_inds_test) if idx < len(test_inp_images)]
    if len(available_test_inds) == 0:
        print("No test images available for visualization.")
        return
        
    inds_to_map = np.random.choice(available_test_inds, size=min(5, len(available_test_inds)), replace=False)
    latent_to_map = x_test[inds_to_map].to(device)
    
    if hasattr(model, 'eval'):
        model.eval()
    if is_icnn:
        mapped = model.push_nograd(latent_to_map)
    else:
        with torch.no_grad():
            mapped = model(latent_to_map)
            
    with torch.no_grad():
        decoded_input = decode(alae_model, latent_to_map)
        decoded_input = ((decoded_input * 0.5 + 0.5) * 255).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()

        decoded_output = decode(alae_model, mapped)
        decoded_output = ((decoded_output * 0.5 + 0.5) * 255).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy()

    num_samples = len(inds_to_map)
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples), dpi=150)
    if num_samples == 1:
        axes = axes[None, :]
        
    for i in range(num_samples):
        axes[i, 0].imshow(decoded_input[i])
        axes[i, 0].set_title("Input (Adult)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(decoded_output[i])
        axes[i, 1].set_title(f"Output ({title_prefix})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison saved to {save_path}")
