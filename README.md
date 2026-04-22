# Flow Matching Experiments

A comprehensive research repository for comparing different Flow Matching methods for generative modeling, with a focus on image-to-image translation tasks using ALAE (Adversarial Latent AutoEncoder) latent spaces.

## Overview

This repository implements and compares three state-of-the-art generative modeling approaches:

- **OFM (Optimal Flow Matching)**: Learns straight trajectories in a single Flow Matching step by solving optimal transport
- **DM (Diffusion Models / Drifting)**: Standard diffusion-based generative modeling approach
- **DMF (Diffusion Models with Friction)**: Enhanced diffusion models with friction dynamics

All methods are evaluated on face attribute transformation tasks (e.g., Adult → Children, Man → Woman) using FFHQ face dataset latent representations.
## Recent Updates (April 21, 2026)

The repository was recently refactored to provide a modular, production-ready codebase:

- ✅ **Decomposed implementations**: Separated OFM, DM, and DMF into independent, maintainable modules
- ✅ **Deterministic experiments**: Added full reproducibility with seed management
- ✅ **Wall-clock timing**: Built-in performance benchmarking
- ✅ **Checkpoint support**: Automatic model checkpointing during training
- ✅ **Unified evaluation**: Consistent FID and CMMD metrics across all methods
- ✅ **Multi-seed runner**: Easy batch experimentation across random seeds

## Repository Structure

```
FlowMatching/
├── experiments/              # Modular experiment framework (NEW)
│   ├── dm_alae.py           # Diffusion Model implementation
│   ├── dmf_alae.py          # Diffusion Model with Friction
│   ├── ofm_alae.py          # Optimal Flow Matching implementation
│   ├── data_loader.py       # Data loading utilities
│   ├── eval_metrics.py      # FID & CMMD evaluation
│   ├── models_and_losses.py # Neural network architectures and loss functions
│   ├── exp_utils.py         # Experiment utilities (seeding, paths, configs)
│   ├── visualization.py     # Plotting and visualization tools
│   └── run_multiple_seeds.py # Multi-seed experiment runner
│
├── CFM/                     # TorchCFM library (Conditional Flow Matching)
├── OFM/                     # Optimal Flow Matching (NeurIPS 2024)
├── DM/                      # Drifting/Diffusion Models (JAX implementation)
├── cmmd_pytorch/            # CMMD metric implementation
│
├── checkpoints/             # Saved model checkpoints
├── models/                  # Trained models
├── pics_in/                 # Input images for inference
├── pics_out/                # Generated output images
│
├── *.py                     # Legacy standalone scripts
├── *.ipynb                  # Jupyter notebooks for exploration
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FlowMatching
```

2. Create and activate a virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required data and models (automatic on first run):
   - FFHQ face latents and metadata
   - Pre-trained ALAE model weights
   - The download happens automatically when you first run any training script

## Quick Start

### Training a Single Model

Train Optimal Flow Matching:
```bash
python experiments/ofm_alae.py --max_steps 10000 --seed 42
```

Train Diffusion Model:
```bash
python experiments/dm_alae.py --max_steps 10000 --seed 42
```

Train Diffusion Model with Friction:
```bash
python experiments/dmf_alae.py --max_steps 10000 --seed 42
```

### Running Multi-Seed Experiments

To run all three methods across multiple random seeds:

```bash
python experiments/run_multiple_seeds.py --max_steps 10000
```

This will train OFM, DM, and DMF with seeds [42, 123, 777] and save results to `experiments/runs/`.

### Disable Metrics (Faster Training)

To skip FID/CMMD evaluation during training:

```bash
python experiments/ofm_alae.py --no_metrics
# or
python experiments/run_multiple_seeds.py --no_metrics
```

## Usage Details

### Command-Line Arguments

Each training script supports the following arguments:

**Common Arguments:**
- `--max_steps`: Maximum training steps (default: 10001)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--seed`: Random seed for reproducibility
- `--no_metrics`: Disable FID/CMMD evaluation
- `--input_data`: Source distribution (ADULT, CHILDREN, MAN, WOMAN)
- `--target_data`: Target distribution (ADULT, CHILDREN, MAN, WOMAN)

**OFM-specific:**
- `--inv_tolerance`: Inverse tolerance for ICNN optimization (default: 1e-2)

**DM/DMF-specific:**
- `--r_list`: List of noise scales for drift loss (default: [0.005, 0.01, 0.02])
- `--friction_on`: Enable/disable friction (DMF only)

### Output Structure

Each training run creates a timestamped directory in `experiments/runs/`:

```
experiments/runs/21-04-2026_ofm_42/
├── config.yaml              # Training configuration
├── metrics.json             # Final FID/CMMD scores
├── model_final.pth          # Final model checkpoint
├── comparison_plot.png      # Visual comparison of results
└── training.log             # Training progress log
```

## Data and Models

### ALAE Model

The repository uses a pre-trained ALAE (Adversarial Latent AutoEncoder) model trained on FFHQ faces:
- **Resolution**: 1024×1024
- **Latent dimension**: 512
- **Dataset**: FFHQ (Flickr-Faces-HQ)

The model is automatically downloaded on first run from the official weights repository.

### Face Attribute Datasets

The data loader supports filtering faces by:
- **Age**: ADULT (≥18), CHILDREN (<18)
- **Gender**: MAN, WOMAN

Data includes:
- 60,000 training latents
- 10,000 test latents
- Age and gender metadata

## Evaluation Metrics

### FID (Fréchet Inception Distance)

Measures the quality and diversity of generated images using InceptionV3 features.
- **Lower is better**
- Standard metric for generative models

### CMMD (CLIP Maximum Mean Discrepancy)

Measures semantic similarity between generated and target distributions using CLIP embeddings.
- **Lower is better**
- More aligned with human perception than FID

Both metrics are computed automatically during training (unless `--no_metrics` is specified).

## Methods Comparison

| Method | Training Complexity | Inference Speed | Trajectory Quality | OT Solution |
|--------|-------------------|-----------------|-------------------|-------------|
| **OFM** | High (ICNN optimization) | Fast (1 step) | Straight paths | ✅ Yes |
| **DM** | Medium | Medium (multiple steps) | Curved paths | ❌ No |
| **DMF** | Medium | Medium (multiple steps) | Improved smoothness | ❌ No |

## Legacy Scripts

The repository also contains legacy standalone scripts for backward compatibility:

- `ofm_alae.py`, `dm_alae.py`: Original monolithic implementations
- `*_gaussian_mixture.py`: 2D Gaussian mixture experiments
- `*_inference.py`: Inference-only scripts
- `compare_dm_ofm.py`: Side-by-side comparison tool

These scripts are superseded by the modular `experiments/` framework but remain for reference.

## Notebooks

Interactive Jupyter notebooks for exploration and visualization:

- `tutorial_training_8_gaussians_to_moons.ipynb`: Tutorial on 2D toy problems
- `dm_alae.ipynb`: Interactive DM experiments
- `dm_gaussian_mixture*.ipynb`: 2D Gaussian mixture visualizations
- `probmodels-flowmatching.ipynb`: Flow matching theory and examples
- `main.ipynb`: Main experimental notebook

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python experiments/ofm_alae.py --batch_size 64
```

### Slow FID/CMMD Computation

Disable metrics during training:
```bash
python experiments/ofm_alae.py --no_metrics
```

### Data Download Issues

Manually download data files if automatic download fails:
- ALAE weights: https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth
- Face latents: Available via Google Drive (links in `data_loader.py`)

## References

### Optimal Flow Matching (OFM)
```bibtex
@inproceedings{kornilov2024optimal,
  title={Optimal Flow Matching: Learning Straight Trajectories in Just One Step},
  author={Kornilov, Nikita and Mokrov, Petr and Gasnikov, Alexander and Korotin, Alexander},
  booktitle={NeurIPS},
  year={2024}
}
```

### Conditional Flow Matching (CFM)
```bibtex
@inproceedings{tong2023improving,
  title={Improving and generalizing flow-based generative models with minibatch optimal transport},
  author={Tong, Alexander and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and Bengio, Yoshua and others},
  booktitle={ICLR},
  year={2024}
}
```

### Drifting Models (DM)
```bibtex
@article{lambert2025generative,
  title={Generative Modeling via Drifting},
  author={Lambert, Alex and others},
  journal={arXiv preprint arXiv:2602.04770},
  year={2025}
}
```

### ALAE
```bibtex
@inproceedings{pidhorskyi2020adversarial,
  title={Adversarial Latent Autoencoders},
  author={Pidhorskyi, Stanislav and Adjeroh, Donald A and Doretto, Gianfranco},
  booktitle={CVPR},
  year={2020}
}
```

## Contributing

This repository is actively maintained. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project incorporates code from multiple sources with different licenses:
- CFM/: MIT License
- OFM/: Check individual repository
- DM/: Check individual repository

Please refer to individual subdirectories for specific license information.

## Citation

If you use this codebase in your research, please cite the relevant papers above.

## Contact

For questions or issues, please open a GitHub issue or contact the repository maintainers.

---

**Last Updated**: April 21, 2026  
**Status**: ✅ Active Development


## Authors

**University of Luxembourg**
- Arkadii Kazanskii
- Tatiana Petrova
- Konstantin Bagrianskii
- Aleksandr Puzikov
- Radu State
