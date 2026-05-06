# Attraction, Repulsion, and Friction: Introducing DMF

> **Paper**: [arXiv:2604.18194](https://arxiv.org/abs/2604.18194)  
> **University of Luxembourg** — Arkadii Kazanskii, Tatiana Petrova, Konstantin Bagrianskii, Aleksandr Puzikov, Radu State

---

A research codebase for **image-to-image translation in ALAE latent space**, comparing five generative modeling methods on face attribute transformations (Adult ↔ Children, Man ↔ Woman) using the FFHQ dataset.

## Methods

| Method | Description | Training | Inference | OT coupling |
|--------|-------------|----------|-----------|-------------|
| **DMF** | Drifting Model with Friction — our proposed method | Low | Fast | — |
| **DM** | Baseline Drifting Model (Lambert et al., 2025) | Low | Fast | — |
| **OFM** | Optimal Flow Matching via ICNN (Kornilov et al., 2024) | High | 1-step | ✅ Global |
| **OT-CFM** | Conditional Flow Matching with mini-batch OT (Tong et al., 2024) | Medium | ODE | ✅ Mini-batch |
| **RF** | Rectified Flow with optional reflow (Liu et al., 2023) | Medium | ODE / 1-step | — |

## Repository Structure

```
FlowMatching/
├── experiments/              # Main experiment framework — all 5 methods
│   ├── otcfm/                # OT-CFM training, inference, multi-seed runner
│   ├── rectified_flow/       # Rectified Flow (1-RF, 2-RF) training & inference
│   ├── dm_alae.py            # Drifting Model
│   ├── dmf_alae.py           # Drifting Model with Friction (ours)
│   ├── ofm_alae.py           # Optimal Flow Matching
│   ├── data_loader.py        # FFHQ data loading & automatic download
│   ├── eval_metrics.py       # FID & CMMD evaluation
│   ├── models_and_losses.py  # MLP architectures and drift loss
│   ├── exp_utils.py          # Seeding, path setup, config & metrics I/O
│   ├── visualization.py      # Face comparison plots
│   └── run_multiple_seeds.py # Multi-seed batch runner (DM + DMF)
│
├── OFM/                      # Optimal Flow Matching (NeurIPS 2024)
│   └── ALAE/                 # Pre-trained ALAE weights & decoder
├── CFM/                      # TorchCFM library (MIT)
├── RectifiedFlow/            # Official RF implementation (ICLR 2023)
├── DM/                       # Drifting Models reference (JAX)
├── cmmd_pytorch/             # CLIP-MMD metric
│
├── checkpoints/              # Saved model checkpoints
├── pics_in/ pics_out/        # Inference image I/O
├── *.ipynb                   # Jupyter notebooks (toy problems, tutorials)
└── requirements.txt
```

## Installation

**Requirements**: Python 3.12, CUDA GPU (recommended), 16 GB+ RAM.

```bash
git clone <repository-url>
cd FlowMatching
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data and model weights are **downloaded automatically** on first run (FFHQ latents + ALAE weights).

## Quick Start

All scripts are run from the repository root. Each creates a timestamped directory under `experiments/runs/` with a config YAML, metrics JSON, checkpoint, and a face-comparison plot.

### Train a single model

```bash
# Drifting Model with Friction (DMF — ours)
python scripts/dmf_alae.py --max_steps 10000 --seed 42

# Drifting Model (baseline)
python scripts/dm_alae.py --max_steps 10000 --seed 42

# Optimal Flow Matching
python scripts/ofm_alae.py --max_steps 10000 --seed 42

# OT-CFM
python scripts/otcfm/otcfm_alae.py --max_steps 10000 --seed 42

# Rectified Flow (1-RF)
python scripts/rectified_flow/rectified_flow_alae.py --max_steps 10000 --seed 42

# Rectified Flow (2-RF — reflow pass)
python scripts/rectified_flow/rectified_flow_alae.py --max_steps 10000 --seed 42 --reflow
```

### Multi-seed batch runs

```bash
# DM + DMF across seeds [42, 123, 777, 1234, 2024]
python scripts/run_multiple_seeds.py --max_steps 30000

# OT-CFM across multiple seeds
python scripts/otcfm/run_multiple_seeds_otcfm.py --max_steps 30000

# Rectified Flow across multiple seeds
python scripts/rectified_flow/run_multiple_seeds_rf.py --max_steps 30000
```

### Face attribute tasks

Use `--input_data` / `--target_data` to select a translation task:

```bash
python scripts/dmf_alae.py --input_data ADULT --target_data CHILDREN --seed 42
python scripts/dmf_alae.py --input_data MAN   --target_data WOMAN    --seed 42
```

Supported values: `ADULT`, `CHILDREN`, `MAN`, `WOMAN`.

### Inference on custom images

```bash
python scripts/otcfm/otcfm_inference.py       --checkpoint scripts/runs/<run_dir>/checkpoint_final.pth
python scripts/rectified_flow/rectified_flow_inference.py --checkpoint scripts/runs/<run_dir>/checkpoint_final.pth
```

## Command-Line Arguments

All training scripts share these common flags:

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_steps` | 10001 | Training iterations |
| `--batch_size` | method-dependent | Mini-batch size |
| `--lr` | method-dependent | Learning rate |
| `--seed` | 42 | Random seed |
| `--input_data` | ADULT | Source distribution |
| `--target_data` | CHILDREN | Target distribution |
| `--no_metrics` | False | Skip FID/CMMD (faster) |

**DMF-specific**: `--friction_on`, `--r_list`  
**OFM-specific**: `--inv_tolerance` (ICNN solver tolerance)

## Output Structure

```
experiments/runs/DD-MM-YYYY-HH-MM_<method>_<seed>/
├── config.yaml              # Full training configuration
├── metrics.json             # FID, CMMD, training_time_seconds
├── checkpoint_final.pth     # Final model weights
├── checkpoint.pth           # Mid-training checkpoint (step 2000)
└── *_transfer_faces.png     # Source → generated face comparison
```

## Data

The FFHQ dataset is used via **512-dimensional ALAE latent codes** (not raw pixels), which dramatically reduces training cost while preserving perceptual quality.

| Split | Size | Content |
|-------|------|---------|
| Train | 60,000 | Latents + age/gender metadata |
| Test  | 10,000 | Latents + reference RGB images |

Attribute filters: **ADULT** (age ≥ 18), **CHILDREN** (age < 18), **MAN** (male), **WOMAN** (female).

The ALAE decoder (1024×1024) reconstructs images from latents for FID/CMMD evaluation and visualization.

## Evaluation Metrics

**FID** (Fréchet Inception Distance) — InceptionV3 feature space, lower is better.  
**CMMD** (CLIP Maximum Mean Discrepancy) — CLIP embedding space, lower is better, more aligned with human perception.

Both are computed every 2000 steps and at the end of training. Use `--no_metrics` to disable during development.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `tutorial_training_8_gaussians_to_moons.ipynb` | 2D toy problem: 8 Gaussians → moons |
| `dm_gaussian_mixture_new.ipynb` | Drifting Model on 2D Gaussian mixtures |
| `probmodels-flowmatching.ipynb` | Flow matching theory walkthrough |
| `main.ipynb` | Main experimental notebook |

## References

```bibtex
@article{kazanskii2026dmf,
  title   = {Attraction, Repulsion, and Friction: Introducing DMF, a Friction-Augmented Drifting Model},
  author  = {Kazanskii, Arkadii and Petrova, Tatiana and Bagrianskii, Konstantin and Puzikov, Aleksandr and State, Radu},
  journal = {arXiv preprint arXiv:2604.18194},
  year    = {2026}
}

@inproceedings{tong2024improving,
  title     = {Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author    = {Tong, Alexander and Fatras, Kilian and Malkin, Nikolay and Huguet, Guillaume and Zhang, Yanlei and Rector-Brooks, Jarrid and Wolf, Guy and Bengio, Yoshua},
  booktitle = {Transactions on Machine Learning Research},
  year      = {2024}
}

@inproceedings{kornilov2024optimal,
  title     = {Optimal Flow Matching: Learning Straight Trajectories in Just One Step},
  author    = {Kornilov, Nikita and Mokrov, Petr and Gasnikov, Alexander and Korotin, Alexander},
  booktitle = {NeurIPS},
  year      = {2024}
}

@inproceedings{liu2023flow,
  title     = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author    = {Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  booktitle = {ICLR},
  year      = {2023}
}

@article{lambert2025generative,
  title   = {Generative Modeling via Drifting},
  author  = {Lambert, Alex and others},
  journal = {arXiv preprint arXiv:2602.04770},
  year    = {2025}
}

@inproceedings{pidhorskyi2020adversarial,
  title     = {Adversarial Latent Autoencoders},
  author    = {Pidhorskyi, Stanislav and Adjeroh, Donald A and Doretto, Gianfranco},
  booktitle = {CVPR},
  year      = {2020}
}
```

## License

This repository incorporates code from multiple sources:
- `CFM/` — MIT License (TorchCFM)
- `OFM/`, `DM/`, `RectifiedFlow/` — see individual subdirectory licenses
- `experiments/` — see paper license

---

**Last updated**: May 2026 &nbsp;|&nbsp; **Status**: Active development
