# Friction-Augmented Drifting Models for Resource-Efficient Domain Translation

> **Paper**: [arXiv:2604.18194](https://arxiv.org/abs/2604.18194)  
> **Authors** — Arkadii Kazanskii, Tatiana Petrova, Andrey Ustyuzhanin, Konstantin Bagrianskii, Aleksandr Puzikov, Radu State

---

A research codebase for **image-to-image translation in ALAE latent space**, comparing five generative modeling methods on face attribute transformations (Adult ↔ Children, Man ↔ Woman) using the FFHQ dataset.

## Methods

| Method | Description | Training | Inference | OT coupling |
|--------|-------------|----------|-----------|-------------|
| **DMF** | Drifting Model with Friction — our proposed method | Low | Fast | — |
| **DM** | Baseline Drifting Model (Deng et al., 2026) | Low | Fast | — |
| **OFM** | Optimal Flow Matching via ICNN (Kornilov et al., 2024) | High | 1-step | ✅ Global |
| **OT-CFM** | Conditional Flow Matching with mini-batch OT (Tong et al., 2024) | Medium | ODE | ✅ Mini-batch |
| **RF** | Rectified Flow with optional reflow (Liu et al., 2023) | Medium | ODE / 1-step | — |

## Repository structure

The training code is in `scripts/`. Archived configurations, metrics and checkpoints are organised by study in `experiments/`.

| Path | Contents |
|---|---|
| `scripts/dm/`, `scripts/dmf/`, `scripts/ofm/` | FFHQ training entry points |
| `scripts/dmf_gamma/` | Drift schedules, sweep runner and drift loss |
| `scripts/otcfm/`, `scripts/rectified_flow/` | Flow-matching baselines |
| `scripts/exp_utils/` | Data loading, FID/CMMD evaluation and run logging |
| `scripts/training_set_ablation/` | Limited-data experiment |
| `OFM/`, `cmmd_pytorch/` | Git submodules for the OFM/ALAE code and CMMD implementation |

## Experiment records

| Folder under `experiments/` | Study |
|---|---|
| `experiment_1/` | FFHQ adult-to-child DM, DMF and OFM. The May OFM runs used batch size 32 and the April OFM runs used 128. Each run has its own configuration and metrics. |
| `experiment_2/` | Five-seed FFHQ drift-schedule comparison, batch size 515 |
| `experiment_3/` | FFHQ male-to-female translation |
| `experiment_4/` | Two-dimensional Gaussian mixture with component means (0,0) and (3,2). The Laplace-kernel FD values are in `results_table2/table2.csv`. The symmetric mixture in the paper's Table 2 is a separate experiment. |
| `experiment_8/` | Gaussian-kernel comparison on the same (0,0)/(3,2) mixture, with the linear Laplace reference carried over from `experiment_4/` |
| `experiment_9/` | Eight-component Gaussian mixture |
| `experiment_10/` | Half-data FFHQ study. `summary_export.csv` aggregates seeds 42, 123 and 777 for both DM and DMF. Regenerate it with `python experiments/experiment_10/runs/export_summary.py` from the repository root. |

The paper reports three-seed summaries for the symmetric two-dimensional mixture and the FFHQ RF, 2-RF and OT-CFM comparisons. Table 4 gives all five Apple Silicon learning-rate pairs. The full-precision scores are in `anc/ffhq_lr_pairs.csv` in the arXiv source package.

## Installation

The archived FFHQ experiments used Python 3.12 and CUDA. Initialise the two Git submodules when cloning.

```bash
git clone --recurse-submodules https://github.com/BagrKonstantin/Friction-Augmented-Drifting-Models.git
cd Friction-Augmented-Drifting-Models
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## FFHQ training

Run these examples from the repository root. They select the adult-to-child configuration with 10,001 updates. The two OFM settings in the paper use batch sizes 32 and 128.

```bash
python -m scripts.dm.run_dm_alae --max_steps 10001 --batch_size 32 --lr 0.0005 --seed 42
python -m scripts.dmf.run_dmf_alae --max_steps 10001 --batch_size 32 --lr 0.0005 --gamma_schedule linear --seed 42
python -m scripts.ofm.run_ofm_alae --max_steps 10001 --batch_size 32 --lr 0.001 --seed 42
```

These FFHQ entry points create a dated run directory under `experiments/`. The saved files include `config.yaml`, the final model and `metrics.json`. Training times in those records are in seconds. Use `--help` for each entry point's arguments.

## Data and evaluation

The adult-to-child loader reads 512-dimensional FFHQ ALAE latents. Its default split uses the first 55,000 rows for training, the next 5,000 for validation and the final 10,000 for testing. Adult and child subsets exclude unknown ages. The reduced-data and sex-attribute experiments have their own loaders and configurations.

FID uses Inception features and CMMD uses CLIP image embeddings of decoded samples. Lower values are better for both metrics.

## References

```bibtex
@article{kazanskii2026dmf,
  title   = {Friction-Augmented Drifting Models for Resource-Efficient Domain Translation},
  author  = {Kazanskii, Arkadii and Petrova, Tatiana and Ustyuzhanin, Andrey and Bagrianskii, Konstantin and Puzikov, Aleksandr and State, Radu},
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

@article{deng2026drifting,
  title   = {Generative Modeling via Drifting},
  author  = {Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal = {arXiv preprint arXiv:2602.04770},
  year    = {2026}
}

@inproceedings{pidhorskyi2020adversarial,
  title     = {Adversarial Latent Autoencoders},
  author    = {Pidhorskyi, Stanislav and Adjeroh, Donald A and Doretto, Gianfranco},
  booktitle = {CVPR},
  year      = {2020}
}
```

## Third-party software

See the upstream [OFM](https://github.com/Jhomanik/Optimal-Flow-Matching) and [CMMD](https://github.com/BagrKonstantin/cmmd-pytorch) repositories for the components included as submodules.
