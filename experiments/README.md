# experiments/

This directory contains the unified experiment framework for all five generative modeling methods evaluated in the paper. Each method shares the same data pipeline, evaluation metrics, and output conventions so that results are directly comparable.

## Directory Layout

```
experiments/
├── otcfm/                    # OT-CFM implementation
├── rectified_flow/           # Rectified Flow implementation (1-RF, 2-RF)
│
├── dm_alae.py                # Drifting Model
├── dmf_alae.py               # Drifting Model with Friction (DMF — ours)
├── ofm_alae.py               # Optimal Flow Matching
│
├── data_loader.py            # FFHQ data loading + automatic download
├── models_and_losses.py      # MLP architecture + drift loss
├── eval_metrics.py           # FID and CMMD evaluation
├── exp_utils.py              # Seeding, paths, config/metrics I/O
├── visualization.py          # Face comparison plots
│
├── run_multiple_seeds.py     # Batch runner: DM + DMF across seeds
├── run_friction_sweep.py     # Sweep over friction coefficients
│
└── runs/                     # Auto-created; one timestamped dir per run
```

## Shared Conventions

All training scripts follow the same pattern:

1. Parse arguments → set seed → load data → train → evaluate → save.
2. Every run writes to `runs/DD-MM-YYYY-HH-MM_<method>_<seed>/`:
   - `config.yaml` — full hyperparameter snapshot
   - `metrics.json` — FID, CMMD, wall-clock training time
   - `checkpoint_final.pth` — final weights
   - `checkpoint.pth` — mid-training checkpoint (step 2000)
   - `*_transfer_faces.png` — source → generated face grid

3. FID/CMMD are evaluated every 2000 steps and at the end. Pass `--no_metrics` to skip.

## Shared Modules

### `data_loader.py`

Downloads and caches FFHQ data on first use:
- `latents.npy` — 70 000 × 512 ALAE latent codes
- `gender.npy`, `age.npy` — metadata for attribute filtering
- `test_images.npy` — reference RGB images for visualization

`load_alae_data(input_data, target_data)` returns a dict with train/test splits and `TensorSampler` objects ready for mini-batch sampling.

Supported distributions: `ADULT` (age ≥ 18), `CHILDREN` (age < 18), `MAN`, `WOMAN`.

### `models_and_losses.py`

**`MLP`**: Residual network used by DM/DMF.  
Input 512 → 3 × (Linear 1024 + LayerNorm + ReLU) → Linear 512 → residual add.

**`drift_loss_pytorch(gen, fixed_pos, fixed_neg, R_list, friction_coeff)`**:  
Implements the attraction/repulsion loss from Lambert et al. (2025), extended with a friction coefficient that dampens the effective force magnitude.

### `eval_metrics.py`

`MetricsEvaluator` lazily loads InceptionV3 (FID) and CLIP (CMMD) on first call, then caches target distribution statistics to avoid recomputing across runs.

### `exp_utils.py`

- `seed_all(seed)` — fully deterministic (torch, numpy, random, CUDA)
- `get_run_dir(method, seed)` — creates the timestamped output directory
- `load_or_compute_target_stats(...)` — cached `.npz` target features

## Running Experiments

All commands are executed from the **repository root**:

```bash
# Individual methods
python experiments/dm_alae.py  --max_steps 10000 --seed 42
python experiments/dmf_alae.py --max_steps 10000 --seed 42
python experiments/ofm_alae.py --max_steps 10000 --seed 42

# See otcfm/ and rectified_flow/ READMEs for those methods

# Multi-seed batch (DM + DMF)
python experiments/run_multiple_seeds.py --max_steps 30000 --no_metrics
```

## Adding a New Method

1. Create a training script that calls `seed_all`, `load_alae_data`, `get_run_dir`, `save_config`, `save_metrics`.
2. Use `MetricsEvaluator` for FID/CMMD.
3. Call `save_comparison_plot` from `visualization.py` for the face grid.
4. Match the output directory structure above.
