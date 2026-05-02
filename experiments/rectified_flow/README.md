# Rectified Flow

Implementation of **Rectified Flow** (Liu et al., 2023) applied to ALAE latent space for face attribute translation. Rectified Flow trains a velocity field on straight-line interpolations between source and target samples, then optionally performs a **reflow** pass to straighten trajectories further — enabling high-quality generation in a single ODE step.

## Background

Rectified Flow defines the training objective as:

$$\mathcal{L} = \mathbb{E}_{t,\, x_0 \sim \pi_0,\, x_1 \sim \pi_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right], \quad x_t = (1 - t)\,x_0 + t\,x_1$$

**1-RF** trains on random (independent) couplings from $\pi_0 \times \pi_1$.  
**2-RF (reflow)** generates new pairs $(x_0, \hat{x}_1)$ using the 1-RF model, then retrains on those straighter trajectories. This iteratively rectifies the flow toward straight lines, making single-step generation viable.

## Files

```
rectified_flow/
├── rectified_flow_alae.py          # Training script (1-RF and 2-RF)
├── rectified_flow_model.py         # RFUNet and TimeConditionedMLP architectures
├── rectified_flow_inference.py     # Inference on saved checkpoint
├── run_multiple_seeds_rf.py        # Batch runner across seeds
└── __init__.py
```

## Architectures

Two velocity field architectures are available:

**`TimeConditionedMLP`** (default for latent space):

```
[z; t] (513-dim) → 3 × (Linear(1024) + Tanh) → Linear(512)
```

**`LatentRFUNet`** (optional UNet-style):  
Reshapes the 512-dim latent to `2×16×16`, applies RFUNet (a lightweight U-Net with time conditioning), then flattens back to 512-dim. Better at capturing spatial structure at higher computational cost.

## Training

### 1-RF (first rectification)

```bash
python experiments/rectified_flow/rectified_flow_alae.py \
    --max_steps 10000 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42 \
    --input_data ADULT \
    --target_data CHILDREN
```

### 2-RF (reflow)

```bash
python experiments/rectified_flow/rectified_flow_alae.py \
    --max_steps 10000 \
    --seed 42 \
    --reflow \
    --rf1_checkpoint experiments/runs/<1rf_run_dir>/checkpoint_final.pth \
    --input_data ADULT \
    --target_data CHILDREN
```

The reflow pass generates synthetic pairs with the 1-RF checkpoint, then trains a fresh model on those pairs.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_steps` | 10001 | Training iterations |
| `--batch_size` | 256 | Mini-batch size |
| `--lr` | 1e-4 | Adam learning rate |
| `--seed` | 42 | Random seed |
| `--reflow` | False | Enable 2-RF reflow pass |
| `--rf1_checkpoint` | — | Required when `--reflow` is set |
| `--use_unet` | False | Use `LatentRFUNet` instead of MLP |
| `--input_data` | ADULT | Source distribution |
| `--target_data` | CHILDREN | Target distribution |
| `--no_metrics` | False | Skip FID/CMMD evaluation |

## Multi-Seed Batch

```bash
python experiments/rectified_flow/run_multiple_seeds_rf.py \
    --max_steps 30000 \
    --no_metrics
```

Runs seeds `[42, 123, 777, 1234, 2024]` sequentially.

## Inference

```bash
python experiments/rectified_flow/rectified_flow_inference.py \
    --checkpoint experiments/runs/<run_dir>/checkpoint_final.pth \
    --input_data ADULT \
    --target_data CHILDREN \
    --n_steps 10
```

For a 2-RF model, a single step (`--n_steps 1`) already produces good results.  
Integration uses `torchdiffeq.odeint` with the Euler or RK45 solver.

## Output

Each run saves to `experiments/runs/DD-MM-YYYY-HH-MM_<1|2>_rectified_flow_<seed>/`:

```
config.yaml
metrics.json                          # FID, CMMD, training_time_seconds
checkpoint_final.pth
checkpoint.pth                        # Saved at step 2000
rectified_flow_transfer_faces.png     # 5-column source → generated grid
```

## Reference

```bibtex
@inproceedings{liu2023flow,
  title     = {Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author    = {Liu, Xingchao and Gong, Chengyue and Liu, Qiang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
```

See also the official implementation in `RectifiedFlow/`.
