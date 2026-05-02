# OT-CFM — Optimal Transport Conditional Flow Matching

Implementation of **OT-CFM** (Tong et al., 2024) applied to ALAE latent space for face attribute translation. OT-CFM improves standard CFM by coupling source and target samples via mini-batch optimal transport, producing straighter probability flow trajectories and lower FID at equivalent NFE budgets.

## Background

Standard CFM trains a velocity field $v_\theta(x_t, t)$ to match a conditional vector field between random pairs $(x_0, x_1) \sim \pi_0 \times \pi_1$. OT-CFM replaces the independent coupling with a **mini-batch OT plan** — the earth mover's distance solution over each batch — so that matched pairs are closer in feature space. This leads to:

- Shorter, more direct trajectories
- Better sample quality at fewer ODE steps
- Faster convergence relative to independent coupling

## Files

```
otcfm/
├── otcfm_alae.py              # Training script
├── otcfm_inference.py         # Inference on saved checkpoint
├── run_multiple_seeds_otcfm.py # Batch runner across seeds
└── __init__.py
```

## Architecture

**`OTCFMMLP`** — time-conditioned velocity field:

```
[z; t] (513-dim) → Linear(512) → 4 × (Linear(512) + LeakyReLU) → Linear(512)
```

Time `t ∈ [0, 1]` is concatenated directly to the latent before the first layer. The model predicts the velocity $v_\theta(z_t, t)$ used to integrate $z_0 \to z_1$.

## Training

```bash
# From repository root
python experiments/otcfm/otcfm_alae.py \
    --max_steps 10000 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42 \
    --input_data ADULT \
    --target_data CHILDREN
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_steps` | 10001 | Training iterations |
| `--batch_size` | 256 | Mini-batch size |
| `--lr` | 1e-4 | Adam learning rate |
| `--sigma` | 0.0 | Gaussian noise on interpolated path |
| `--seed` | 42 | Random seed |
| `--input_data` | ADULT | Source distribution |
| `--target_data` | CHILDREN | Target distribution |
| `--no_metrics` | False | Skip FID/CMMD evaluation |

**Training objective:**

For a mini-batch OT-coupled pair $(x_0, x_1)$ and $t \sim \mathcal{U}[0,1]$:

$$x_t = (1 - t)\,x_0 + t\,x_1 + \sigma\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

The OT plan is recomputed each mini-batch using the **POT** library (`ot.emd`).

## Multi-Seed Batch

```bash
python experiments/otcfm/run_multiple_seeds_otcfm.py \
    --max_steps 30000 \
    --no_metrics
```

Runs seeds `[42, 123, 777, 1234, 2024]` sequentially.

## Inference

```bash
python experiments/otcfm/otcfm_inference.py \
    --checkpoint experiments/runs/<run_dir>/checkpoint_final.pth \
    --input_data ADULT \
    --target_data CHILDREN \
    --n_steps 10
```

Integration is performed with `torchdiffeq.odeint` (RK45 solver by default). Use `--n_steps` to control the number of ODE steps (fewer = faster, lower quality).

## Output

Each run saves to `experiments/runs/DD-MM-YYYY-HH-MM_otcfm_<seed>/`:

```
config.yaml
metrics.json                  # FID, CMMD, training_time_seconds
checkpoint_final.pth
checkpoint.pth                # Saved at step 2000
otcfm_transfer_faces.png      # 5-column source → generated grid
```

## Reference

```bibtex
@article{tong2024improving,
  title   = {Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author  = {Tong, Alexander and Fatras, Kilian and Malkin, Nikolay and Huguet, Guillaume
             and Zhang, Yanlei and Rector-Brooks, Jarrid and Wolf, Guy and Bengio, Yoshua},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```

See also the original [TorchCFM](https://github.com/atong01/conditional-flow-matching) library in `CFM/`.
