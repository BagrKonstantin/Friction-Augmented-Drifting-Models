#!/usr/bin/env python3
"""
run_kernel_comparison.py  —  Experiment 8: Laplace vs Gaussian kernel
Gaussian N(0,I) → Gaussian Mixture  ×  seeds {42, 123, 777}

Conditions:
  A: Laplace kernel,  linear γ  — loaded from ../Experiment_4/results_table2/table2.csv
  B: Gaussian kernel, linear γ  (friction 0→1)
  C: Gaussian kernel, γ = 0     (no friction)

Metrics: FD (Fréchet Distance), MMD² (B/C only), wall-clock time.

Outcomes:
  A ≈ B  →  robust to kernel choice
  B < A  →  switch default to Gaussian
  A < B  →  keep Laplace + add Gaussian ablation

Usage:
    python run_kernel_comparison.py
    python run_kernel_comparison.py --dry-run        # 500-iter smoke test
    python run_kernel_comparison.py --device cuda
"""

import sys, os, time, argparse, csv
sys.path.append('/home/arc/Projects/')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP4_CSV = os.path.join(_HERE, '..', 'Experiment_4', 'results_table2', 'table2.csv')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from tqdm import tqdm
import random

# ── Hyperparameters ───────────────────────────────────────────────────────────

MAX_ITER     = 10_000
BATCH_SIZE   = 16
LR           = 5e-4
TEMPERATURES = [0.02, 0.05, 0.15]
N_EVAL       = 10_000
SEEDS        = [42, 123, 777]

MIX_PARAMS = [
    {"mean": [0, 0], "std": [1, 1]},
    {"mean": [3, 2], "std": [1, 1]},
]

# ── Drift Field ─────────────────────────────────────────────────────────────────

def _cdist(x, y, eps=1e-8):
    """Pairwise Euclidean distances.  x: (N, D), y: (M, D) -> (N, M)"""
    xn = (x ** 2).sum(-1, keepdim=True)       # (N, 1)
    yn = (y ** 2).sum(-1, keepdim=True)        # (M, 1)
    dist_sq = xn + yn.T - 2 * (x @ y.T)
    return torch.sqrt(torch.clamp(dist_sq, min=eps))


def compute_V_multi_temperature(
    gen,
    pos,
    neg=None,
    temperatures=None,
    mask_self=True,
    normalize_each=True,
):
    """
    Compute the drift field V for *gen* particles.

    Parameters
    ----------
    gen  : (N, D)  generated particles
    pos  : (M, D)  positive target particles  (attraction)
    neg  : (K, D)  negative / self particles  (repulsion); defaults to gen
    temperatures : list[float]   bandwidth values R (Gaussian σ)
    mask_self    : bool  mask the gen↔gen diagonal when neg is gen
    normalize_each : bool  normalise the force contribution of each R

    Returns
    -------
    V : (N, D)  drift field in *original* (unscaled) space
    """
    if temperatures is None:
        temperatures = [0.05]
    if neg is None:
        neg = gen

    N, D = gen.shape
    M = pos.shape[0]
    K = neg.shape[0]

    # Concatenate targets: [neg | pos]
    all_targets = torch.cat([neg, pos], dim=0)   # (K+M, D)

    dist = _cdist(gen, all_targets)              # (N, K+M)

    # Adaptive scale: mean pairwise distance
    scale = dist.mean().clamp(min=1e-3)
    dist_normed = dist / scale

    # Self-mask: prevent each gen particle from interacting with itself
    if mask_self and (neg is gen or (neg.data_ptr() == gen.data_ptr())):
        eye = torch.eye(N, device=gen.device, dtype=gen.dtype)
        self_mask = F.pad(eye, (0, M)) * 1e4    # (N, K+M)
        dist_normed = dist_normed + self_mask

    gen_scaled     = gen         / scale         # (N, D)
    targets_scaled = all_targets / scale         # (K+M, D)

    V = torch.zeros_like(gen_scaled)

    for R in temperatures:
        # ── Gaussian kernel ──────────────────────────────────────────────────
        # affinity ∝ exp(-d² / (2·R²))
        logits = -(dist_normed ** 2) / (2.0 * R ** 2)

        aff   = F.softmax(logits, dim=-1)        # row-normalised  (N, K+M)
        aff_T = F.softmax(logits, dim=-2)        # col-normalised  (N, K+M)
        affinity = torch.sqrt(torch.clamp(aff * aff_T, min=1e-6))

        aff_neg = affinity[:, :K]                # (N, K) — repulsion block
        aff_pos = affinity[:, K:]                # (N, M) — attraction block

        sum_pos = aff_pos.sum(-1, keepdim=True)
        sum_neg = aff_neg.sum(-1, keepdim=True)

        coeff_neg = -aff_neg * sum_pos           # repel from neg
        coeff_pos =  aff_pos * sum_neg           # attract to pos

        coeff = torch.cat([coeff_neg, coeff_pos], dim=-1)   # (N, K+M)

        # Force = weighted sum of targets minus self-correction
        force_R = coeff @ targets_scaled - coeff.sum(-1, keepdim=True) * gen_scaled

        if normalize_each:
            fn = force_R.pow(2).mean().sqrt().clamp(min=1e-8)
            force_R = force_R / fn

        V = V + force_R

    # Rescale back to original space
    return V * scale


def normalize_drift_fn(V, eps=1e-8):
    """Unit-normalise the drift field (global RMS norm)."""
    norm = V.pow(2).mean().sqrt().clamp(min=eps)
    return V / norm


# ── Utilities ─────────────────────────────────────────────────────────────────

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def frechet_distance(real: np.ndarray, gen: np.ndarray) -> float:
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sr = np.cov(real, rowvar=False)
    sg = np.cov(gen,  rowvar=False)
    cov = sqrtm(sr @ sg)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float((mu_r - mu_g) @ (mu_r - mu_g) + np.trace(sr + sg - 2 * cov))


def mmd_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased MMD² with Gaussian kernel (median-heuristic bandwidth)."""
    sub = min(2000, len(x), len(y))
    xy  = np.concatenate([x[:sub], y[:sub]], axis=0)
    dists = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    bw = float(np.median(dists[dists > 0])) ** 2 + 1e-8

    def k(a, b):
        d2 = ((a[:, None] - b[None, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * bw))

    n, m = len(x), len(y)
    kxx = k(x, x); np.fill_diagonal(kxx, 0)
    kyy = k(y, y); np.fill_diagonal(kyy, 0)
    kxy = k(x, y)
    return float(kxx.sum() / (n * (n-1)) + kyy.sum() / (m * (m-1)) - 2 * kxy.mean())


# ── Distributions ─────────────────────────────────────────────────────────────

class GaussianMixture:
    def __init__(self, parameters, device='cpu'):
        self.device = device
        self.means  = torch.tensor([p["mean"] for p in parameters],
                                   dtype=torch.float32, device=device)
        self.stds   = torch.tensor([p["std"]  for p in parameters],
                                   dtype=torch.float32, device=device)
        self.K = len(parameters)

    def sample(self, n):
        idx = torch.randint(0, self.K, (n,), device=self.device)
        return self.means[idx] + self.stds[idx] * torch.randn(n, 2, device=self.device)

    def sample_np(self, n):
        return self.sample(n).cpu().numpy()


class StandardNormal:
    def __init__(self, device='cpu'):
        self.device = device

    def sample(self, n):
        return torch.randn(n, 2, device=self.device)


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dim=2, hidden=256, n_hidden=3):
        super().__init__()
        layers, d = [], dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Load condition A from Experiment_4 ────────────────────────────────────────

def load_condition_a() -> dict:
    """
    Read DMF row from Experiment_4 table2.csv.
    Returns {seed: fd} for each seed in SEEDS.
    """
    seed_cols = {42: 'seed_42', 123: 'seed_123', 777: 'seed_777'}
    with open(EXP4_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['method'] == 'DMF':
                return {s: float(row[col]) for s, col in seed_cols.items()}
    raise ValueError(f"DMF row not found in {EXP4_CSV}")


# ── Training (conditions B and C) ─────────────────────────────────────────────

def train(condition, seed, device, n_iters, verbose=False):
    """
    Returns (fd, mmd2, elapsed_seconds).
    condition: 'B' Gaussian+linear_γ | 'C' Gaussian+γ=0
    """
    seed_everything(seed)
    src = StandardNormal(device)
    tgt = GaussianMixture(MIX_PARAMS, device)

    model = MLP().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    t0  = time.time()
    bar = tqdm(range(n_iters), desc=f"  {condition} seed={seed}", leave=False)

    for it in bar:
        z     = src.sample(BATCH_SIZE).to(device)
        x_pos = tgt.sample(BATCH_SIZE).to(device)
        x_gen = model(z)

        fc = (it / (n_iters - 1)) if condition == 'B' else 0.0

        with torch.no_grad():
            V = compute_V_multi_temperature(
                x_gen.detach(), x_pos,
                temperatures=TEMPERATURES,
                mask_self=True,
                normalize_each=True,
            )
            V      = normalize_drift_fn(V)
            target = x_gen.detach() + (1.0 - fc) * V

        loss = ((x_gen - target) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if verbose and it % 1000 == 0:
            bar.write(f"    {condition} seed={seed} iter={it:5d}  loss={loss.item():.5f}  fc={fc:.3f}")

    elapsed = time.time() - t0

    # Reset RNG to training seed before evaluation — matches run_table2.py pattern
    seed_everything(seed)
    model.eval()
    with torch.no_grad():
        x_gen_eval = model(src.sample(N_EVAL).to(device)).cpu().numpy()
    x_real = tgt.sample_np(N_EVAL)

    return frechet_distance(x_real, x_gen_eval), mmd_squared(x_real, x_gen_eval), elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

CONDITIONS = {
    'A': 'Laplace  + linear γ (Exp4)',
    'B': 'Gaussian + linear γ',
    'C': 'Gaussian + γ=0',
}


def fmt_val(v, spec):
    return 'N/A'.rjust(len(spec.format(0.0))) if np.isnan(v) else spec.format(v)


def print_table(label, key, spec, results):
    seed_hdrs = [f"seed {s}" for s in SEEDS]
    w = len(spec.format(0.0))
    hdr = f"  {'Condition':<30}" + "".join(f"  {h:>{w}}" for h in seed_hdrs) + f"  {'mean±std':>{2*w+3}}"
    sep = f"  {'-'*30}" + "".join(f"  {'-'*w}" for _ in SEEDS) + f"  {'-'*(2*w+3)}"
    print(f"\n  [{label}]")
    print(hdr); print(sep)
    for cond, desc in CONDITIONS.items():
        arr  = np.array(results[cond][key])
        cols = "".join(f"  {fmt_val(v, spec)}" for v in arr)
        if np.all(np.isnan(arr)):
            summary = 'N/A (loaded — no samples saved)'
        else:
            valid = arr[~np.isnan(arr)]
            summary = f"{valid.mean():{spec.strip()[2:-1]}} ± {valid.std():{spec.strip()[2:-1]}}"
        print(f"  {desc:<30}{cols}  {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    n_iters = 500 if args.dry_run else MAX_ITER
    device  = args.device
    tag     = '[DRY RUN] ' if args.dry_run else ''
    print(f"{tag}n_iters={n_iters}  device={device}  seeds={SEEDS}")
    print(f"  Condition A: loading from {os.path.relpath(EXP4_CSV)}")

    out_dir = os.path.join(_HERE, 'results_kernel_comparison')
    os.makedirs(out_dir, exist_ok=True)

    results = {c: {'fd': [], 'mmd2': [], 'time': []} for c in CONDITIONS}

    # ── Condition A: load pre-computed FD from Experiment_4 ───────────────────
    exp4_fd = load_condition_a()
    for seed in SEEDS:
        results['A']['fd'].append(exp4_fd[seed])
        results['A']['mmd2'].append(float('nan'))   # samples not saved by run_table2
        results['A']['time'].append(float('nan'))

    # ── Conditions B & C: train fresh ─────────────────────────────────────────
    for seed in SEEDS:
        print(f"\n{'='*62}\n  seed = {seed}\n{'='*62}")
        for cond in ('B', 'C'):
            fd, mmd2, elapsed = train(cond, seed, device, n_iters, verbose=True)
            results[cond]['fd'].append(fd)
            results[cond]['mmd2'].append(mmd2)
            results[cond]['time'].append(elapsed)
            print(f"  {cond} ({CONDITIONS[cond]})  FD={fd:.5f}  MMD²={mmd2:.6f}  t={elapsed:.0f}s")

    print_table('FD — lower is better',   'fd',   '  {:11.5f}', results)
    print_table('MMD² — lower is better', 'mmd2', '  {:11.6f}', results)
    print_table('Time (s)',               'time', '  {:11.0f}', results)

    # ── Outcome verdict ───────────────────────────────────────────────────────
    fd_A  = np.mean(results['A']['fd']);  std_A = np.std(results['A']['fd'])
    fd_B  = np.mean(results['B']['fd']);  std_B = np.std(results['B']['fd'])
    diff  = fd_A - fd_B
    thr   = max(std_A, std_B)
    print(f"\n  FD(A) - FD(B) = {diff:+.5f}  (threshold ±{thr:.5f})")
    if abs(diff) <= thr:
        verdict = "A ≈ B  →  robust to kernel. Either can be default."
    elif diff > 0:
        verdict = "B < A  →  Gaussian kernel is better. Consider switching default."
    else:
        verdict = "A < B  →  Laplace kernel is better. Keep Laplace + Gaussian as ablation."
    print(f"  {verdict}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, 'kernel_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(
            ['cond', 'kernel', 'friction', 'source'] +
            [f'seed{s}_fd'   for s in SEEDS] + ['mean_fd',   'std_fd'  ] +
            [f'seed{s}_mmd2' for s in SEEDS] + ['mean_mmd2', 'std_mmd2'] +
            [f'seed{s}_time' for s in SEEDS] + ['mean_time']
        )
        meta = {
            'A': ('Laplace',  'linear', 'Experiment_4/results_table2/table2.csv'),
            'B': ('Gaussian', 'linear', 'trained'),
            'C': ('Gaussian', 'zero',   'trained'),
        }
        for cond in CONDITIONS:
            ker, fric, src = meta[cond]
            fd_v  = results[cond]['fd']
            mmd_v = results[cond]['mmd2']
            t_v   = results[cond]['time']
            valid_fd   = [v for v in fd_v  if not np.isnan(v)]
            valid_mmd  = [v for v in mmd_v if not np.isnan(v)]
            valid_t    = [v for v in t_v   if not np.isnan(v)]
            w.writerow(
                [cond, ker, fric, src] +
                [f'{v:.6f}' if not np.isnan(v) else '' for v in fd_v] +
                [f'{np.mean(valid_fd):.6f}',  f'{np.std(valid_fd):.6f}' ] +
                [f'{v:.7f}' if not np.isnan(v) else '' for v in mmd_v] +
                [f'{np.mean(valid_mmd):.7f}' if valid_mmd else '',
                 f'{np.std(valid_mmd):.7f}'  if valid_mmd else ''] +
                [f'{v:.1f}' if not np.isnan(v) else '' for v in t_v] +
                [f'{np.mean(valid_t):.1f}' if valid_t else '']
            )
    print(f"\n  Results saved → {csv_path}")


if __name__ == '__main__':
    main()
