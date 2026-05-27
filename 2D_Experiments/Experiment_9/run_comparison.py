#!/usr/bin/env python3
"""
run_comparison.py  —  Comparison (toy): OFM vs plain-DM vs DMF
Gaussian N(0,I) → 8 Gaussians  ×  seeds {42, 123, 1234}
    note: 777 breaks the OFM :(

Hyperparameters fixed to DMF-v2:
    MLP: 3 hidden layers × 256 units, ReLU
    MAX_ITER=5000, BATCH_SIZE=32, LR=5e-4
    R_list=(0.02, 0.3), friction: linear 0→1 over training

Criterion: DMF FD < plain-DM FD by > 1 std.

Usage:
    python run_comparison.py            # runs all, prints comparison table
    python run_comparison.py --dry-run  # 500-iter smoke test
"""

import sys, os, time, argparse, csv
sys.path.append('/home/arc/Projects/')
sys.path.append('/home/arc/Projects/OFM')
from OFM.ofmsrc.tools import EMA

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD
from scipy.linalg import sqrtm
from tqdm import tqdm
import random
import ot

# ── Hyperparameters (DMF v2) ──────────────────────────────────────────────────
MAX_ITER   = 5_000
BATCH_SIZE = 16 
LR         = 5e-4
R_LIST     = (0.02, 0.3)
N_EVAL     = 10_000
SEEDS      = [42,123,1234]
DEVICE     = 'cpu'

import math as _math
_r = 3.0
_s = _r / _math.sqrt(2)
MIX_PARAMS = [
    {"mean": [ _r,   0], "std": [0.4, 0.4]},
    {"mean": [-_r,   0], "std": [0.4, 0.4]},
    {"mean": [  0,  _r], "std": [0.4, 0.4]},
    {"mean": [  0, -_r], "std": [0.4, 0.4]},
    {"mean": [ _s,  _s], "std": [0.4, 0.4]},
    {"mean": [ _s, -_s], "std": [0.4, 0.4]},
    {"mean": [-_s,  _s], "std": [0.4, 0.4]},
    {"mean": [-_s, -_s], "std": [0.4, 0.4]},
]

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Utilities ─────────────────────────────────────────────────────────────────

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def frechet_distance(real: np.ndarray, gen: np.ndarray) -> float:
    mu_r, mu_g = real.mean(0), gen.mean(0)
    sr = np.cov(real, rowvar=False)
    sg = np.cov(gen,  rowvar=False)
    cov = sqrtm(sr @ sg)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float((mu_r - mu_g) @ (mu_r - mu_g) + np.trace(sr + sg - 2 * cov))


def wasserstein2(real: np.ndarray, gen: np.ndarray, n_sub: int = 2000, seed: int = 0) -> float:
    rng   = np.random.RandomState(seed)
    idx_r = rng.choice(len(real), size=min(n_sub, len(real)), replace=False)
    idx_g = rng.choice(len(gen),  size=min(n_sub, len(gen)),  replace=False)
    r, g  = real[idx_r], gen[idx_g]
    M     = ot.dist(r, g, metric='sqeuclidean')
    w2_sq = ot.emd2(np.ones(len(r)) / len(r), np.ones(len(g)) / len(g), M)
    return float(np.sqrt(w2_sq))


def l2_uvp(real: np.ndarray, w2_val: float) -> float:
    """L2-UVP = W2² / E[‖y‖²] × 100%"""
    return float(w2_val**2 / np.mean(np.sum(real**2, axis=1)) * 100.0)


# ── Distributions ─────────────────────────────────────────────────────────────

class GaussianMixture:
    def __init__(self, parameters, device='cpu'):
        self.device = device
        self.means = torch.tensor([p["mean"] for p in parameters],
                                  dtype=torch.float32, device=device)
        self.stds  = torch.tensor([p["std"]  for p in parameters],
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


# ── MLP model (3 hidden layers × 256, shared by DM and DMF) ──────────────────

class MLP(nn.Module):
    """
    Residual MLP: output = x + net(x).
    Last layer zero-initialized → model(x) = x at init,
    preserving the source distribution from the first iteration.
    """
    def __init__(self, dim=2, hidden=256, n_hidden=3):
        super().__init__()
        layers, d = [], dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


# ── Drift loss (PyTorch, Laplace kernel) ──────────────────────────────────────

def _cdist(x, y, eps=1e-8):
    xydot  = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn",  x, x)
    ynorms = torch.einsum("bmd,bmd->bm",  y, y)
    return torch.sqrt(torch.clamp(xnorms[:,:,None] + ynorms[:,None,:] - 2*xydot, min=eps))


def drift_loss(gen, fixed_pos, R_list, friction_coeff=0.0):
    """
    gen, fixed_pos: (B, C, 2)  — single batch dimension added externally.
    Returns scalar loss.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    old_gen   = gen.detach()
    targets   = torch.cat([old_gen, fixed_pos], dim=1)          # neg=old_gen
    targets_w = torch.ones(B, C_g + C_p, device=gen.device)

    with torch.no_grad():
        dist         = _cdist(old_gen, targets)
        scale        = (dist * targets_w[:,None,:]).mean(dim=(1,2)) / targets_w.mean(dim=1)
        scale_in     = torch.clamp(scale / (S**0.5), min=1e-3)
        old_s        = old_gen / scale_in[:,None,None]
        targets_s    = targets / scale_in[:,None,None]
        dist_n       = dist   / torch.clamp(scale[:,None,None], min=1e-3)

        # mask self-interactions
        mask = F.pad(torch.eye(C_g, device=gen.device), (0, C_p)).unsqueeze(0) * 100.
        dist_n = dist_n + mask

        force = torch.zeros_like(old_s)
        for R in R_list:
            logits   = -(dist_n / R)
            aff      = F.softmax(logits, dim=-1)
            aff_T    = F.softmax(logits, dim=-2)
            affinity = torch.sqrt(torch.clamp(aff * aff_T, min=1e-6))

            aff_neg  = affinity[:, :, :C_g]
            aff_pos  = affinity[:, :, C_g:]

            sum_pos  = aff_pos.sum(-1, keepdim=True)
            sum_neg  = aff_neg.sum(-1, keepdim=True)
            coeff    = torch.cat([-aff_neg * sum_pos, aff_pos * sum_neg], dim=2)

            fR       = torch.einsum("biy,byx->bix", coeff, targets_s)
            fR       = fR - coeff.sum(-1, keepdim=True) * old_s
            fn       = fR.pow(2).mean(dim=(1,2))
            force    = force + fR / torch.sqrt(torch.clamp(fn, min=1e-8))[:,None,None]

        goal = old_s + (1.0 - friction_coeff) * force

    gen_s = gen / scale_in[:,None,None]
    return ((gen_s - goal)**2).mean()


# ── Train plain-DM or DMF ─────────────────────────────────────────────────────

def train_dm_or_dmf(seed, friction_on, n_iters, verbose=False):
    seed_everything(seed)
    src = StandardNormal(DEVICE)
    tgt = GaussianMixture(MIX_PARAMS, DEVICE)

    model = MLP().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    name  = "DMF" if friction_on else "DM "

    _s_np, _s_torch = np.random.get_state(), torch.get_rng_state()
    ema = EMA(0, betas=[0.999, 0.99])
    np.random.set_state(_s_np); torch.set_rng_state(_s_torch)

    bar = tqdm(range(n_iters), desc=f"{name} seed={seed}", leave=False)
    for it in bar:
        z     = src.sample(BATCH_SIZE)
        x_pos = tgt.sample(BATCH_SIZE)
        x_gen = model(z)

        fc   = (it / (n_iters - 1)) if friction_on else 0.0
        loss = drift_loss(x_gen.unsqueeze(0), x_pos.unsqueeze(0), R_LIST, fc)

        optim.zero_grad()
        loss.backward()
        optim.step()

        _s_np, _s_torch = np.random.get_state(), torch.get_rng_state()
        ema(model)
        np.random.set_state(_s_np); torch.set_rng_state(_s_torch)

        if verbose and it % 1000 == 0:
            bar.write(f"  {name} seed={seed} iter={it:5d} loss={loss.item():.4f} fc={fc:.3f}")

    model.eval()
    seed_everything(seed)
    with torch.no_grad():
        x_gen_eval = model(src.sample(N_EVAL)).cpu().numpy()
    x_real = tgt.sample_np(N_EVAL)
    w2 = wasserstein2(x_real, x_gen_eval)
    return frechet_distance(x_real, x_gen_eval), w2, l2_uvp(x_real, w2)


# ── Train OFM ─────────────────────────────────────────────────────────────────

def train_ofm(seed, n_iters, verbose=False):
    from ofmsrc.icnn import ICNN2CPF
    from ofmsrc.model_tools import (
        id_pretrain_model, ofm_loss,
        TorchlbfgsInvParams, BruteforceIHVParams,
    )

    seed_everything(seed)
    src = StandardNormal(DEVICE)
    tgt = GaussianMixture(MIX_PARAMS, DEVICE)

    prior = TD.Normal(
        torch.tensor([0., 0.], device=DEVICE),
        torch.tensor([4., 4.], device=DEVICE),
    )
    model = ICNN2CPF(dim=2, dimh=1024, num_hidden_layers=2, batch_size=1024).to(DEVICE)
    id_pretrain_model(model, prior, n_max_iterations=200, lr=1e-2)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    _s_np, _s_torch = np.random.get_state(), torch.get_rng_state()
    ema = EMA(0, betas=[0.999, 0.99])
    np.random.set_state(_s_np); torch.set_rng_state(_s_torch)

    bar = tqdm(range(n_iters), desc=f"OFM seed={seed}", leave=False)
    for it in bar:
        X = src.sample(BATCH_SIZE)
        Y = tgt.sample(BATCH_SIZE)
        t = (torch.rand(BATCH_SIZE) + 1e-8).to(DEVICE)

        loss, true_loss = ofm_loss(
            model, X, Y, t,
            TorchlbfgsInvParams(lbfgs_params=dict(tolerance_grad=0.01), max_iter=50),
            BruteforceIHVParams(),
            stats={},
        )
        optim.zero_grad(); loss.backward()
        optim.step(); model.convexify()

        _s_np, _s_torch = np.random.get_state(), torch.get_rng_state()
        ema(model)
        np.random.set_state(_s_np); torch.set_rng_state(_s_torch)

        if verbose and it % 1000 == 0:
            bar.write(f"  OFM seed={seed} iter={it:5d} loss={true_loss.item():.4f}")

    model.eval()
    seed_everything(seed)
    x_src_np = src.sample(N_EVAL).cpu().numpy()
    x_src_t  = torch.tensor(x_src_np, dtype=torch.float32, requires_grad=True).to(DEVICE)
    x_gen    = model.push_nograd(x_src_t).detach().cpu().numpy()
    x_real   = tgt.sample_np(N_EVAL)
    w2 = wasserstein2(x_real, x_gen)
    return frechet_distance(x_real, x_gen), w2, l2_uvp(x_real, w2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 500 iters instead of full run")
    args = parser.parse_args()

    n_iters = 500 if args.dry_run else MAX_ITER
    if args.dry_run:
        print(f"[DRY RUN] n_iters={n_iters}, N_EVAL={N_EVAL}")

    fd_results  = {"DM": [], "DMF": [], "OFM": []}
    w2_results  = {"DM": [], "DMF": [], "OFM": []}
    uvp_results = {"DM": [], "DMF": [], "OFM": []}

    seed_labels = [str(s) for s in SEEDS]

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  seed = {seed}")
        print(f"{'='*60}")

        t0 = time.time()
        fd, w2, uvp = train_dm_or_dmf(seed, friction_on=False, n_iters=n_iters, verbose=True)
        fd_results["DM"].append(fd); w2_results["DM"].append(w2); uvp_results["DM"].append(uvp)
        print(f"  DM   FD={fd:.5f}  W2={w2:.5f}  L2-UVP={uvp:.4f}%   ({time.time()-t0:.0f}s)")

        t0 = time.time()
        fd, w2, uvp = train_dm_or_dmf(seed, friction_on=True,  n_iters=n_iters, verbose=True)
        fd_results["DMF"].append(fd); w2_results["DMF"].append(w2); uvp_results["DMF"].append(uvp)
        print(f"  DMF  FD={fd:.5f}  W2={w2:.5f}  L2-UVP={uvp:.4f}%   ({time.time()-t0:.0f}s)")

        t0 = time.time()
        fd, w2, uvp = train_ofm(seed, n_iters=n_iters, verbose=True)
        fd_results["OFM"].append(fd); w2_results["OFM"].append(w2); uvp_results["OFM"].append(uvp)
        print(f"  OFM  FD={fd:.5f}  W2={w2:.5f}  L2-UVP={uvp:.4f}%   ({time.time()-t0:.0f}s)")

    # ── Summary table ─────────────────────────────────────────────────────────
    hdr = "  ".join(f"{'seed '+s:>10}" for s in seed_labels)
    for metric, results in [("FD", fd_results), ("W2", w2_results), ("L2-UVP%", uvp_results)]:
        print(f"\n{'='*60}")
        print(f"  Comparison (toy): Gaussian → 8 Gaussians  [{metric}]")
        print(f"{'='*60}")
        print(f"  {'Method':<8}  {hdr}  {'mean±std':>16}")
        print(f"  {'-'*8}  " + "  ".join(["-"*10]*len(SEEDS)) + f"  {'-'*16}")
        for method, vals in results.items():
            arr  = np.array(vals)
            cols = "  ".join(f"{v:10.5f}" for v in arr)
            print(f"  {method:<8}  {cols}  {arr.mean():7.5f} ± {arr.std():7.5f}")

    # ── Criterion check (FD) ──────────────────────────────────────────────────
    dm_mean,  dm_std  = np.mean(fd_results["DM"]),  np.std(fd_results["DM"])
    dmf_mean, dmf_std = np.mean(fd_results["DMF"]), np.std(fd_results["DMF"])
    margin = dm_mean - dmf_mean
    threshold = dm_std
    print(f"\n  DMF < DM by {margin:.5f}  (threshold = 1 std of DM = {threshold:.5f})")
    if margin > threshold:
        print("  ✓  CRITERION MET: DMF FD < plain-DM FD by > 1 std")
    else:
        print("  ✗  CRITERION NOT MET")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "comparison.csv")
    seed_cols = [f"seed_{s}" for s in SEEDS]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric"] + seed_cols + ["mean", "std"])
        for method in ["DM", "DMF", "OFM"]:
            for metric, results in [("FD", fd_results), ("W2", w2_results), ("L2-UVP%", uvp_results)]:
                arr = np.array(results[method])
                w.writerow([method, metric] + [f"{v:.6f}" for v in arr] +
                           [f"{arr.mean():.6f}", f"{arr.std():.6f}"])
    print(f"\n  Results saved → {csv_path}")


if __name__ == "__main__":
    main()
