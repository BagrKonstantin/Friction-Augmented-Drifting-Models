#!/usr/bin/env python
# coding: utf-8

# parameters
DEVICE = 'cpu'
MODEL  = 'v2cpf'

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import torch
import torch.distributions as TD
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from tqdm import tqdm
from IPython.display import clear_output
import random, os, sys

sys.path.append('')
sns.set_theme(style="ticks", palette="pastel")

from OFM.ofmsrc.icnn import ICNN2CPF
from OFM.ofmsrc.model_tools import (
    id_pretrain_model, ofm_loss,
    TorchlbfgsInvParams, BruteforceIHVParams
)
from OFM.ofmsrc.dgm_utils import visualize_2d_samples_groups, visualize_2d_map
from OFM.ofmsrc.tools import EMA
from OFM.ofmsrc.distributions import Sampler, StandardNormalSampler


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# ── Gaussian Mixture Sampler ──────────────────────────────────────────────────

class GaussianMixture(Sampler):
    """
    parameters: list of dicts with keys:
        "mean" : list of length dim
        "std"  : list of length dim  (diagonal covariance)
    All components equally weighted.
    """
    def __init__(self, parameters, device='cpu'):
        super().__init__(device=device)
        self.dim   = len(parameters[0]["mean"])
        self.means = torch.tensor(
            [p["mean"] for p in parameters], dtype=torch.float32, device=device)
        self.stds  = torch.tensor(
            [p["std"]  for p in parameters], dtype=torch.float32, device=device)
        self.K = len(parameters)
        # also keep numpy version for KDE plotting
        self.means_np = self.means.cpu().numpy()
        self.stds_np  = self.stds.cpu().numpy()

    def sample(self, batch_size=10):
        idx   = torch.randint(0, self.K, (batch_size,), device=self.device)
        means = self.means[idx]
        stds  = self.stds[idx]
        return means + stds * torch.randn(batch_size, self.dim, device=self.device)

    def sample_np(self, n):
        """Returns numpy array of shape (n, dim)."""
        return self.sample(n).cpu().numpy()


# ── Distributions ─────────────────────────────────────────────────────────────

seed_everything()

parameters = [
    {"mean": [0, 0],  "std": [1,   1  ]},
    {"mean": [3, 2],  "std": [0.5, 0.5]},
]
mixture        = GaussianMixture(parameters, device=DEVICE)
source_sampler = StandardNormalSampler(device=DEVICE)
target_sampler = mixture

visualize_2d_samples_groups([
    source_sampler.sample(5024).cpu().numpy(),
    target_sampler.sample(5024).cpu().numpy()
], 'Source & Target', s=3)


# ── Model ─────────────────────────────────────────────────────────────────────

seed_everything()

_prior = TD.Normal(
    torch.tensor([0., 0.]).to(DEVICE),
    torch.tensor([4., 4.]).to(DEVICE))

def get_model(name):
    assert name == 'v2cpf'
    model = ICNN2CPF(dim=2, dimh=1024, num_hidden_layers=2, batch_size=1024).to(DEVICE)
    id_pretrain_model(model, _prior, n_max_iterations=200, lr=1e-2)
    return model

model = get_model(MODEL).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-2)


# ── Output directory ──────────────────────────────────────────────────────────

EXP_DIR = f'./gaumix_{MODEL}'
os.makedirs(EXP_DIR, exist_ok=True)

samples_path    = os.path.join(EXP_DIR, 'samples.png')
curr_model_path = os.path.join(EXP_DIR, 'model.pth')
loss_path       = os.path.join(EXP_DIR, 'loss.png')


# ── Training ──────────────────────────────────────────────────────────────────

MAX_ITER      = 20000
BATCH_SIZE    = 256
INV_TOLERANCE = 0.01

ema          = EMA(0, betas=[0.999, 0.99])
loss_history = []

print('Fitting the Gaussian Mixture')
with tqdm(range(MAX_ITER)) as tbar:
    for iteration in tbar:
        X = source_sampler.sample(BATCH_SIZE).to(DEVICE)
        Y = target_sampler.sample(BATCH_SIZE).to(DEVICE)
        t = (torch.rand(BATCH_SIZE) + 1e-8).to(DEVICE)

        metrics = {}
        loss, true_loss = ofm_loss(
            model, X, Y, t,
            TorchlbfgsInvParams(lbfgs_params=dict(tolerance_grad=INV_TOLERANCE), max_iter=5),
            BruteforceIHVParams(), stats=metrics
        )
        optim.zero_grad(); loss.backward()
        optim.step(); model.convexify()
        ema(model)

        loss_history.append(true_loss.item())
        tbar.set_postfix(loss=loss.item(), tloss=true_loss.item())

        if iteration % 500 == 0:
            clear_output(wait=True)
            print(f"Iteration {iteration}  |  Loss: {true_loss.item():.4f}")
            plt.plot(loss_history); plt.savefig(loss_path); plt.show()
            X_vis = source_sampler.sample(5000).to(DEVICE)
            Y_vis = model.push_nograd(X_vis)
            visualize_2d_map(
                X_vis.cpu().numpy(), Y_vis.cpu().numpy(), 'Learned mapping',
                data_label='src', mapped_data_label='mpd',
                s=3, figsize=(8, 8), save_path=samples_path)
            torch.save(model.state_dict(), curr_model_path)
            for beta in [0.999, 0.99]:
                torch.save(ema.get_model(beta),
                           os.path.join(EXP_DIR, f'model_{beta}.pth'))


# ═══════════════════════════════════════════════════════════════════════════════
# POST-TRAINING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Build trajectory via Euler integration of the OFM vector field ────────────
# OFM maps x0 -> x1 = ∇Ψ(x0) directly (no ODE needed in theory), but we
# interpolate along the straight OT path to get intermediate states for vis.

N_GEN    = 500
N_STEPS  = 100
time_pts = np.linspace(0.0, 1.0, N_STEPS)

model.eval()
with torch.no_grad():
    x0_np  = source_sampler.sample(N_GEN).cpu().numpy()          # (N, 2)
    x0_t   = torch.tensor(x0_np, dtype=torch.float32).to(DEVICE)
    x1_t   = model.push_nograd(x0_t)                              # OFM map
    x1_np  = x1_t.cpu().numpy()

# straight OT trajectories: x(t) = (1-t)*x0 + t*x1
trajectory = np.stack(
    [(1 - t) * x0_np + t * x1_np for t in time_pts], axis=0
)   # (N_STEPS, N_GEN, 2)

# velocity along each straight path: constant = x1 - x0
velocity_const = x1_np - x0_np   # (N_GEN, 2)  — same at every t


# ── Helper: angle-based point colors (keyed to initial position) ──────────────

def get_angle_colors(positions):
    angles     = np.arctan2(positions[:, 1], positions[:, 0])
    angles_deg = (np.degrees(angles) + 360) % 360
    colors     = np.zeros((len(positions), 3))
    for i, angle in enumerate(angles_deg):
        seg   = int(angle / 120)
        local = angle - seg * 120
        if   seg == 0: colors[i] = [1 - local/120, local/120, 0]
        elif seg == 1: colors[i] = [0, 1 - local/120, local/120]
        else:          colors[i] = [local/120, 0, 1 - local/120]
    return colors


# ── 1. Trajectory snapshots at 6 times ───────────────────────────────────────

desired_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
indices       = [np.argmin(np.abs(time_pts - tv)) for tv in desired_times]
point_colors  = get_angle_colors(x0_np)   # color locked to t=0 position

xx, yy  = np.mgrid[-5:7:100j, -4:6:100j]
grid_pts = np.vstack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('OFM trajectory snapshots  (Gaussian → Gaussian Mixture)', fontsize=14)

for i, idx in enumerate(indices):
    ax  = axes[i // 3, i % 3]
    t   = time_pts[idx]
    x_t = trajectory[idx]

    # target density at time t via forward interpolation of the mixture
    x1_ref   = mixture.sample_np(5000)
    eps      = np.random.randn(5000, 2)
    x_fwd    = t * x1_ref + (1 - t) * eps
    kernel   = gaussian_kde(x_fwd.T)
    density  = kernel(grid_pts).reshape(xx.shape)

    ax.contourf(xx, yy, density, levels=12, cmap='Blues', alpha=0.4)
    ax.contour (xx, yy, density, levels=12, alpha=0.6)
    ax.scatter(x_t[:, 0], x_t[:, 1], c=point_colors, alpha=0.6, s=10)
    ax.set_title(f't = {t:.2f}')
    ax.set_xlim(-5, 7); ax.set_ylim(-4, 6)
    ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'trajectory_snapshots.png'), dpi=120)
plt.show()


# ── 2. Thermodynamic analogy ──────────────────────────────────────────────────

def volume(x):
    cov = np.cov(x.T)
    return float(np.sqrt(np.linalg.det(cov)))

def temperature(v):
    return float(np.mean(np.sum(v**2, axis=1)))

volumes, temps, pressures, energies = [], [], [], []

for i, t in enumerate(time_pts):
    x = trajectory[i]
    v = velocity_const

    V = volume(x)
    T = temperature(v)
    P = T / V if V > 1e-8 else np.nan

    volumes.append(V)
    temps.append(T)
    pressures.append(P)
    energies.append(np.mean(np.sum(v**2, axis=1) / 2))

V_arr = np.array(volumes)
T_arr = np.array(temps)
P_arr = np.array(pressures)
E_arr = np.array(energies)

dV = np.diff(V_arr); dT = np.diff(T_arr); dP = np.diff(P_arr)


# ── 2a. P-V / P-T / V-T phase diagrams ───────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Thermodynamic phase diagrams  (OFM  Gaussian → Gaussian Mixture)', fontsize=13)

arrow_kw = dict(angles='xy', scale_units='xy', scale=1, width=0.004, color='steelblue', alpha=0.7)

# P–V
ax = axes[0]
ax.plot(V_arr, P_arr, lw=2, color='steelblue')
ax.quiver(V_arr[:-1], P_arr[:-1], dV, dP, **arrow_kw)
ax.set_xlabel('Volume  V(t)'); ax.set_ylabel('Pressure  P(t)')
ax.set_title('P–V diagram')

# P–T
ax = axes[1]
ax.plot(T_arr, P_arr, lw=2, color='tomato')
ax.quiver(T_arr[:-1], P_arr[:-1], dT, dP,
          **{**arrow_kw, 'color': 'tomato'})
ax.set_xlabel('Temperature  T(t)'); ax.set_ylabel('Pressure  P(t)')
ax.set_title('P–T diagram')

# V–T
ax = axes[2]
ax.plot(T_arr, V_arr, lw=2, color='seagreen')
ax.quiver(T_arr[:-1], V_arr[:-1], dT, dV,
          **{**arrow_kw, 'color': 'seagreen'})
ax.set_xlabel('Temperature  T(t)'); ax.set_ylabel('Volume  V(t)')
ax.set_title('V–T diagram')

plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'phase_diagrams.png'), dpi=120)
plt.show()


# ── 2b. Individual time-series for V, T, P, E ────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Thermodynamic quantities over OFM transport time', fontsize=13)

for ax, y, label, color in zip(
        axes.ravel(),
        [V_arr,  T_arr,         P_arr,      E_arr],
        ['Volume  V(t)', 'Temperature  T(t)', 'Pressure  P(t)', 'Mean kinetic energy  E(t)'],
        ['steelblue', 'tomato', 'mediumpurple', 'darkorange']
):
    ax.plot(time_pts, y, lw=2, color=color)
    ax.set_xlabel('t'); ax.set_ylabel(label)
    ax.set_title(label); ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'thermo_timeseries.png'), dpi=120)
plt.show()


# ── 2c. Annotated P–V with time colormap ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(V_arr, P_arr, c=time_pts, cmap='plasma', s=18, zorder=3)
ax.plot(V_arr, P_arr, lw=1, alpha=0.4, color='grey')
plt.colorbar(sc, ax=ax, label='Transport time t')
ax.set_xlabel('Volume  V(t)'); ax.set_ylabel('Pressure  P(t)')
ax.set_title('P–V diagram (colored by time)')
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(EXP_DIR, 'pv_time_colored.png'), dpi=120)
plt.show()

print("All figures saved to", EXP_DIR)
