#!/usr/bin/env python3
"""
plot_w2_comparison.py  —  Plot W2 vs iteration and vs wall-clock time.

Loads w2_<METHOD>_seed<SEED>_{history,iters,times}.npy files produced by
run_comparison.py and plots mean ± std across seeds for each method.

Usage:
    python plot_w2_comparison.py [--dir results_comparison] [--out w2_comparison.pdf]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

METHODS = ["DM", "DMF", "OFM"]
SEEDS   = [42, 123, 777]
COLORS  = {"OFM": "#6ACC65", "DM": "#D65F5F", "DMF": "#B47CC7"}


def load_method(out_dir, method, seeds):
    """Load tracking arrays for all seeds of a method.

    Returns (histories, iters_list, times_list) where each is a list of arrays,
    one per seed that was found.
    """
    histories, iters_list, times_list = [], [], []
    for seed in seeds:
        h = os.path.join(out_dir, f"w2_{method}_seed{seed}_history.npy")
        i = os.path.join(out_dir, f"w2_{method}_seed{seed}_iters.npy")
        t = os.path.join(out_dir, f"w2_{method}_seed{seed}_times.npy")
        if os.path.exists(h) and os.path.exists(i) and os.path.exists(t):
            histories.append(np.load(h))
            iters_list.append(np.load(i))
            times_list.append(np.load(t))
        else:
            print(f"  [{method} seed={seed}] files not found, skipping")
    return histories, iters_list, times_list


def plot_with_band(ax, x_arrays, y_arrays, color, label):
    """Plot mean line ± std band; x is interpolated to a common grid."""
    if not y_arrays:
        return

    # Use the x-values of the first seed as reference grid
    x_ref = x_arrays[0]
    interp_ys = []
    for x, y in zip(x_arrays, y_arrays):
        if len(x) < 2:
            continue
        interp_ys.append(np.interp(x_ref, x, y))

    if not interp_ys:
        return

    ys  = np.stack(interp_ys, axis=0)   # (n_seeds, T)
    mu  = ys.mean(axis=0)
    std = ys.std(axis=0)

    ax.plot(x_ref, mu, color=color, linewidth=1.8, label=label)
    ax.fill_between(x_ref, mu - std, mu + std, color=color, alpha=0.20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results_comparison",
                        help="Directory with w2_*.npy files")
    parser.add_argument("--out", default="w2_comparison.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), args.dir)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Directory not found: {out_dir}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in METHODS:
        color = COLORS[method]
        histories, iters_list, times_list = load_method(out_dir, method, SEEDS)

        if not histories:
            print(f"  [{method}] no data loaded")
            continue

        n_found = len(histories)
        print(f"  [{method}] loaded {n_found}/{len(SEEDS)} seeds, "
              f"{len(histories[0])} checkpoints each")

        label = f"{method} (n={n_found})"
        plot_with_band(ax1, iters_list, histories, color, label)
        plot_with_band(ax2, times_list, histories, color, label)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("W2")
    ax1.set_title("W2 vs Iteration")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.set_xlabel("Wall-clock time (s)")
    ax2.set_ylabel("W2")
    ax2.set_title("W2 vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
