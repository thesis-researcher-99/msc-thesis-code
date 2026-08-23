"""
Targeted follow-up to the Ginestet Sim 5 grid: reruns GCN ONLY, at Nv=40
ONLY, with hidden_channels=32 instead of the original 16, across all
topologies / n_samples / effect_sizes already covered by the main grid.

Motivation: the main grid (ginestet_sim5_grid_parallel.py) showed GCN's
power plateauing well below 1.0 at Nv=40 even at the largest effect size,
while Ginestet's own test (run on the same simulated data) continued
climbing toward 1.0 given a large enough effect -- suggesting a GCN
capacity/receptive-field ceiling at hidden_channels=16, rather than pure
problem difficulty shared across methods. This script tests whether
doubling hidden_channels resolves it.

This does NOT rerun Nv=10/20/30 (those results from the main grid are
already valid and unaffected) or any other method (Ginestet/Dubey/Lovato/
KNN/SVM are untouched by this hidden_channels change). Uses a SEPARATE
checkpoint file so it never collides with or overwrites the main grid's
checkpoint.
"""

import time
import itertools
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import torch

from data_generation.topology_covariance_datagen import generate_data
from methods import gcn

# ----------------------------------------------------------------------
# Config -- matches the main grid's settings for the dimensions we're
# NOT changing, restricted to Nv=40 only.
# ----------------------------------------------------------------------
TOPOLOGIES = ["block", "small_world"]
D_NODES = 40                                  # fixed -- this is the whole point of the rerun
N_SAMPLES_VALUES = [100, 400]
EFFECT_SIZES = [0, 1, 2, 3, 4]
T_TIMEPOINTS = 50
B_REPLICATES = 100
INNER_B = 500                                 # unused here (no permutation test needed for
                                               # a direct power comparison of GCN alone, but
                                               # kept for consistency if you extend this script)
ALPHA = 0.05
HIDDEN_CHANNELS_NEW = 32                      # the one thing being changed
HIDDEN_CHANNELS_OLD = 16                      # for reference / labeling only
GCN_EPOCHS = 150
N_JOBS = 22

CHECKPOINT_PATH = "gcn_hidden32_Nv40_checkpoint.pkl"

completed_conditions = set()
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    pvals = checkpoint["pvals"]  # actually stores GCN predictions' derived accuracy, see below
    completed_conditions = checkpoint["completed_conditions"]
    print(f"Resuming from checkpoint: {len(completed_conditions)} conditions already done.")
else:
    pvals = {
        topo: {n: {e: [] for e in EFFECT_SIZES} for n in N_SAMPLES_VALUES}
        for topo in TOPOLOGIES
    }


# ----------------------------------------------------------------------
# Single-replicate worker -- GCN only, hidden_channels=32
# ----------------------------------------------------------------------
def run_one_replicate(topology, n_samples, effect_size, rep):
    torch.set_num_threads(1)

    data = generate_data(topology=topology, d_nodes=D_NODES, n_samples=n_samples,
                          T=T_TIMEPOINTS, effect_size=effect_size, random_state=rep, test_size=0.5)
    G_all, y = data["G_all"], data["y"]
    idx_train, idx_test = data["idx_train"], data["idx_test"]

    y_pred, y_test = gcn.get_predictions(
        G_all, y, idx_train, idx_test,
        hidden_channels=HIDDEN_CHANNELS_NEW,
        epochs=GCN_EPOCHS, seed=rep
    )

    # Same as the main grid: derive a p-value via the permutation test on
    # classifier predictions, so power is computed identically to the
    # main sweep's GCN results and the two are directly comparable.
    from testing.permutation_test import permutation_test
    _, p_val, _ = permutation_test(y_pred, y_test, B=INNER_B, random_state=rep)

    return rep, p_val


# ----------------------------------------------------------------------
# Sweep -- one (topology, n_samples, effect_size) condition at a time;
# 100 replicates dispatched in parallel across N_JOBS workers.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    total_conditions = len(TOPOLOGIES) * len(N_SAMPLES_VALUES) * len(EFFECT_SIZES)
    condition_idx = 0
    t_sweep_start = time.time()

    for topology, n_samples, effect_size in itertools.product(
            TOPOLOGIES, N_SAMPLES_VALUES, EFFECT_SIZES):
        condition_idx += 1
        condition_key = (topology, n_samples, effect_size)

        if condition_key in completed_conditions:
            print(f"[{condition_idx}/{total_conditions}] {topology}, n={n_samples}, "
                  f"eta={effect_size}: already done (checkpoint), skipping")
            continue

        t0 = time.time()

        rep_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_replicate)(topology, n_samples, effect_size, rep)
            for rep in range(B_REPLICATES)
        )

        for rep, p_val in rep_results:
            pvals[topology][n_samples][effect_size].append(p_val)

        elapsed = time.time() - t0
        print(f"[{condition_idx}/{total_conditions}] {topology}, n={n_samples}, "
              f"eta={effect_size}: {elapsed:.1f}s ({B_REPLICATES} reps, {N_JOBS} workers)")

        completed_conditions.add(condition_key)
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({"pvals": pvals, "completed_conditions": completed_conditions}, f)

    t_sweep_end = time.time()
    print(f"\nSweep done in {t_sweep_end - t_sweep_start:.1f}s "
          f"({(t_sweep_end - t_sweep_start) / 60:.1f} min).\n")


    # ------------------------------------------------------------------
    # Power (point estimate)
    # ------------------------------------------------------------------
    def power(p_list, alpha=ALPHA):
        arr = np.asarray(p_list)
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr < alpha))


    # ------------------------------------------------------------------
    # Comparison figure: hidden_channels=32 (this run) vs. hidden_channels=16
    # (pulled from the main grid's checkpoint, if available) at Nv=40,
    # side by side -- directly answers "did doubling hidden_channels help".
    # ------------------------------------------------------------------
    old_pvals = None
    main_checkpoint_path = "ginestet_sim5_grid_checkpoint.pkl"
    if os.path.exists(main_checkpoint_path):
        with open(main_checkpoint_path, "rb") as f:
            main_checkpoint = pickle.load(f)
        try:
            old_pvals = main_checkpoint["pvals"]["GCN"]
        except KeyError:
            print(f"Warning: could not find GCN results in {main_checkpoint_path}; "
                  f"plotting new results only.")

    topology_labels = {"block": "Block", "small_world": "Small World"}

    fig, axes = plt.subplots(len(TOPOLOGIES), len(N_SAMPLES_VALUES),
                              figsize=(5 * len(N_SAMPLES_VALUES), 4 * len(TOPOLOGIES)),
                              sharex=True, sharey=True, squeeze=False)

    for row, topology in enumerate(TOPOLOGIES):
        for col, n_samples in enumerate(N_SAMPLES_VALUES):
            ax = axes[row][col]

            powers_new = [power(pvals[topology][n_samples][e]) for e in EFFECT_SIZES]
            ax.plot(EFFECT_SIZES, powers_new, marker='o', color='C0',
                     label=f'Hidden channels = {HIDDEN_CHANNELS_NEW}')

            if old_pvals is not None:
                try:
                    powers_old = [power(old_pvals[topology][D_NODES][n_samples][e])
                                  for e in EFFECT_SIZES]
                    ax.plot(EFFECT_SIZES, powers_old, marker='s', linestyle='--', color='C1',
                             label=f'Hidden channels = {HIDDEN_CHANNELS_OLD} (original)')
                except KeyError:
                    pass  # main grid checkpoint doesn't have this condition; skip overlay

            ax.axhline(ALPHA, color='gray', linewidth=0.8)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f'{topology_labels[topology]}, n={n_samples}')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='lower right')

    fig.supxlabel(r'Effect Size ($\eta$)')
    fig.supylabel('Power')
    plt.tight_layout()
    fname = f'gcn_hidden_comparison_Nv{D_NODES}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fname}")
