"""
Reproduces the STRUCTURE of Ginestet et al. (2017) Fig. 3 -- topology as
rows, network size (d_nodes) as columns, sample size (n) as separate
lines within each panel, power vs effect size (eta) on each axis -- but
for all SIX methods in this project, not just Ginestet's own test.
Produces one such grid figure per method.

COST WARNING: the paper's own full grid is
  2 topologies x 4 d_nodes x 4 n_samples x 5 effect_sizes = 160 conditions
each requiring B_REPLICATES runs across all 6 methods. Based on this
project's own timing data (~275s for a SINGLE (topology, d_nodes=10,
n_samples=100, one effect_size) condition across all 6 methods, at
B_REPLICATES=100, SEQUENTIALLY), the full grid is a many-hours-to-multi-day
job if run sequentially.

PARALLELIZATION: the 100 replicates within each condition are independent,
so they're dispatched across N_JOBS worker processes via joblib. This does
NOT change the checkpointing granularity -- checkpoints are still saved
per CONDITION (all 100 reps done), not per replicate, since that's the
natural unit of "resumable work" here (a condition either finished or it
didn't; partial-condition results aren't saved to keep the checkpoint
logic simple).

This script STILL DEFAULTS TO A SMALL VALIDATION SUBSET (D_NODES_VALUES,
N_SAMPLES_VALUES below) rather than the full paper grid. Run this subset
first, check the printed per-condition timings (now with parallel
speedup), and only widen D_NODES_VALUES / N_SAMPLES_VALUES to the full
{10,20,30,40} / {100,200,300,400} once you have a real per-condition time
to extrapolate from -- do NOT just uncomment the full grid and walk away.
"""

import time
import itertools
import pickle
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import beta
from joblib import Parallel, delayed
import torch

from generate_multi_topology import generate_data
from methods import kernel_svm, gcn, ginestet2017, dubey2019, knn, lovato2020
from permutation_test import permutation_test

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TOPOLOGIES = ["block", "small_world"]
D_NODES_VALUES = [10,20,30,40]                   # extremes only -- was [10,20,30,40]
N_SAMPLES_VALUES = [100, 200, 300, 400]      # extremes only -- was [100,200,300,400]
EFFECT_SIZES = [0, 1, 2, 3, 4]
T_TIMEPOINTS = 50
B_REPLICATES = 100
INNER_B = 500
ALPHA = 0.05
GCN_EPOCHS = 150   # ceiling; early stopping inside gcn.get_predictions handles the rest
N_JOBS = os.cpu_count() or 1        # matches os.cpu_count() on this machine

CHECKPOINT_PATH = "ginestet_sim5_grid_checkpoint.pkl"

methods_list = ["KNN", "SVM", "GCN", "Ginestet", "Dubey", "Lovato"]

# pvals[method][topology][d_nodes][n_samples][effect_size] -> list of p-values
# Resume from a checkpoint if one exists, instead of starting from scratch.
completed_conditions = set()
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    pvals = checkpoint["pvals"]
    completed_conditions = checkpoint["completed_conditions"]
    print(f"Resuming from checkpoint: {len(completed_conditions)} conditions already done.")

    for m in methods_list:
        pvals.setdefault(m, {})
        for topo in TOPOLOGIES:
            pvals[m].setdefault(topo, {})
            for d in D_NODES_VALUES:
                pvals[m][topo].setdefault(d, {})
                for n in N_SAMPLES_VALUES:
                    pvals[m][topo][d].setdefault(n, {})
                    for e in EFFECT_SIZES:
                        pvals[m][topo][d][n].setdefault(e, [])
else:
    pvals = {
        m: {
            topo: {d: {n: {e: [] for e in EFFECT_SIZES} for n in N_SAMPLES_VALUES}
                   for d in D_NODES_VALUES}
            for topo in TOPOLOGIES
        }
        for m in methods_list
    }


def prepare_shared(G_all):
    laplacians = [nx.laplacian_matrix(G).toarray() for G in G_all]
    n = len(laplacians)

    D_sq = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = np.sum((laplacians[i] - laplacians[j]) ** 2)
            D_sq[i, j] = D_sq[j, i] = d_sq

    D_matrix = np.sqrt(D_sq)

    nonzero = D_sq[D_sq > 0]
    median_sq_dist = np.median(nonzero) if len(nonzero) > 0 else 1.0
    gamma = 1.0 / median_sq_dist
    K_matrix = np.exp(-gamma * D_sq)

    return laplacians, D_sq, D_matrix, K_matrix


# ----------------------------------------------------------------------
# Single-replicate worker -- runs ALL SIX methods for one rep within a
# given (topology, d_nodes, n_samples, effect_size) condition. Fully
# self-contained so joblib can dispatch it to any worker process.
# ----------------------------------------------------------------------
def run_one_replicate(topology, d_nodes, n_samples, effect_size, rep):
    torch.set_num_threads(1)  # avoid thread contention across the 22 worker processes

    data = generate_data(topology=topology, d_nodes=d_nodes, n_samples=n_samples,
                          T=T_TIMEPOINTS, effect_size=effect_size, random_state=rep)
    G_all, y = data["G_all"], data["y"]
    idx_train, idx_test = data["idx_train"], data["idx_test"]

    laplacians, D_sq, D_matrix, K_matrix = prepare_shared(G_all)

    result = {}

    y_pred, y_test = knn.get_predictions(D_matrix, y, idx_train, idx_test)
    _, p_val, _ = permutation_test(y_pred, y_test, B=INNER_B, random_state=rep)
    result["KNN"] = p_val

    y_pred, y_test = kernel_svm.get_predictions(K_matrix, y, idx_train, idx_test)
    _, p_val, _ = permutation_test(y_pred, y_test, B=INNER_B, random_state=rep)
    result["SVM"] = p_val

    y_pred, y_test = gcn.get_predictions(G_all, y, idx_train, idx_test,
                                          epochs=GCN_EPOCHS, seed=rep)
    _, p_val, _ = permutation_test(y_pred, y_test, B=INNER_B, random_state=rep)
    result["GCN"] = p_val

    result["Ginestet"] = ginestet2017.run_test(G_all, y, laplacians=laplacians)["p_value"]
    result["Dubey"] = dubey2019.run_test(G_all, y, laplacians=laplacians)["p_value"]
    result["Lovato"] = lovato2020.run_test(G_all, y, B=INNER_B, random_state=rep,
                                            laplacians=laplacians)["p_combined"]

    return rep, result


# ----------------------------------------------------------------------
# Sweep -- one condition at a time; within each condition, all
# B_REPLICATES reps are dispatched in parallel across N_JOBS workers.
# Checkpoint saved after every CONDITION (not every replicate).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    total_conditions = len(TOPOLOGIES) * len(D_NODES_VALUES) * len(N_SAMPLES_VALUES) * len(EFFECT_SIZES)
    condition_idx = 0
    t_sweep_start = time.time()

    for topology, d_nodes, n_samples, effect_size in itertools.product(
            TOPOLOGIES, D_NODES_VALUES, N_SAMPLES_VALUES, EFFECT_SIZES):
        condition_idx += 1
        condition_key = (topology, d_nodes, n_samples, effect_size)

        if condition_key in completed_conditions:
            print(f"[{condition_idx}/{total_conditions}] {topology}, d={d_nodes}, n={n_samples}, "
                  f"eta={effect_size}: already done (checkpoint), skipping")
            continue

        t0 = time.time()

        rep_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_replicate)(topology, d_nodes, n_samples, effect_size, rep)
            for rep in range(B_REPLICATES)
        )

        for rep, result in rep_results:
            for method, p_val in result.items():
                pvals[method][topology][d_nodes][n_samples][effect_size].append(p_val)

        elapsed = time.time() - t0
        print(f"[{condition_idx}/{total_conditions}] {topology}, d={d_nodes}, n={n_samples}, "
              f"eta={effect_size}: {elapsed:.1f}s ({B_REPLICATES} reps, {N_JOBS} workers)")

        # Save progress after EVERY condition -- if this crashes or gets
        # interrupted, rerunning the script picks up from here instead of
        # starting over. Partial (within-condition) progress is NOT saved;
        # an interrupted condition reruns from rep 0 next time.
        completed_conditions.add(condition_key)
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({"pvals": pvals, "completed_conditions": completed_conditions}, f)

    t_sweep_end = time.time()
    print(f"\nSweep done in {t_sweep_end - t_sweep_start:.1f}s "
          f"({(t_sweep_end - t_sweep_start) / 60:.1f} min). "
          f"Generating paper-style grid figures (one per method)...\n")


    # ------------------------------------------------------------------
    # Power (point estimate only, matching the paper's own Fig. 3 style)
    # ------------------------------------------------------------------
    def power(p_list, alpha=ALPHA):
        arr = np.asarray(p_list)
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr < alpha))


    # ------------------------------------------------------------------
    # One grid figure per method: rows = topology, columns = d_nodes,
    # lines within each panel = n_samples -- matches Fig. 3's layout.
    # ------------------------------------------------------------------
    line_styles = ['-', '--', ':', '-.']

    for m in methods_list:
        fig, axes = plt.subplots(len(TOPOLOGIES), len(D_NODES_VALUES),
                                  figsize=(4 * len(D_NODES_VALUES), 3.5 * len(TOPOLOGIES)),
                                  sharex=True, sharey=True, squeeze=False)

        for row, topology in enumerate(TOPOLOGIES):
            for col, d_nodes in enumerate(D_NODES_VALUES):
                ax = axes[row][col]
                for n_idx, n_samples in enumerate(N_SAMPLES_VALUES):
                    powers = [power(pvals[m][topology][d_nodes][n_samples][e]) for e in EFFECT_SIZES]
                    ax.plot(EFFECT_SIZES, powers,
                            linestyle=line_styles[n_idx % len(line_styles)],
                            marker='o', markersize=3, color='black',
                            label=f'n={n_samples}')
                ax.axhline(ALPHA, color='gray', linewidth=0.8)
                ax.set_ylim(-0.02, 1.02)
                if row == 0:
                    ax.set_title(f'Nv={d_nodes}')
                if col == 0:
                    ax.set_ylabel(f'{topology}\nPower')

        axes[0][-1].legend(fontsize=8, loc='lower right')
        fig.supxlabel('Effect Size (eta)')
        fig.suptitle(f'{m}: Power Curves (Ginestet Simulation 5 structure, B_REPLICATES={B_REPLICATES})')
        plt.tight_layout()
        fname = f'ginestet_sim5_grid_{m}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fname}")

    print("\nDone.")
