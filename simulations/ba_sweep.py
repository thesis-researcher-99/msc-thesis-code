import time
import pickle
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import beta
from joblib import Parallel, delayed
import torch

from generate_barabasi_albert import generate_data
from methods import kernel_svm, gcn, ginestet2017, dubey2019, knn, lovato2020
from permutation_test import permutation_test

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
B_REPLICATES = 100
INNER_B = 500
ALPHA = 0.05
GCN_EPOCHS = 150   # ceiling; early stopping inside gcn.get_predictions handles the rest
N_JOBS = os.cpu_count() or 1        # matches os.cpu_count() on this machine
N_SAMPLES = 450

gamma2_values = np.array([2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5,
                           2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0])

methods_list = ["KNN", "SVM", "GCN", "Ginestet", "Dubey", "Lovato"]

CHECKPOINT_PATH = f"gamma2_sweep_checkpoint_n{N_SAMPLES}.pkl"

# ----------------------------------------------------------------------
# Resume from checkpoint if one exists
# ----------------------------------------------------------------------
completed_gamma2 = set()
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)

    # Safety check: confirm this checkpoint was actually computed with the
    # N_SAMPLES currently configured, not just that the filename matches.
    checkpoint_n_samples = checkpoint.get("n_samples")
    if checkpoint_n_samples != N_SAMPLES:
        raise ValueError(
            f"Checkpoint {CHECKPOINT_PATH} was computed with n_samples="
            f"{checkpoint_n_samples}, but current config has N_SAMPLES={N_SAMPLES}. "
            f"Refusing to resume from a mismatched checkpoint -- delete or rename "
            f"the checkpoint file if you intend to start a fresh run."
        )

    pvals = checkpoint["pvals"]
    completed_gamma2 = checkpoint["completed_gamma2"]
    print(f"Resuming from checkpoint: {len(completed_gamma2)}/{len(gamma2_values)} "
          f"gamma2 values already done (n_samples={checkpoint_n_samples}).")

    for m in methods_list:
        pvals.setdefault(m, {})
        for g in gamma2_values:
            pvals[m].setdefault(g, [])
else:
    pvals = {m: {g: [] for g in gamma2_values} for m in methods_list}


# ----------------------------------------------------------------------
# Shared, label-independent precomputation (done ONCE per rep, reused by
# every method).
# ----------------------------------------------------------------------
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
# Single-replicate worker -- runs ALL SIX methods for one (gamma2, rep)
# combination. Fully self-contained so joblib can dispatch it to any
# worker process independently.
# ----------------------------------------------------------------------
def run_one_replicate(gamma2, rep):
    torch.set_num_threads(1)

    data = generate_data(n_samples=N_SAMPLES, n_nodes=10, random_state=rep,
                          gamma1=2.5, gamma2=gamma2, m=2, test_size=0.5)
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
# Sweep -- one gamma2 value at a time; within each, all B_REPLICATES reps
# dispatched in parallel across N_JOBS workers. Checkpoint saved after
# every gamma2 value.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    t_sweep_start = time.time()

    for g_idx, gamma2 in enumerate(gamma2_values):
        if gamma2 in completed_gamma2:
            print(f"[{g_idx+1}/{len(gamma2_values)}] gamma2={gamma2}: "
                  f"already done (checkpoint), skipping")
            continue

        t0 = time.time()

        rep_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_replicate)(gamma2, rep) for rep in range(B_REPLICATES)
        )

        for rep, result in rep_results:
            for method, p_val in result.items():
                pvals[method][gamma2].append(p_val)

        elapsed = time.time() - t0
        print(f"[{g_idx+1}/{len(gamma2_values)}] gamma2={gamma2}: {elapsed:.1f}s "
              f"({B_REPLICATES} reps, {N_JOBS} workers)")

        # Save progress after EVERY gamma2 value -- if this crashes or gets
        # interrupted, rerunning the script picks up from here instead of
        # starting over. Partial (within-gamma2) progress is NOT saved; an
        # interrupted gamma2 value reruns from rep 0 next time.
        completed_gamma2.add(gamma2)
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({"pvals": pvals, "completed_gamma2": completed_gamma2,
                         "n_samples": N_SAMPLES}, f)

    t_sweep_end = time.time()
    print(f"\nTotal sweep time: {t_sweep_end - t_sweep_start:.1f} seconds "
          f"({(t_sweep_end - t_sweep_start) / 60:.2f} minutes)")

    # ------------------------------------------------------------------
    # Power + Clopper-Pearson confidence interval per (method, gamma2)
    # ------------------------------------------------------------------
    def power_and_ci(p_list, alpha=ALPHA, ci_alpha=0.05):
        arr = np.asarray(p_list)
        k = int(np.sum(arr < alpha))
        n = len(arr)

        if n == 0:
            return 0.0, 0.0, 0.0

        p_hat = k / n
        ci_low = 0.0 if k == 0 else beta.ppf(ci_alpha / 2, k, n - k + 1)
        ci_high = 1.0 if k == n else beta.ppf(1 - ci_alpha / 2, k + 1, n - k)

        return p_hat, ci_low, ci_high

    # ------------------------------------------------------------------
    # Combined plot (point estimates only)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for m in methods_list:
        power = [power_and_ci(pvals[m][g])[0] for g in gamma2_values]
        ax1.plot(gamma2_values, power, marker='o', label=m)
    ax1.axhline(ALPHA, color='r', linestyle='--', alpha=0.5, label=f'alpha={ALPHA}')
    ax1.set_xlabel('gamma2')
    ax1.set_ylabel('Power (rejection rate)')
    ax1.set_title(f'Power vs gamma2 (B_REPLICATES={B_REPLICATES}, n_samples={N_SAMPLES}, point estimates)')
    ax1.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'power_curve_combined_n{N_SAMPLES}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Faceted plot with Clopper-Pearson CI bands
    # ------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, methods_list):
        p_hats, los, his = [], [], []
        for g in gamma2_values:
            p_hat, lo, hi = power_and_ci(pvals[m][g])
            p_hats.append(p_hat)
            los.append(lo)
            his.append(hi)
        p_hats, los, his = np.array(p_hats), np.array(los), np.array(his)

        ax.plot(gamma2_values, p_hats, marker='o', color='C0')
        ax.fill_between(gamma2_values, los, his, alpha=0.25, color='C0')
        ax.set_title(m)
        ax.set_ylim(-0.02, 1.02)

    fig2.supxlabel('gamma2')
    fig2.supylabel('Power (rejection rate)')
    fig2.suptitle(f'Power vs gamma2 with 95% Clopper-Pearson CI '
                  f'(B_REPLICATES={B_REPLICATES}, n_samples={N_SAMPLES})')
    plt.tight_layout()
    plt.savefig(f'power_curve_faceted_ci_n{N_SAMPLES}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"Saved: power_curve_combined_n{N_SAMPLES}.png, "
          f"power_curve_faceted_ci_n{N_SAMPLES}.png")
