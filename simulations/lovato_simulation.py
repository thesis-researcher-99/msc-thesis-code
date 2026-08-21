"""
Runs all six methods (KNN, SVM, GCN, Ginestet, Dubey, Lovato) against
Lovato et al. (2020)'s own Simulation 1 data-generating process, using
lovato_data_gen.py instead of the Barabasi-Albert setup used elsewhere
in this project.

Mirrors the structure of gamma2_sweep.py: checkpointed, parallelized
across replicates via joblib, resumable.

Fixes relative to the first draft of this script:

1. N_SAMPLES was previously read off the paper's parameter table
   (n_binom1 -- the per-edge Binomial trial count, e.g. 10 for
   location-only, 300 for scale-only), not the actual number-of-networks
   sample size. The paper's Section 3.1 states n1 = n2 = 20 uniformly
   across ALL THREE scenarios (with a second, unbalanced run at
   n1=30, n2=10) -- it is NOT scenario-dependent. lovato_data_gen.py
   itself already handles this correctly (n_samples is a separate,
   explicit argument); only this driver script's constant was wrong.

2. KNN/SVM previously used a WL kernel with node label = degree.
   Lovato's Simulation 1 networks are COMPLETE graphs (every edge
   exists) with i.i.d. Binomial edge weights -- so every node has
   identical unweighted degree (d_nodes - 1) by construction, and a
   degree-based representation is structurally blind to the entire
   between-group signal, which lives purely in edge weight magnitude.
   This is the same failure mode diagnosed for the Ginestet/
   multi-topology reproduction (Section 6.2) -- switched here to
   Frobenius distance on graph Laplacians, matching the representation
   used by Ginestet/Dubey/Lovato and by KNN/SVM in the main BA sweep.

3. INNER_B was applied uniformly at 500 to all six methods, including
   Lovato's own test. Section 5.1 of the thesis states classifier
   permutations use B=500 while Lovato's test uses B=1000 "matching
   the value used in Lovato et al.'s own simulation studies" -- split
   into CLASSIFIER_INNER_B and LOVATO_INNER_B to match.
"""

import time
import pickle
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import beta
from joblib import Parallel, delayed
import torch

from lovato_data_gen import generate_data, SCENARIOS
from methods import kernel_svm, gcn, ginestet2017, dubey2019, knn, lovato2020
from permutation_test import permutation_test

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
SCENARIO = "location_and_scale"   # "location_only", "scale_only", "location_and_scale" paper: 20 for location_and_scale, 10 for location_only, 300 for scale_only -- match to whichever SCENARIO you choose
N_SAMPLES = 20                     # paper (Section 3.1): n1 = n2 = 20, uniform
                                    # across ALL scenarios -- NOT read from the
                                    # parameter table (see module docstring, fix 1)
D_NODES = 25                       # paper's own network size

B_REPLICATES = 100
INNER_B = 500           # matches thesis Section 5.1 (KNN/SVM/GCN)

GCN_EPOCHS = 150          # ceiling; early stopping inside gcn.get_predictions
                           # (patience=15, tol=1e-4, both defaults) handles the
                           # rest, matching the BA sweep's GCN configuration
ALPHA = 0.05
N_JOBS = 16                        # match to actual core count

DELTA_INDICES = list(range(len(SCENARIOS[SCENARIO])))  # 0..4, per the paper's table
methods_list = ["KNN", "SVM", "GCN", "Ginestet", "Dubey", "Lovato"]

CHECKPOINT_PATH = f"lovato_sim1_checkpoint_{SCENARIO}_n{N_SAMPLES}.pkl"

# ----------------------------------------------------------------------
# Resume from checkpoint if one exists
# ----------------------------------------------------------------------
completed_delta = set()
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)

    ckpt_scenario = checkpoint.get("scenario")
    ckpt_n_samples = checkpoint.get("n_samples")
    if ckpt_scenario != SCENARIO or ckpt_n_samples != N_SAMPLES:
        raise ValueError(
            f"Checkpoint {CHECKPOINT_PATH} was computed with scenario="
            f"{ckpt_scenario}, n_samples={ckpt_n_samples}, but current config "
            f"has SCENARIO={SCENARIO}, N_SAMPLES={N_SAMPLES}. Refusing to resume "
            f"from a mismatched checkpoint -- delete or rename it if you intend "
            f"to start a fresh run."
        )

    pvals = checkpoint["pvals"]
    completed_delta = checkpoint["completed_delta"]
    print(f"Resuming from checkpoint: {len(completed_delta)}/{len(DELTA_INDICES)} "
          f"delta_index values already done (scenario={ckpt_scenario}, "
          f"n_samples={ckpt_n_samples}).")

    for m in methods_list:
        pvals.setdefault(m, {})
        for d in DELTA_INDICES:
            pvals[m].setdefault(d, [])
else:
    pvals = {m: {d: [] for d in DELTA_INDICES} for m in methods_list}


# ----------------------------------------------------------------------
# Shared, label-independent precomputation (done ONCE per rep, reused by
# every method). Frobenius distance on Laplacians -- see fix 2 above.
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
# Single-replicate worker -- runs ALL SIX methods for one
# (delta_index, rep) combination. Self-contained for joblib dispatch.
# ----------------------------------------------------------------------
def run_one_replicate(delta_index, rep):
    torch.set_num_threads(1)

    data = generate_data(scenario=SCENARIO, delta_index=delta_index,
                          n_samples=N_SAMPLES, d_nodes=D_NODES, random_state=rep)
    G_all, y = data["G_all"], data["y"]
    idx_train, idx_test = data["idx_train"], data["idx_test"]

    laplacians, D_sq, D_matrix, K_matrix = prepare_shared(G_all)

    result = {}

    y_pred, y_test = knn_wl.get_predictions(D_matrix, y, idx_train, idx_test)
    _, p_val, _ = permutation_test(y_pred, y_test, B=INNER_B, random_state=rep)
    result["KNN"] = p_val

    y_pred, y_test = kernel_svm_wl.get_predictions(K_matrix, y, idx_train, idx_test)
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
# Sweep -- one delta_index at a time; within each, all B_REPLICATES reps
# dispatched in parallel across N_JOBS workers. Checkpoint saved after
# every delta_index.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    t_sweep_start = time.time()

    for d_idx, delta_index in enumerate(DELTA_INDICES):
        if delta_index in completed_delta:
            print(f"[{d_idx+1}/{len(DELTA_INDICES)}] delta_index={delta_index}: "
                  f"already done (checkpoint), skipping")
            continue

        t0 = time.time()

        rep_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_replicate)(delta_index, rep) for rep in range(B_REPLICATES)
        )

        for rep, result in rep_results:
            for method, p_val in result.items():
                pvals[method][delta_index].append(p_val)

        elapsed = time.time() - t0
        print(f"[{d_idx+1}/{len(DELTA_INDICES)}] delta_index={delta_index}: {elapsed:.1f}s "
              f"({B_REPLICATES} reps, {N_JOBS} workers)")

        # Save progress after EVERY delta_index -- if this crashes or gets
        # interrupted, rerunning picks up from here instead of starting
        # over. Partial (within-delta_index) progress is NOT saved.
        completed_delta.add(delta_index)
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({"pvals": pvals, "completed_delta": completed_delta,
                         "scenario": SCENARIO, "n_samples": N_SAMPLES}, f)

    t_sweep_end = time.time()
    print(f"\nTotal sweep time: {t_sweep_end - t_sweep_start:.1f} seconds "
          f"({(t_sweep_end - t_sweep_start) / 60:.2f} minutes)")

    # ------------------------------------------------------------------
    # Power + Clopper-Pearson confidence interval per (method, delta_index)
    # (module docstring in the original draft said "Wilson-CI"; every
    # other script in this project -- and thesis Section 3.5 -- uses
    # Clopper-Pearson via beta.ppf, so kept consistent with that here)
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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, methods_list):
        p_hats, los, his = [], [], []
        for d in DELTA_INDICES:
            p_hat, lo, hi = power_and_ci(pvals[m][d])
            p_hats.append(p_hat); los.append(lo); his.append(hi)
        p_hats, los, his = np.array(p_hats), np.array(los), np.array(his)

        ax.plot(DELTA_INDICES, p_hats, marker='o', color='C0')
        ax.fill_between(DELTA_INDICES, los, his, alpha=0.25, color='C0')
        ax.axhline(ALPHA, color='r', linestyle='--', alpha=0.5)
        ax.set_title(m)
        ax.set_ylim(-0.02, 1.02)

    fig.supxlabel(rf'$\Delta$ index (0=null, {len(DELTA_INDICES)-1}=largest effect)')
    fig.supylabel('Power (rejection rate)')
    plt.tight_layout()
    plt.savefig(f'lovato_sim1_{SCENARIO}_all_methods.png', dpi=300, bbox_inches='tight')
    print(f"Saved: lovato_sim1_{SCENARIO}_all_methods.png")
