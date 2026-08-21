"""
Diagnostic: is the BA-graph data-generating process itself asymmetric
around gamma=2.5, independent of any test statistic?

For each g in the gamma2 grid, generates a fresh batch of gamma=2.5
graphs and a batch of gamma=g graphs, and computes the mean pairwise
Frobenius distance between their Laplacians (same distance every method
in this project ultimately builds on). If this distance-vs-g curve is
itself asymmetric around g=2.5 -- steeper on the g<2.5 side than the
g>2.5 side -- that explains the shared left-right power asymmetry seen
across all six methods, as a property of generate_ba_with_attractiveness's
A = m*(gamma-3) reparametrization (which hits a validity-boundary
clipping regime near gamma=2, but reduces cleanly to plain BA at
gamma=3), rather than anything about the test statistics themselves.

Only touches data generation -- doesn't run any of the six methods.

Random number generation: ONE numpy.random.Generator is seeded per
replicate (rep) and used sequentially for BOTH the gamma1 reference
batch and every gamma2 comparison batch within that replicate. This
avoids the per-graph seed-arithmetic pattern (seed=1000*rep+i) found
elsewhere in this project to introduce spurious, batch-size-dependent
correlations between nominally independent draws.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from data_gen import generate_ba_with_attractiveness

N_SAMPLES = 100          # graphs per group, per g
N_NODES = 10
M = 2
GAMMA1 = 2.5
GAMMA2_VALUES = np.round(np.arange(2.0, 3.01, 0.1), 2)
N_REPS = 20              # repeat the whole comparison this many times,
                          # for a mean +/- spread rather than one noisy estimate


def mean_pairwise_frobenius(laplacians_a, laplacians_b):
    """Mean Frobenius distance between every graph in group a and every
    graph in group b (cross-distances only, not within-group)."""
    dists = []
    for La in laplacians_a:
        for Lb in laplacians_b:
            dists.append(np.linalg.norm(La - Lb, 'fro'))
    return np.mean(dists)


if __name__ == "__main__":
    results = {g: [] for g in GAMMA2_VALUES}

    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)  # one stream for this whole replicate

        # Fresh gamma=2.5 reference batch, drawn first from this rep's stream.
        G_ref = [generate_ba_with_attractiveness(N_NODES, M, GAMMA1, rng=rng)
                 for _ in range(N_SAMPLES)]
        L_ref = [nx.laplacian_matrix(G).toarray() for G in G_ref]

        for g in GAMMA2_VALUES:
            # Continues drawing from the SAME stream -- no seed arithmetic,
            # no dependence on N_SAMPLES or number of gamma2 values swept.
            G_g = [generate_ba_with_attractiveness(N_NODES, M, g, rng=rng)
                   for _ in range(N_SAMPLES)]
            L_g = [nx.laplacian_matrix(G).toarray() for G in G_g]

            d = mean_pairwise_frobenius(L_ref, L_g)
            results[g].append(d)

        print(f"rep {rep} done")

    means = np.array([np.mean(results[g]) for g in GAMMA2_VALUES])
    stds = np.array([np.std(results[g]) for g in GAMMA2_VALUES])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(GAMMA2_VALUES, means, marker='o', color='C2')
    ax.fill_between(GAMMA2_VALUES, means - stds, means + stds, alpha=0.2, color='C2')
    ax.axvline(GAMMA1, color='gray', linestyle=':', alpha=0.7, label=f'$\gamma_1$={GAMMA1} (null)')
    ax.set_xlabel('$\gamma_2$')
    ax.set_ylabel('Mean pairwise Frobenius distance to $\gamma_1$=2.5 graphs')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("ba_distance_asymmetry_check.png", dpi=300, bbox_inches="tight")
    print("Saved: ba_distance_asymmetry_check.png")

    print("\n=== Symmetric-offset comparison ===")
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5]:
        g_low = round(GAMMA1 - offset, 2)
        g_high = round(GAMMA1 + offset, 2)
        if g_low in results and g_high in results:
            d_low = np.mean(results[g_low])
            d_high = np.mean(results[g_high])
            print(f"  offset={offset}: dist(2.5, {g_low})={d_low:.3f}  "
                  f"dist(2.5, {g_high})={d_high:.3f}  "
                  f"ratio(low/high)={d_low/d_high:.3f}")
          
