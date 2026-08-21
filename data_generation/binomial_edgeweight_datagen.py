"""
Lovato et al. (2020) Simulation 1 data generator.

Reproduces the location-only, scale-only, and location-and-scale
scenarios from Section 3.1 / Appendix B (Table: "Parameters of the
independent Binomial dist location and scale alternatives").

Each network is a complete graph on d vertices (paper: d=25) with i.i.d.
edge weights drawn from Binomial(n_binom, p), symmetrized, zero diagonal.
mu = n_binom * p, sigma^2 = n_binom * p * (1-p) for a Binomial(n_binom, p).

Group G1 uses (n_binom1, p1); group G2 uses (n_binom2, p2), varied per
scenario/row of the paper's table to achieve the target mu2, sigma2^2.
"""

import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split


# Paper's own parameter table (Simulation 1). Each entry: (n1, n2, p1, p2)
# for the two groups' Binomial(n_binom, p) edge-weight distributions.
# mu = n_binom * p; sigma^2 = n_binom * p * (1 - p).
LOCATION_ONLY = [
    dict(n_binom1=10, p1=0.50000, n_binom2=10, p2=0.50000),
    dict(n_binom1=10, p1=0.50625, n_binom2=10, p2=0.49375),
    dict(n_binom1=10, p1=0.51250, n_binom2=10, p2=0.48750),
    dict(n_binom1=10, p1=0.51875, n_binom2=10, p2=0.48125),
    dict(n_binom1=10, p1=0.52500, n_binom2=10, p2=0.47500),
]

SCALE_ONLY = [
    dict(n_binom1=300, p1=0.20000, n_binom2=300, p2=0.20000),
    dict(n_binom1=300, p1=0.20000, n_binom2=375, p2=0.16000),
    dict(n_binom1=300, p1=0.20000, n_binom2=500, p2=0.12000),
    dict(n_binom1=300, p1=0.20000, n_binom2=750, p2=0.08000),
    dict(n_binom1=300, p1=0.20000, n_binom2=1500, p2=0.04000),
]

LOCATION_AND_SCALE = [
    dict(n_binom1=20, p1=0.10000, n_binom2=20, p2=0.10000),
    dict(n_binom1=20, p1=0.10000, n_binom2=21, p2=0.10000),
    dict(n_binom1=20, p1=0.10000, n_binom2=22, p2=0.10000),
    dict(n_binom1=20, p1=0.10000, n_binom2=23, p2=0.10000),
    dict(n_binom1=20, p1=0.10000, n_binom2=24, p2=0.10000),
]

SCENARIOS = {
    "location_only": LOCATION_ONLY,
    "scale_only": SCALE_ONLY,
    "location_and_scale": LOCATION_AND_SCALE,
}


def generate_complete_graph_binomial(d, n_binom, p, rng):
    """One network: complete graph on d vertices, i.i.d. Binomial(n_binom, p)
    edge weights, symmetrized, zero diagonal."""
    weights = rng.binomial(n_binom, p, size=(d, d)).astype(float)
    weights = np.triu(weights, k=1)
    weights = weights + weights.T  # symmetrize
    np.fill_diagonal(weights, 0)

    G = nx.from_numpy_array(weights)
    return G


def generate_data(scenario="location_and_scale", delta_index=4, n_samples=20,
                   d_nodes=25, test_size=0.5, random_state=42):
    """
    scenario: one of "location_only", "scale_only", "location_and_scale"
      (matches the paper's Table B.1 / Simulation 1 rows).
    delta_index: which row of the scenario's parameter table to use
      (0 = null / smallest effect, up to len(table)-1 = largest effect).
    n_samples: n1 = n2 = n_samples (paper: 10 for location-only,
      300 for scale-only, 20 for location-and-scale -- pass explicitly
      to match, defaults here are for location-and-scale).
    d_nodes: number of vertices per network (paper: 25).

    Returns G_all, y, and a train/test split, matching the structure of
    the BA-graph data_gen.py used elsewhere in this project.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {list(SCENARIOS.keys())}")

    params = SCENARIOS[scenario][delta_index]
    rng = np.random.default_rng(random_state)

    G_pop1 = [generate_complete_graph_binomial(d_nodes, params["n_binom1"], params["p1"], rng)
              for _ in range(n_samples)]
    G_pop2 = [generate_complete_graph_binomial(d_nodes, params["n_binom2"], params["p2"], rng)
              for _ in range(n_samples)]

    G_all = G_pop1 + G_pop2
    y = np.array([0] * n_samples + [1] * n_samples)

    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "G_all": G_all, "y": y,
        "y_train": y_train, "y_test": y_test,
        "idx_train": idx_train, "idx_test": idx_test,
        "params": params,
    }


if __name__ == "__main__":
    # Quick sanity check: confirm realized mu/sigma^2 roughly match the
    # table's stated mu1/mu2/sigma1^2/sigma2^2 for one scenario/row.
    data = generate_data(scenario="location_and_scale", delta_index=4,
                          n_samples=20, d_nodes=25, random_state=0)
    G_all, y = data["G_all"], data["y"]

    G1_weights = np.concatenate([
        nx.to_numpy_array(G)[np.triu_indices(25, k=1)] for G, lab in zip(G_all, y) if lab == 0
    ])
    G2_weights = np.concatenate([
        nx.to_numpy_array(G)[np.triu_indices(25, k=1)] for G, lab in zip(G_all, y) if lab == 1
    ])
    print(f"G1: mean={G1_weights.mean():.3f}, var={G1_weights.var():.3f} "
          f"(target mu1=2.0, sigma1^2=1.8)")
    print(f"G2: mean={G2_weights.mean():.3f}, var={G2_weights.var():.3f} "
          f"(target mu2=2.4, sigma2^2=2.16)")
