import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split


def generate_ba_with_attractiveness(n, m, gamma, rng):
    """
    Barabasi-Albert-style growth with initial attractiveness A (Dorogovtsev,
    Mendes & Samukhin, 2000), giving a tunable power-law exponent gamma.

    gamma = 3 + A/m  =>  A = m * (gamma - 3)

    Valid for gamma in (2, 3): A must satisfy -m < A <= 0 for this range.

    rng: a numpy.random.Generator instance (NOT an integer seed). Draws
    are made sequentially from this shared stream -- callers should pass
    the SAME rng object across all graphs generated within one replicate,
    seeding it ONCE per replicate (see generate_data) rather than deriving
    a fresh seed per graph. Per-graph seed arithmetic (e.g. seed=base+i)
    does not guarantee statistical independence between the resulting
    graphs, and can introduce spurious dependence on n_samples if the
    seed range used depends on it -- both diagnosed as real issues in
    this project's earlier implementation.
    """
    A = m * (gamma - 3)

    # Seed graph: m+1 fully connected nodes, each starts with degree m
    G = nx.complete_graph(m + 1)
    degrees = dict(G.degree())

    for new_node in range(m + 1, n):
        candidates = list(G.nodes())
        weights = np.array([degrees[v] + A for v in candidates], dtype=float)
        weights = np.clip(weights, 1e-6, None)  # guard against numerical edge cases
        probs = weights / weights.sum()

        targets = rng.choice(candidates, size=m, replace=False, p=probs)

        G.add_node(new_node)
        degrees[new_node] = 0
        for t in targets:
            G.add_edge(new_node, t)
            degrees[t] += 1
            degrees[new_node] += 1

    return G


def generate_data(n_samples=100, n_nodes=10, test_size=0.5, random_state=42,
                   gamma1=2.5, gamma2=2.5, m=2):
    """
    Reproduces Dubey & Muller (2019)'s Barabasi-Albert simulation setting:
    scale-free networks with tunable power-law exponent gamma, Frobenius
    metric on graph Laplacians. gamma2 swept in [2, 3] reproduces their
    power curve.

    Random number generation: a single numpy.random.Generator is seeded
    once from random_state and used to draw ALL 2*n_samples graphs
    (both groups) sequentially. This guarantees:
      - Reproducibility: a fixed random_state always produces the exact
        same dataset.
      - Independence across replicates: different random_state values
        produce statistically independent datasets.
      - Invariance to n_samples: the sequence of graphs drawn for a given
        random_state does not depend on n_samples, so changing n_samples
        does not silently change the effective population being sampled
        from (as it did under the previous seed=random_state+i scheme).
    """
    rng = np.random.default_rng(random_state)

    X_pop0, X_pop1 = [], []
    G_pop0, G_pop1 = [], []

    for _ in range(n_samples):
        G = generate_ba_with_attractiveness(n_nodes, m, gamma1, rng=rng)
        L = nx.laplacian_matrix(G).toarray()
        evals = np.linalg.eigvalsh(L)
        X_pop0.append(evals)
        G_pop0.append(G)

    for _ in range(n_samples):
        G = generate_ba_with_attractiveness(n_nodes, m, gamma2, rng=rng)
        L = nx.laplacian_matrix(G).toarray()
        evals = np.linalg.eigvalsh(L)
        X_pop1.append(evals)
        G_pop1.append(G)

    X = np.vstack([np.array(X_pop0), np.array(X_pop1)])
    G_all = G_pop0 + G_pop1
    y = np.array([0] * n_samples + [1] * n_samples)

    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = X[idx_train], X[idx_test]

    return {
        "G_all": G_all, "X": X, "y": y,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "idx_train": idx_train, "idx_test": idx_test
    }
