"""
Ginestet et al. (2017) Section 5 simulation data generator.

Two DOCUMENTED INTERPRETIVE CHOICES were required where the paper's own
text is ambiguous or produces a degenerate result taken literally.

1. S2m formula. The paper states "S2m := C(eta-1)S1m" (Section 5.1).
   Taken literally this gives S2m=0 at eta=1 (degenerate) and a
   NEGATIVE-definite matrix for eta<1 (invalid covariance). We instead
   use:
       S2m := (1 - C*eta) * S1m
   which (a) reduces to S2m=S1m exactly at eta=0 (a proper null), (b)
   stays PSD for all eta in the simulated range [0,4] with C=0.03
   (worst case: 1 - 0.03*4 = 0.88 > 0), and (c) reproduces the
   qualitative shape of the paper's own Fig. 3 (power rising smoothly
   from ~0.05 at eta=0). This uses exactly the same symbols (C, eta, 1,
   S1m) as the paper's stated formula, just recombined -- plausibly a
   typesetting/transcription artifact in the original, though this
   cannot be confirmed with certainty.

2. Covariance -> Laplacian step. Section 5.2's data-generating process
   description (steps i-iii) does not mention any thresholding, unlike
   Section 6's real-data analysis (which explicitly thresholds at
   correlation 0.25). We therefore use each subject's sample covariance
   matrix DIRECTLY as a weighted adjacency matrix (diagonal zeroed) and
   build the Laplacian from that -- no threshold applied.
"""

import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split


C_CONST = 0.03  # paper's stated constant, Section 5.1


def make_block_diagonal_topology(d, rng):
    """A1: two-block structure, within-block edge prob p1=4/d, between-block p2=1/(2d)."""
    d_ceil, d_floor = -(-d // 2), d // 2  # ceil(d/2), floor(d/2)
    p1 = 4 / d
    p2 = 1 / (2 * d)

    A = np.zeros((d, d), dtype=int)
    X = (rng.random((d_ceil, d_ceil)) < p1).astype(int)
    Y = (rng.random((d_floor, d_floor)) < p1).astype(int)
    R = (rng.random((d_ceil, d_floor)) < p2).astype(int)

    A[:d_ceil, :d_ceil] = X
    A[d_ceil:, d_ceil:] = Y
    A[:d_ceil, d_ceil:] = R
    A[d_ceil:, :d_ceil] = R.T

    A = np.triu(A, k=1)
    A = A + A.T
    return A


def make_small_world_topology(d, rng, target_edges=None):
    """A2: Watts-Strogatz ring lattice + rewiring, edge count matched to A1's."""
    if target_edges is None:
        d_ceil, d_floor = -(-d // 2), d // 2
        p1, p2 = 4 / d, 1 / (2 * d)
        target_edges = int(round(
            p1 * d_ceil * (d_ceil - 1) / 2 + p1 * d_floor * (d_floor - 1) / 2 + p2 * d_ceil * d_floor
        ))

    k = max(2, int(round(2 * target_edges / d)))
    if k % 2 != 0:
        k += 1
    k = min(k, d - 1 if (d - 1) % 2 == 0 else d - 2)

    seed = int(rng.integers(0, 2**31 - 1))
    G = nx.watts_strogatz_graph(d, k, p=0.1, seed=seed)
    A = nx.to_numpy_array(G).astype(int)
    return A


def make_S1m(A, rng, lam=4.0, mu1=1.0, mu2=0.0, sigma2=0.2):
    """S1m: diagonal ~ Exp(lam), off-diagonal ~ mixture Normal conditioned on A,
    then projected to nearest PD matrix in Frobenius norm."""
    d = A.shape[0]
    S = np.zeros((d, d))

    diag_vals = rng.exponential(scale=1 / lam, size=d)
    np.fill_diagonal(S, diag_vals)

    sigma = np.sqrt(sigma2)
    for a in range(d):
        for b in range(a + 1, d):
            if A[a, b] == 1:
                val = rng.normal(mu1, sigma)
            else:
                val = rng.normal(mu2, sigma)
            val = abs(val)
            S[a, b] = S[b, a] = val

    return nearest_pd(S)


def nearest_pd(A, eps=1e-8):
    """Nearest PD matrix in Frobenius norm via eigenvalue clipping."""
    A_sym = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def generate_subject_laplacian(S, T, rng):
    """One subject: T i.i.d. draws from N(0, S), sample covariance of the
    resulting time series used directly as a weighted adjacency matrix."""
    d = S.shape[0]
    X = rng.multivariate_normal(mean=np.zeros(d), cov=S, size=T)
    sample_cov = np.cov(X, rowvar=False)

    W = sample_cov.copy()
    np.fill_diagonal(W, 0)
    W = np.abs(W)

    G = nx.from_numpy_array(W)
    return G


def generate_data(topology="block", d_nodes=10, n_samples=100, T=50,
                   effect_size=2, test_size=0.5, random_state=42):
    if topology not in ("block", "small_world"):
        raise ValueError('topology must be "block" or "small_world"')

    rng = np.random.default_rng(random_state)

    A = (make_block_diagonal_topology(d_nodes, rng) if topology == "block"
         else make_small_world_topology(d_nodes, rng))

    S1 = make_S1m(A, rng)
    S2 = (1 - C_CONST * effect_size) * S1

    G_pop1 = [generate_subject_laplacian(S1, T, rng) for _ in range(n_samples)]
    G_pop2 = [generate_subject_laplacian(S2, T, rng) for _ in range(n_samples)]

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
        "S1": S1, "S2": S2,
    }


if __name__ == "__main__":
    for eta in [0, 1, 2, 3, 4]:
        data = generate_data(topology="block", d_nodes=10, n_samples=5,
                              effect_size=eta, random_state=0)
        S1, S2 = data["S1"], data["S2"]
        eigvals_S2 = np.linalg.eigvalsh(S2)
        print(f"eta={eta}: ||S1-S2||_F={np.linalg.norm(S1-S2,'fro'):.4f}, "
              f"min eigenvalue of S2={eigvals_S2.min():.6f} "
              f"({'PD, ok' if eigvals_S2.min() > 0 else 'NOT PD -- problem'})")
