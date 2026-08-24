"""
Ginestet et al. (2017) -- "Hypothesis Testing For Network Data In
Functional Neuroimaging" (Annals of Applied Statistics).
 
Two-sample Wald-type T2 test on graph Laplacians: each group's Laplacians
are half-vectorized (upper-triangular, off-diagonal entries) into R^p,
p = d(d-1)/2; group means are compared via a Hotelling-type quadratic
form, using a pooled covariance estimate (Ledoit-Wolf shrinkage, then
projected to the nearest positive-definite matrix via nearest_pd()) for
numerical stability at high dimension relative to sample size. Under H0
(equal population mean networks), T2 is asymptotically chi^2 with p
degrees of freedom.
 
Entry point: run_test(G_all, y, shrinkage=True, laplacians=None) ->
    {"statistic": T2, "p_value": ..., "dof": p}
Full-sample hypothesis test -- no train/test split.
"""
import networkx as nx
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf

def nearest_pd(A, eps=1e-8):
    """Project a symmetric matrix to the nearest positive-definite matrix
    by clipping negative/near-zero eigenvalues."""
    A_sym = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    eigvals_clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def network_two_sample_test(L1_list, L2_list, shrinkage=True):
    n1, n2 = len(L1_list), len(L2_list)
    n = n1 + n2
    d = L1_list[0].shape[0]

    def half_vec(L):
        triu_indices = np.triu_indices(d, k=1)
        return L[triu_indices]

    phi_L1 = np.array([half_vec(L) for L in L1_list])
    phi_L2 = np.array([half_vec(L) for L in L2_list])

    mean_L1 = np.mean(phi_L1, axis=0)
    mean_L2 = np.mean(phi_L2, axis=0)

    if shrinkage:
        cov1 = LedoitWolf().fit(phi_L1).covariance_
        cov2 = LedoitWolf().fit(phi_L2).covariance_
    else:
        cov1 = np.cov(phi_L1, rowvar=False)
        cov2 = np.cov(phi_L2, rowvar=False)

    pooled_cov = (n1 * cov1 + n2 * cov2) / (n - 2)
    pooled_cov = nearest_pd(pooled_cov)

    inv_cov = np.linalg.inv(pooled_cov)

    mean_diff = mean_L1 - mean_L2
    T2_obs = (n1 * n2 / n) * (mean_diff.T @ inv_cov @ mean_diff)

    p = int(d * (d - 1) / 2)
    p_value = chi2.sf(T2_obs, df=p)

    return T2_obs, p_value, p


def run_test(G_all, y, shrinkage=True, laplacians=None):
    """
    Runs the Ginestet et al. (2017) T2 test on the FULL dataset
    (no train/test split — this is a hypothesis test, not a classifier).
 
    laplacians: optional list of precomputed Laplacian arrays, in the same
    order as G_all. If provided, skips recomputing them here -- pass this
    in when the caller (e.g. a sweep script) has already built the
    Laplacians once for shared reuse across methods, to avoid redundant
    recomputation. If None (default), computes them internally as before --
    fully backward compatible with existing standalone calls.
    """
    if laplacians is None:
        laplacians = [nx.laplacian_matrix(G).toarray() for G in G_all]

    L1_list = [L for L, label in zip(laplacians, y) if label == 0]
    L2_list = [L for L, label in zip(laplacians, y) if label == 1]

    T2_obs, p_value, dof = network_two_sample_test(L1_list, L2_list, shrinkage=shrinkage)
    return {"statistic": T2_obs, "p_value": p_value, "dof": dof}
