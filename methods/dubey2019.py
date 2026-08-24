import numpy as np
import networkx as nx
from scipy.stats import chi2


def frechet_mean_and_sq_dists(L_list):
    """Euclidean (Frobenius) Frechet mean of a list of Laplacians,
    plus the squared distances of each point to that mean."""
    L_stack = np.stack(L_list)  # (n_j, d, d)
    mu = np.mean(L_stack, axis=0)
    sq_dists = np.array([np.sum((L - mu) ** 2) for L in L_list])  # d^2(mu, Y_i)
    return mu, sq_dists


def sigma_hat_sq(sq_dists):
    """sigma_j^2 = (1/n_j) sum d^4(mu_j, Y_i) - [(1/n_j) sum d^2(mu_j, Y_i)]^2"""
    return np.mean(sq_dists ** 2) - np.mean(sq_dists) ** 2


def dubey_test_statistic(groups):
    """
    groups: list of k lists of Laplacian matrices (one list per population).
    Implements Dubey & Muller (2019) Frechet ANOVA k-sample test statistic T_n.

    KNOWN LIMITATION -- small-sample instability: sigma2s[j] (the estimated
    Frechet variance-of-variance) sits in the denominator below. At small
    within-group sample sizes it can come out at/near zero, producing a
    RuntimeWarning (divide by zero / invalid value) and an unreliable
    T_n/p-value, rather than raising an error. Not observed at any sample
    size used in this thesis (n=25/50/100/450 all checked clean); relevant
    mainly if reusing this function on new data with small n per group.
    """
    k = len(groups)
    n_j = np.array([len(g) for g in groups])
    n = n_j.sum()
    lambdas = n_j / n

    Vs = np.empty(k)
    sigma2s = np.empty(k)
    for j, g in enumerate(groups):
        mu_j, sq_dists_j = frechet_mean_and_sq_dists(g)
        Vs[j] = np.mean(sq_dists_j)
        sigma2s[j] = sigma_hat_sq(sq_dists_j)

    all_L = [L for g in groups for L in g]
    _, sq_dists_pooled = frechet_mean_and_sq_dists(all_L)
    V_p = np.mean(sq_dists_pooled)

    # F_n: targets differences in Frechet means
    F_n = V_p - np.sum(lambdas * Vs)

    # U_n: targets differences in Frechet variances (generalized Levene's test)
    U_n = 0.0
    for j in range(k):
        for l in range(j + 1, k):
            U_n += (lambdas[j] * lambdas[l] / (sigma2s[j] * sigma2s[l])) * (Vs[j] - Vs[l]) ** 2

    # NOTE: sigma2s can be ~0 at small n_j, causing a silent divide-by-zero
    # here (RuntimeWarning) and an unreliable T_n. See docstring above.
    term1 = n * U_n / np.sum(lambdas / sigma2s)
    term2 = n * F_n ** 2 / np.sum(lambdas ** 2 * sigma2s)

    T_n = term1 + term2
    return T_n, F_n, U_n, Vs, sigma2s


def run_test(G_all, y, laplacians=None):
    """
    Runs the Dubey et al. (2019) Frechet ANOVA test on the FULL dataset
    (no train/test split — a hypothesis test, not a classifier).

    Asymptotic null: T_n ~ chi2(df), df = k-1 (Theorem 2, Dubey & Muller
    2019) -- NOT 2*(k-1); see project notes for the derivation/confirmation
    of this correction.
      - Under H0, the F_n (means) term is oP(1) and vanishes asymptotically;
        only the U_n (variances, generalized Levene's) term contributes to
        the limiting chi2(k-1) distribution.

    laplacians: optional list of precomputed Laplacian arrays, in the same
    order as G_all. If provided, skips recomputing them here -- pass this
    in when the caller has already built the Laplacians once for shared
    reuse across methods. If None (default), computes them internally --
    fully backward compatible with existing standalone calls.
    """
    if laplacians is None:
        laplacians = [nx.laplacian_matrix(G).toarray() for G in G_all]

    labels = np.unique(y)
    groups = [[L for L, lab in zip(laplacians, y) if lab == g] for g in labels]

    T_n, F_n, U_n, Vs, sigma2s = dubey_test_statistic(groups)
    k = len(groups)
    dof = k - 1

    p_value = chi2.sf(T_n, df=dof)

    return {"statistic": T_n, "F_n": F_n, "U_n": U_n, "p_value": p_value, "dof": dof}
