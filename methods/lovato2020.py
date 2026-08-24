"""
Lovato et al. (2020) -- "Model-free two-sample test for network-valued
data" (Statistics & Probability Letters / arXiv).
 
Inter-point-distance permutation test on graph Laplacians under the
Frobenius distance: two statistics targeting different moments --
T_IP-Student (mean differences, a Welch-t-style ratio of cross-group to
within-group squared distances) and T_IP-Fisher (variance differences, a
ratio of within-group variance estimates) -- combined via Tippett's
non-parametric combination function into T_IP-StudentFisher, sensitive to
differences in either moment. Significance for each statistic uses the
exact Phipson & Smyth (2010) permutation p-value (after Dwass, 1957),
which guarantees exact Type I error control regardless of sample size or
number of permutations sampled -- unlike ginestet2017.py/dubey2019.py,
which rely on asymptotic chi^2 approximations.
 
Entry point: run_test(G_all, y, B=1000, random_state=42, combine=True,
    laplacians=None) -> dict with T_student/p_student, T_fisher/p_fisher,
    mt (total number of distinct permutations), and (if combine=True)
    T_combined/p_combined -- the last is what run_one_replicate() in the
    sweep scripts actually uses.
Full-sample hypothesis test -- no train/test split.
"""
import math
import numpy as np
import networkx as nx
from scipy.stats import binom, rankdata
from scipy.integrate import quad


def prepare_sq_distance_matrix(G_all, laplacians=None):
    """Precompute squared Frobenius distances between ALL graphs' Laplacians once.
    Label-independent — safe to reuse across every permutation.

    laplacians: optional list of precomputed Laplacian arrays, in the same
    order as G_all. If provided, skips recomputing them here. If None
    (default), computes them internally as before -- fully backward
    compatible with existing standalone calls.
    """
    if laplacians is None:
        laplacians = [nx.laplacian_matrix(G).toarray() for G in G_all]
    n = len(laplacians)
    D_sq = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_sq = np.sum((laplacians[i] - laplacians[j]) ** 2)  # rho_FR^2
            D_sq[i, j] = D_sq[j, i] = d_sq
    return D_sq


def compute_ip_statistics(D_sq, idx1, idx2):
    """T_IP-Student and T_IP-Fisher for a given group split."""
    n1, n2 = len(idx1), len(idx2)

    cross = D_sq[np.ix_(idx1, idx2)]
    cross_mean = cross.mean()

    D1 = D_sq[np.ix_(idx1, idx1)]
    sigma1_sq = D1[np.triu_indices(n1, k=1)].sum() / (n1 * (n1 - 1))

    D2 = D_sq[np.ix_(idx2, idx2)]
    sigma2_sq = D2[np.triu_indices(n2, k=1)].sum() / (n2 * (n2 - 1))

    T_student = (cross_mean - (sigma1_sq + sigma2_sq)) / (sigma1_sq / n1 + sigma2_sq / n2)
    T_fisher = max(sigma1_sq / sigma2_sq, sigma2_sq / sigma1_sq)

    return T_student, T_fisher


def total_num_permutations(n1, n2, two_sided_equal=True):
    """mt = (n1+n2)! / n1! / n2!, halved if two-sided and n1 == n2 (paper, Section 2.4)."""
    mt = math.comb(n1 + n2, n1)
    if two_sided_equal and n1 == n2:
        mt = mt // 2
    return mt


def phipson_smyth_pvalue(tobs, tperm_array, mt):
    """
    Exact permutation p-value, Eq. (2.5) of Lovato et al. (2020), after
    Dwass (1957) / Phipson & Smyth (2010). Guarantees an exact test
    (P_H0[p <= alpha] = alpha) regardless of m or mt.
    """
    m = len(tperm_array)
    b = int(np.sum(tperm_array > tobs))  # number of permuted stats strictly greater than observed

    if mt < 10_000:
        bt = np.arange(0, mt + 1)
        pt = (bt + 1) / (mt + 1)
        F_vals = binom.cdf(b, m, pt)
        p = np.sum(F_vals) / (mt + 1)
    else:
        approx = (b + 1) / (m + 1)
        try:
            upper = 0.5 / (mt + 1)
            integral, _ = quad(lambda pt: binom.cdf(b, m, pt), 0, upper)
        except OverflowError:
            # mt too large to represent as a float -- the correction
            # integral's interval width is already ~0 for any mt this
            # large, so its contribution is negligible; skip it.
            integral = 0.0
        p = approx - integral

    return p


def run_test(G_all, y, B=1000, random_state=42, combine=True, laplacians=None):
    """
    Lovato et al. (2020) IP-Student / IP-Fisher permutation test.
    Literal reproduction: exact Phipson & Smyth p-value (Eq. 2.5) +
    Non-Parametric Combination via Tippett's function (IP-StudentFisher).

    laplacians: optional list of precomputed Laplacian arrays, in the same
    order as G_all. If provided, skips recomputing them here -- pass this
    in when the caller has already built the Laplacians once for shared
    reuse across methods. If None (default), computes them internally --
    fully backward compatible with existing standalone calls.
    """
    D_sq = prepare_sq_distance_matrix(G_all, laplacians=laplacians)
    y = np.array(y)
    n = len(y)

    idx1_obs = np.where(y == 0)[0]
    idx2_obs = np.where(y == 1)[0]
    n1, n2 = len(idx1_obs), len(idx2_obs)

    T_student_obs, T_fisher_obs = compute_ip_statistics(D_sq, idx1_obs, idx2_obs)

    rng = np.random.default_rng(random_state)
    all_idx = np.arange(n)
    T_student_perm = np.empty(B)
    T_fisher_perm = np.empty(B)

    for b_i in range(B):
        perm = rng.permutation(all_idx)
        idx1_p, idx2_p = perm[:n1], perm[n1:]
        T_student_perm[b_i], T_fisher_perm[b_i] = compute_ip_statistics(D_sq, idx1_p, idx2_p)

    mt = total_num_permutations(n1, n2, two_sided_equal=True)

    p_student = phipson_smyth_pvalue(T_student_obs, T_student_perm, mt)
    p_fisher = phipson_smyth_pvalue(T_fisher_obs, T_fisher_perm, mt)

    result = {
        "T_student": T_student_obs, "p_student": p_student,
        "T_fisher": T_fisher_obs, "p_fisher": p_fisher,
        "mt": mt,
    }

    if combine:
        # Concatenate observed + permuted into vectors of size B+1 (paper, Section 2.4)
        student_all = np.concatenate(([T_student_obs], T_student_perm))
        fisher_all = np.concatenate(([T_fisher_obs], T_fisher_perm))

        # Rank in DECREASING order, divided by B+1 -> "intermediate p-values"
        pi_student = rankdata(-student_all, method="ordinal") / (B + 1)
        pi_fisher = rankdata(-fisher_all, method="ordinal") / (B + 1)

        # Tippett's combining function: psi(x, y) = 1 - min(x, y)
        combined_all = 1 - np.minimum(pi_student, pi_fisher)
        T_combined_obs = combined_all[0]
        T_combined_perm = combined_all[1:]

        p_combined = phipson_smyth_pvalue(T_combined_obs, T_combined_perm, mt)
        result["T_combined"] = T_combined_obs
        result["p_combined"] = p_combined

    return result
