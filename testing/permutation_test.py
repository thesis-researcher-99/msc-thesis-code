import numpy as np

def permutation_test(y_pred, y_test, B=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    observed_error = np.mean(y_pred != y_test)

    permuted_errors = np.empty(B)
    for b in range(B):
        permuted_labels = rng.permutation(y_test)
        permuted_errors[b] = np.mean(y_pred != permuted_labels)

    p_value = (np.sum(permuted_errors <= observed_error) + 1) / (B + 1)
    return observed_error, p_value, permuted_errors

def permutation_test_refit(fit_predict_fn, y, B=1000, random_state=None):
    """Refit-per-permutation test. fit_predict_fn(y_labels) fits on the FULL
    dataset using y_labels and returns predictions on that SAME full dataset
    (resubstitution). Called once with true y for the observed statistic,
    then once per permutation with shuffled y for the null draws -- so the
    model is genuinely refit under H1 and under every draw from H0."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    y_pred_obs = fit_predict_fn(y)
    observed_error = np.mean(y_pred_obs != y)

    permuted_errors = np.empty(B)
    for b in range(B):
        y_perm = rng.permutation(y)
        y_pred_perm = fit_predict_fn(y_perm)
        permuted_errors[b] = np.mean(y_pred_perm != y_perm)

    p_value = (np.sum(permuted_errors <= observed_error) + 1) / (B + 1)
    return observed_error, p_value, permuted_errors
