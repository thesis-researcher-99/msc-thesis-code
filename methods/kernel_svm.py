"""
Kernel SVM classifier two-sample test.
 
Classifier-based two-sample test (Lopez-Paz & Oquab, 2017 C2ST framework):
fits an SVC with a precomputed Gram/kernel matrix to discriminate between 
the two groups, then feeds its predictions to a permutation test (see 
testing/permutation_test.py) to obtain a p-value from the classifier's 
held-out misclassification error.
 
Entry points:
  get_predictions(kernel_matrix, y, idx_train, idx_test) -> (y_pred, y_test)
      Fixed train/test split -- fit once, predict once. Feeds into the
      generic permutation test, which holds predictions fixed and permutes
      labels.
  fit_predict_full(kernel_matrix, y) -> y_pred
      Resubstitution variant (fit and predict on all points) -- used by
      the refit-per-permutation sweep variant instead of the fixed-split one.
Neither function computes a p-value itself -- see testing/permutation_test.py.
"""
import numpy as np
from sklearn.svm import SVC

def get_predictions(kernel_matrix, y, idx_train, idx_test):
    """Fit ONCE on true labels using a precomputed kernel, predict ONCE."""
    K_train = kernel_matrix[np.ix_(idx_train, idx_train)]
    K_test = kernel_matrix[np.ix_(idx_test, idx_train)]

    svm_model = SVC(kernel="precomputed", C=1.0)
    svm_model.fit(K_train, y[idx_train])

    y_pred = svm_model.predict(K_test)
    y_true_test = y[idx_test]
    return y_pred, y_true_test

def fit_predict_full(kernel_matrix, y):
    """Fit on ALL points, predict on the SAME all points (resubstitution)."""
    svm_model = SVC(kernel="precomputed", C=1.0)
    svm_model.fit(kernel_matrix, y)
    return svm_model.predict(kernel_matrix)
