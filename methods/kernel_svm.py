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
    svm_model = SVC(kernel="precomputed", C=1.0)
    svm_model.fit(kernel_matrix, y)
    return svm_model.predict(kernel_matrix)
