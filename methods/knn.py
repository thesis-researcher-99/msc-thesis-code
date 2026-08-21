import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def get_predictions(distance_matrix, y, idx_train, idx_test):
    """Fit ONCE on true labels, predict ONCE. Feeds into the generic permutation test."""
    D_train = distance_matrix[np.ix_(idx_train, idx_train)]
    D_test = distance_matrix[np.ix_(idx_test, idx_train)]

    knn_model = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    knn_model.fit(D_train, y[idx_train])

    y_pred = knn_model.predict(D_test)
    y_true_test = y[idx_test]
    return y_pred, y_true_test

def fit_predict_full(distance_matrix, y):
    """Fit on ALL points, predict on the SAME all points (resubstitution).
    distance_matrix is label-independent and precomputed once outside the
    permutation loop; only the KNN fit is redone per call."""
    knn_model = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    knn_model.fit(distance_matrix, y)
    return knn_model.predict(distance_matrix)
