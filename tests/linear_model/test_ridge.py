import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris

from scikit_endpoint.map import convert_estimator
from scikit_endpoint.utils import shape

METHODS = ["decision_function", "predict", "_predict_proba_lr"]


def test_ridge():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for fit_intercept in [True, False]:
            clf = RidgeClassifier(fit_intercept=fit_intercept)
            clf.fit(X, y_)
            clf_ = convert_estimator(clf)

            for method in METHODS:
                scores = getattr(clf, method)(X)
                scores_ = getattr(clf_, method)(X_)
                assert np.allclose(scores.shape, shape(scores_))
                assert np.allclose(scores, scores_)
