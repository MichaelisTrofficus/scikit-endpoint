import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scikit_endpoint.feature_selection import VarianceThresholdPure

data = [[0, 1, 2, 3, 4], [0, 2, 2, 3, 5], [1, 1, 2, 4, 0]]


def test_variance_threshold():
    # Test VarianceThreshold with custom variance.
    estimator = VarianceThreshold(threshold=0.4)
    estimator.fit(data)
    estimator_ = VarianceThresholdPure(estimator)
    np.allclose(estimator.transform(data), estimator_.transform(data))
