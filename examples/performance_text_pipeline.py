"""
Performance comparison between sklearn and scikit_endpoint for a
text pipeline. The pipeline unions a `TfidfVectorizer` and a `HashingVectorizer`
followed by a `RandomForestClassifier` as the estimator.
In the case of model object size, unpickle latency,
and prediction latency for a single record, we see
outperformance with scikit_endpoint.

We see substantial outperformance with scikit_endpoint for
single record prediction.

Example Run
-----------
Pickle Size sklearn: 5546533
Pickle Size pure-predict: 3334779
Difference: 0.6012366644172135
Unpickle time sklearn: 0.048230886459350586
Unpickle time pure-predict: 0.03762626647949219
Difference: 0.7801280308460417
Predict 1 record sklearn: 0.09419584274291992
Predict 1 record pure-predict: 0.00426483154296875
Difference: 0.04527621834233559
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion

from scikit_endpoint.map import convert_estimator
from scikit_endpoint.utils import performance_comparison

N_ESTIMATORS = 100
MAX_DEPTH = None

categories = ["rec.autos", "sci.space"]
X, y = fetch_20newsgroups(subset="train", categories=categories, return_X_y=True)
vec1 = HashingVectorizer()
vec2 = TfidfVectorizer()
feats = FeatureUnion([("vec1", vec1), ("vec2", vec2)])
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
clf = Pipeline(steps=[("feats", feats), ("rf", rf)])
clf.fit(X, y)
clf_ = convert_estimator(clf)
performance_comparison(clf, clf_, X)
