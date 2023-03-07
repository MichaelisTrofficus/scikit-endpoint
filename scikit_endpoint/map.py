"""
Mappings from `sklearn` class names to `scikit_endpoint` equivalents
"""

MAPPING = {
    "LogisticRegression": "scikit_endpoint.linear_model.LogisticRegressionPure",
    "RidgeClassifier": "scikit_endpoint.linear_model.RidgeClassifierPure",
    "SGDClassifier": "scikit_endpoint.linear_model.SGDClassifierPure",
    "Perceptron": "scikit_endpoint.linear_model.PerceptronPure",
    "PassiveAggressiveClassifier": "scikit_endpoint.linear_model.PassiveAggressiveClassifierPure",
    "LinearSVC": "scikit_endpoint.svm.LinearSVCPure",
    "DecisionTreeClassifier": "scikit_endpoint.tree.DecisionTreeClassifierPure",
    "DecisionTreeRegressor": "scikit_endpoint.tree.DecisionTreeRegressorPure",
    "ExtraTreeClassifier": "scikit_endpoint.tree.ExtraTreeClassifierPure",
    "ExtraTreeRegressor": "scikit_endpoint.tree.ExtraTreeRegressorPure",
    "RandomForestClassifier": "scikit_endpoint.ensemble.RandomForestClassifierPure",
    "BaggingClassifier": "scikit_endpoint.ensemble.BaggingClassifierPure",
    "GradientBoostingClassifier": "scikit_endpoint.ensemble.GradientBoostingClassifierPure",
    "XGBClassifier": "scikit_endpoint.xgboost.XGBClassifierPure",
    "ExtraTreesClassifier": "scikit_endpoint.ensemble.ExtraTreesClassifierPure",
    "GaussianNB": "scikit_endpoint.naive_bayes.GaussianNBPure",
    "MultinomialNB": "scikit_endpoint.naive_bayes.MultinomialNBPure",
    "ComplementNB": "scikit_endpoint.naive_bayes.ComplementNBPure",
    "SimpleImputer": "scikit_endpoint.impute.SimpleImputerPure",
    "MissingIndicator": "scikit_endpoint.impute.MissingIndicatorPure",
    "DummyClassifier": "scikit_endpoint.dummy.DummyClassifierPure",
    "Pipeline": "scikit_endpoint.pipeline.PipelinePure",
    "FeatureUnion": "scikit_endpoint.pipeline.FeatureUnionPure",
    "OneHotEncoder": "scikit_endpoint.preprocessing.OneHotEncoderPure",
    "OrdinalEncoder": "scikit_endpoint.preprocessing.OrdinalEncoderPure",
    "StandardScaler": "scikit_endpoint.preprocessing.StandardScalerPure",
    "MinMaxScaler": "scikit_endpoint.preprocessing.MinMaxScalerPure",
    "MaxAbsScaler": "scikit_endpoint.preprocessing.MaxAbsScalerPure",
    "Normalizer": "scikit_endpoint.preprocessing.NormalizerPure",
    "DictVectorizer": "scikit_endpoint.feature_extraction.DictVectorizerPure",
    "TfidfVectorizer": "scikit_endpoint.feature_extraction.text.TfidfVectorizerPure",
    "CountVectorizer": "scikit_endpoint.feature_extraction.text.CountVectorizerPure",
    "TfidfTransformer": "scikit_endpoint.feature_extraction.text.TfidfTransformerPure",
    "HashingVectorizer": "scikit_endpoint.feature_extraction.text.HashingVectorizerPure",
    "VarianceThreshold": "scikit_endpoint.feature_selection.VarianceThresholdPure",
}


def _instantiate_class(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def convert_estimator(est, min_version=None):
    """Convert scikit-learn estimator to its scikit_endpoint counterpart"""
    est_name = est.__class__.__name__
    pure_est_name = MAPPING.get(est_name)
    if pure_est_name is None:
        raise ValueError(
            "Cannot find 'scikit_endpoint' counterpart for {}".format(est_name)
        )
    module = ".".join(pure_est_name.split(".")[:-1])
    name = pure_est_name.split(".")[-1]
    return _instantiate_class(module, name)(est)
