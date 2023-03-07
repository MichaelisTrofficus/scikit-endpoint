from ._base import SelectorMixinPure
from ..utils import check_types, check_version


class VarianceThresholdPure(SelectorMixinPure):
    """
    Pure python implementation of `VarianceThreshold`.
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.mask = estimator._get_support_mask().tolist()
        check_types(self)

    def _get_support_mask(self):
        return self.mask
