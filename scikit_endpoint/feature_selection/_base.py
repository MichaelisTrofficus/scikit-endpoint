from itertools import compress
from abc import ABCMeta, abstractmethod


class SelectorMixinPure(metaclass=ABCMeta):
    def get_support(self):
        mask = self._get_support_mask()
        return mask

    @abstractmethod
    def _get_support_mask(self):
        raise NotImplementedError()

    def transform(self, X):
        return self._transform(X)

    def _transform(self, X):
        mask = self.get_support()
        return [list(compress(x, mask)) for x in X]
