"""
The :mod:`scikit_endpoint.impute` module implements a variety of imputer transformers
"""

from ._base import SimpleImputerPure, MissingIndicatorPure

__all__ = ["SimpleImputerPure", "MissingIndicatorPure"]
