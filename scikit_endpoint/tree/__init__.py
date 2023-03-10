"""
The :mod:`scikit_endpoint.tree` module implements a variety of tree models
"""

from ._classes import (
    DecisionTreeClassifierPure,
    ExtraTreeClassifierPure,
    DecisionTreeRegressorPure,
    ExtraTreeRegressorPure,
)

__all__ = [
    "DecisionTreeClassifierPure",
    "ExtraTreeClassifierPure",
    "DecisionTreeRegressorPure",
    "ExtraTreeRegressorPure",
]
