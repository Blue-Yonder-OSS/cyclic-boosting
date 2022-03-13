"""
Smoothers of target profiles (see :mod:`nbpy.target_profile`) and factor profiles
(see :mod:`nbpy.factor_model`) or for the regularization of plots.
"""
from __future__ import absolute_import, division, print_function

from cyclic_boosting.smoothing import extrapolate  # noqa
from cyclic_boosting.smoothing import meta_smoother  # noqa
from cyclic_boosting.smoothing import multidim  # noqa
from cyclic_boosting.smoothing import onedim  # noqa
from cyclic_boosting.smoothing.meta_smoother import RegressionType

__all__ = ["RegressionType"]
