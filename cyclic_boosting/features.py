from collections import namedtuple

import numpy as np

from cyclic_boosting import flags
from cyclic_boosting.smoothing import multidim, onedim
from cyclic_boosting.utils import (
    arange_multi,
    clone,
    get_X_column,
    multidim_binnos_to_lexicographic_binnos,
)

FeatureID = namedtuple("FeatureID", ["feature_group", "feature_type"])


class FeatureTypes(object):
    standard = None
    external = "external"


class Feature(object):
    """Wrapper for information regarding a single feature group in the cyclic
    boosting algorithm

    Parameters
    ----------

    feature_id: :class:`FeatureID`
        feature ID is a named tuple consisting of the name of a feature or
        tuple of such names in the multidimensional case and the feature type.

    feature_property: int
        feature property, see :mod:`flags`

    smoother: subclass of :class:`cyclic_boosting.smoothing.onedim.AbstractBinSmoother`
        smoother for the bins of the feature group

    minimal_factor_change: float
        minimum change of the results in the feature group's bins that are
        considered progress in a cyclic boosting iteration

    Note
    ----

    The suffix ``_link`` in attributes refers to values valid in link space.
    """

    def __init__(
        self, feature_id, feature_property, smoother, minimal_factor_change=1e-4
    ):
        self.feature_id = feature_id
        self.feature_group = feature_id.feature_group
        if len(self.feature_group) < 2:
            self.smootherb = onedim.BinValuesSmoother()
        else:
            self.smootherb = multidim.BinValuesSmoother()
        self.is_fitted = False
        self.feature_property = feature_property
        self.smoother = smoother
        self.learn_rate = 0.0

        self.unfitted_factors_link = None
        self.factors_link = None
        self.factors_link_old = None
        self.unfitted_uncertainties_link = None
        self.uncertainties_link = None

        # set by bind_data method
        self.lex_binned_data = None
        self.n_multi_bins_finite = None

        self.minimal_factor_change = minimal_factor_change
        self.stop_iterations = False

        self.feature_type = feature_id.feature_type
        self.factor_sum = None

    @property
    def dim(self):
        """Dimension of the feature group, i.e. the number of its features"""
        return len(self.feature_group)

    @property
    def is_1dim(self):
        """
        True if feature is 1-dimensional, False otherwise
        """
        return self.dim == 1

    @property
    def missing_not_learned(self):
        """
        True if ``flags.MISSING_NOT_LEARNED`` can be found in feature property.
        """
        return any(
            flags.missing_not_learned_set(feature_prop)
            for feature_prop in self.feature_property
        )

    @property
    def n_bins(self):
        """Number of bins, including the extra bin for missing and infinite
        values
        """
        return self.n_bins_finite + 1

    @property
    def n_bins_finite(self):
        """Number of bins, excluding the extra bin for missing and infinite
        values
        """
        return int(np.prod(self.n_multi_bins_finite))

    @property
    def bin_centers(self):
        if self.n_bins_finite == 0:
            return np.ndarray(shape=(0, len(self.n_multi_bins_finite)))
        else:
            return arange_multi(self.n_multi_bins_finite)

    @property
    def finite_bin_weightsums(self):
        """Array of finite bin weightsums"""
        return self.bin_weightsums[:-1]

    @property
    def nan_bin_weightsum(self):
        return self.bin_weightsums[-1]

    def bind_data(self, X, weights, first):
        """
        Binds data from X belonging to the feature and calculates the
        following features:

        * lex_binned_data: The binned data transformed to a 1-dimensional array
            containing lexical binned data.
        * n_multi_bins_finite: Number of bins for each column in feature
            (needed for plotting).
        * finite_bin_weightsums: Array containing the sum of weights for
             each bin.
        * nan_bin_weightsum: The sum of weights for the nan-bin.
        * bin_centers: Array containing the center of each bin.
        """
        binnumbers = get_X_column(X, self.feature_group, array_for_1_dim=False)
        (
            self.lex_binned_data,
            self.n_multi_bins_finite,
        ) = multidim_binnos_to_lexicographic_binnos(
            binnumbers, self.n_multi_bins_finite
        )
        self.bin_weightsums = np.bincount(
            self.lex_binned_data, weights=weights, minlength=self.n_bins
        )

    @property
    def unfitted_factor_link_nan_bin(self):
        return self.unfitted_factors_link[-1]

    @property
    def factor_link_nan_bin(self):
        return self.factors_link[-1]

    def unbind_data(self):
        """Clear some of the references set in :meth:`bind_data`."""
        self.lex_binned_data = None

    def unbind_factor_data(self):
        self.unfitted_factors_link = None
        self.factors_link_old = None
        self.fitted_aggregated = None
        self.unfitted_uncertainties_link = None
        self.uncertainties_link = None
        self.factors_link = [self.factors_link[-1]]

    def _get_data_for_smoother(self, uncertainties_link):
        """Prepare ``X_for_smoother`` as needed by the smoother's ``fit`` method.

        The bin centers, bin weightsums and the uncertainties in link space
        must already have been calculated.
        """
        n_bins = self.n_bins_finite
        n_bin_dims = self.dim
        X_for_smoother = np.empty((n_bins + 1, n_bin_dims + 2))

        X_for_smoother[:n_bins, :n_bin_dims] = self.bin_centers
        X_for_smoother[n_bins, :n_bin_dims] = np.nan
        X_for_smoother[:, n_bin_dims] = self.bin_weightsums
        X_for_smoother[:, -1] = uncertainties_link
        return X_for_smoother

    def update_factors(
        self, unfitted_factors, unfitted_uncertainties, neutral_factor_link, learn_rate
    ):
        """Call the smoother on the bin results if necessary.

        Parameters
        ----------

        unfitted_factors: ndarray
            bin means in link space as returned by the method
            :meth:`cyclic_boosting.base.CyclicBoostingBase.calc_parameters`

        unfitted_uncertainties: ndarray
            bin uncertainties in link space as returned by the method
            :meth:`cyclic_boosting.base.CyclicBoostingBase.calc_parameters`

        neutral_factor_link: float
            neutral value in link space, currently always 0
        """
        self.learn_rate = learn_rate
        self.factors_link_old = self.factors_link.copy()
        self.unfitted_factors_link = unfitted_factors
        self.unfitted_uncertainties_link = unfitted_uncertainties
        if self.missing_not_learned:
            # do not learn missing features by regularizing the factor
            # for the nan bin to the neutral factor
            unfitted_factors[-1] = neutral_factor_link
            self.factors_link[-1] = neutral_factor_link
        else:
            self.factors_link[-1] = unfitted_factors[-1] * learn_rate

        X_for_smoother = self._get_data_for_smoother(unfitted_uncertainties)

        if self.n_bins_finite > 0:
            self.smoother.fit(X_for_smoother[:-1], unfitted_factors[:-1].copy())
            if not self.is_fitted:
                self.smootherb.fit(X_for_smoother[:-1], unfitted_factors[:-1].copy())
                self.smootherb.smoothed_y_ = None
        else:
            self.smoother = None
            self.smootherb = None
        return X_for_smoother


def create_feature_id(feature_group_or_id, default_type=None):
    """Convenience function to convert feature_groups
    into :class:`FeatureID`s
    """
    if isinstance(feature_group_or_id, FeatureID):
        return feature_group_or_id
    elif isinstance(feature_group_or_id, tuple):
        return FeatureID(feature_group=feature_group_or_id, feature_type=default_type)
    else:
        return FeatureID(
            feature_group=(feature_group_or_id,), feature_type=default_type
        )


def create_features(feature_groups_or_ids, feature_properties, smoother_choice):
    """
    Parameters
    ----------

    feature_group: list of feature groups or feature ids
        Each feature group is a name of a feature or a tuple of such names in the
        multidimensional case.

    feature_properties: dict of feature properties

    smoother_choice: subclass of :class:`cyclic_boosting.SmootherChoice`
        Selects smoothers
    """
    if feature_groups_or_ids is None:
        feature_groups_or_ids = []

    def make_feature_for_group_or_id(feature_group_or_id):
        feature_id = create_feature_id(feature_group_or_id)

        feature_property = flags.read_feature_property(
            feature_properties, feature_id.feature_group, default=flags.HAS_MISSING
        )
        smoother = smoother_choice.choice_fct(
            feature_id.feature_group, feature_property, feature_id.feature_type
        )
        return Feature(feature_id, feature_property, clone(smoother))

    features = [
        make_feature_for_group_or_id(feature_group_or_id)
        for feature_group_or_id in feature_groups_or_ids
    ]

    return FeatureList(features)


class FeatureList(object):
    """Iterable providing the information about a list of features in the
    form of :class:`Feature` objects

    Parameters
    ----------

    features: list of :class:`Feature`
        List of :class:`Feature` that is normally created by
        :func:`create_features`.
    """

    def __init__(self, features):
        self.features = features

    @property
    def feature_groups(self):
        """
        Obtain the feature_groups for all features.
        """
        return [feature.feature_group for feature in self.features]

    @property
    def feature_ids(self):
        """
        Obtain the feature_groups for all features.
        """
        return [feature.feature_id for feature in self.features]

    def iter_fitting(self):
        """Generator yielding only the features with attribute
        ``stop_iterations == False``
        """
        for feature in self.features:
            if not feature.stop_iterations:
                yield feature

    def __iter__(self):
        for feature in self.features:
            yield feature

    def __len__(self):
        return len(self.features)

    def __getitem__(self, feature_group_or_id):
        """Selects feature specified by ``feature_id``

        Parameters
        ----------

        feature_group_or_id: `string`, `int` or `tuple` of `string` or :class:`FeatureID`
           feature identifier

        Returns
        -------

        class:`cyclic_boosting.base.Feature`
           Feature instance
        """
        feature_id = create_feature_id(feature_group_or_id)

        for feature in self.features:
            if feature.feature_id == feature_id:
                return feature
        raise KeyError(
            "Feature {0} is not known in {1}".format(feature_id, self.feature_ids)
        )

    def get_feature(self, feature_group, feature_type=None):
        """Selects feature specified by ``feature_group`` and ``feature_type``

        Parameters
        ----------

        feature_group: `string`, `int` or `tuple` of `string` or `int`
           name or index of feature group

        feature_type: `None` or `string`
           type of feature

        Returns
        -------

        class:`cyclic_boosting.base.Feature`
           Feature instance
        """
        feature_id = create_feature_id(feature_group, feature_type)
        return self[feature_id]
