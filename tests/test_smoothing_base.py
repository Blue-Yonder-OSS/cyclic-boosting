import numpy as np
import six

from cyclic_boosting import smoothing, utils


def check_sklearn_style_cloning(est):
    """Verify that an estimator is correctly cloned.

    Cloning of an estimator or transformer by passing attributes with the same
    names as the constructor parameters to its constructor. This cloning is
    used in meta-estimators.

    This function also verifies that attributes mentioned in the :obj:`set`
    ``no_deepcopy`` (optionally defined statically in the estimator class), if
    present, are not deeply copied but referenced.

    An :class:`AssertionError` is raised if the cloning is incorrect.
    """

    def _compare_clones(est, cloned_est):
        """Auxiliary function for :func:`check_sklearn_style_cloning` comparing
        an estimator and its clone.
        """
        class_name = est.__class__.__name__

        object_params = est.get_params(deep=False)
        new_object_params = cloned_est.get_params(deep=False)

        if hasattr(est, "no_deepcopy"):
            no_deepcopy = est.no_deepcopy
        else:
            no_deepcopy = set()

        for key in object_params:
            assert hasattr(
                cloned_est, key
            ), "Attribute '{}' missing in cloned estimator of class '{}'.".format(
                key, class_name
            )

        assert sorted(object_params.keys()) == sorted(new_object_params.keys()), (
            "Estimator of class '{}' was incorrectly cloned. Some attributes "
            "are missing.".format(class_name)
        )

        for name, param in six.iteritems(object_params):
            new_param = new_object_params[name]
            if name in no_deepcopy:
                assert param is new_param, (
                    "Attribute '{}' of estimator of class '{}' was cloned in "
                    "contradiction to the specification of 'no_deepcopy'.".format(
                        name, class_name
                    )
                )
            else:
                clone_of_param = utils.clone(param, safe=False)
                assert (param is new_param) == (
                        param is clone_of_param
                ), "Attribute '{}' of estimator of class '{}' was not cloned.".format(
                    name, class_name
                )

    def check_subestimators(est, cloned_est):
        """Check subestimators.

        :param est: Possibly a meta-estimator.
        :type est: Estimator

        :param cloned_est: Corresponding clone.
        :type cloned_est: Estimator
        """
        _compare_clones(est, cloned_est)
        if hasattr(est, "get_subestimators"):
            for subest, subest_cloned in zip(
                est.get_subestimators(prototypes=True),
                cloned_est.get_subestimators(prototypes=True),
            ):
                check_subestimators(subest, subest_cloned)

    check_subestimators(est, utils.clone(est))


def test_cloning():
    for smoother in [
        smoothing.onedim.BinValuesSmoother(),
        smoothing.onedim.RegularizeToPriorExpectationSmoother(2, threshold=4),
        smoothing.onedim.RegularizeToOneSmoother(threshold=4),
        smoothing.onedim.UnivariateSplineSmoother(k=4, s=5),
        smoothing.onedim.OrthogonalPolynomialSmoother(),
        smoothing.onedim.SeasonalSmoother(),
        smoothing.onedim.PolynomialSmoother(k=3),
        smoothing.onedim.LSQUnivariateSpline([-1, 0, 1]),
        smoothing.onedim.IsotonicRegressor(),
        smoothing.multidim.BinValuesSmoother(),
        smoothing.multidim.RegularizeToPriorExpectationSmoother(2, threshold=4),
        smoothing.multidim.RegularizeToOneSmoother(threshold=4),
    ]:
        check_sklearn_style_cloning(smoother)


def test_exceptions():
    smoother = smoothing.multidim.BinValuesSmoother()
    np.testing.assert_raises_regex(
        ValueError,
        'Please call the method "fit" before "predict" and "set_n_bins"',
        smoother.predict,
        None,
    )
    smoother = smoothing.onedim.PolynomialSmoother(k=2)
    np.testing.assert_raises_regex(
        ValueError,
        "The PolynomialSmoother has not been fitted!",
        smoother.predict,
        None,
    )


def compare_smoother(est1, est2, X, y, dim=1):
    est1.fit(X.copy(), y.copy())
    pred1 = est1.predict(X[:, :dim].copy())
    pred2 = None
    if est2 is not None:
        est2.fit(X.copy(), y.copy())
        pred2 = est2.predict(X[:, :dim].copy())
    return pred1, pred2


def get_data(onedim=True):
    if onedim:
        y = np.array([0.91, 0.92, 0.93, 0.94, 1.75, 1.80, 0.40, 0.92])
        n = len(y)
        X = np.c_[
            np.arange(n), np.ones(n), [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.05]
        ]
    else:
        X = np.c_[
            np.repeat(np.arange(5), 5) * 1.0,
            np.tile(np.arange(5), 5) * 1.0,
            np.ones(25),
            np.ones(25),
        ]
        y = (X[:, 0] + 1) * (X[:, 1] + 1)
    return X, y


def test_meta_smoother_zero_threshold_onedim_weighted_mean_smoother():
    est1 = smoothing.onedim.WeightedMeanSmoother()
    est2 = smoothing.onedim.PriorExpectationMetaSmoother(
        smoothing.onedim.WeightedMeanSmoother(), prior_expectation=1, threshold=0.0
    )
    X, y = get_data()
    pred1, pred2 = compare_smoother(est1, est2, X, y)
    np.testing.assert_allclose(pred1, pred2)

def test_meta_smoother_zero_threshold_onedim_orthogonal_smoother():
    est1 = smoothing.onedim.OrthogonalPolynomialSmoother()
    est2 = smoothing.onedim.PriorExpectationMetaSmoother(
        smoothing.onedim.OrthogonalPolynomialSmoother(),
        prior_expectation=1,
        threshold=0.0,
    )
    X, y = get_data()
    pred1, pred2 = compare_smoother(est1, est2, X, y)
    np.testing.assert_allclose(pred1, pred2)

def test_meta_smoother_zero_threshold_multdim_weighted_mean_smoother():
    X, y = get_data(onedim=False)
    est1 = smoothing.multidim.WeightedMeanSmoother()
    est2 = smoothing.multidim.PriorExpectationMetaSmoother(
        smoothing.multidim.WeightedMeanSmoother(), np.mean(y), threshold=0.0
    )
    pred1, pred2 = compare_smoother(est1, est2, X, y, dim=2)
    np.testing.assert_allclose(pred1, pred2)


def test_meta_smoother_high_threshold_onedim_weighted_mean_smoother():
    est = smoothing.onedim.PriorExpectationMetaSmoother(
        smoothing.onedim.WeightedMeanSmoother(), prior_expectation=1, threshold=20.0
    )
    X, y = get_data()
    pred1, _2 = compare_smoother(est, None, X, y)
    np.testing.assert_allclose(pred1, 1)


def test_meta_smoother_high_threshold_onedim_orthogonal_smoother():
    est = smoothing.onedim.PriorExpectationMetaSmoother(
        smoothing.onedim.OrthogonalPolynomialSmoother(),
        prior_expectation=1,
        threshold=20.0,
    )
    X, y = get_data()
    pred1, _2 = compare_smoother(est, None, X, y)
    np.testing.assert_allclose(pred1, 1)


def test_meta_smoother_high_threshold_multdim_weighted_mean_smoother():
    X, y = get_data(onedim=False)
    est = smoothing.multidim.PriorExpectationMetaSmoother(
        smoothing.multidim.WeightedMeanSmoother(), np.mean(y), threshold=20.0
    )
    pred1, _2 = compare_smoother(est, None, X, y, dim=2)
    np.testing.assert_allclose(pred1, np.mean(y))


def test_minimum_number_of_columns():
    X, y = get_data(onedim=True)
    est = smoothing.multidim.PriorExpectationMetaSmoother(
        smoothing.multidim.WeightedMeanSmoother(), np.mean(y), threshold=20.0
    )
    with np.testing.assert_raises(ValueError):
        est.fit(X, y)
