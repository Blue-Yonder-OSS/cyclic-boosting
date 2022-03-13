import contextlib
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import six

from cyclic_boosting import utils


def long_test(test_func):
    """
    Decorator to skip long tests.

    :param test_func: Test function or method.
    :type test_func: callable
    """
    if os.environ.get("LONGTESTS", "") in ["", "0"]:
        return unittest.skip("skip long tests")(test_func)
    else:
        return test_func


def _compare_clones(est, cloned_est):
    """Auxiliary function for :func:`check_sklearn_style_cloning` comparing
    an estimator and its clone.

    :param est: Estimator to check.
    :type est: :class:`nbpy.estimator.Estimator` or
        :class:`nbpy.estimator.Transformer`

    :param est: The corresponding clone.
    :type est: :class:`nbpy.estimator.Estimator` or
        :class:`nbpy.estimator.Transformer`
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


def check_sklearn_style_cloning(est):
    """Verify that an estimator is correctly cloned.

    :func:`sklearn.base.clone()` and :func:`nbpy.estimator.clone` clone an
    estimator or transformer by passing attributes with the same names as the
    constructor parameters (as determined by the method
    :meth:`sklearn.base.BaseEstimator.get_params`) to its constructor.

    This cloning is used in meta-estimators such as
    :class:`nbpy.nclassify.NClassify` and :class:`nbpy.ext.group_by.GroupBy`.

    This function also verifies that attributes mentioned in the :obj:`set`
    ``no_deepcopy`` (optionally defined statically in the estimator class), if
    present, are not deeply copied but referenced.

    An :class:`AssertionError` is raised if the cloning is incorrect.

    This test fails for some `sklearn` estimators such as
    :class:`sklearn.linear_model.SGDClassifier` that save deprecated
    constructor parameters (in this case ``rho``) in attributes corresponding
    to other constructor parameters.

    Please call this test for all estimators you write if you want them to be
    usable in cloning meta-estimators.

    :param est: Estimator to check.
    :type est: :class:`nbpy.estimator.Estimator` or
        :class:`nbpy.estimator.Transformer`
    """

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


def pickle_and_unpickle_in_temp_file(obj):
    """Pickle and unpickle an object in a temporary file.

    The file is always deleted before leaving this function.

    :param obj: Object to be pickled.
    :type obj: :obj:`object`

    :return: The object after pickling and unpickling it.
    """
    with temp_filename_touched_and_removed() as filename:
        with open(filename, "wb") as fout:
            six.moves.cPickle.dump(obj, fout)
        with open(filename, "rb") as fin:
            return six.moves.cPickle.load(fin)


@contextlib.contextmanager
def temp_filename_touched_and_removed(suffix=""):
    """Context manager returning a filename of a touched and closed temporary
    file.

    At the end of the scope, the file is automatically removed.

    :param suffix: Let the filename end with suffix.
    :type suffix: :obj:`str`

    :return: Name of a touched and closed temporary file. The path of the
        file is chosen by :mod:`tempfile`.
    :rtype: :obj:`str`

    >>> import os.path
    >>> filename1 = None
    >>> from cyclic_boosting.testing_utils import temp_filename_touched_and_removed
    >>> with temp_filename_touched_and_removed() as filename:
    ...     filename1 = filename
    ...     os.path.exists(filename)
    True
    >>> os.path.exists(filename1)
    False
    """
    filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fout:
            filename = fout.name
        yield filename
    finally:
        if filename is not None and os.path.exists(filename):
            os.remove(filename)


@contextlib.contextmanager
def temp_dirname_created_and_removed(suffix="", prefix="tmp", base_dir=None):
    """Context manager returning the name of new temporary directory.

    At the end of the scope, the directory is automatically removed.
    All parameters are passed to :func:`tempfile.mkdtemp`.

    :param suffix: If specified, the file name will end with that suffix,
        otherwise there will be no suffix.
    :type suffix: string

    :param prefix: If specified, the file name will begin with that prefix;
        otherwise, a default prefix is used.
    :type prefix: string

    :param base_dir: If base_dir is specified,
        the file will be created in that directory.
        For details see ``dir`` parameter in :func:`tempfile.mkdtemp`.
    :type base_dir: str

    :return: Name of the temporary directory. The path of the
        directory is chosen by :func:`tempfile.mkdtemp`.
    :rtype: :obj:`str`

    >>> import os.path
    >>> dirname1 = None
    >>> from cyclic_boosting.testing_utils import temp_dirname_created_and_removed
    >>> with temp_dirname_created_and_removed() as dirname:
    ...     dirname1 = dirname
    ...     os.path.exists(dirname)
    True
    >>> os.path.exists(dirname1)
    False
    """
    dirname = None
    try:
        dirname = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=base_dir)
        yield dirname
    finally:
        if dirname is not None:
            shutil.rmtree(dirname)


def generate_binned_data(n_samples, n_features=10, seed=123):
    """
    Generate uncorrelated binned data for a sales problem.

    :param n_samples: number of samples that ought to be created.
    :type n_samples: int

    :param seed: random seed used in data creation
    :type seed: int

    :returns: Randomly generated binned feature matrix and binned target array.
    :rtype: :obj:`tuple` of :class:`pandas.DataFrame` and
        :class:`numpy.ndarray`

    """
    np.random.seed(seed)
    X = pd.DataFrame()
    y = np.random.randint(0, 10, n_samples)
    for i in range(0, n_features):
        n_bins = np.random.randint(1, 100)
        X[str(i)] = np.random.randint(0, n_bins, n_samples)
    return X, y
