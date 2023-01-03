import numpy as np

from cyclic_boosting.binning import _binary_search as binary_search


def test_le():
    z = np.array([3.0, 10.0, 20.0, 25.0])
    n = len(z)

    np.testing.assert_equal(
        [
            binary_search.le(z, z_searched, 0)
            for z_searched in [-1, 100, 11, 26, 2, 3, 10, 20, 25]
        ],
        [-1, n - 1, 1, 3, -1, 0, 1, 2, 3],
    )

    z = np.linspace(0.83, 1, 18)
    np.testing.assert_equal(binary_search.le(z, (0.83 + 1) / 2, 0), 8)

    z = np.concatenate([[0.52, 0.82], np.linspace(0.83, 1, 18)])

    np.testing.assert_equal(binary_search.le(z, (0.83 + 1) / 2, 0), 2 + 8)
    np.testing.assert_equal(binary_search.le(z, 1.0, 0), len(z) - 1)


def test_le_interp():
    z = np.array([3.0, 10.0, 20.0, 25.0])
    u = np.array([10.0, 12.0, 13.0, 50.0])
    outl, _ = -1, 100

    np.testing.assert_equal(
        [
            binary_search.le_interp(z, z_searched, u, outl, 0)
            for z_searched in [10, 15, 18, 0, 3, 30, 25, 20]
        ],
        [12, 12.5, 12 + 8 / 10, outl, 10, 50, 50, 13],
    )


def test_calculate_index_array():
    x = np.array([3.0, 2.0, 1.0, 4.0, np.nan, 2.1, 2.00001])
    x_unique = np.array([1.0, 2.0, 3.0])
    index_arr = np.arange(len(x_unique), dtype=float)
    result_arr = np.zeros(len(x))
    epsilon = 0.0001
    binary_search.eq_multi(x_unique, x, index_arr, epsilon, result_arr)
    xref = np.array([2.0, 1.0, 0.0, np.nan, np.nan, np.nan, 1.0])
    np.testing.assert_allclose(result_arr, xref)


def test_calculate_index_array_boolean_index():
    x = np.array([3.0, 2.0, 1.0, 4, np.nan, 2.1, 2.00001])
    x_unique = np.array([1.0, 2.0, 3.0])
    is_valid = x > 2
    index_arr = np.arange(len(x_unique), dtype=float)
    result_arr = np.zeros(len(x[is_valid]))
    epsilon = 0.0001

    binary_search.eq_multi(x_unique, x[is_valid], index_arr, epsilon, result_arr)

    xref = np.array([2.0, np.nan, np.nan, 1])
    np.testing.assert_allclose(result_arr, xref)


def test_calculate_index_array_order():
    x = np.floor(np.array([2.0, 5, 0]))

    x_unique = np.array([0.0, 2, 5])
    index_arr = np.arange(len(x_unique), dtype=float)
    result_arr = np.zeros(len(x))
    epsilon = 1e-9
    binary_search.eq_multi(x_unique, x, index_arr, epsilon, result_arr)
    xref = np.array([1.0, 2, 0])
    np.testing.assert_allclose(result_arr, xref)
