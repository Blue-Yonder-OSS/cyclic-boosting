import numpy as np

from cyclic_boosting import learning_rate


def test_constant_learn_rate_one():
    assert (
        learning_rate.constant_learn_rate_one(
            np.random.randint(100), np.random.randint(100), None
        )
        == 1.0
    )


def test_linear_learn_rate():
    iterations = np.arange(1, 11)
    max_iter = np.max(iterations)
    expected = np.linspace(1 / max_iter, 1, len(iterations))
    np.testing.assert_allclose(
        expected, learning_rate.linear_learn_rate(iterations, max_iter)
    )


def test_logistic_learn_rate():
    for i in range(30):
        max_iter = i + 2
        iterations = np.arange(1, max_iter)
        learn_rate = learning_rate.logistic_learn_rate(iterations, max_iter)
        assert 1 - learn_rate[-1] < 0.05
        assert np.all(learn_rate > 0.0)
        assert np.all(learn_rate < 1.0)
        assert np.all(np.diff(learn_rate) > 0.0)


def test_half_linear_learn_rate():
    expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rates = learning_rate.half_linear_learn_rate(np.arange(1, 11), 10)
    np.testing.assert_allclose(rates, expected)

    expected = np.array([0.66666667, 1.0, 1.0])
    rates = learning_rate.half_linear_learn_rate(np.arange(1, 4), 3)
    np.testing.assert_allclose(rates, expected)

    expected = np.array([1.0, 1.0])
    rates = learning_rate.half_linear_learn_rate(np.arange(1, 3), 2)
    np.testing.assert_allclose(rates, expected)

    expected = np.array([1.0])
    rates = learning_rate.half_linear_learn_rate(np.arange(1, 2), 1)
    np.testing.assert_allclose(rates, expected)
