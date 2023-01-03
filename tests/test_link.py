import numpy as np

from cyclic_boosting import link


def test_loglink_simple():
    loglink = link.LogLinkMixin()
    x = np.linspace(1, 100)
    np.testing.assert_allclose(loglink.unlink_func(loglink.link_func(x)), x)

    assert loglink.link_func(np.e ** 10) == 10.0
    assert loglink.is_in_range(np.e ** 10)
    assert loglink.is_in_range(np.asarray(np.linspace(0.5, 10)))
    assert not loglink.is_in_range(np.asarray(np.linspace(0.0, 10)))
    assert not loglink.is_in_range(np.asarray(np.linspace(-10.0, 10)))
    assert not loglink.is_in_range(np.ones(10) * -1)


def test_logitlink_simple():
    logitlink = link.LogitLinkMixin()
    x = np.linspace(0.001, 0.999)
    np.testing.assert_allclose(logitlink.unlink_func(logitlink.link_func(x)), x)

    assert logitlink.link_func(0.5) == 0.0
    assert logitlink.link_func(0.25) == np.log(1.0 / 3)
    assert logitlink.link_func(0.75) == np.log(3.0)
    assert not logitlink.is_in_range(np.asarray(np.linspace(0.5, 10)))
    assert not logitlink.is_in_range(np.linspace(-1, 2))
    assert not logitlink.is_in_range(np.ones(10) * 1.01)


def test_identity_simple():
    inv_link = link.IdentityLinkMixin()
    x = np.linspace(0.001, 1000)
    np.testing.assert_allclose(inv_link.link_func(x), x)

    np.testing.assert_allclose(inv_link.unlink_func(inv_link.link_func(x)), x)
