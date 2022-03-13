from __future__ import absolute_import, division, print_function

import numpy as np


def constant_learn_rate_one(iteration, maximal_iteration, feature=None):
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate returned is always 1.

    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature


    .. plot::
        :include-source:

        from matplotlib import pyplot as plt
        import numpy as np
        from cyclic_boosting.plot_utils import _nbpy_style_figure
        from cyclic_boosting import learning_rate

        iterations = np.arange(1, 11)
        max_iter = 10

        learn_rate = [learning_rate.constant_learn_rate_one(x, max_iter) for x in iterations]
        plt.close("all")
        with _nbpy_style_figure(figsize=(13., 8.)):
            plt.plot(iterations, learn_rate)
    """
    return 1.0


def linear_learn_rate(iteration, maximal_iteration, feature=None):
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate is linear increasing each iteration until it reaches 1 in
    the last iteration.


    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature


    .. plot::
        :include-source:

        from matplotlib import pyplot as plt
        import numpy as np
        from cyclic_boosting.plot_utils import _nbpy_style_figure
        from cyclic_boosting import learning_rate

        iterations = np.arange(1, 11)
        max_iter = 10

        learn_rate = learning_rate.linear_learn_rate(iterations, max_iter)
        plt.close("all")
        with _nbpy_style_figure(figsize=(13., 8.)):
            plt.plot(iterations, learn_rate)
    """
    return iteration * (1.0 / maximal_iteration)


def logistic_learn_rate(iteration, maximal_iteration, feature=None):
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate has a logistic form.

    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature


    .. plot::
        :include-source:

        from matplotlib import pyplot as plt
        import numpy as np
        from cyclic_boosting.plot_utils import _nbpy_style_figure
        from cyclic_boosting import learning_rate

        iterations = np.arange(1, 11)
        max_iter = 10

        learn_rate = learning_rate.logistic_learn_rate(iterations, max_iter)
        plt.close("all")
        with _nbpy_style_figure(figsize=(13., 8.)):
            plt.plot(iterations, learn_rate)
    """
    saturation_value = 0.999999999
    x_t = maximal_iteration / np.log((1 - saturation_value) / (1 + saturation_value))
    return (1.0 / (1.0 + np.exp(iteration / x_t)) - 0.5) * 2


def half_linear_learn_rate(iteration, maximal_iteration, feature=None):
    """Function to specify the learning rate of a cyclic boosting iteration.
    The learning rate is linear increasing each iteration until it reaches 1 in
    half of the iterations.


    Parameters
    ----------

    iteration: int
        Current major cyclic boosting iteration.

    maximal_iteration: int
        Maximal cyclic boosting iteration.

    feature: :class:`cyclic_boosting.base.Feature`
        Feature


    .. plot::
        :include-source:

        from matplotlib import pyplot as plt
        import numpy as np
        from cyclic_boosting.plot_utils import _nbpy_style_figure
        from cyclic_boosting import learning_rate

        iterations = np.arange(1, 11)
        max_iter = 10

        learn_rate = learning_rate.half_linear_learn_rate(iterations, max_iter)
        plt.close("all")
        with _nbpy_style_figure(figsize=(13., 8.)):
            plt.plot(iterations, learn_rate)
    """
    return np.minimum(
        linear_learn_rate(iteration, maximal_iteration * 0.5, feature), 1.0
    )
