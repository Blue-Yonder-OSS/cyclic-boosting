"""contain the evaluation functions referenced by evaluator classes."""
import numpy as np
from numpy import abs, square, nanmean, nanmedian, nansum


def mean_deviation(y, yhat) -> float:
    """Calculate the mean deviation (MD).

    The mean deviation is calculated by averaging the error from the true
    value. The smaller the value, the more accurate the model.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Mean deviation (MD)
    """
    return nanmean(y - yhat)


def mean_absolute_deviation(y, yhat) -> float:
    """Calculate the mean absolute deviation (MAD).

    The mean absolute deviation is calculated by averaging the absolute
    difference between the mean of the true value and the predicted value.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Mean absolute deviation (MAD)
    """
    return nanmean(abs(nanmean(y) - yhat))


def mean_square_error(y, yhat) -> float:
    """Calculate the mean square error (MSE).

    Mean square error is calculated by averaging the squares of the difference
    between the true value and the predicted value. The smaller the value, the
    more accurate the model.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Mean square error (MSE)
    """
    return nanmean(square(y - yhat))


def mean_absolute_error(y, yhat) -> float:
    """Calculate the mean absolute error (MAE).

    Mean absolute error is calculated as the average of the absolute
    difference between the true value and the predicted value. The smaller the
    value, the more accurate the model.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Mean absolute error (MAE)
    """
    return nanmean(abs(y - yhat))


def median_absolute_error(y, yhat) -> float:
    """Calculate the median absolute error (MedAE).

    The median absolute error is calculated as the median of the absolute
    difference between the true value and the predicted value. The smaller the
    value, the more accurate the model.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Median absolute error (MedAE)
    """
    return nanmedian(abs(y - yhat))


def weighted_absolute_percentage_error(y, yhat) -> float:
    """Calculate the weighted absolute percentage error (WAPE).

    The Weighted Absolute Error Rate measures the overall deviation of the
    predicted value from the observed value. It is calculated by taking the
    sum of the observed values and the sum of the predicted values and
    calculating the error between these two values. The smaller the value, the
    more accurate the model.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Weighted absolute percentage error (WAPE)
    """
    return nansum(abs(y - yhat) * y) / nansum(y)


def symmetric_mean_absolute_percentage_error(y, yhat) -> float:
    """Calculate the symmetric mean absolute percentage error (SMAPE).

    The symmetric mean absolute percentage error is calculated by dividing the
    absolute error by the average of the absolute values of the observed and
    predicted values. The smaller the value, the more accurate the model.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    Returns
    -------
    float
        Symmetric mean absolute percentage error (SMAPE)
    """
    smape = 100.0 * nanmean(abs(y - yhat) / ((abs(y) + abs(yhat)) / 2.0))
    return smape


def mean_y(y, **args) -> float:
    """Calculate the mean.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    Returns
    -------
    float
        Mean
    """
    return nanmean(y)


def coefficient_of_determination(y, yhat, k) -> float:
    """Calculate the coefficient of determination (COD).

    The coefficient of determination measures how well the independent
    variable(s) explain the variability of the dependent variable. It ranges
    from 0 to 1, with 0 indicating no explanatory power and 1 indicating
    perfect explanatory power.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    k : int
        Number of features (explanatory variables)

    Returns
    -------
    float
        Coefficient of determination (COD)
    """
    n = len(y)
    numerator = nansum((y - yhat) ** 2) / (n - k - 1)
    denominator = nansum((y - nanmean(y)) ** 2) / (n - 1)
    return 1 - (numerator / denominator)


def F(y, yhat, k) -> float:
    """Calculate the F-value.

    The F value, also known as the ratio of variances, is a statistical
    measure used in analysis of variance (ANOVA) and other methods. It
    compares the variability between group means with the variability within
    groups. It is calculated as the ratio of the "mean square of the
    regression variation" to the "mean square of the residual variation." The
    larger the value, the more meaningful the obtained regression equation is.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    k : int
        Number of features (explanatory variables)

    Returns
    -------
    float
        F-value
    """
    n = len(y)
    residual_variance = nansum((y - yhat) ** 2) / (n - k - 1)
    regression_variance = nansum((yhat - nanmean(y)) ** 2) / k
    return regression_variance / residual_variance


def F_quantile(
    y,
    yhat_q1,
    yhat_q2,
    q1,
    k,
) -> float:
    """Calculate the F value considering the quartile point.

    The F value, also known as the ratio of variances, is a statistical
    measure used in analysis of variance (ANOVA) and other methods. It
    compares the variability between group means with the variability within
    groups. It is calculated as the ratio of the "mean square of the
    regression variation" to the "mean square of the residual variation." The
    larger the value, the more meaningful the obtained regression equation is.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat_q1 : numpy.ndarray
        Predicted value at quantile 1

    yhat_q2 : numpy.ndarray
        Predicted value at quantile 2

    q1 : float
        Quantile

    k : int
        Number of features (explanatory variables)

    Returns
    -------
    float
        F-value
    """
    # from scipy.stats import f
    q2 = 1 - q1
    n = len(y)
    q_range = q2 - q1
    rss_indep = sum(square(v) for v in [yhat_q1, yhat_q2])
    rss_part = (rss_indep[1] - rss_indep[0]) / (q_range)
    sse_full = nansum(square(y - nanmean(y)))
    rss_full = sse_full - nansum(rss_indep)
    F_value = rss_part / (rss_full / (n - k - 1))
    # p_value = 1 - f.cdf(F_value, 1, n - k - 1)
    print("F-value: %.4f" % F_value)
    return F_value


def mean_pinball_loss(y, yhat, alpha) -> float:
    """Calculate the mean pinball loss.

    The pinball loss is a type of loss function used in quantile regression.
    It calculates the cost differently for overestimation and underestimation,
    and is asymmetric, which means it penalizes overestimation and
    underestimation differently depending on a specified quantile.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    yhat : numpy.ndarray
        Predicted value

    alpha : float
        Quantile

    Returns
    -------
    float
        Mean pinball loss
    """
    diff = y - yhat
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    output_errors = np.average(loss, axis=0)
    return output_errors


def probability_distribution_accuracy(y, pd_func) -> float:
    """Calculate probability distribution accuracy.

    Accuracy of the probability distribution calculated based on Wasserstein
    distance between the cumulative distribution function (CDF) of the
    predicted probability distribution at each observed value and the uniform
    distribution. The value range is from 0 to 1. The closer to 1, the better
    the accuracy. For more details, https://arxiv.org/abs/2009.07052.

    Parameters
    ----------
    y : numpy.ndarray
        Ground truth

    pd_func : list of scipy.stats._distn_infrastructure.rv_frozen
        List of instances obtained from the predict_proba function of the
        tornado predictor with the option output="func". Each instance is a
        fitted method collection. For more information,
        https://docs.scipy.org/doc/scipy/reference/stats.html

    Returns
    -------
    float
        Probability distribution accuracy
    """
    cdf_values = np.array([])
    for i, dist in enumerate(pd_func):
        cdf_value = dist.cdf(y[i])
        cdf_values = np.append(cdf_values, cdf_value)
    cdf_values = cdf_values[~np.isnan(cdf_values)]

    counts, _ = np.histogram(cdf_values, bins=100)
    n_cdf_bins = len(counts)
    pmf = counts / np.sum(counts)
    unif = np.full_like(pmf, 1.0 / len(pmf))

    cdf_pmf = np.cumsum(pmf)
    cdf_unif = np.cumsum(unif)
    wasser_distance = 2.0 * np.sum(np.abs(cdf_pmf - cdf_unif)) / len(cdf_pmf)
    wasser_distance = wasser_distance * n_cdf_bins / (n_cdf_bins - 1.0)

    acc = np.nanmean(1 - 2 * wasser_distance)
    return acc
