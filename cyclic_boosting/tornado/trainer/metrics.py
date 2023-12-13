import numpy as np
from numpy import abs, square, nanmean, nanmedian, nansum


def mean_deviation(y, yhat) -> float:
    return nanmean(y - yhat)


def mean_absolute_deviation(y, yhat) -> float:
    return nanmean(abs(nanmean(y) - yhat))


def mean_square_error(y, yhat) -> float:
    return nanmean(square(y - yhat))


def mean_absolute_error(y, yhat) -> float:
    return nanmean(abs(y - yhat))


def median_absolute_error(y, yhat) -> float:
    return nanmedian(abs(y - yhat))


def weighted_absolute_percentage_error(y, yhat) -> float:
    return nansum(abs(y - yhat) * y) / nansum(y)


def symmetric_mean_absolute_percentage_error(y, yhat) -> float:
    smape = 100.0 * nanmean(abs(y - yhat) / ((abs(y) + abs(yhat)) / 2.0))
    return smape


def mean_y(y, **args) -> float:
    return nanmean(y)


def coefficient_of_determination(y, yhat, k) -> float:
    n = len(y)
    numerator = nansum((y - yhat) ** 2) / (n - k - 1)
    denominator = nansum((y - nanmean(y)) ** 2) / (n - 1)
    return 1 - (numerator / denominator)


def F(y, yhat, k) -> float:
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
    diff = y - yhat
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    output_errors = np.average(loss, axis=0)
    return output_errors
