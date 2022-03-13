import numpy as np
import scipy.special


def _calc_factors_from_posterior(alpha_posterior, beta_posterior):
    # The posterior distribution of f_j (factor in bin j)
    # follows a gamma distribution. See Leonard Held page ...

    # We want to use the median as estimate because it is more stable against
    # the log transformation. But calculating the median is much more
    # expensive for large values of alpha_posterior and beta_posterior.

    # Therefore we omit calculating the median and take the mean instead since
    # mean -> median for large values of alpha_posterior and beta_posterior

    noncritical_posterior = (alpha_posterior <= 1e12) & (beta_posterior <= 1e12)
    # Median of the gamma distribution
    posterior_gamma = (
        scipy.special.gammaincinv(alpha_posterior[noncritical_posterior], 0.5)
        / beta_posterior[noncritical_posterior]
    )

    factors = alpha_posterior / beta_posterior
    factors[noncritical_posterior] = posterior_gamma
    return np.log(factors)


__all__ = []
