import numpy as np
from functions.model import model


def log_likelihood(params, x, y, yerr,
                   polynomial_degree=2,
                   include_slow_harmonics=True,
                   base_period_slow_harmonics=30,
                   slow_harmonics=None):
    """
    Compute the natural logarithm of the model likelihood,
    assuming independent Gaussian errors.

    Parameters
    ----------
    params : array-like
        Model parameter values.
    x, y : array-like
        Observed data (centered dates and observed values).
    yerr : array-like
        Uncertainty associated with each value in y.
    polynomial_degree : int, optional
        Degree of the polynomial trend. Allowed values are 1, 2, and 3.
    include_slow_harmonics : bool, optional
        Whether to include slow harmonics.
    base_period_slow_harmonics : float, optional
        Base period of the slow harmonics (in years).
    slow_harmonics : list, optional
        List of selected slow-harmonic orders
        (e.g. [2, 3, 4, 7, 8]).

    Returns
    -------
    float
        Log-likelihood value.
    """

    model_values = model(x,
                         *params,
                         polynomial_degree=polynomial_degree,
                         include_slow_harmonics=include_slow_harmonics,
                         base_period_slow_harmonics=base_period_slow_harmonics,
                         slow_harmonics=slow_harmonics)

    residuals = (y - model_values) / yerr
    return -0.5 * np.sum(residuals**2 + np.log(2 * np.pi * yerr**2))


def log_prior(params,
              polynomial_degree=2,
              polynomial_ranges=None,
              a0_range=(300, 400),
              a1_range=(0, 5),
              a2_range=(-0.1, 0.1),
              a3_range=(-0.01, 0.01),
              slow_harmonic_ranges=None,
              harmonic_ranges=None,
              slow_harmonics=None):
    """
    Compute the logarithm of the prior probability for a full parameter set,
    assuming uniform priors within the specified ranges.

    Parameters
    ----------
    params : array-like
        Full set of model parameters.
    polynomial_degree : int, optional
        Degree of the polynomial trend. Allowed values are 1, 2, and 3.
    polynomial_ranges : list of tuple, optional
        Allowed ranges for the polynomial coefficients.
        If provided, it overrides a0_range, a1_range, a2_range, and a3_range.
    a0_range, a1_range, a2_range, a3_range : tuple, optional
        Allowed ranges for the polynomial coefficients.
    slow_harmonic_ranges : list of tuple, optional
        Allowed ranges for each slow-harmonic coefficient pair (bLk, cLk).
        It can be empty if slow harmonics are not used.
    harmonic_ranges : list of tuple, optional
        Allowed ranges for the seasonal harmonic coefficients.
        It must have length 10: b1, c1, bp1, cp1, b2, c2, b3, c3, b4, c4.
    slow_harmonics : list, optional
        List of selected slow-harmonic orders.

    Returns
    -------
    float
        0.0 if all parameters lie within their allowed ranges,
        -np.inf otherwise.
    """

    if polynomial_degree not in [1, 2, 3]:
        return -np.inf

    if slow_harmonics is None:
        slow_harmonics = []

    if slow_harmonic_ranges is None:
        slow_harmonic_ranges = []

    if harmonic_ranges is None:
        harmonic_ranges = [(-5, 5)] * 10

    if polynomial_ranges is None:
        polynomial_ranges = [a0_range, a1_range]

        if polynomial_degree >= 2:
            polynomial_ranges.append(a2_range)

        if polynomial_degree >= 3:
            polynomial_ranges.append(a3_range)

    # Check that the lists of parameter ranges have the expected length
    if len(polynomial_ranges) != polynomial_degree + 1:
        return -np.inf

    if len(harmonic_ranges) != 10:  # There are 10 seasonal-harmonic coefficients
        return -np.inf

    if len(slow_harmonic_ranges) != 2 * len(slow_harmonics):
        # There are 2 * len(slow_harmonics) slow-harmonic coefficients
        return -np.inf

    n_poly = len(polynomial_ranges)
    nL = len(slow_harmonic_ranges)

    polynomial_params = params[:n_poly]
    bcsL = params[n_poly:n_poly + nL]
    bcs = params[n_poly + nL:]

    # Check total parameter length
    if len(params) != n_poly + nL + len(harmonic_ranges):
        return -np.inf

    # Check parameter ranges
    if any(p < r[0] or p > r[1] for p, r in zip(polynomial_params, polynomial_ranges)):
        return -np.inf

    if any(p < r[0] or p > r[1] for p, r in zip(bcsL, slow_harmonic_ranges)):
        return -np.inf

    if any(p < r[0] or p > r[1] for p, r in zip(bcs, harmonic_ranges)):
        return -np.inf

    return 0.0


def log_probability(params, x, y, yerr,
                    polynomial_degree=2,
                    polynomial_ranges=None,
                    a0_range=(300, 400),
                    a1_range=(0, 5),
                    a2_range=(-0.1, 0.1),
                    a3_range=(-0.01, 0.01),
                    slow_harmonic_ranges=None,
                    harmonic_ranges=None,
                    include_slow_harmonics=True,
                    base_period_slow_harmonics=30,
                    slow_harmonics=None):
    """
    Compute the total log-probability for a full parameter set by combining
    the log-prior and the log-likelihood.

    Parameters
    ----------
    params : array-like
        Model parameters.
    x, y, yerr : array-like
        Observed data and their uncertainties.
    polynomial_degree : int, optional
        Degree of the polynomial trend. Allowed values are 1, 2, and 3.
    polynomial_ranges : list of tuple, optional
        Allowed ranges for the polynomial coefficients.
        If provided, it overrides a0_range, a1_range, a2_range, and a3_range.
    a0_range, a1_range, a2_range, a3_range : tuple, optional
        Allowed ranges for the polynomial coefficients.
    slow_harmonic_ranges : list of tuple, optional
        Allowed ranges for the slow-harmonic coefficients.
    harmonic_ranges : list of tuple, optional
        Allowed ranges for the seasonal harmonic coefficients.
        It must have length 10.
    include_slow_harmonics : bool, optional
        Whether to include slow harmonics.
    base_period_slow_harmonics : float, optional
        Base period of the slow harmonics (in years).
    slow_harmonics : list, optional
        List of selected slow-harmonic orders
        (e.g. [2, 3, 4, 7, 8]).

    Returns
    -------
    float
        Total log-probability value.
    """

    lp = log_prior(params,
                   polynomial_degree=polynomial_degree,
                   polynomial_ranges=polynomial_ranges,
                   a0_range=a0_range,
                   a1_range=a1_range,
                   a2_range=a2_range,
                   a3_range=a3_range,
                   harmonic_ranges=harmonic_ranges,
                   slow_harmonic_ranges=slow_harmonic_ranges,
                   slow_harmonics=slow_harmonics)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(params, x, y, yerr,
                               polynomial_degree=polynomial_degree,
                               include_slow_harmonics=include_slow_harmonics,
                               base_period_slow_harmonics=base_period_slow_harmonics,
                               slow_harmonics=slow_harmonics)