import numpy as np
from astropy.timeseries import LombScargle


def simulate_white_noise(n_points, sigma, rng):
    """
    Generate a Gaussian white-noise series.
    """
    return rng.normal(0.0, sigma, size=n_points)


def simulate_red_noise_ar1(n_points, alpha, sigma, rng):
    """
    Generate a stationary AR(1) red-noise series.
    """
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    if not np.isfinite(alpha) or not np.isfinite(sigma):
        raise ValueError("alpha and sigma must be finite")
    if abs(alpha) >= 1.0:
        raise ValueError("alpha must satisfy abs(alpha) < 1 for a stationary AR(1) process")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    noise = np.empty(n_points)
    noise[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - alpha**2))
    for i in range(1, n_points):
        noise[i] = alpha * noise[i - 1] + rng.normal(0.0, sigma)
    return noise


def estimate_red_noise_ar1_parameters(residuals):
    """
    Estimate the parameters of an AR(1) red-noise model from a residual series.
    """
    residuals = np.asarray(residuals)

    if len(residuals) < 3:
        raise ValueError("At least three residuals are required to estimate AR(1) parameters")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Residuals must be finite to estimate AR(1) parameters")

    centered_residuals = residuals - np.mean(residuals)

    y0 = centered_residuals[:-1]
    y1 = centered_residuals[1:]

    denominator = np.dot(y0, y0)
    if denominator == 0.0:
        raise ValueError("Residual variance is zero; AR(1) parameters cannot be estimated")

    alpha = np.dot(y0, y1) / denominator
    if not np.isfinite(alpha):
        raise ValueError("Estimated AR(1) alpha is not finite")
    if abs(alpha) >= 1.0:
        raise ValueError("Estimated AR(1) alpha is not stationary")

    innovations = y1 - alpha * y0
    sigma = np.std(innovations, ddof=1)
    if not np.isfinite(sigma):
        raise ValueError("Estimated AR(1) innovation sigma is not finite")

    return alpha, sigma


def compute_peak_exceedance_percentage(timestamps, residuals_for_noise_parameters, peak_power, peak_frequency, peak_frequency_sigma, noise_type, n_simulations=100000,
                                       window_half_width_in_sigmas=2.0, samples_per_peak=10, normalization="standard", rng_seed=123):
    """
    Compute the percentage of simulated noise realizations whose maximum Lomb-Scargle power within a frequency window around a peak exceeds
    the observed peak power.
    """
    timestamps = np.asarray(timestamps)
    residuals_for_noise_parameters = np.asarray(residuals_for_noise_parameters)

    rng = np.random.default_rng(rng_seed)

    n_points = len(timestamps)
    max_powers = np.empty(n_simulations)

    fmin = peak_frequency - window_half_width_in_sigmas * peak_frequency_sigma
    fmax = peak_frequency + window_half_width_in_sigmas * peak_frequency_sigma

    if noise_type == "white":
        sigma = np.std(residuals_for_noise_parameters, ddof=1)
    elif noise_type == "red":
        alpha, sigma = estimate_red_noise_ar1_parameters(residuals_for_noise_parameters)
    else:
        raise ValueError("noise_type must be 'white' or 'red'")

    for i in range(n_simulations):
        if noise_type == "white":
            simulated_series = simulate_white_noise(n_points, sigma, rng)
        else:
            simulated_series = simulate_red_noise_ar1(n_points, alpha, sigma, rng)

        ls_simulated = LombScargle(timestamps, simulated_series, fit_mean=True, center_data=True, normalization=normalization)

        _, simulated_power = ls_simulated.autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)

        max_powers[i] = np.max(simulated_power)

    exceedance_percentage = 100.0 * np.mean(max_powers > peak_power)

    return exceedance_percentage
