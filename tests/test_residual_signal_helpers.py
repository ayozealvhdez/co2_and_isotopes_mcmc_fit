"""
Tests for helper functions used by residual signal analysis.

The tests cover deterministic pieces of the residual-analysis workflow:
Gaussian peak profiles, white and red-noise simulations, AR(1) estimates, and
empirical peak exceedance.
"""

import numpy as np

from functions.lombscargle_fap import (
    compute_peak_exceedance_percentage,
    estimate_red_noise_ar1_parameters,
    simulate_red_noise_ar1,
    simulate_white_noise,
)
from functions.utilities import gauss


def test_gaussian_profile_has_requested_amplitude_at_center():
    """Evaluate the Gaussian peak at its centre."""
    x = np.asarray([0.0, 1.0, 2.0])

    y = gauss(x, amplitude=5.0, mu=1.0, sigma=0.5)

    assert np.isclose(y[1], 5.0)
    assert y[0] < y[1]
    assert y[2] < y[1]


def test_white_noise_simulation_is_reproducible_with_seeded_rng():
    """Repeat white-noise simulations with a fixed random seed."""
    rng_1 = np.random.default_rng(123)
    rng_2 = np.random.default_rng(123)

    noise_1 = simulate_white_noise(5, 2.0, rng_1)
    noise_2 = simulate_white_noise(5, 2.0, rng_2)

    np.testing.assert_allclose(noise_1, noise_2)
    assert noise_1.shape == (5,)


def test_red_noise_simulation_uses_stationary_ar1_initial_value():
    """Start AR(1) red noise from its stationary distribution."""
    alpha = 0.8
    sigma = 0.5

    rng = np.random.default_rng(321)
    noise = simulate_red_noise_ar1(6, alpha=alpha, sigma=sigma, rng=rng)

    rng_expected = np.random.default_rng(321)
    expected = np.empty(6)
    expected[0] = rng_expected.normal(0.0, sigma / np.sqrt(1.0 - alpha**2))
    for i in range(1, 6):
        expected[i] = alpha * expected[i - 1] + rng_expected.normal(0.0, sigma)

    np.testing.assert_allclose(noise, expected)


def test_red_noise_simulation_rejects_nonstationary_alpha():
    """Reject non-stationary AR(1) coefficients."""
    rng = np.random.default_rng(321)

    try:
        simulate_red_noise_ar1(6, alpha=1.0, sigma=0.5, rng=rng)
    except ValueError as exc:
        assert "stationary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-stationary alpha")


def test_ar1_parameter_estimate_recovers_known_positive_autocorrelation():
    """Recover positive serial correlation in a simple residual series."""
    residuals = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])

    alpha, sigma = estimate_red_noise_ar1_parameters(residuals)

    assert np.isfinite(alpha)
    assert np.isfinite(sigma)
    assert alpha > 0.0
    assert sigma >= 0.0


def test_ar1_parameter_estimate_rejects_constant_residuals():
    """Reject residuals with zero variance."""
    residuals = np.ones(5)

    try:
        estimate_red_noise_ar1_parameters(residuals)
    except ValueError as exc:
        assert "variance" in str(exc)
    else:
        raise AssertionError("Expected ValueError for constant residuals")


def fap_test_inputs():
    """Return a small deterministic setup for empirical FAP calculations."""
    timestamps = np.linspace(0.0, 20.0, 41)
    residuals_for_noise_parameters = np.asarray([
        0.20, 0.10, 0.15, 0.05, -0.02, -0.06, -0.04, 0.01,
        0.08, 0.12, 0.06, -0.01, -0.05, -0.03, 0.02, 0.07,
        0.11, 0.05, -0.02, -0.07, -0.04,
    ])

    return timestamps, residuals_for_noise_parameters


def test_peak_exceedance_white_noise_extreme_thresholds():
    """Return exact exceedance limits for white-noise thresholds."""
    timestamps, residuals_for_noise_parameters = fap_test_inputs()

    exceed_all = compute_peak_exceedance_percentage(
        timestamps=timestamps,
        residuals_for_noise_parameters=residuals_for_noise_parameters,
        peak_power=-np.inf,
        peak_frequency=0.20,
        peak_frequency_sigma=0.03,
        noise_type="white",
        n_simulations=5,
        samples_per_peak=2,
        rng_seed=123,
    )

    exceed_none = compute_peak_exceedance_percentage(
        timestamps=timestamps,
        residuals_for_noise_parameters=residuals_for_noise_parameters,
        peak_power=np.inf,
        peak_frequency=0.20,
        peak_frequency_sigma=0.03,
        noise_type="white",
        n_simulations=5,
        samples_per_peak=2,
        rng_seed=123,
    )

    assert exceed_all == 100.0
    assert exceed_none == 0.0


def test_peak_exceedance_red_noise_extreme_thresholds():
    """Return exact exceedance limits for AR(1) red-noise thresholds."""
    timestamps, residuals_for_noise_parameters = fap_test_inputs()

    exceed_all = compute_peak_exceedance_percentage(
        timestamps=timestamps,
        residuals_for_noise_parameters=residuals_for_noise_parameters,
        peak_power=-np.inf,
        peak_frequency=0.20,
        peak_frequency_sigma=0.03,
        noise_type="red",
        n_simulations=5,
        samples_per_peak=2,
        rng_seed=456,
    )

    exceed_none = compute_peak_exceedance_percentage(
        timestamps=timestamps,
        residuals_for_noise_parameters=residuals_for_noise_parameters,
        peak_power=np.inf,
        peak_frequency=0.20,
        peak_frequency_sigma=0.03,
        noise_type="red",
        n_simulations=5,
        samples_per_peak=2,
        rng_seed=456,
    )

    assert exceed_all == 100.0
    assert exceed_none == 0.0


def test_peak_exceedance_decreases_for_larger_peak_power():
    """Decrease exceedance when the peak-power threshold increases."""
    timestamps, residuals_for_noise_parameters = fap_test_inputs()

    low_threshold_exceedance = compute_peak_exceedance_percentage(
        timestamps=timestamps,
        residuals_for_noise_parameters=residuals_for_noise_parameters,
        peak_power=0.0,
        peak_frequency=0.20,
        peak_frequency_sigma=0.03,
        noise_type="white",
        n_simulations=10,
        samples_per_peak=2,
        rng_seed=789,
    )

    high_threshold_exceedance = compute_peak_exceedance_percentage(
        timestamps=timestamps,
        residuals_for_noise_parameters=residuals_for_noise_parameters,
        peak_power=1.0,
        peak_frequency=0.20,
        peak_frequency_sigma=0.03,
        noise_type="white",
        n_simulations=10,
        samples_per_peak=2,
        rng_seed=789,
    )

    assert 0.0 <= high_threshold_exceedance <= low_threshold_exceedance <= 100.0


def test_peak_exceedance_rejects_unknown_noise_type():
    """Reject unsupported noise-model labels."""
    timestamps, residuals_for_noise_parameters = fap_test_inputs()

    try:
        compute_peak_exceedance_percentage(
            timestamps=timestamps,
            residuals_for_noise_parameters=residuals_for_noise_parameters,
            peak_power=0.1,
            peak_frequency=0.20,
            peak_frequency_sigma=0.03,
            noise_type="pink",
            n_simulations=1,
            samples_per_peak=2,
            rng_seed=123,
        )
    except ValueError as exc:
        assert "noise_type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown noise type")
