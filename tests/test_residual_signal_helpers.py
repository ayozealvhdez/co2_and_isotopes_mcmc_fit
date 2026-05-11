"""
Tests for helper functions used by residual signal analysis.

These tests focus on small deterministic pieces of the residual-analysis logic:
the Gaussian profile used to refine candidate peaks and the synthetic noise
generators used for empirical false-alarm calculations.
"""

import numpy as np

import context  # noqa: F401
from functions.lombscargle_fap import (
    estimate_red_noise_ar1_parameters,
    simulate_red_noise_ar1,
    simulate_white_noise,
)
from functions.utilities import gauss


def test_gaussian_profile_has_requested_amplitude_at_center():
    x = np.asarray([0.0, 1.0, 2.0])

    y = gauss(x, amplitude=5.0, mu=1.0, sigma=0.5)

    assert np.isclose(y[1], 5.0)
    assert y[0] < y[1]
    assert y[2] < y[1]


def test_white_noise_simulation_is_reproducible_with_seeded_rng():
    rng_1 = np.random.default_rng(123)
    rng_2 = np.random.default_rng(123)

    noise_1 = simulate_white_noise(5, 2.0, rng_1)
    noise_2 = simulate_white_noise(5, 2.0, rng_2)

    np.testing.assert_allclose(noise_1, noise_2)
    assert noise_1.shape == (5,)


def test_red_noise_simulation_uses_first_order_autoregression():
    rng = np.random.default_rng(321)

    noise = simulate_red_noise_ar1(6, alpha=0.8, sigma=0.0, rng=rng)

    np.testing.assert_allclose(noise, np.zeros(6))


def test_ar1_parameter_estimate_recovers_known_positive_autocorrelation():
    residuals = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])

    alpha, sigma = estimate_red_noise_ar1_parameters(residuals)

    assert np.isfinite(alpha)
    assert np.isfinite(sigma)
    assert alpha > 0.0
    assert sigma >= 0.0
