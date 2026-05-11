"""
Tests for the Bayesian probability functions.

These tests check the flat bounded prior, the Gaussian log-likelihood with its
normalization term, and the posterior as the sum of prior and likelihood.
"""

import numpy as np

import context  # noqa: F401
from functions.mcmc_probability import log_likelihood, log_prior, log_probability
from functions.model import model


def test_log_likelihood_matches_normalized_gaussian_expression():
    x = np.asarray([0.0, 0.25, 0.5])
    yerr = np.asarray([0.2, 0.3, 0.4])
    params = np.asarray([
        1.0,
        0.2,
        0.01,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
    y_model = model(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )
    y = y_model + np.asarray([0.1, -0.2, 0.05])

    expected = -0.5 * np.sum(((y - y_model) / yerr) ** 2 + np.log(2.0 * np.pi * yerr**2))
    actual = log_likelihood(
        params,
        x,
        y,
        yerr,
        polynomial_degree=2,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    np.testing.assert_allclose(actual, expected)


def test_log_prior_accepts_complete_parameter_vector_inside_ranges():
    params = np.asarray([
        350.0,
        2.0,
        0.01,
        0.2,
        -0.2,
        0.1,
        -0.1,
        1.0,
        -1.0,
        0.1,
        -0.1,
        0.2,
        -0.2,
        0.05,
        -0.05,
        0.01,
        -0.01,
    ])

    actual = log_prior(
        params,
        polynomial_degree=2,
        polynomial_ranges=[(300, 400), (0, 5), (-0.1, 0.1)],
        slow_harmonic_ranges=[(-2, 2), (-2, 2), (-2, 2), (-2, 2)],
        harmonic_ranges=[
            (-5, 5),
            (-5, 5),
            (-1, 1),
            (-1, 1),
            (-5, 5),
            (-5, 5),
            (-1, 1),
            (-1, 1),
            (-0.5, 0.5),
            (-0.5, 0.5),
        ],
        slow_harmonics=[2, 3],
    )

    assert actual == 0.0


def test_log_prior_rejects_parameter_outside_any_range():
    params = np.zeros(13)
    params[0] = 999.0

    actual = log_prior(
        params,
        polynomial_degree=2,
        polynomial_ranges=[(300, 400), (-1, 1), (-1, 1)],
        slow_harmonic_ranges=[],
        harmonic_ranges=[(-1, 1)] * 10,
        slow_harmonics=[],
    )

    assert actual == -np.inf


def test_log_prior_rejects_inconsistent_range_lengths():
    params = np.zeros(13)

    bad_polynomial_ranges = log_prior(
        params,
        polynomial_degree=2,
        polynomial_ranges=[(-1, 1), (-1, 1)],
        slow_harmonic_ranges=[],
        harmonic_ranges=[(-1, 1)] * 10,
        slow_harmonics=[],
    )
    bad_harmonic_ranges = log_prior(
        params,
        polynomial_degree=2,
        polynomial_ranges=[(-1, 1), (-1, 1), (-1, 1)],
        slow_harmonic_ranges=[],
        harmonic_ranges=[(-1, 1)] * 9,
        slow_harmonics=[],
    )

    assert bad_polynomial_ranges == -np.inf
    assert bad_harmonic_ranges == -np.inf


def test_log_probability_returns_minus_infinity_when_prior_rejects():
    x = np.asarray([0.0, 0.25])
    y = np.asarray([1.0, 1.1])
    yerr = np.asarray([0.1, 0.1])
    params = np.zeros(13)
    params[0] = 999.0

    actual = log_probability(
        params,
        x,
        y,
        yerr,
        polynomial_degree=2,
        polynomial_ranges=[(300, 400), (-1, 1), (-1, 1)],
        slow_harmonic_ranges=[],
        harmonic_ranges=[(-1, 1)] * 10,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    assert actual == -np.inf


def test_log_probability_equals_likelihood_for_valid_flat_prior():
    x = np.asarray([0.0, 0.25])
    y = np.asarray([1.0, 1.1])
    yerr = np.asarray([0.1, 0.1])
    params = np.zeros(13)
    params[0] = 1.0

    expected = log_likelihood(
        params,
        x,
        y,
        yerr,
        polynomial_degree=2,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )
    actual = log_probability(
        params,
        x,
        y,
        yerr,
        polynomial_degree=2,
        polynomial_ranges=[(-10, 10), (-10, 10), (-10, 10)],
        slow_harmonic_ranges=[],
        harmonic_ranges=[(-10, 10)] * 10,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    np.testing.assert_allclose(actual, expected)
