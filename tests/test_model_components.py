"""
Tests for the mathematical model components.

These tests check that the implementation follows the model definition used in
the paper: a polynomial long-term term, annual seasonal harmonics up to the
fourth harmonic with a linearly varying first harmonic, and an optional
low-frequency Fourier component.
"""

import numpy as np

from functions.model import model, model_components


def build_params(polynomial_coeffs, slow_pairs=None, seasonal_coeffs=None):
    """Build a parameter vector using the order expected by functions.model."""
    if slow_pairs is None:
        slow_pairs = []
    if seasonal_coeffs is None:
        seasonal_coeffs = np.zeros(10)

    params = list(polynomial_coeffs)
    for b_slow, c_slow in slow_pairs:
        params.extend([b_slow, c_slow])
    params.extend(seasonal_coeffs)
    return np.asarray(params, dtype=float)


def test_polynomial_component_uses_configured_degree():
    """Check that p(t) is evaluated with all terms up to the selected degree."""
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    polynomial_coeffs = [2.0, -0.5, 0.25, 0.1]
    params = build_params(polynomial_coeffs)

    poly, seasonal, low_frequency = model_components(
        x,
        *params,
        polynomial_degree=3,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    expected_poly = (
        polynomial_coeffs[0]
        + polynomial_coeffs[1] * x
        + polynomial_coeffs[2] * x**2
        + polynomial_coeffs[3] * x**3
    )

    np.testing.assert_allclose(poly, expected_poly)
    np.testing.assert_allclose(seasonal, np.zeros_like(x))
    np.testing.assert_allclose(low_frequency, np.zeros_like(x))


def test_first_annual_harmonic_coefficients_vary_linearly_with_time():
    """Check the b1 + bp1*t and c1 + cp1*t terms of the annual harmonic."""
    x = np.asarray([0.125, 0.375, 1.125, 1.375])
    polynomial_coeffs = [0.0, 0.0, 0.0]
    seasonal_coeffs = np.asarray([
        2.0,
        -1.0,
        0.5,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])
    params = build_params(polynomial_coeffs, seasonal_coeffs=seasonal_coeffs)

    poly, seasonal, low_frequency = model_components(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    expected = (
        (2.0 + 0.5 * x) * np.sin(2.0 * np.pi * x)
        + (-1.0 - 0.25 * x) * np.cos(2.0 * np.pi * x)
    )

    np.testing.assert_allclose(poly, np.zeros_like(x))
    np.testing.assert_allclose(seasonal, expected)
    np.testing.assert_allclose(low_frequency, np.zeros_like(x))


def test_seasonal_component_includes_fixed_harmonics_two_to_four():
    """Check that the fixed annual harmonics k = 2, 3 and 4 are included."""
    x = np.asarray([0.1, 0.2, 0.3])
    polynomial_coeffs = [0.0, 0.0, 0.0]
    seasonal_coeffs = np.asarray([
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        -0.5,
        0.25,
        0.1,
        -0.2,
    ])
    params = build_params(polynomial_coeffs, seasonal_coeffs=seasonal_coeffs)

    _, seasonal, _ = model_components(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    expected = (
        1.0 * np.sin(2.0 * np.pi * 2 * x)
        + 2.0 * np.cos(2.0 * np.pi * 2 * x)
        - 0.5 * np.sin(2.0 * np.pi * 3 * x)
        + 0.25 * np.cos(2.0 * np.pi * 3 * x)
        + 0.1 * np.sin(2.0 * np.pi * 4 * x)
        - 0.2 * np.cos(2.0 * np.pi * 4 * x)
    )

    np.testing.assert_allclose(seasonal, expected)


def test_low_frequency_component_uses_selected_harmonics_and_base_period():
    """Check l(t) for selected low-frequency harmonics and a 30-year base period."""
    x = np.asarray([0.0, 1.5, 3.0])
    polynomial_coeffs = [0.0, 0.0, 0.0]
    slow_pairs = [(1.0, -0.5), (0.25, 0.75)]
    seasonal_coeffs = np.zeros(10)
    params = build_params(polynomial_coeffs, slow_pairs, seasonal_coeffs)

    poly, seasonal, low_frequency = model_components(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=True,
        base_period_slow_harmonics=30.0,
        slow_harmonics=[2, 7],
    )

    expected = (
        1.0 * np.sin(2.0 * np.pi * 2 * x / 30.0)
        - 0.5 * np.cos(2.0 * np.pi * 2 * x / 30.0)
        + 0.25 * np.sin(2.0 * np.pi * 7 * x / 30.0)
        + 0.75 * np.cos(2.0 * np.pi * 7 * x / 30.0)
    )

    np.testing.assert_allclose(poly, np.zeros_like(x))
    np.testing.assert_allclose(seasonal, np.zeros_like(x))
    np.testing.assert_allclose(low_frequency, expected)


def test_model_equals_sum_of_reported_components():
    """Check that f(t) equals p(t) + s(t) + l(t)."""
    x = np.linspace(0.0, 2.0, 6)
    params = build_params(
        polynomial_coeffs=[1.0, 0.2, -0.01],
        slow_pairs=[(0.4, -0.2)],
        seasonal_coeffs=[0.5, 0.1, 0.01, -0.02, 0.2, -0.1, 0.05, 0.03, -0.02, 0.04],
    )

    poly, seasonal, low_frequency = model_components(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=True,
        base_period_slow_harmonics=30.0,
        slow_harmonics=[2],
    )
    total = model(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=True,
        base_period_slow_harmonics=30.0,
        slow_harmonics=[2],
    )

    np.testing.assert_allclose(total, poly + seasonal + low_frequency)


def test_low_frequency_component_is_zero_when_disabled():
    """Check that l(t) is zero when no slow harmonics are included."""
    x = np.asarray([0.0, 1.0, 2.0])
    params = build_params(
        polynomial_coeffs=[0.0, 0.0, 0.0],
        slow_pairs=[],
        seasonal_coeffs=np.zeros(10),
    )

    _, _, low_frequency = model_components(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=False,
        slow_harmonics=[],
    )

    np.testing.assert_allclose(low_frequency, np.zeros_like(x))


def test_low_frequency_derivative_matches_analytical_expression():
    """Check the analytical derivative of the low-frequency component."""
    x = np.asarray([0.5, 1.5, 2.5])
    params = build_params(
        polynomial_coeffs=[0.0, 0.0, 0.0],
        slow_pairs=[(1.5, -0.4)],
        seasonal_coeffs=np.zeros(10),
    )
    omega = 2.0 * np.pi * 2 / 30.0

    _, _, _, derivative = model_components(
        x,
        *params,
        polynomial_degree=2,
        include_slow_harmonics=True,
        base_period_slow_harmonics=30.0,
        slow_harmonics=[2],
        return_dlf=True,
    )

    expected = 1.5 * omega * np.cos(omega * x) - (-0.4) * omega * np.sin(omega * x)
    np.testing.assert_allclose(derivative, expected)


def test_invalid_polynomial_degree_raises_value_error():
    """Check that unsupported polynomial degrees are rejected."""
    x = np.asarray([0.0, 1.0])
    params = build_params([0.0, 0.0, 0.0])

    try:
        model(
            x,
            *params,
            polynomial_degree=4,
            include_slow_harmonics=False,
            slow_harmonics=[],
        )
    except ValueError:
        return

    raise AssertionError("model should raise ValueError for polynomial_degree = 4")
