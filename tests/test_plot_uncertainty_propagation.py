"""
Tests for posterior-sample uncertainty propagation.

Derived curves and uncertainty bands must be computed from complete joint
posterior parameter vectors. These tests use synthetic samples to protect that
row-wise propagation.
"""

import numpy as np

from functions.model import model


def posterior_percentile_band(x, samples, polynomial_degree, slow_harmonics=None):
    """Propagate complete posterior rows and return the 16th, 50th and 84th percentiles."""
    if slow_harmonics is None:
        slow_harmonics = []

    y_fits = []

    for sample in samples:
        y_fit = model(
            x,
            *sample,
            polynomial_degree=polynomial_degree,
            include_slow_harmonics=len(slow_harmonics) > 0,
            base_period_slow_harmonics=30.0,
            slow_harmonics=slow_harmonics,
        )
        y_fits.append(y_fit)

    y_fits = np.asarray(y_fits)
    return np.percentile(y_fits, [16, 50, 84], axis=0)


def test_row_wise_posterior_propagation_preserves_joint_samples():
    """Use complete posterior vectors when forming percentile bands."""
    x = np.asarray([0.0, 0.25, 0.5])
    sample_low = np.asarray([1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sample_high = np.asarray([3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    samples = np.vstack([sample_low, sample_high])

    p16, p50, p84 = posterior_percentile_band(
        x,
        samples,
        polynomial_degree=2,
        slow_harmonics=[],
    )

    y_low = model(x, *sample_low, polynomial_degree=2, include_slow_harmonics=False, slow_harmonics=[])
    y_high = model(x, *sample_high, polynomial_degree=2, include_slow_harmonics=False, slow_harmonics=[])
    expected = np.percentile(np.vstack([y_low, y_high]), [16, 50, 84], axis=0)

    np.testing.assert_allclose(p16, expected[0])
    np.testing.assert_allclose(p50, expected[1])
    np.testing.assert_allclose(p84, expected[2])
