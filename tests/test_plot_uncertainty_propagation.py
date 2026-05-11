"""
Tests for posterior-sample uncertainty propagation.

The paper requires derived curves and uncertainty bands to be computed by
evaluating complete joint posterior parameter vectors. These tests check that
the project plotting scripts use that pattern and demonstrate the expected
row-wise propagation on synthetic posterior samples.
"""

from pathlib import Path

import numpy as np

from functions.model import model


PLOT_SCRIPT_PATHS = [
    PROJECT_ROOT / "scripts" / "fit" / "plot_fit_co2.py",
    PROJECT_ROOT / "scripts" / "fit" / "plot_fit_delta13c.py",
    PROJECT_ROOT / "scripts" / "fit" / "plot_fit_delta14c.py",
]


def posterior_percentile_band(x, samples, polynomial_degree, slow_harmonics=None):
    """Evaluate the model row by row and return the 16th, 50th and 84th percentiles."""
    if slow_harmonics is None:
        slow_harmonics = []

    y_fits = np.asarray([
        model(
            x,
            *sample,
            polynomial_degree=polynomial_degree,
            include_slow_harmonics=len(slow_harmonics) > 0,
            base_period_slow_harmonics=30.0,
            slow_harmonics=slow_harmonics,
        )
        for sample in samples
    ])
    return np.percentile(y_fits, [16, 50, 84], axis=0)


def test_row_wise_posterior_propagation_preserves_joint_samples():
    """Check that uncertainty bands are computed from complete posterior rows."""
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


def test_plot_scripts_evaluate_model_for_each_joint_sample():
    """Check that plotting scripts evaluate the model for each saved sample."""
    for script_path in PLOT_SCRIPT_PATHS:
        text = script_path.read_text(encoding="utf-8")

        assert "for pars in samples" in text, f"{Path(script_path).name} should loop over joint samples"
        assert "model(" in text, f"{Path(script_path).name} should evaluate the model"
        assert "np.percentile" in text, f"{Path(script_path).name} should compute posterior percentiles"
        assert "[16, 50, 84]" in text, f"{Path(script_path).name} should use 16/50/84 percentiles"
        assert "axis=0" in text, f"{Path(script_path).name} should take percentiles at each grid point"


def test_plot_scripts_read_saved_joint_samples_not_parameter_sigmas():
    """Check that plotting scripts use saved posterior samples, not parameter sigmas."""
    for script_path in PLOT_SCRIPT_PATHS:
        text = script_path.read_text(encoding="utf-8")

        assert "samples_for_MC.txt" in text
        assert "best_fit_and_residuals.txt" in text
        assert "fit_summary_" not in text
        assert "sigmas" not in text
