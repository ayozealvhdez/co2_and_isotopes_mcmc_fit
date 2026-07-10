"""
Tests for run paths and fit metrics.

These tests protect deterministic names and simple summary quantities used in
fit outputs.
"""

import os

import numpy as np

from functions.chi2 import calculate_chi2
from functions.paths import model_tag, run_plots_directory, run_results_directory


def test_model_tag_records_polynomial_degree_and_slow_harmonics():
    """Build stable model tags from polynomial and slow-harmonic settings."""
    assert model_tag(False, polynomial_degree=1) == "poly1_noSlow"
    assert model_tag(False, polynomial_degree=2) == "poly2_noSlow"
    assert model_tag(False, polynomial_degree=3) == "poly3_noSlow"
    assert model_tag(True, slow_base_period=30, slow_harmonics=[2, 3, 7], polynomial_degree=2) == "poly2_withSlow_P30_K2-3-7"


def test_model_tag_rejects_unsupported_polynomial_degree():
    """Reject polynomial degrees outside the model definition."""
    try:
        model_tag(False, polynomial_degree=4)
    except ValueError:
        return

    raise AssertionError("model_tag should reject polynomial_degree = 4")


def test_run_directories_are_built_from_run_configuration():
    """Build results and plots paths from the same run configuration."""
    project_root = os.path.normpath("/project")

    results_dir = run_results_directory(
        project_root,
        "co2",
        "IZO",
        "monthly",
        True,
        slow_base_period=30,
        slow_harmonics=[2, 3],
        polynomial_degree=2,
    )
    plots_dir = run_plots_directory(
        project_root,
        "co2",
        "IZO",
        "monthly",
        True,
        slow_base_period=30,
        slow_harmonics=[2, 3],
        polynomial_degree=2,
    )

    expected_base = os.path.join(
        project_root,
        "results_and_plots",
        "co2",
        "izo",
        "monthly",
        "poly2_withSlow_P30_K2-3",
    )
    assert results_dir == os.path.join(expected_base, "results")
    assert plots_dir == os.path.join(expected_base, "plots")


def test_chi2_matches_manual_calculation():
    """Compute chi2, degrees of freedom and reduced chi2 for a small fit."""
    observed_means = np.asarray([1.0, 2.0, 4.0])
    observed_stds = np.asarray([0.5, 1.0, 2.0])
    y_fit = np.asarray([1.5, 1.0, 2.0])

    chi2, dof, reduced_chi2 = calculate_chi2(
        observed_means,
        observed_stds,
        y_fit,
        n_parameters=1,
    )

    expected_chi2 = ((1.0 - 1.5) / 0.5) ** 2 + ((2.0 - 1.0) / 1.0) ** 2 + ((4.0 - 2.0) / 2.0) ** 2
    np.testing.assert_allclose(chi2, expected_chi2)
    assert dof == 2
    np.testing.assert_allclose(reduced_chi2, expected_chi2 / 2)


def test_chi2_rejects_non_positive_degrees_of_freedom():
    """Reject chi2 summaries with no positive degrees of freedom."""
    try:
        calculate_chi2(
            observed_means=np.asarray([1.0, 2.0]),
            observed_stds=np.asarray([0.1, 0.1]),
            y_fit=np.asarray([1.0, 2.0]),
            n_parameters=2,
        )
    except ValueError:
        return

    raise AssertionError("calculate_chi2 should reject non-positive degrees of freedom")
