"""
Scientific tests for the paper-specific figure calculations.

These tests intentionally live next to the additional paper figure scripts
rather than in the project-wide tests/ directory. The calculations tested here
are tied to the paper figures and are not intended to define a reusable API.

Run from the project root with:
python scripts/additional_paper_figures/tests_paper_figures.py
"""

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "functions").is_dir() and (parent / "scripts").is_dir()
)

for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from functions.grids import daily_grid_for_year
from scripts.additional_paper_figures.paper_figure_calculations import (
    build_monthly_midpoint_grid,
    compute_annual_amplitude_band,
    compute_low_frequency_band,
    compute_mean_seasonal_band,
    compute_polynomial_band,
    load_nino34_anomaly,
    map_monthly_series_to_grid,
    pearson_correlation_by_lag,
)


def degree1_sample(
        a0=0.0,
        a1=0.0,
        slow_pairs=None,
        b1=0.0,
        c1=0.0,
        bp1=0.0,
        cp1=0.0,
        b2=0.0,
        c2=0.0,
        b3=0.0,
        c3=0.0,
        b4=0.0,
        c4=0.0):
    """Build one model-parameter vector for a degree-1 polynomial model."""
    if slow_pairs is None:
        slow_pairs = []

    params = [a0, a1]

    for bL, cL in slow_pairs:
        params.extend([bL, cL])

    params.extend([b1, c1, bp1, cp1, b2, c2, b3, c3, b4, c4])

    return np.asarray(params, dtype=float)


def read_script(script_name):
    """Return the source text of a paper figure script."""
    return (SCRIPT_DIR / script_name).read_text(encoding="utf-8")


def test_polynomial_band_uses_posterior_percentiles_of_p_of_t():
    """Check Fig. 04 p(t) percentiles against an analytic polynomial example."""
    decimal_year_grid = np.array([2000.0, 2001.0, 2002.0])
    timezero = 2000.0
    samples = np.array([
        degree1_sample(a0=1.0, a1=0.5),
        degree1_sample(a0=2.0, a1=0.5),
        degree1_sample(a0=3.0, a1=0.5),
    ])

    p16, p50, p84 = compute_polynomial_band(
        samples,
        decimal_year_grid,
        timezero,
        polynomial_degree=1,
        include_slow_harmonics=False,
        base_period_slow_harmonics=30,
        slow_harmonics=[],
    )

    x_grid = decimal_year_grid - timezero
    expected_curves = np.array([sample[0] + sample[1] * x_grid for sample in samples])
    expected_p16, expected_p50, expected_p84 = np.percentile(expected_curves, [16, 50, 84], axis=0)

    np.testing.assert_allclose(p16, expected_p16)
    np.testing.assert_allclose(p50, expected_p50)
    np.testing.assert_allclose(p84, expected_p84)


def test_mean_seasonal_cycle_averages_time_varying_first_harmonic():
    """Check Fig. 05 mean s(t) keeps the bp1*t time dependence before averaging."""
    phase_grid = np.array([0.0, 0.25, 0.5, 0.75])
    years_for_mean = np.array([2000, 2001])
    timezero = 2000.0
    samples = np.array([
        degree1_sample(b1=2.0, bp1=1.0),
        degree1_sample(b1=2.0, bp1=1.0),
        degree1_sample(b1=2.0, bp1=1.0),
    ])

    p16, p50, p84 = compute_mean_seasonal_band(
        samples,
        phase_grid,
        years_for_mean,
        timezero,
        polynomial_degree=1,
        include_slow_harmonics=False,
        base_period_slow_harmonics=30,
        slow_harmonics=[],
    )

    mean_year_offset = np.mean(years_for_mean - timezero)
    expected = (2.0 + mean_year_offset + phase_grid) * np.sin(2.0 * np.pi * phase_grid)

    np.testing.assert_allclose(p16, expected, atol=1e-12)
    np.testing.assert_allclose(p50, expected, atol=1e-12)
    np.testing.assert_allclose(p84, expected, atol=1e-12)


def test_annual_amplitude_band_uses_peak_to_trough_seasonal_component():
    """Check Fig. 06 annual amplitude is max(s) - min(s) for each sample and year."""
    years = np.array([2001])
    timezero = 2000.0
    amplitudes = np.array([1.0, 2.0, 3.0])
    samples = np.array([degree1_sample(b1=amplitude) for amplitude in amplitudes])

    p16, p50, p84 = compute_annual_amplitude_band(
        samples,
        years,
        timezero,
        polynomial_degree=1,
        include_slow_harmonics=False,
        base_period_slow_harmonics=30,
        slow_harmonics=[],
    )

    x_grid = daily_grid_for_year(2001) - timezero
    expected_amplitudes = np.array([
        (
            np.max(amplitude * np.sin(2.0 * np.pi * x_grid))
            - np.min(amplitude * np.sin(2.0 * np.pi * x_grid))
        )
        for amplitude in amplitudes
    ])
    expected_p16, expected_p50, expected_p84 = np.percentile(expected_amplitudes, [16, 50, 84])

    np.testing.assert_allclose(p16, [expected_p16])
    np.testing.assert_allclose(p50, [expected_p50])
    np.testing.assert_allclose(p84, [expected_p84])


def test_low_frequency_band_returns_ell_and_analytical_derivative():
    """Check Fig. 07 ell(t) and d ell / dt against an analytic slow harmonic."""
    decimal_years = np.array([2000.0, 2001.0, 2002.0])
    timezero = 2000.0
    base_period = 4.0
    samples = np.array([
        degree1_sample(slow_pairs=[(2.0, -1.0)]),
        degree1_sample(slow_pairs=[(2.0, -1.0)]),
        degree1_sample(slow_pairs=[(2.0, -1.0)]),
    ])

    _, p50_lf, _, _, p50_dlf, _ = compute_low_frequency_band(
        samples,
        decimal_years,
        timezero,
        polynomial_degree=1,
        include_slow_harmonics=True,
        base_period_slow_harmonics=base_period,
        slow_harmonics=[1],
    )

    x = decimal_years - timezero
    omega = 2.0 * np.pi / base_period
    expected_lf = 2.0 * np.sin(omega * x) - np.cos(omega * x)
    expected_dlf = 2.0 * omega * np.cos(omega * x) + omega * np.sin(omega * x)

    np.testing.assert_allclose(p50_lf, expected_lf, atol=1e-12)
    np.testing.assert_allclose(p50_dlf, expected_dlf, atol=1e-12)


def test_monthly_midpoint_grid_builds_complete_yyyymm_axis():
    """Check Fig. 07 monthly grid uses all month keys from start to end year."""
    midpoint_dates, decimal_years, grid_keys = build_monthly_midpoint_grid(2001, 2001)

    assert len(midpoint_dates) == 12
    assert len(decimal_years) == 12
    assert grid_keys[0] == 200101
    assert grid_keys[-1] == 200112
    assert np.all(np.diff(decimal_years) > 0)
    assert midpoint_dates[0] == np.datetime64("2001-01-16T12:00:00")


def test_monthly_series_mapping_sorts_values_and_does_not_interpolate():
    """Check Fig. 07 maps monthly values to the common grid without interpolation."""
    years = np.array([2000, 2000, 2000, 2000])
    months = np.array([3, 1, 2, 4])
    values = np.array([3.0, 1.0, 2.0, np.nan])
    grid_keys = np.array([200001, 200002, 200003, 200004, 200005])

    mapped = map_monthly_series_to_grid(years, months, values, grid_keys)

    np.testing.assert_allclose(mapped[:3], [1.0, 2.0, 3.0])
    assert np.isnan(mapped[3])
    assert np.isnan(mapped[4])


def test_nino34_loader_uses_noaa_monthly_anomaly_column_and_midmonth_time():
    """Check Fig. 07 Nino 3.4 loading uses column 10 and midmonth decimal years."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "enso_index.txt"
        filepath.write_text(
            "YR MON A B C D E F G NINO34\n"
            "2000 1 0 0 0 0 0 0 0 1.5\n"
            "2000 2 0 0 0 0 0 0 0 -0.5\n"
            "2000 3 0 0 0 0 0 0 0 nan\n"
            "2001 1 0 0 0 0 0 0 0 9.0\n",
            encoding="utf-8",
        )

        years, months, decimal_years, nino34 = load_nino34_anomaly(filepath, 2000.0, 2000.25)

    np.testing.assert_array_equal(years, [2000, 2000])
    np.testing.assert_array_equal(months, [1, 2])
    np.testing.assert_allclose(decimal_years, [2000.0 + 0.5 / 12.0, 2000.0 + 1.5 / 12.0])
    np.testing.assert_allclose(nino34, [1.5, -0.5])


def test_lagged_correlation_identifies_positive_lag_when_nino_leads():
    """Check Fig. 07 lag convention: positive lag means y leads x."""
    rng = np.random.default_rng(123)
    y = rng.normal(size=80)
    lead_months = 3
    x = rng.normal(size=80)
    x[lead_months:] = y[:-lead_months]
    y[10] = np.nan
    x[10 + lead_months] = np.nan

    lags, r_values, best_lag, best_r = pearson_correlation_by_lag(x, y, max_lag=8)

    assert best_lag == lead_months
    np.testing.assert_allclose(best_r, 1.0, atol=1e-12)
    assert r_values[np.where(lags == lead_months)[0][0]] == best_r


def test_figure_scripts_use_tested_scientific_helpers():
    """Check that figures call the local scientific helper functions tested here."""
    expected_helpers = {
        "fig04.py": ["compute_polynomial_band"],
        "fig05.py": ["compute_mean_seasonal_band"],
        "fig06.py": ["compute_annual_amplitude_band"],
        "fig07.py": [
            "build_monthly_midpoint_grid",
            "compute_low_frequency_band",
            "load_nino34_anomaly",
            "map_monthly_series_to_grid",
            "pearson_correlation_by_lag",
        ],
    }

    for script_name, helper_names in expected_helpers.items():
        text = read_script(script_name)

        for helper_name in helper_names:
            assert f"{helper_name}(" in text


def run_tests():
    """Run all local paper-figure tests."""
    test_functions = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    n_passed = 0
    n_failed = 0

    print("Step 1: Run paper figure calculation tests")
    print(f"Test file: {Path(__file__).resolve()}")
    print(f"Project root: {PROJECT_ROOT}")
    print("-------------------------------------------------------")

    for test_function in test_functions:
        try:
            test_function()
        except Exception:
            n_failed += 1
            print(f"FAILED  {test_function.__name__}")
            traceback.print_exc()
        else:
            n_passed += 1
            print(f"PASSED  {test_function.__name__}")

    print("-------------------------------------------------------")
    print("Step 2: Summary")
    print(f"Passed: {n_passed}")
    print(f"Failed: {n_failed}")

    if n_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
