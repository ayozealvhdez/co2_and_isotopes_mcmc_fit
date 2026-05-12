"""
Tests for reusable executable scripts.

The fitting, plotting and residual-analysis scripts are currently script-style
files with top-level execution. These tests therefore inspect their source code
without importing them, avoiding MCMC runs, file writes and plotting side effects.

The checks below avoid enforcing a particular paper configuration. User-editable
settings such as the model structure, MCMC length, burn-in, selected run and
periodogram range may change without breaking this reusable test suite.
"""

import ast
from pathlib import Path


FIT_SCRIPTS = {
    "co2": PROJECT_ROOT / "scripts" / "fit" / "fit_mcmc_co2.py",
    "delta13c": PROJECT_ROOT / "scripts" / "fit" / "fit_mcmc_delta13c.py",
    "delta14c": PROJECT_ROOT / "scripts" / "fit" / "fit_mcmc_delta14c.py",
}

PLOT_SCRIPTS = {
    "co2": PROJECT_ROOT / "scripts" / "fit" / "plot_fit_co2.py",
    "delta13c": PROJECT_ROOT / "scripts" / "fit" / "plot_fit_delta13c.py",
    "delta14c": PROJECT_ROOT / "scripts" / "fit" / "plot_fit_delta14c.py",
}

RESIDUAL_SCRIPTS = {
    "co2": PROJECT_ROOT / "scripts" / "residual_analysis" / "residual_signals_co2.py",
    "delta13c": PROJECT_ROOT / "scripts" / "residual_analysis" / "residual_signals_delta13c.py",
    "delta14c": PROJECT_ROOT / "scripts" / "residual_analysis" / "residual_signals_delta14c.py",
}

MODEL_SETTING_NAMES = [
    "timezero",
    "polynomial_degree",
    "include_slow_harmonics",
    "base_period_slow_harmonics",
    "slow_harmonics",
]

FIT_SETTING_NAMES = [
    "site_acronym",
    "input_file",
    "start_month",
    "end_month",
    "nwalkers",
    "nsteps",
    "discard",
    "corner_mode",
    "number_of_saved_samples",
]

PLOT_GRID_SETTING_NAMES = [
    "start_decimal_year_for_grid",
    "end_decimal_year_for_grid",
    "step_years_for_grid",
]

RESIDUAL_PERIODICITY_SETTING_NAMES = [
    "fmin",
    "fmax",
    "samples_per_peak",
    "lomb_scargle_normalization",
    "fap_levels",
    "n_simulations_for_fap",
]


def read_script(script_path):
    """Return the source text of a reusable script."""
    return Path(script_path).read_text(encoding="utf-8")


def assigned_names(script_path):
    """Return top-level assigned variable names from a script without executing it."""
    tree = ast.parse(read_script(script_path))
    names = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)

    return names


def assert_names_are_assigned(script_path, names):
    """Check that each expected configuration name is assigned at top level."""
    script_names = assigned_names(script_path)

    for name in names:
        assert name in script_names


def assert_text_contains(script_path, required_strings):
    """Check that each required code fragment appears in a script."""
    text = read_script(script_path)

    for required_string in required_strings:
        assert required_string in text


def test_fit_scripts_expose_user_editable_settings():
    """Check that fit scripts keep their main user-editable settings explicit."""
    for script_path in FIT_SCRIPTS.values():
        assert_names_are_assigned(script_path, FIT_SETTING_NAMES)
        assert_names_are_assigned(script_path, MODEL_SETTING_NAMES)


def test_fit_scripts_use_model_settings_in_paths_priors_and_model_calls():
    """Check that fit scripts pass user-selected model settings to shared helpers."""
    required_strings = [
        "if polynomial_degree not in [1, 2, 3]:",
        "model_tag(include_slow_harmonics",
        "run_results_directory(project_root",
        "run_plots_directory(project_root",
        "polynomial_degree=polynomial_degree",
        "include_slow_harmonics=include_slow_harmonics",
        "base_period_slow_harmonics=base_period_slow_harmonics",
        "slow_harmonics=slow_harmonics",
    ]

    for script_path in FIT_SCRIPTS.values():
        assert_text_contains(script_path, required_strings)


def test_fit_scripts_use_sampler_settings_without_fixed_values():
    """Check that sampler settings are wired through without enforcing their values."""
    required_strings = [
        "p0 = np.zeros((nwalkers, ndim))",
        "emcee.EnsembleSampler(nwalkers, ndim",
        "sampler.run_mcmc(p0, nsteps",
        "sampler.get_chain(discard=discard, flat=True)",
        '["nwalkers"',
        '["nsteps"',
        '["discard"',
    ]

    for script_path in FIT_SCRIPTS.values():
        assert_text_contains(script_path, required_strings)


def test_fit_scripts_build_parameter_names_in_model_order():
    """Check that saved parameter names follow the model parameter order."""
    required_strings = [
        'param_names = [f"a{i}" for i in range(polynomial_degree + 1)]',
        'param_names.extend([f"bL{k}", f"cL{k}"])',
        "['b1', 'c1', 'bp1', 'cp1'",
        "'b2', 'c2', 'b3', 'c3', 'b4', 'c4']",
    ]

    for script_path in FIT_SCRIPTS.values():
        assert_text_contains(script_path, required_strings)


def test_fit_scripts_save_required_numerical_outputs():
    """Check that fit scripts write the expected numerical output files."""
    required_strings = [
        "fit_summary_",
        "best_fit_and_residuals.txt",
        "samples_for_MC.txt",
        "# decimal_year\\tobserved\\tyerr\\tnvalues\\tfit\\tresidual",
    ]

    for script_path in FIT_SCRIPTS.values():
        assert_text_contains(script_path, required_strings)


def test_plot_scripts_expose_selected_run_and_grid_settings():
    """Check that plot scripts keep selected-run and plotting-grid settings explicit."""
    for script_path in PLOT_SCRIPTS.values():
        assert_names_are_assigned(script_path, ["site_acronym"])
        assert_names_are_assigned(script_path, MODEL_SETTING_NAMES)
        assert_names_are_assigned(script_path, PLOT_GRID_SETTING_NAMES)


def test_plot_scripts_use_selected_settings_to_read_and_evaluate_runs():
    """Check that plot scripts use selected settings when reading and evaluating runs."""
    required_strings = [
        "run_results_directory(project_root",
        "run_plots_directory(project_root",
        "samples_for_MC.txt",
        "best_fit_and_residuals.txt",
        "x_fit_grid = grid_decimal_dates - timezero",
        "for pars in samples",
        "polynomial_degree=polynomial_degree",
        "include_slow_harmonics=include_slow_harmonics",
        "base_period_slow_harmonics=base_period_slow_harmonics",
        "slow_harmonics=slow_harmonics",
    ]

    for script_path in PLOT_SCRIPTS.values():
        assert_text_contains(script_path, required_strings)


def test_residual_scripts_expose_periodogram_settings():
    """Check that residual-analysis scripts keep periodogram settings explicit."""
    for script_path in RESIDUAL_SCRIPTS.values():
        assert_names_are_assigned(script_path, RESIDUAL_PERIODICITY_SETTING_NAMES)


def test_residual_scripts_use_configured_periodogram_settings():
    """Check that residual-analysis scripts pass configured settings to periodogram calls."""
    required_strings = [
        "run_results_directory(project_root",
        "LombScargle(",
        "autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=samples_per_peak)",
        'false_alarm_level(fap_levels, method="baluev"',
        "np.ones_like",
        "center_data=False",
        "fit_mean=False",
        "find_peaks",
        "curve_fit(gauss",
        "compute_peak_exceedance_percentage(",
        "n_simulations=n_simulations_for_fap",
        "samples_per_peak=samples_per_peak",
        "normalization=lomb_scargle_normalization",
    ]

    for script_path in RESIDUAL_SCRIPTS.values():
        assert_text_contains(script_path, required_strings)
