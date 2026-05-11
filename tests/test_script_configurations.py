"""
Tests for scientific configurations written in executable scripts.

The fitting, plotting and residual-analysis scripts are currently script-style
files with top-level execution. These tests therefore inspect their source code
without importing them, avoiding MCMC runs, file writes and plotting side effects.
"""

import ast
from pathlib import Path

import context


FIT_SCRIPTS = {
    "co2": context.PROJECT_ROOT / "scripts" / "fit" / "fit_mcmc_co2.py",
    "delta13c": context.PROJECT_ROOT / "scripts" / "fit" / "fit_mcmc_delta13c.py",
    "delta14c": context.PROJECT_ROOT / "scripts" / "fit" / "fit_mcmc_delta14c.py",
}

PLOT_SCRIPTS = {
    "co2": context.PROJECT_ROOT / "scripts" / "fit" / "plot_fit_co2.py",
    "delta13c": context.PROJECT_ROOT / "scripts" / "fit" / "plot_fit_delta13c.py",
    "delta14c": context.PROJECT_ROOT / "scripts" / "fit" / "plot_fit_delta14c.py",
}

RESIDUAL_SCRIPTS = {
    "co2": context.PROJECT_ROOT / "scripts" / "residual_analysis" / "residual_signals_co2.py",
    "delta13c": context.PROJECT_ROOT / "scripts" / "residual_analysis" / "residual_signals_delta13c.py",
    "delta14c": context.PROJECT_ROOT / "scripts" / "residual_analysis" / "residual_signals_delta14c.py",
}

PAPER_MODEL_CONFIG = {
    "co2": {
        "polynomial_degree": 2,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [2, 3, 4, 7, 8],
    },
    "delta13c": {
        "polynomial_degree": 2,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [2, 3],
    },
    "delta14c": {
        "polynomial_degree": 3,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [2],
    },
}


def literal_assignments(script_path):
    """Return top-level literal assignments from a script without executing it."""
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    values = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value

    return values


def test_fit_scripts_use_common_timezero_and_paper_model_configuration():
    for observable, script_path in FIT_SCRIPTS.items():
        values = literal_assignments(script_path)
        expected = PAPER_MODEL_CONFIG[observable]

        assert values["timezero"] == 1985.0
        assert values["include_slow_harmonics"] is True
        assert values["polynomial_degree"] == expected["polynomial_degree"]
        assert values["base_period_slow_harmonics"] == expected["base_period_slow_harmonics"]
        assert values["slow_harmonics"] == expected["slow_harmonics"]


def test_fit_scripts_keep_sampler_settings_traceable():
    for script_path in FIT_SCRIPTS.values():
        values = literal_assignments(script_path)
        text = Path(script_path).read_text(encoding="utf-8")

        assert values["nwalkers"] == 128
        assert values["nsteps"] > 0
        assert "discard = int(0.5 * nsteps)" in text


def test_fit_scripts_build_parameter_names_in_model_order():
    for script_path in FIT_SCRIPTS.values():
        text = Path(script_path).read_text(encoding="utf-8")

        assert 'param_names = [f"a{i}" for i in range(polynomial_degree + 1)]' in text
        assert 'param_names.extend([f"bL{k}", f"cL{k}"])' in text
        assert "['b1', 'c1', 'bp1', 'cp1'" in text
        assert "'b2', 'c2', 'b3', 'c3', 'b4', 'c4']" in text


def test_fit_scripts_save_required_numerical_outputs():
    required_outputs = [
        "fit_summary_",
        "best_fit_and_residuals.txt",
        "samples_for_MC.txt",
        "# decimal_year\\tobserved\\tyerr\\tnvalues\\tfit\\tresidual",
    ]

    for script_path in FIT_SCRIPTS.values():
        text = Path(script_path).read_text(encoding="utf-8")

        for required_output in required_outputs:
            assert required_output in text


def test_plot_scripts_use_same_model_configuration_as_final_fits():
    for observable, script_path in PLOT_SCRIPTS.items():
        values = literal_assignments(script_path)
        expected = PAPER_MODEL_CONFIG[observable]

        assert values["timezero"] == 1985.0
        assert values["include_slow_harmonics"] is True
        assert values["polynomial_degree"] == expected["polynomial_degree"]
        assert values["base_period_slow_harmonics"] == expected["base_period_slow_harmonics"]
        assert values["slow_harmonics"] == expected["slow_harmonics"]


def test_residual_scripts_use_expected_frequency_range_and_fap_levels():
    for script_path in RESIDUAL_SCRIPTS.values():
        values = literal_assignments(script_path)

        assert values["fmin"] == 0.025
        assert values["fmax"] == 0.5
        assert values["samples_per_peak"] == 10
        assert values["lomb_scargle_normalization"] == "standard"
        assert values["fap_levels"] == [0.1587, 0.00135, 3.17e-5]


def test_residual_scripts_use_baluev_thresholds_and_sampling_windows():
    for script_path in RESIDUAL_SCRIPTS.values():
        text = Path(script_path).read_text(encoding="utf-8")

        assert 'method="baluev"' in text
        assert "np.ones_like" in text
        assert "center_data=False" in text
        assert "fit_mean=False" in text
        assert "find_peaks" in text
        assert "curve_fit(gauss" in text
