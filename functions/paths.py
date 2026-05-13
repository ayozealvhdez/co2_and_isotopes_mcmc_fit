import os
from pathlib import Path


def find_project_root(file_path):
    """
    Find the project root by searching parent directories for the
    'functions' and 'scripts' folders.
    """
    p = Path(file_path).resolve()
    for parent in p.parents:
        if (parent / "functions").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Project root not found.")


def model_tag(use_slow_harmonics, slow_base_period=30, slow_harmonics=None, polynomial_degree=2):
    """
    Return the subfolder name associated with the model configuration.

    Examples
    --------
    linear without slow harmonics:
        poly1_noSlow

    quadratic without slow harmonics:
        poly2_noSlow

    cubic without slow harmonics:
        poly3_noSlow

    quadratic with slow harmonics:
        poly2_withSlow_P30_K2-3-4-7-8

    cubic with slow harmonics:
        poly3_withSlow_P30_K2
    """
    if polynomial_degree not in [1, 2, 3]:
        raise ValueError("polynomial_degree must be 1, 2, or 3")

    poly_tag = f"poly{polynomial_degree}_"

    if use_slow_harmonics and slow_harmonics:
        ks = "-".join(map(str, slow_harmonics))
        return f"{poly_tag}withSlow_P{slow_base_period}_K{ks}"

    return f"{poly_tag}noSlow"


def run_directory(project_root, observable, site_acronym, data_frequency, use_slow_harmonics, slow_base_period=30, slow_harmonics=None, polynomial_degree=2):
    """
    Return the base directory associated with one specific run.

    The directory structure is:
        results_and_plots/observable/site/frequency/model_tag

    Examples
    --------
    CO2 monthly run at IZO with slow harmonics and a quadratic polynomial:
        results_and_plots/co2/izo/monthly/poly2_withSlow_P30_K2-3-4-7-8

    delta13CO2 discrete run at MLO without slow harmonics and a quadratic polynomial:
        results_and_plots/delta13c/mlo/discrete/poly2_noSlow

    Delta14CO2 monthly run at IZO without slow harmonics and a cubic polynomial:
        results_and_plots/delta14c/izo/monthly/poly3_noSlow
    """
    site_tag = site_acronym.lower()
    model_tag_str = model_tag(use_slow_harmonics, slow_base_period, slow_harmonics, polynomial_degree=polynomial_degree)
    return os.path.join(project_root, "results_and_plots", observable, site_tag, data_frequency, model_tag_str)


def run_results_directory(project_root, observable, site_acronym, data_frequency, use_slow_harmonics, slow_base_period=30, slow_harmonics=None, polynomial_degree=2):
    """
    Return the results directory associated with one specific run.

    This is the folder where numerical outputs are stored, such as:
    - MCMC samples
    - best-fit parameters
    - fitted values and residuals
    - summary files

    Example
    -------
    results_and_plots/co2/izo/monthly/poly2_withSlow_P30_K2-3-4-7-8/results
    """
    return os.path.join(run_directory(project_root, observable, site_acronym, data_frequency, use_slow_harmonics, slow_base_period, slow_harmonics, polynomial_degree=polynomial_degree), "results")


def run_plots_directory(project_root, observable, site_acronym, data_frequency, use_slow_harmonics, slow_base_period=30, slow_harmonics=None, polynomial_degree=2):
    """
    Return the plots directory associated with one specific run.

    This is the folder where figures of runs are stored, such as:
    - trace plots
    - corner plots
    - fitted time series

    Example
    -------
    results_and_plots/co2/izo/monthly/poly2_withSlow_P30_K2-3-4-7-8/plots
    """
    return os.path.join(run_directory(project_root, observable, site_acronym, data_frequency, use_slow_harmonics, slow_base_period, slow_harmonics, polynomial_degree=polynomial_degree), "plots")


def comparison_directory(project_root, comparison_name):
    """
    Return the directory associated with comparison figures.

    Comparison figures are not linked to a single run, because they may combine
    different observables, sites, frequencies, or model configurations.

    Example
    -------
    results_and_plots/comparisons/co2_and_isotopes_series
    """
    return os.path.join(project_root, "results_and_plots", "comparisons", comparison_name)
