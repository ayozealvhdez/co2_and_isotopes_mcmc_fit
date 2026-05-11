import os
import numpy as np

from functions.model import model
from functions.paths import find_project_root, model_tag, run_results_directory


project_root = find_project_root(__file__)

runs = [
    {
        "name": "co2_izo",
        "observable": "co2",
        "site": "IZO",
        "data_tag": "monthly",
        "timezero": 1985.0,
        "polynomial_degree": 2,
        "include_slow_harmonics": True,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [2,3,4,7,8],
    },
    {
        "name": "co2_mlo",
        "observable": "co2",
        "site": "MLO",
        "data_tag": "monthly",
        "timezero": 1985.0,
        "polynomial_degree": 2,
        "include_slow_harmonics": True,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [2,3,4,7,8],
    },
    {
        "name": "delta13c_izo",
        "observable": "delta13c",
        "site": "IZO",
        "data_tag": "monthly",
        "timezero": 1985.0,
        "polynomial_degree": 2,
        "include_slow_harmonics": False,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [],
    },
    {
        "name": "delta13c_mlo",
        "observable": "delta13c",
        "site": "MLO",
        "data_tag": "monthly",
        "timezero": 1985.0,
        "polynomial_degree": 2,
        "include_slow_harmonics": False,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [],
    },
    {
        "name": "delta14c_izo",
        "observable": "delta14c",
        "site": "IZO",
        "data_tag": "monthly",
        "timezero": 1985.0,
        "polynomial_degree": 3,
        "include_slow_harmonics": False,
        "base_period_slow_harmonics": 30,
        "slow_harmonics": [],
    },
]


for run in runs:
    print("-------------------------------------------------------")
    print(f"Checking {run['name']}")

    results_dir = run_results_directory(project_root, run["observable"], run["site"], run["data_tag"], run["include_slow_harmonics"], run["base_period_slow_harmonics"], run["slow_harmonics"], polynomial_degree=run["polynomial_degree"])

    residuals_path = os.path.join(results_dir, "best_fit_and_residuals.txt")
    samples_path = os.path.join(results_dir, "samples_for_MC.txt")

    model_tag_str = model_tag(run["include_slow_harmonics"], run["base_period_slow_harmonics"], run["slow_harmonics"], polynomial_degree=run["polynomial_degree"])
    fit_summary_path = os.path.join(results_dir, f"fit_summary_{model_tag_str}.txt")

    if not os.path.exists(residuals_path):
        raise FileNotFoundError(f"Missing file: {residuals_path}")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Missing file: {samples_path}")
    if not os.path.exists(fit_summary_path):
        raise FileNotFoundError(f"Missing file: {fit_summary_path}")

    data = np.loadtxt(residuals_path, comments="#", ndmin=2)
    samples = np.loadtxt(samples_path, comments="#", ndmin=2)

    if data.shape[1] != 6:
        raise ValueError(f"{run['name']}: best_fit_and_residuals.txt should have 6 columns, found {data.shape[1]}")

    decimal_year = data[:, 0]
    observed = data[:, 1]
    yerr = data[:, 2]
    nvalues = data[:, 3]
    saved_fit = data[:, 4]
    saved_residuals = data[:, 5]

    if np.any(~np.isfinite(data)):
        raise ValueError(f"{run['name']}: non-finite values found in best_fit_and_residuals.txt")
    if np.any(~np.isfinite(samples)):
        raise ValueError(f"{run['name']}: non-finite values found in samples_for_MC.txt")
    if np.any(yerr <= 0):
        raise ValueError(f"{run['name']}: non-positive uncertainties found")
    if np.any(nvalues < 1):
        raise ValueError(f"{run['name']}: nvalues < 1 found")

    expected_nparams = run["polynomial_degree"] + 1 + 2 * len(run["slow_harmonics"]) * int(run["include_slow_harmonics"]) + 10
    if samples.shape[1] != expected_nparams:
        raise ValueError(f"{run['name']}: samples have {samples.shape[1]} columns, expected {expected_nparams}")

    if samples.shape[0] != 50000:
        print(f"WARNING: {run['name']}: samples_for_MC.txt has {samples.shape[0]} samples, not 50000")

    medians = np.median(samples, axis=0)
    x = decimal_year - run["timezero"]

    recomputed_fit = model(x, *medians, polynomial_degree=run["polynomial_degree"], include_slow_harmonics=run["include_slow_harmonics"], base_period_slow_harmonics=run["base_period_slow_harmonics"], slow_harmonics=run["slow_harmonics"])
    recomputed_residuals = observed - recomputed_fit

    max_fit_diff = np.max(np.abs(saved_fit - recomputed_fit))
    max_residual_diff = np.max(np.abs(saved_residuals - recomputed_residuals))

    print(f"Results directory: {results_dir}")
    print(f"N data = {len(data)}")
    print(f"N samples = {len(samples)}")
    print(f"N params = {samples.shape[1]}")
    print(f"Max fit difference = {max_fit_diff:.6e}")
    print(f"Max residual difference = {max_residual_diff:.6e}")

    if max_fit_diff > 1e-3:
        print("WARNING: saved fit does not exactly match the fit recomputed from posterior medians.")
    if max_residual_diff > 1e-3:
        print("WARNING: saved residuals do not exactly match observed - recomputed_fit.")

print("-------------------------------------------------------")
print("All checks completed.")