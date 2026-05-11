"""
For the three observables (CO2, delta13C, delta14C), plot the fitted series, the posterior median model, the 68% confidence band, and the residuals.

Layout:
- Left column: CO2.
- Middle column: delta13C.
- Right column: delta14C.

For each observable:
- Upper panel: observed data, posterior median model, and 68% confidence band.
- Lower panel: residuals.

The script reads:
- 'samples_for_MC.txt', containing 50,000 posterior samples drawn from the MCMC chain during the execution of 'fit_mcmc_<observable>.py'.
- 'best_fit_and_residuals.txt', containing observed values, uncertainties, nvalues, best-fit values, and residuals at the observation timestamps.

To select the version of these files corresponding to the run you want to plot, the matching run configuration must be specified in the 'SELECTED RUNS' block.

The best-fit curve and the confidence band are computed by evaluating the model on a regular grid in decimal years for the 50,000 posterior samples stored in 'samples_for_MC.txt' and taking the 16th, 50th, and 84th percentiles at each grid point.

The result is stored in:
results_and_plots/comparisons/all_observables_fitted/
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.model import model
from functions.paths import find_project_root, run_results_directory, comparison_directory



# -------------------------------------------------------
# ------------------- SELECTED RUNS ---------------------
# -------------------------------------------------------

# ---------- CO2 IZO ----------
co2_site_acronym = "IZO"
co2_data_frequency = "monthly"
co2_polynomial_degree = 2
co2_include_slow_harmonics = True
co2_base_period_slow_harmonics = 30
co2_slow_harmonics = [2,3,4,7,8]
co2_timezero = 1985.0

# ---------- delta13C IZO ----------
d13c_site_acronym = "IZO"
d13c_recompute_monthly_series = True
d13c_polynomial_degree = 2
d13c_include_slow_harmonics = True
d13c_base_period_slow_harmonics = 30
d13c_slow_harmonics = [2,3]
d13c_timezero = 1985.0

# ---------- delta14C IZO ----------
d14c_site_acronym = "IZO"
d14c_recompute_monthly_series = True
d14c_polynomial_degree = 3
d14c_include_slow_harmonics = True
d14c_base_period_slow_harmonics = 30
d14c_slow_harmonics = [2]
d14c_timezero = 1985.0



# -------------------------------------------------------
# ----------------- GRIDS CONFIGURATION -----------------
# -------------------------------------------------------

co2_range = (1985, 2025)
d13c_range = (1992, 2025)
d14c_range = (1985, 2024)

n_grid = 1000



# -------------------------------------------------------
# ---------- FUNCTION TO COMPUTE UNCERTAINTY BAND --------
# -------------------------------------------------------

def compute_model_band(samples, decimal_year_grid, timezero, polynomial_degree, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics):
    x_grid = decimal_year_grid - timezero
    model_curves = np.empty((len(samples), len(decimal_year_grid)))

    for i, params in enumerate(samples):
        model_curves[i] = model(x_grid, *params, polynomial_degree=polynomial_degree, include_slow_harmonics=include_slow_harmonics, base_period_slow_harmonics=base_period_slow_harmonics, slow_harmonics=slow_harmonics)

    p16 = np.percentile(model_curves, 16, axis=0)
    p50 = np.percentile(model_curves, 50, axis=0)
    p84 = np.percentile(model_curves, 84, axis=0)

    return p16, p50, p84



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

d13c_data_tag = "monthly" if d13c_recompute_monthly_series else "discrete"
d14c_data_tag = "monthly" if d14c_recompute_monthly_series else "discrete"

co2_results_dir = run_results_directory(project_root, "co2", co2_site_acronym, co2_data_frequency, co2_include_slow_harmonics, co2_base_period_slow_harmonics, co2_slow_harmonics, polynomial_degree=co2_polynomial_degree)
d13c_results_dir = run_results_directory(project_root, "delta13c", d13c_site_acronym, d13c_data_tag, d13c_include_slow_harmonics, d13c_base_period_slow_harmonics, d13c_slow_harmonics, polynomial_degree=d13c_polynomial_degree)
d14c_results_dir = run_results_directory(project_root, "delta14c", d14c_site_acronym, d14c_data_tag, d14c_include_slow_harmonics, d14c_base_period_slow_harmonics, d14c_slow_harmonics, polynomial_degree=d14c_polynomial_degree)

co2_samples_path = os.path.join(co2_results_dir, "samples_for_MC.txt")
co2_best_fit_and_residuals_path = os.path.join(co2_results_dir, "best_fit_and_residuals.txt")

d13c_samples_path = os.path.join(d13c_results_dir, "samples_for_MC.txt")
d13c_best_fit_and_residuals_path = os.path.join(d13c_results_dir, "best_fit_and_residuals.txt")

d14c_samples_path = os.path.join(d14c_results_dir, "samples_for_MC.txt")
d14c_best_fit_and_residuals_path = os.path.join(d14c_results_dir, "best_fit_and_residuals.txt")

plot_dir = comparison_directory(project_root, "fig03_all_observables_fitted")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig03.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load files")

co2_samples = np.loadtxt(co2_samples_path, comments="#", ndmin=2)
co2_data = np.loadtxt(co2_best_fit_and_residuals_path, comments="#", ndmin=2)
co2_time = co2_data[:, 0]
co2_observed = co2_data[:, 1]
co2_yerr = co2_data[:, 2]
co2_nvalues = co2_data[:, 3].astype(int)
co2_fit = co2_data[:, 4]
co2_residuals = co2_data[:, 5]

d13c_samples = np.loadtxt(d13c_samples_path, comments="#", ndmin=2)
d13c_data = np.loadtxt(d13c_best_fit_and_residuals_path, comments="#", ndmin=2)
d13c_time = d13c_data[:, 0]
d13c_observed = d13c_data[:, 1]
d13c_yerr = d13c_data[:, 2]
d13c_nvalues = d13c_data[:, 3].astype(int)
d13c_fit = d13c_data[:, 4]
d13c_residuals = d13c_data[:, 5]

d14c_samples = np.loadtxt(d14c_samples_path, comments="#", ndmin=2)
d14c_data = np.loadtxt(d14c_best_fit_and_residuals_path, comments="#", ndmin=2)
d14c_time = d14c_data[:, 0]
d14c_observed = d14c_data[:, 1]
d14c_yerr = d14c_data[:, 2]
d14c_nvalues = d14c_data[:, 3].astype(int)
d14c_fit = d14c_data[:, 4]
d14c_residuals = d14c_data[:, 5]

print(f"Loaded CO2 samples from: {co2_samples_path}")
print(f"Loaded CO2 best fit and residuals from: {co2_best_fit_and_residuals_path}")
print(f"Loaded delta13C samples from: {d13c_samples_path}")
print(f"Loaded delta13C best fit and residuals from: {d13c_best_fit_and_residuals_path}")
print(f"Loaded delta14C samples from: {d14c_samples_path}")
print(f"Loaded delta14C best fit and residuals from: {d14c_best_fit_and_residuals_path}")
print("-------------------------------------------------------")



print("Step 2: Compute posterior median models and 68% confidence bands")

co2_decimal_year_grid = np.linspace(co2_range[0], co2_range[1], n_grid)
co2_p16, co2_p50, co2_p84 = compute_model_band(co2_samples, co2_decimal_year_grid, co2_timezero, co2_polynomial_degree, co2_include_slow_harmonics, co2_base_period_slow_harmonics, co2_slow_harmonics)

d13c_decimal_year_grid = np.linspace(d13c_range[0], d13c_range[1], n_grid)
d13c_p16, d13c_p50, d13c_p84 = compute_model_band(d13c_samples, d13c_decimal_year_grid, d13c_timezero, d13c_polynomial_degree, d13c_include_slow_harmonics, d13c_base_period_slow_harmonics, d13c_slow_harmonics)

d14c_decimal_year_grid = np.linspace(d14c_range[0], d14c_range[1], n_grid)
d14c_p16, d14c_p50, d14c_p84 = compute_model_band(d14c_samples, d14c_decimal_year_grid, d14c_timezero, d14c_polynomial_degree, d14c_include_slow_harmonics, d14c_base_period_slow_harmonics, d14c_slow_harmonics)

print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(17, 7.5), sharex="col", gridspec_kw={"height_ratios": [3, 1]})
fig.subplots_adjust(wspace=0.28, hspace=0.06)

ax11, ax12, ax13 = axes[0]
ax21, ax22, ax23 = axes[1]

data_style = dict(fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
residual_style = dict(fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)

# Upper-left panel: CO2 observed data and fitted model
ax11.errorbar(co2_time, co2_observed, yerr=co2_yerr, **data_style, label="Observed data")
ax11.plot(co2_decimal_year_grid, co2_p50, "b-", lw=0.8, label="Posterior median model")
ax11.fill_between(co2_decimal_year_grid, co2_p16, co2_p84, color="b", alpha=0.25, linewidth=0, label="68% confidence band")
ax11.set_ylabel("CO$_2$ (ppm)", fontsize=14)

# Lower-left panel: CO2 residuals
ax21.axhline(0, color="0.5", lw=1.0, linestyle="--")
ax21.errorbar(co2_time, co2_residuals, yerr=co2_yerr, **residual_style)
ax21.set_xlabel("Year", fontsize=14)
ax21.set_ylabel("Residuals (ppm)", fontsize=13)
ax21.set_xlim(1985,2025)

# Upper-middle panel: delta13C observed data and fitted model
ax12.errorbar(d13c_time, d13c_observed, yerr=d13c_yerr, **data_style)
ax12.plot(d13c_decimal_year_grid, d13c_p50, "b-", lw=0.8)
ax12.fill_between(d13c_decimal_year_grid, d13c_p16, d13c_p84, color="b", alpha=0.25, linewidth=0)
ax12.set_ylabel(r"$\delta^{13}$C-CO$_2$ ($\perthousand$)", fontsize=14)

# Lower-middle panel: delta13C residuals
ax22.axhline(0, color="0.5", lw=1.0, linestyle="--")
ax22.errorbar(d13c_time, d13c_residuals, yerr=d13c_yerr, **residual_style)
ax22.set_xlabel("Year", fontsize=14)
ax22.set_ylabel(r"Residuals ($\perthousand$)", fontsize=13)
ax22.set_xlim(1985,2025)

# Upper-right panel: delta14C observed data and fitted model
ax13.errorbar(d14c_time, d14c_observed, yerr=d14c_yerr, **data_style)
ax13.plot(d14c_decimal_year_grid, d14c_p50, "b-", lw=0.8)
ax13.fill_between(d14c_decimal_year_grid, d14c_p16, d14c_p84, color="b", alpha=0.25, linewidth=0)
ax13.set_ylabel(r"$\Delta^{14}$C-CO$_2$ ($\perthousand$)", fontsize=14)

# Lower-right panel: delta14C residuals
ax23.axhline(0, color="0.5", lw=1.0, linestyle="--")
ax23.errorbar(d14c_time, d14c_residuals, yerr=d14c_yerr, **residual_style)
ax23.set_xlabel("Year", fontsize=14)
ax23.set_ylabel(r"Residuals ($\perthousand$)", fontsize=13)
ax23.set_xlim(1985,2025)

# Axis formatting
for ax in (ax11, ax12, ax13, ax21, ax22, ax23):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=12, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
