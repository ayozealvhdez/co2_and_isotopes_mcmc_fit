"""
Plot the fitted Delta14CO2 series, the posterior median model, the 68% confidence band, the posterior median p(t) + l(t) component with its 68% confidence band, and the residuals from a previous MCMC run performed with 'fit_mcmc_delta14c.py'.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chain during the execution of 'fit_mcmc_delta14c.py'
- 'best_fit_and_residuals.txt', containing observed values, uncertainties, nvalues, best-fit values and residuals at the observation timestamps

To select the version of these files corresponding to the run you want to plot, the matching run configuration must be specified in the 'SELECTED RUN' block.

The posterior median model curve, the posterior median p(t) + l(t) curve, and their confidence bands are computed on a regular grid in decimal years for all posterior samples by taking the 16th, 50th (median), and 84th percentiles at each grid point.

The script stores the following plot in results_and_plots/delta14c/<site_acronym.lower()>/<data_tag>/<model_tag>/plots/
- Upper panel: Observed Delta14CO2 series alongside the posterior median model and the posterior median p(t) + l(t) component, with shaded bands indicating 68% CL derived from Monte Carlo with joint posterior vectors. Lower panel: residuals of the fit.
"""


# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.model import model_components
from functions.paths import find_project_root, run_results_directory, run_plots_directory



# -------------------------------------------------------
# -------------------- SELECTED RUN ---------------------
# -------------------------------------------------------
site_acronym = "IZO"
recompute_monthly_series = True
polynomial_degree = 3
include_slow_harmonics = True
base_period_slow_harmonics = 30
slow_harmonics = [2]

timezero = 1985.0  # Must match the value used in the fitting run



# -------------------------------------------------------
# ---------------- GRID CONFIGURATION -------------------
# -------------------------------------------------------
start_decimal_year_for_grid = 1985.0
end_decimal_year_for_grid = 2024.0
step_years_for_grid = 1 / 100   # resolution for the posterior median model curve and uncertainty band



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------
project_root = find_project_root(__file__)
data_tag = "monthly" if recompute_monthly_series else "discrete"

results_dir = run_results_directory(project_root, "delta14c", site_acronym, data_tag, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics, polynomial_degree=polynomial_degree)
plots_dir = run_plots_directory(project_root, "delta14c", site_acronym, data_tag, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics, polynomial_degree=polynomial_degree)
os.makedirs(plots_dir, exist_ok=True)

samples_path = os.path.join(results_dir, "samples_for_MC.txt")
residuals_path = os.path.join(results_dir, "best_fit_and_residuals.txt")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Load posterior samples and residuals")

samples = np.loadtxt(samples_path, comments="#", ndmin=2)
residual_data = np.loadtxt(residuals_path, comments="#", ndmin=2)

decimal_dates = residual_data[:, 0]
delta14c = residual_data[:, 1]
yerr = residual_data[:, 2]
residuals = residual_data[:, 5]

print(f"Loaded {len(samples)} posterior samples from: {samples_path}")
print(f"Loaded observed values and residuals from: {residuals_path}")
print("-------------------------------------------------------")



print(f"Step 2: Build the grid of timestamps used to calculate the model curve and uncertainty band")
grid_decimal_dates = np.arange(start_decimal_year_for_grid, end_decimal_year_for_grid + step_years_for_grid, step_years_for_grid)
x_fit_grid = grid_decimal_dates - timezero
print("-------------------------------------------------------")



print(f"Step 3: Compute posterior median model and p(t) + l(t) curves with 68% confidence bands from {len(samples)} posterior samples")
y_fits_grid = np.empty((len(samples), len(grid_decimal_dates)))
trend_fits_grid = np.empty((len(samples), len(grid_decimal_dates)))

for i, pars in enumerate(samples):
    poly, seasonal, lf = model_components(
        x_fit_grid,
        *pars,
        polynomial_degree=polynomial_degree,
        include_slow_harmonics=include_slow_harmonics,
        base_period_slow_harmonics=base_period_slow_harmonics,
        slow_harmonics=slow_harmonics,
    )
    y_fits_grid[i] = poly + seasonal + lf
    trend_fits_grid[i] = poly + lf

p16, p50, p84 = np.percentile(y_fits_grid, [16, 50, 84], axis=0)
trend_p16, trend_p50, trend_p84 = np.percentile(trend_fits_grid, [16, 50, 84], axis=0)

print("-------------------------------------------------------")



print("Step 4: Plot the fitted series, uncertainty band, and residuals")

fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    figsize=(15, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)


# Upper panel: fit, p(t) + l(t), and data
trend_color = "crimson"

ax1.errorbar(
    decimal_dates,
    delta14c,
    yerr=yerr,
    fmt="ko",
    markersize=3,
    elinewidth=0.8,
    capsize=2,
    capthick=0.8,
)
ax1.fill_between(
    grid_decimal_dates,
    p16,
    p84,
    color="blue",
    alpha=0.15,
    zorder=1,
)
ax1.fill_between(
    grid_decimal_dates,
    trend_p16,
    trend_p84,
    color=trend_color,
    alpha=0.16,
    linewidth=0,
    zorder=1,
)
ax1.plot(
    grid_decimal_dates,
    p50,
    "b",
    linewidth=0.8,
    zorder=2,
)
ax1.plot(
    grid_decimal_dates,
    trend_p50,
    color=trend_color,
    linewidth=1.1,
    zorder=3,
)
ax1.set_ylabel(r"$\Delta^{14}$CO$_2$ ($\perthousand$)", size=20)
ax1.tick_params(axis="both", labelsize=18, direction="in", top=True, right=True, length=6)
ax1.minorticks_on()
ax1.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
ax1.set_xlim(start_decimal_year_for_grid, end_decimal_year_for_grid)


# Lower panel: residuals
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.errorbar(
    decimal_dates,
    residuals,
    yerr=yerr,
    fmt="ko",
    markersize=3,
    elinewidth=0.8,
    capsize=2,
    capthick=0.8,
)
ax2.set_xlabel("Year", size=20)
ax2.set_ylabel(r"$\Delta^{14}$CO$_2$ - $f(t)$ ($\perthousand$)", size=20)
ax2.tick_params(axis="both", labelsize=18, direction="in", top=True, right=True, length=6)
ax2.minorticks_on()
ax2.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)

fig.tight_layout()
fig.subplots_adjust(left=0.09, right=0.95, top=0.94, bottom=0.10)

ax1.yaxis.set_label_coords(-0.05, 0.5)
ax2.yaxis.set_label_coords(-0.05, 0.5)

output_path = os.path.join(plots_dir, "mcmc_fit_with_band.png")
fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)
plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
