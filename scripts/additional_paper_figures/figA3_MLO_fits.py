"""
For the two MLO observables available in this repository (CO2 and delta13CO2),
plot the fitted series, the posterior median model, the 68% confidence band,
the Thoning-like seasonally adjusted trend p(t) + l(t), and the residuals.

Layout:
- Upper block: CO2, spanning the full figure width.
- Lower-left block: delta13CO2.
- Lower-right block: empty Delta14CO2 placeholder, because no MLO Delta14CO2
  data are available.

For each observable:
- Upper panel: observed data, posterior median model and 68% confidence band,
  plus the Thoning-like seasonally adjusted trend p(t) + l(t) and its 68%
  confidence band.
- Lower panel: residuals.

The script reads:
- 'samples_for_MC.txt', containing posterior samples drawn from the MCMC chain
  during the execution of 'fit_mcmc_<observable>.py'.
- 'best_fit_and_residuals.txt', containing observed values, uncertainties,
  nvalues, best-fit values, and residuals at the observation timestamps.

To select the version of these files corresponding to the run you want to plot,
the matching run configuration must be specified in the 'SELECTED RUNS' block.

The posterior median model curve, the Thoning-like seasonally adjusted trend
p(t) + l(t), and their confidence bands are computed by evaluating the relevant
model components on a regular grid in decimal years for the posterior samples
stored in 'samples_for_MC.txt' and taking the 16th, 50th, and 84th percentiles
at each grid point.

The Thoning-like trend is computed within this model framework; it is not a
reproduction of the NOAA CCGCRV filtering procedure.

The result is stored in:
results_and_plots/comparisons/figA3_MLO_fits/figA3_MLO_fits.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.model import model_components
from functions.paths import find_project_root, run_results_directory, comparison_directory



# -------------------------------------------------------
# ------------------- SELECTED RUNS ---------------------
# -------------------------------------------------------

# ---------- CO2 MLO ----------
co2_site_acronym = "MLO"
co2_data_frequency = "monthly"
co2_polynomial_degree = 2
co2_include_slow_harmonics = True
co2_base_period_slow_harmonics = 30
co2_slow_harmonics = [2, 3, 4, 7, 8]
co2_timezero = 1985.0

# ---------- delta13CO2 MLO ----------
d13c_site_acronym = "MLO"
d13c_recompute_monthly_series = True
d13c_polynomial_degree = 2
d13c_include_slow_harmonics = True
d13c_base_period_slow_harmonics = 30
d13c_slow_harmonics = [2, 3]
d13c_timezero = 1985.0



# -------------------------------------------------------
# ----------------- GRIDS CONFIGURATION -----------------
# -------------------------------------------------------

co2_range = (1985, 2025)
d13c_range = (1992, 2025)

n_grid = 1000



# -------------------------------------------------------
# ---------- FUNCTION TO COMPUTE UNCERTAINTY BAND --------
# -------------------------------------------------------

def compute_model_and_trend_bands(samples, decimal_year_grid, timezero, polynomial_degree, include_slow_harmonics, base_period_slow_harmonics, slow_harmonics):
    """
    Compute posterior bands for f(t) and for the seasonally adjusted trend p(t) + l(t).
    """
    x_grid = decimal_year_grid - timezero
    model_curves = np.empty((len(samples), len(decimal_year_grid)))
    trend_curves = np.empty((len(samples), len(decimal_year_grid)))

    for i, params in enumerate(samples):
        poly, seasonal, lf = model_components(
            x_grid,
            *params,
            polynomial_degree=polynomial_degree,
            include_slow_harmonics=include_slow_harmonics,
            base_period_slow_harmonics=base_period_slow_harmonics,
            slow_harmonics=slow_harmonics,
        )
        model_curves[i] = poly + seasonal + lf
        trend_curves[i] = poly + lf

    model_p16, model_p50, model_p84 = np.percentile(model_curves, [16, 50, 84], axis=0)
    trend_p16, trend_p50, trend_p84 = np.percentile(trend_curves, [16, 50, 84], axis=0)

    return (model_p16, model_p50, model_p84), (trend_p16, trend_p50, trend_p84)



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

d13c_data_tag = "monthly" if d13c_recompute_monthly_series else "discrete"

co2_results_dir = run_results_directory(project_root, "co2", co2_site_acronym, co2_data_frequency, co2_include_slow_harmonics, co2_base_period_slow_harmonics, co2_slow_harmonics, polynomial_degree=co2_polynomial_degree)
d13c_results_dir = run_results_directory(project_root, "delta13c", d13c_site_acronym, d13c_data_tag, d13c_include_slow_harmonics, d13c_base_period_slow_harmonics, d13c_slow_harmonics, polynomial_degree=d13c_polynomial_degree)

co2_samples_path = os.path.join(co2_results_dir, "samples_for_MC.txt")
co2_best_fit_and_residuals_path = os.path.join(co2_results_dir, "best_fit_and_residuals.txt")

d13c_samples_path = os.path.join(d13c_results_dir, "samples_for_MC.txt")
d13c_best_fit_and_residuals_path = os.path.join(d13c_results_dir, "best_fit_and_residuals.txt")

plot_dir = comparison_directory(project_root, "figA3_MLO_fits")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "figA3_MLO_fits.png")



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

print(f"Loaded CO2 samples from: {co2_samples_path}")
print(f"Loaded CO2 best fit and residuals from: {co2_best_fit_and_residuals_path}")
print(f"Loaded delta13CO2 samples from: {d13c_samples_path}")
print(f"Loaded delta13CO2 best fit and residuals from: {d13c_best_fit_and_residuals_path}")
print("-------------------------------------------------------")



print("Step 2: Compute posterior median models and 68% confidence bands")

co2_decimal_year_grid = np.linspace(co2_range[0], co2_range[1], n_grid)
(co2_p16, co2_p50, co2_p84), (co2_trend_p16, co2_trend_p50, co2_trend_p84) = compute_model_and_trend_bands(co2_samples, co2_decimal_year_grid, co2_timezero, co2_polynomial_degree, co2_include_slow_harmonics, co2_base_period_slow_harmonics, co2_slow_harmonics)

d13c_decimal_year_grid = np.linspace(d13c_range[0], d13c_range[1], n_grid)
(d13c_p16, d13c_p50, d13c_p84), (d13c_trend_p16, d13c_trend_p50, d13c_trend_p84) = compute_model_and_trend_bands(d13c_samples, d13c_decimal_year_grid, d13c_timezero, d13c_polynomial_degree, d13c_include_slow_harmonics, d13c_base_period_slow_harmonics, d13c_slow_harmonics)

print("-------------------------------------------------------")



print("Step 3: Plot the figure")

fig = plt.figure(figsize=(12.4, 10.2))
outer_gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[1.35, 1.0],
    hspace=0.18,
    wspace=0.22,
)

co2_gs = outer_gs[0, :].subgridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.0], hspace=0.06)
d13c_gs = outer_gs[1, 0].subgridspec(nrows=2, ncols=1, height_ratios=[2.2, 0.8], hspace=0.06)
d14c_gs = outer_gs[1, 1].subgridspec(nrows=2, ncols=1, height_ratios=[2.2, 0.8], hspace=0.06)

ax11 = fig.add_subplot(co2_gs[0, 0])
ax21 = fig.add_subplot(co2_gs[1, 0], sharex=ax11)
ax12 = fig.add_subplot(d13c_gs[0, 0])
ax22 = fig.add_subplot(d13c_gs[1, 0], sharex=ax12)
ax13 = fig.add_subplot(d14c_gs[0, 0])
ax23 = fig.add_subplot(d14c_gs[1, 0], sharex=ax13)
d14c_note_ax = fig.add_subplot(outer_gs[1, 1])

trend_color = "crimson"

# Upper block: CO2 observed data and fitted model
ax11.errorbar(co2_time, co2_observed, yerr=co2_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8, label="Observed data")
ax11.fill_between(co2_decimal_year_grid, co2_p16, co2_p84, color="b", alpha=0.25, linewidth=0, label="68% confidence band")
ax11.fill_between(co2_decimal_year_grid, co2_trend_p16, co2_trend_p84, color=trend_color, alpha=0.16, linewidth=0, label=r"68% band for trend")
ax11.plot(co2_decimal_year_grid, co2_p50, "b-", lw=0.8, label="Posterior median model")
ax11.plot(co2_decimal_year_grid, co2_trend_p50, color=trend_color, lw=1.1, label=r"Thoning-like trend, $p(t)+l(t)$")
ax11.set_ylabel("CO$_2$ (ppm)", fontsize=14)

# CO2 residuals
ax21.axhline(0, color="0.5", lw=1.0, linestyle="--")
ax21.errorbar(co2_time, co2_residuals, yerr=co2_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
ax21.set_xlabel("Year", fontsize=14)
ax21.set_ylabel("Residuals (ppm)", fontsize=13)
ax21.set_xlim(1985, 2025)

# Lower block: delta13CO2 observed data and fitted model
ax12.errorbar(d13c_time, d13c_observed, yerr=d13c_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
ax12.fill_between(d13c_decimal_year_grid, d13c_p16, d13c_p84, color="b", alpha=0.25, linewidth=0)
ax12.fill_between(d13c_decimal_year_grid, d13c_trend_p16, d13c_trend_p84, color=trend_color, alpha=0.16, linewidth=0)
ax12.plot(d13c_decimal_year_grid, d13c_p50, "b-", lw=0.8)
ax12.plot(d13c_decimal_year_grid, d13c_trend_p50, color=trend_color, lw=1.1)
ax12.set_ylabel(r"$\delta^{13}$CO$_2$ ($\perthousand$)", fontsize=14)

# delta13CO2 residuals
ax22.axhline(0, color="0.5", lw=1.0, linestyle="--")
ax22.errorbar(d13c_time, d13c_residuals, yerr=d13c_yerr, fmt="ko", markersize=3, elinewidth=0.8, capsize=2, capthick=0.8)
ax22.set_xlabel("Year", fontsize=14)
ax22.set_ylabel(r"Residuals ($\perthousand$)", fontsize=13)
ax22.set_xlim(1985, 2025)

# Lower-right block: no Delta14CO2 data are available for MLO
for ax in (ax13, ax23):
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

d14c_note_ax.patch.set_alpha(0.0)
d14c_note_ax.set_frame_on(False)
d14c_note_ax.set_xticks([])
d14c_note_ax.set_yticks([])
d14c_note_ax.text(
    0.5,
    0.5,
    r"No MLO $\Delta^{14}$CO$_2$ data",
    transform=d14c_note_ax.transAxes,
    ha="center",
    va="center",
    fontsize=14,
    color="0.35",
)

# Axis formatting
for ax in (ax11, ax12, ax21, ax22):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=12, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)

plt.setp(ax11.get_xticklabels(), visible=False)
plt.setp(ax12.get_xticklabels(), visible=False)

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
