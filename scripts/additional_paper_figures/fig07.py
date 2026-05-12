"""
Plot the CO2 low-frequency component, its derivative, and the Nino 3.4 index.

Panels:
- Upper: low-frequency component ell(t) for IZO and MLO CO2.
- Middle: first derivative d ell / dt for IZO and MLO CO2.
- Lower: monthly Nino 3.4 sea-surface temperature anomaly index.

The low-frequency components and their derivatives are computed from joint
posterior Monte Carlo draws saved by the selected CO2 MCMC runs. Shaded regions
show the 16th-84th percentile range. The script also computes Pearson
correlations between Nino 3.4 and d ell / dt as a function of lag.

The script reads:
- 'samples_for_MC.txt' from the selected CO2 MCMC runs.
- 'enso_index.txt', containing the monthly Nino 3.4 anomaly index.

The result is stored in:
results_and_plots/comparisons/fig07_correlation_nino34/fig07.png
"""



# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

from functions.paths import find_project_root, run_results_directory, comparison_directory
from scripts.additional_paper_figures.paper_figure_calculations import (
    build_monthly_midpoint_grid,
    compute_low_frequency_band,
    load_nino34_anomaly,
    map_monthly_series_to_grid,
    pearson_correlation_by_lag,
)



# -------------------------------------------------------
# ------------------- SELECTED RUNS ---------------------
# -------------------------------------------------------

# ---------- CO2 IZO ----------
co2_izo_site_acronym = "IZO"
co2_izo_data_frequency = "monthly"
co2_izo_polynomial_degree = 2
co2_izo_include_slow_harmonics = True
co2_izo_base_period_slow_harmonics = 30
co2_izo_slow_harmonics = [2,3,4,7,8]
co2_izo_timezero = 1985.0

# ---------- CO2 MLO ----------
co2_mlo_site_acronym = "MLO"
co2_mlo_data_frequency = "monthly"
co2_mlo_polynomial_degree = 2
co2_mlo_include_slow_harmonics = True
co2_mlo_base_period_slow_harmonics = 30
co2_mlo_slow_harmonics = [2,3,4,7,8]
co2_mlo_timezero = 1985.0



# -------------------------------------------------------
# ---------------- GRID CONFIGURATION -------------------
# -------------------------------------------------------

start_year = 1985
end_year = 2024

xlim_min = 1985
xlim_max = 2025

max_lag_months = 60



def set_symmetric_ylim(ax, curves, extra_fraction=0.08):
    """
    Set symmetric y limits around zero from the plotted curves.
    """
    max_abs = max(np.nanmax(np.abs(curve)) for curve in curves)

    if max_abs == 0:
        max_abs = 1.0

    ax.set_ylim(-(1.0 + extra_fraction) * max_abs, (1.0 + extra_fraction) * max_abs)



# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

project_root = find_project_root(__file__)

co2_izo_results_dir = run_results_directory(project_root, "co2", co2_izo_site_acronym, co2_izo_data_frequency, co2_izo_include_slow_harmonics, co2_izo_base_period_slow_harmonics, co2_izo_slow_harmonics, polynomial_degree=co2_izo_polynomial_degree)
co2_mlo_results_dir = run_results_directory(project_root, "co2", co2_mlo_site_acronym, co2_mlo_data_frequency, co2_mlo_include_slow_harmonics, co2_mlo_base_period_slow_harmonics, co2_mlo_slow_harmonics, polynomial_degree=co2_mlo_polynomial_degree)

co2_izo_samples_path = os.path.join(co2_izo_results_dir, "samples_for_MC.txt")
co2_mlo_samples_path = os.path.join(co2_mlo_results_dir, "samples_for_MC.txt")

climatic_indexes_dir = os.path.join(project_root, "data", "climatic_indexes")
nino34_path = os.path.join(climatic_indexes_dir, "enso_index.txt")

plot_dir = comparison_directory(project_root, "fig07_correlation_nino34")
os.makedirs(plot_dir, exist_ok=True)

output_path = os.path.join(plot_dir, "fig07.png")



# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Build monthly grid")

dates, decimal_years, grid_keys = build_monthly_midpoint_grid(start_year, end_year)

print(f"Monthly grid from {start_year}-01 to {end_year}-12")
print("-------------------------------------------------------")



print("Step 2: Load posterior samples")

co2_izo_samples = np.loadtxt(co2_izo_samples_path, comments="#", ndmin=2)
co2_mlo_samples = np.loadtxt(co2_mlo_samples_path, comments="#", ndmin=2)

print(f"Loaded IZO CO2 samples from: {co2_izo_samples_path}")
print(f"Loaded MLO CO2 samples from: {co2_mlo_samples_path}")
print("-------------------------------------------------------")



print("Step 3: Compute low-frequency components and derivatives")

izo_p16_lf, izo_p50_lf, izo_p84_lf, izo_p16_dlf, izo_p50_dlf, izo_p84_dlf = compute_low_frequency_band(
    co2_izo_samples,
    decimal_years,
    co2_izo_timezero,
    co2_izo_polynomial_degree,
    co2_izo_include_slow_harmonics,
    co2_izo_base_period_slow_harmonics,
    co2_izo_slow_harmonics,
)

mlo_p16_lf, mlo_p50_lf, mlo_p84_lf, mlo_p16_dlf, mlo_p50_dlf, mlo_p84_dlf = compute_low_frequency_band(
    co2_mlo_samples,
    decimal_years,
    co2_mlo_timezero,
    co2_mlo_polynomial_degree,
    co2_mlo_include_slow_harmonics,
    co2_mlo_base_period_slow_harmonics,
    co2_mlo_slow_harmonics,
)

print("-------------------------------------------------------")



print("Step 4: Load Nino 3.4 anomaly index")

n34_years, n34_months, n34_decimal_years, n34 = load_nino34_anomaly(nino34_path, decimal_years[0], decimal_years[-1])
n34_grid = map_monthly_series_to_grid(n34_years, n34_months, n34, grid_keys)

print(f"Loaded Nino 3.4 data from: {nino34_path}")
print("-------------------------------------------------------")



print("Step 5: Compute lagged Pearson correlations")

lags, r_izo, best_lag_izo, best_r_izo = pearson_correlation_by_lag(izo_p50_dlf, n34_grid, max_lag_months)
_, r_mlo, best_lag_mlo, best_r_mlo = pearson_correlation_by_lag(mlo_p50_dlf, n34_grid, max_lag_months)

print(f"IZO d ell / dt vs Nino 3.4: best lag = {best_lag_izo} months; r = {best_r_izo:.3f}")
print(f"MLO d ell / dt vs Nino 3.4: best lag = {best_lag_mlo} months; r = {best_r_mlo:.3f}")
print("Lag convention: positive lag means Nino 3.4 leads d ell / dt.")
print("-------------------------------------------------------")



print("Step 6: Plot the figure")

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
fig.subplots_adjust(hspace=0.10)

izo_line_style = dict(color="k", linewidth=1.2, zorder=5)
izo_band_style = dict(color="0.45", alpha=0.30, linewidth=0, zorder=2)

mlo_line_style = dict(color="r", linewidth=1.2, alpha=0.50, zorder=4)
mlo_band_style = dict(color="r", alpha=0.15, linewidth=0, zorder=1)

# Upper panel: low-frequency component
ax1.fill_between(decimal_years, mlo_p16_lf, mlo_p84_lf, **mlo_band_style)
ax1.plot(decimal_years, mlo_p50_lf, **mlo_line_style, label="MLO")
ax1.fill_between(decimal_years, izo_p16_lf, izo_p84_lf, **izo_band_style)
ax1.plot(decimal_years, izo_p50_lf, **izo_line_style, label="IZO")
ax1.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
ax1.set_ylabel(r"$\ell(t)$ CO$_2$ (ppm)", fontsize=16)
set_symmetric_ylim(ax1, [izo_p16_lf, izo_p84_lf, mlo_p16_lf, mlo_p84_lf])

# Middle panel: low-frequency derivative
ax2.fill_between(decimal_years, mlo_p16_dlf, mlo_p84_dlf, **mlo_band_style)
ax2.plot(decimal_years, mlo_p50_dlf, **mlo_line_style)
ax2.fill_between(decimal_years, izo_p16_dlf, izo_p84_dlf, **izo_band_style)
ax2.plot(decimal_years, izo_p50_dlf, **izo_line_style)
ax2.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
ax2.set_ylabel(r"$\mathrm{d}\ell/\mathrm{d}t$ (ppm yr$^{-1}$)", fontsize=16)
set_symmetric_ylim(ax2, [izo_p16_dlf, izo_p84_dlf, mlo_p16_dlf, mlo_p84_dlf])

# Lower panel: Nino 3.4 anomaly
ax3.plot(n34_decimal_years, n34, color="k", linewidth=1.0)
ax3.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
ax3.set_xlabel("Year", fontsize=16)
ax3.set_ylabel(r"Nino 3.4 anomaly ($^\circ$C)", fontsize=16)

# Axis formatting
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=14, length=6, width=1)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.8)
    ax.set_xlim(xlim_min, xlim_max)

ax1.legend(loc="best")

fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.1)

plt.show()
plt.close(fig)

print(f"Saved in '{output_path}'")
print("-------------------------------------------------------")
